"""
Main launcher for the Drone Control and Display System.
This script can run both components together or separately.
"""

import asyncio
import threading
import argparse
import sys
import signal
import logging

from configs.airsim_config import DroneConfig, ControlState # type: ignore
from utils.util import event_bus # type: ignore
from utils.logger import LOGGER # type: ignore

from task.display import DisplayApplication # type: ignore
from task.control import ControlApplication # type: ignore




class DroneSystem:
    """
    Main application that coordinates both control and display components.
    """
    
    def __init__(self, config: DroneConfig, run_display=True, run_control=True):
        self.config = config
        self.run_display = run_display
        self.run_control = run_control
        
        # Set up shared state and event bus
        self.state = ControlState(gimbal_pitch=config.initial_pitch)
        
        # Display components
        self.display_app = None
        self.display_thread = None
        
        # Control components
        self.control_app = None
        self.control_task = None
        
        # Exit management
        self.exit_event = threading.Event()
        
        # Configure signal handling
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            LOGGER.info(f"Received signal {sig}, initiating shutdown...")
            self.state.exit_flag = True
            self.exit_event.set()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _start_display(self):
        """Start the display component in a separate thread"""
        if not self.run_display:
            return
        
        LOGGER.info("Starting display component...")
        
        def display_thread_runner():
            try:
                LOGGER.info("Initializing display application...")
                self.display_app = DisplayApplication(self.config, self.state, event_bus)
                
                # Run the display application
                exit_code = self.display_app.run()
                LOGGER.info(f"Display application exited with code {exit_code}")
                
                # Signal exit if display component exits
                self.state.exit_flag = True
                self.exit_event.set()
                
            except Exception as e:
                LOGGER.error(f"Error in display thread: {e}")
                self.state.exit_flag = True
                self.exit_event.set()
        
        # Create and start display thread
        self.display_thread = threading.Thread(
            target=display_thread_runner,
            name="DisplayThread",
            daemon=True
        )
        self.display_thread.start()
        LOGGER.info("Display thread started.")
    
    async def _start_control(self):
        """Start the control component"""
        if not self.run_control:
            return
        
        LOGGER.info("Starting control component...")
        
        try:
            self.control_app = ControlApplication(self.config, self.state, event_bus)
            self.control_task = asyncio.create_task(self.control_app.run())
            
            # Wait for control task to complete or exit event
            done, pending = await asyncio.wait(
                [self.control_task, asyncio.create_task(self._wait_for_exit())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            LOGGER.info("Control component stopped.")
            
        except Exception as e:
            LOGGER.error(f"Error starting control component: {e}")
            self.state.exit_flag = True
            self.exit_event.set()
    
    async def _wait_for_exit(self):
        """Async task to wait for exit event"""
        while not self.exit_event.is_set() and not self.state.exit_flag:
            await asyncio.sleep(0.1)
        return True
    
    async def run(self):
        """Run the drone system (both display and control components)"""
        try:
            # Start display component in separate thread if enabled
            if self.run_display:
                self._start_display()
                # Give display a moment to initialize
                await asyncio.sleep(0.5)
            
            # Start control component if enabled
            if self.run_control:
                await self._start_control()
            # If only running display, just wait for exit
            else:
                await self._wait_for_exit()
            
            return 0
            
        except KeyboardInterrupt:
            LOGGER.info("System interrupted by user.")
            return 130
        except Exception as e:
            LOGGER.error(f"Error in drone system: {e}")
            return 1
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources before exit"""
        LOGGER.info("Cleaning up system resources...")
        
        # Set exit flags
        self.state.exit_flag = True
        self.exit_event.set()
        
        # Clean up control component
        if self.control_app:
            LOGGER.info("Cleaning up control application...")
            await self.control_app.cleanup()
        
        # Wait for display thread to finish
        if self.display_thread and self.display_thread.is_alive():
            LOGGER.info("Waiting for display thread to finish...")
            self.display_thread.join(timeout=3.0)
            if self.display_thread.is_alive():
                LOGGER.warning("Display thread did not exit cleanly.")
        
        LOGGER.info("System cleanup complete.")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="AirSim Drone Control and Display System")
    
    # Component selection
    parser.add_argument(
        "--display-only",
        action="store_true",
        help="Run only the display component"
    )
    parser.add_argument(
        "--control-only",
        action="store_true",
        help="Run only the control component"
    )
    
    # Connection parameters
    parser.add_argument(
        "--ip", 
        type=str, 
        default="172.19.160.1", 
        help="IP address of AirSim server"
    )
    parser.add_argument(
        "--mavsdk-port",
        type=int,
        default=14540,
        help="UDP port for MAVSDK connection"
    )
    parser.add_argument(
        "--vehicle", 
        type=str, 
        default="PX4", 
        help="Vehicle name in AirSim"
    )
    
    # Camera parameters
    parser.add_argument(
        "--camera", 
        type=str, 
        default="front_center", 
        help="Primary camera name"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Request specific image width"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Request specific image height"
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=960,
        help="Output display width"
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=540,
        help="Output display height"
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Disable image resizing"
    )
    
    # Drone parameters
    parser.add_argument(
        "--takeoff-alt",
        type=float,
        default=5.0,
        help="Takeoff altitude in meters"
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=5.0,
        help="Maximum drone speed in m/s"
    )
    
    # Other options
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    
    # Determine which components to run
    run_display = not args.control_only
    run_control = not args.display_only
    
    # Create configuration
    config = DroneConfig()
    config.connection.ip_address = args.ip
    config.connection.vehicle_name = args.vehicle
    config.system_address = f"udp://:{args.mavsdk_port}"
    config.default_takeoff_altitude = args.takeoff_alt
    config.max_speed = args.max_speed
    config.camera.camera_name = args.camera
    config.camera.width = args.width
    config.camera.height = args.height
    config.display.output_width = args.output_width
    config.display.output_height = args.output_height
    config.display.resize_output = not args.no_resize
    
    # Initialize and run the system
    system = DroneSystem(config, run_display, run_control)
    return await system.run()


if __name__ == "__main__":
    # Run the main function
    loop = asyncio.get_event_loop()
    
    try:
        exit_code = loop.run_until_complete(main())
    except KeyboardInterrupt:
        LOGGER.info("Program interrupted by user.")
        exit_code = 130
    finally:
        # Clean up pending tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        
        # Run loop until tasks are cancelled
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()
    
    # Exit with appropriate code
    sys.exit(exit_code)