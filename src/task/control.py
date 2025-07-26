import asyncio
import threading
import time
import tkinter as tk
import numpy as np
import sys
import logging
from typing import Dict, Optional, Any

from configs.airsim_config import DroneConfig, ControlState # type: ignore
from utils.util import EventBus, event_bus # type: ignore
from utils.logger import LOGGER # type: ignore


from mavsdk import System  # type: ignore
from mavsdk.telemetry import FlightMode  # type: ignore
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed  # type: ignore



# =============================================================================
# DRONE CONTROLLER
# =============================================================================


class DroneController:
    """Controller for MAVSDK drone operations with improved async handling"""

    def __init__(self, config: DroneConfig, event_bus: EventBus):
        self.config = config
        self.event_bus = event_bus
        self.drone = System()
        self._telemetry_task: Optional[Any] = None

        # Events to monitor drone state
        self.connected_event = asyncio.Event()
        self.armed_event = asyncio.Event()
        self.in_air_event = asyncio.Event()
        self.offboard_started_event = asyncio.Event()

        # Subscribe to relevant events
        event_bus.subscribe("drone_exit", self._handle_exit_request)

    async def _handle_exit_request(self, data):
        """Handle exit request from other components"""
        LOGGER.info("Drone controller received exit request")
        # Start landing sequence
        await self.land_drone()

    async def is_connected(self) -> bool:
        """Check if drone is connected"""
        connected = False
        async for state in self.drone.core.connection_state():
            connected = state.is_connected
            break
        return connected

    async def connect_drone(self) -> bool:
        """Connect to drone using MAVSDK"""            
        LOGGER.info(f"Connecting to drone at {self.config.system_address}...")
        try:
            await self.drone.connect(system_address=self.config.system_address)

            LOGGER.info("Waiting for drone connection...")
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    LOGGER.info("Drone connected!")
                    self.connected_event.set()
                    self.event_bus.publish("drone_connected", True)
                    return True

            LOGGER.error("Drone connection loop finished unexpectedly.")
            return False

        except asyncio.TimeoutError:
            LOGGER.error(
                f"Drone connection timed out after {self.config.connection_timeout} seconds"
            )
            return False
        except Exception as e:
            LOGGER.error(f"Drone connection error: {e}")
            return False

    async def check_drone_health(self) -> bool:
        """Check drone health status"""            
        if not await self.is_connected():
            LOGGER.error("Health check failed: Drone not connected.")
            return False

        try:
            LOGGER.info("Checking drone health...")
            async for health in self.drone.telemetry.health():
                is_healthy = (
                    health.is_gyrometer_calibration_ok
                    or health.is_accelerometer_calibration_ok
                    or health.is_magnetometer_calibration_ok
                )
                is_armable = health.is_armable

                LOGGER.info(
                    f"Health: GyroOK={health.is_gyrometer_calibration_ok}, "
                    f"AccelOK={health.is_accelerometer_calibration_ok}, "
                    f"MagOK={health.is_magnetometer_calibration_ok}, "
                    f"LocalPosOK={health.is_local_position_ok}, "
                    f"GlobalPosOK={health.is_global_position_ok}, "
                    f"Armable={is_armable}"
                )

                self.event_bus.publish(
                    "drone_health",
                    {
                        "is_healthy": is_healthy,
                        "is_armable": is_armable,
                        "health_data": {
                            "gyro_ok": health.is_gyrometer_calibration_ok,
                            "accel_ok": health.is_accelerometer_calibration_ok,
                            "mag_ok": health.is_magnetometer_calibration_ok,
                            "local_pos_ok": health.is_local_position_ok,
                            "global_pos_ok": health.is_global_position_ok,
                        },
                    },
                )

                if is_healthy and is_armable:
                    LOGGER.info("Drone health OK and armable!")
                    return True
                if is_healthy and not is_armable:
                    LOGGER.warning(
                        "Drone sensors OK, but currently not armable. Check pre-arm conditions."
                    )
                    return True
                await asyncio.sleep(1)

            LOGGER.error("Health telemetry stream ended unexpectedly.")
            return False

        except asyncio.TimeoutError:
            LOGGER.error("Timeout waiting for health telemetry.")
            return False
        except Exception as e:
            LOGGER.error(f"Error during health check: {e}")
            return False

    async def arm_drone(self) -> bool:
        """Arm the drone"""            
        if not await self.is_connected():
            LOGGER.error("Arming failed: Drone not connected.")
            return False
        try:
            LOGGER.info("Arming drone...")
            await self.drone.action.arm()
            LOGGER.info("Drone armed!")
            self.armed_event.set()
            self.event_bus.publish("drone_armed", True)
            return True
        except Exception as e:
            LOGGER.error(f"Arming failed: {e}")
            return False

    async def takeoff_drone(self) -> bool:
        """Takeoff to specified altitude"""
        if not await self.is_connected():
            LOGGER.error("Takeoff failed: Drone not connected.")
            return False

        altitude = self.config.default_takeoff_altitude
        try:
            LOGGER.info(f"Setting takeoff altitude to {altitude} meters...")
            await self.drone.action.set_takeoff_altitude(altitude)

            LOGGER.info("Commanding takeoff...")
            await self.drone.action.takeoff()

            # Monitor altitude to confirm takeoff
            takeoff_timeout = 30  # seconds
            start_time = asyncio.get_event_loop().time()
            LOGGER.info("Monitoring takeoff progress...")

            async for position in self.drone.telemetry.position():
                current_alt = position.relative_altitude_m
                LOGGER.info(f"Current altitude: {current_alt:.2f} m")

                # Publish altitude info for GUI
                self.event_bus.publish("drone_altitude", current_alt)

                if current_alt >= altitude * 0.90:  # Allow slightly lower
                    LOGGER.info("Target altitude reached!")
                    self.in_air_event.set()
                    self.event_bus.publish("drone_in_air", True)
                    return True

                if asyncio.get_event_loop().time() - start_time > takeoff_timeout:
                    LOGGER.warning(
                        f"Takeoff timeout after {takeoff_timeout}s at altitude {current_alt:.2f}m. Proceeding anyway."
                    )
                    self.in_air_event.set()
                    self.event_bus.publish("drone_in_air", True)
                    return True  # Proceed even if full altitude wasn't confirmed

            LOGGER.warning(
                "Position telemetry stream ended before confirming takeoff altitude."
            )
            return True

        except asyncio.TimeoutError:
            LOGGER.error("Timeout during takeoff sequence (waiting for telemetry).")
            return False
        except Exception as e:
            LOGGER.error(f"Takeoff failed: {e}")
            return False

    async def land_drone(self):
        """Land the drone"""
            
        if not await self.is_connected():
            LOGGER.warning("Cannot land: Drone not connected.")
            return

        try:
            # Check if already landed/disarmed
            is_armed = await self.drone.telemetry.armed().__aiter__().__anext__()
            in_air = await self.drone.telemetry.in_air().__aiter__().__anext__()

            if not is_armed or not in_air:
                LOGGER.info("Drone already on ground or disarmed. No landing needed.")
                self.in_air_event.clear()
                self.event_bus.publish("drone_landed", True)
                return

            LOGGER.info("Commanding drone to land...")
            await self.drone.action.land()
            LOGGER.info("Drone landing command sent. Monitoring landing...")
            self.event_bus.publish("drone_landing", True)

            # Monitor until disarmed or timeout
            land_timeout = 60  # seconds
            start_time = asyncio.get_event_loop().time()

            async for armed_status in self.drone.telemetry.armed():
                if not armed_status:
                    LOGGER.info("Drone disarmed. Landing complete.")
                    self.in_air_event.clear()
                    self.armed_event.clear()
                    self.event_bus.publish("drone_landed", True)
                    return  # Success

                if asyncio.get_event_loop().time() - start_time > land_timeout:
                    LOGGER.warning(
                        f"Landing timeout after {land_timeout}s. Drone might still be armed."
                    )
                    return  # Timeout

            LOGGER.warning("Armed telemetry stream ended before confirming disarm.")

        except asyncio.TimeoutError:
            LOGGER.error("Timeout during landing sequence (waiting for telemetry).")
        except Exception as e:
            LOGGER.error(f"Landing error: {e}")

    async def _telemetry_updater(self, state: ControlState):
        """Continuously updates telemetry in the background."""            
        LOGGER.info("Starting telemetry updater task.")
        try:
            # Subscribe to multiple streams concurrently
            async def subscribe_position():
                async for position in self.drone.telemetry.position():
                    if state.exit_flag:
                        break

                    position_data = {
                        "latitude_deg": position.latitude_deg,
                        "longitude_deg": position.longitude_deg,
                        "absolute_altitude_m": position.absolute_altitude_m,
                        "relative_altitude_m": position.relative_altitude_m,
                    }

                    state.last_position = position_data
                    self.event_bus.publish("drone_position", position_data)

            async def subscribe_attitude():
                async for attitude in self.drone.telemetry.attitude_euler():
                    if state.exit_flag:
                        break

                    attitude_data = {
                        "roll_deg": attitude.roll_deg,
                        "pitch_deg": attitude.pitch_deg,
                        "yaw_deg": attitude.yaw_deg,
                    }

                    state.last_attitude = attitude_data
                    self.event_bus.publish("drone_attitude", attitude_data)

            # Add flight mode subscription
            async def subscribe_flight_mode():
                async for flight_mode in self.drone.telemetry.flight_mode():
                    if state.exit_flag:
                        break
                    self.event_bus.publish("drone_flight_mode", flight_mode)

            await asyncio.gather(
                subscribe_position(), subscribe_attitude(), subscribe_flight_mode()
            )

        except asyncio.CancelledError:
            LOGGER.info("Telemetry updater task cancelled.")
        except Exception as e:
            LOGGER.error(f"Error in telemetry updater: {e}")
        finally:
            LOGGER.info("Telemetry updater task finished.")

    async def start_telemetry_updates(self, state: ControlState):
        """Start the background telemetry update task."""
            
        if self._telemetry_task and not self._telemetry_task.done():
            LOGGER.warning("Telemetry task already running.")
            return
        self._telemetry_task = asyncio.create_task(self._telemetry_updater(state))

    async def stop_telemetry_updates(self):
        """Stop the background telemetry update task."""            
        if self._telemetry_task and not self._telemetry_task.done():
            LOGGER.info("Stopping telemetry updater task...")
            self._telemetry_task.cancel()
            try:
                await asyncio.wait_for(self._telemetry_task, timeout=1.0)
            except asyncio.TimeoutError:
                LOGGER.warning("Timeout waiting for telemetry task to cancel.")
            except asyncio.CancelledError:
                pass  # Expected
            LOGGER.info("Telemetry updater task stopped.")
        self._telemetry_task = None

    async def run_manual_control_loop(self, state: ControlState):
        """Run the main drone control loop. Reads state updated by GUI."""
        if not self.is_connected():
            LOGGER.warning("MAVSDK not available. Running in simulation mode only.")
            # Run a simulated control loop that just processes the state
            try:
                # Start simulated telemetry updates
                LOGGER.info("Starting simulated telemetry...")
                
                async def publish_simulated_telemetry():
                    while not state.exit_flag:
                        # Simulate position updates
                        position_data = {
                            "latitude_deg": 0.0,
                            "longitude_deg": 0.0,
                            "absolute_altitude_m": 100.0,
                            "relative_altitude_m": 5.0,
                        }
                        state.last_position = position_data
                        self.event_bus.publish("drone_position", position_data)
                        
                        # Simulate attitude updates
                        attitude_data = {
                            "roll_deg": 0.0,
                            "pitch_deg": 0.0,
                            "yaw_deg": 0.0,
                        }
                        state.last_attitude = attitude_data
                        self.event_bus.publish("drone_attitude", attitude_data)
                        
                        # Simulate flight mode
                        self.event_bus.publish("drone_flight_mode", "OFFBOARD")
                        
                        await asyncio.sleep(0.2)
                
                # Start simulated telemetry
                telemetry_task = asyncio.create_task(publish_simulated_telemetry())
                
                # Main control loop - just wait for exit
                while not state.exit_flag:
                    # Output current control state for debugging
                    if hasattr(state, "velocity_forward") and (
                        state.velocity_forward != 0 or
                        state.velocity_right != 0 or
                        state.velocity_down != 0 or
                        state.yawspeed != 0
                    ):
                        LOGGER.debug(
                            f"Simulated Control: fwd={state.velocity_forward:.1f}, "
                            f"right={state.velocity_right:.1f}, down={state.velocity_down:.1f}, "
                            f"yaw={state.yawspeed:.1f}"
                        )
                    
                    await asyncio.sleep(0.1)
                
                # Clean up
                telemetry_task.cancel()
                try:
                    await telemetry_task
                except asyncio.CancelledError:
                    pass
                
                return
                
            except asyncio.CancelledError:
                LOGGER.info("Simulated control loop cancelled.")
                return
            except Exception as e:
                LOGGER.error(f"Error in simulated control loop: {e}")
                state.exit_flag = True
                return
            
        # Real drone control
        if not await self.is_connected():
            LOGGER.error("Cannot run control loop: drone not connected")
            return

        LOGGER.info("Preparing for flight control...")
        offboard_started = False

        try:
            # Start background telemetry updates
            await self.start_telemetry_updates(state)

            # --- Start Offboard Mode ---
            LOGGER.info("Priming Offboard setpoints...")
            # Send initial setpoints before starting
            initial_velocity_cmd = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            await self.drone.offboard.set_velocity_body(initial_velocity_cmd)
            await asyncio.sleep(0.1)  # Short delay

            try:
                LOGGER.info("Starting Offboard mode...")
                await self.drone.offboard.start()
                offboard_started = True
                self.offboard_started_event.set()
                self.event_bus.publish("offboard_started", True)
                LOGGER.info("Offboard mode started successfully.")
            except OffboardError as e:
                LOGGER.error(f"Failed to start Offboard mode: {e}")
                LOGGER.info("Attempting to land.")
                await self.land_drone()
                return  # Cannot continue without Offboard

            # --- Main Control Loop ---
            LOGGER.info("\n--- Flight Control Active ---")
            LOGGER.info("Use Tkinter GUI window for control.")
            LOGGER.info("Press ESC in GUI to stop.\n")

            last_print_time = time.monotonic()
            control_error_count = 0
            max_control_errors = 10
            offboard_check_interval = 2.0  # Check if still in offboard every 2s
            last_offboard_check_time = time.monotonic()

            while not state.exit_flag:
                current_time_monotonic = time.monotonic()

                # --- Check if still in Offboard Mode periodically ---
                if (
                    current_time_monotonic - last_offboard_check_time
                    > offboard_check_interval
                ):
                    try:
                        current_mode = (
                            await self.drone.telemetry.flight_mode()
                            .__aiter__()
                            .__anext__()
                        )
                        if current_mode != FlightMode.OFFBOARD:  # Use mavsdk enum
                            LOGGER.error(
                                f"Drone left Offboard mode! Current mode: {current_mode}. Landing."
                            )
                            state.exit_flag = True
                            self.event_bus.publish("offboard_lost", current_mode)
                            break  # Exit control loop
                        last_offboard_check_time = current_time_monotonic
                    except Exception as mode_err:
                        LOGGER.warning(f"Could not verify flight mode: {mode_err}")

                # --- Send Drone Velocity Command ---
                try:
                    await self.drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(
                            state.velocity_forward,
                            state.velocity_right,
                            state.velocity_down,
                            state.yawspeed,
                        )
                    )
                    control_error_count = 0  # Reset error count on success

                except OffboardError as e:
                    control_error_count += 1
                    LOGGER.warning(
                        f"Offboard command error ({control_error_count}/{max_control_errors}): {e}"
                    )
                    if control_error_count >= max_control_errors:
                        LOGGER.error(
                            "Too many consecutive Offboard errors, stopping control loop."
                        )
                        state.exit_flag = True  # Signal exit
                        self.event_bus.publish(
                            "offboard_error", f"Too many errors: {e}"
                        )
                    await asyncio.sleep(0.1)  # Wait before retrying after error
                    continue  # Skip rest of loop iteration
                except Exception as e:
                    control_error_count += 1
                    LOGGER.warning(
                        f"Control command error ({control_error_count}/{max_control_errors}): {e}"
                    )
                    if control_error_count >= max_control_errors:
                        LOGGER.error(
                            "Too many consecutive control errors, stopping control loop."
                        )
                        state.exit_flag = True
                        self.event_bus.publish("control_error", f"Too many errors: {e}")
                    await asyncio.sleep(0.1)
                    continue

                # --- Periodic Logging ---
                if current_time_monotonic - last_print_time >= 1.0:
                    log_msg = (
                        f"Vel (F,R,D,Y): {state.velocity_forward:+.1f}, {state.velocity_right:+.1f}, "
                        f"{state.velocity_down:+.1f}, {state.yawspeed:+.1f} | "
                        f"Gimbal (P,R,Y): {state.gimbal_pitch:+.1f}, {state.gimbal_roll:+.1f}, {state.gimbal_yaw:+.1f}"
                    )
                    LOGGER.debug(log_msg)  # Use debug level for frequent logs
                    last_print_time = current_time_monotonic

                await asyncio.sleep(0.02)

        except asyncio.CancelledError:
            LOGGER.info("Control loop task cancelled.")
        except Exception as e:
            LOGGER.error(f"Unhandled error during flight control: {e}")
            state.exit_flag = True  # Ensure exit on error
            self.event_bus.publish("drone_control_error", str(e))
        finally:
            LOGGER.info("Exiting control loop and cleaning up...")

            # Stop telemetry updates first
            await self.stop_telemetry_updates()

            # Stop Offboard mode if it was started
            if offboard_started:
                LOGGER.info("Stopping Offboard mode...")
                try:
                    # Send zero velocity before stopping
                    await self.drone.offboard.set_velocity_body(
                        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                    )
                    await asyncio.sleep(0.1)
                    await self.drone.offboard.stop()
                    self.offboard_started_event.clear()
                    self.event_bus.publish("offboard_stopped", True)
                    LOGGER.info("Offboard mode stopped.")
                except OffboardError as e:
                    LOGGER.error(f"Error stopping offboard mode: {e}")
                except Exception as e:
                    LOGGER.error(f"Unexpected error stopping offboard mode: {e}")

            # Landing is now handled in the main finally block
            LOGGER.info("Control loop cleanup complete.")


# =============================================================================
# CONTROL GUI
# =============================================================================


class DroneControlGUI:
    """Enhanced GUI for drone and gimbal control without embedded camera view"""

    def __init__(
        self, root: tk.Tk, config: DroneConfig, state: ControlState, event_bus: EventBus
    ):
        self.root = root
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self.active_key_labels: Dict = {}

        # Subscribe to events
        event_bus.subscribe("drone_position", self._handle_position_update)
        event_bus.subscribe("drone_attitude", self._handle_attitude_update)
        event_bus.subscribe("drone_flight_mode", self._handle_flight_mode_update)
        event_bus.subscribe("drone_connected", self._handle_connection_status)
        event_bus.subscribe("drone_armed", self._handle_armed_status)
        event_bus.subscribe("drone_in_air", self._handle_in_air_status)
        event_bus.subscribe("offboard_started", self._handle_offboard_status)
        event_bus.subscribe("offboard_error", self._handle_error_message)
        event_bus.subscribe("control_error", self._handle_error_message)
        event_bus.subscribe("drone_control_error", self._handle_error_message)

        self.root.title("Drone Control Interface")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")

        # Fonts
        self.title_font = ("Arial", 16, "bold")
        self.header_font = ("Arial", 12, "bold")
        self.text_font = ("Arial", 10)
        self.button_font = ("Arial", 11, "bold")

        # Colors
        self.title_color = "#2C3E50"
        self.status_active_color = "#27AE60"
        self.status_warning_color = "#F39C12"
        self.status_error_color = "#E74C3C"

        self.main_container = tk.Frame(root, bg="#f0f0f0", padx=15, pady=15)
        self.main_container.pack(fill="both", expand=True)

        self.create_header()
        self.create_main_layout()
        self.create_control_reference()
        self.create_status_section()
        self.create_visual_indicators()
        self.create_control_buttons()

        # Status message label at the bottom
        self.status_message_label = tk.Label(
            self.main_container, text="", font=("Arial", 10), bg="#f0f0f0", fg="#333333"
        )
        self.status_message_label.pack(fill="x", pady=(5, 0))

        self.last_key_process_time = time.monotonic()
        self.last_telemetry_update_time = time.monotonic()

        # Ensure GUI has focus initially
        self.root.focus_force()

        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.root.protocol("WM_DELETE_WINDOW", self.exit_control)

        # Start the UI update loop
        self._update_gui_loop_id = self.root.after(50, self.update_gui_loop)
        
        # Update status
        self.update_status_message("Control interface ready. Waiting for drone connection.")

    def _handle_position_update(self, data):
        """Handle position update event"""
        if hasattr(self, "altitude_label"):
            alt = data.get("relative_altitude_m", 0.0)
            self.altitude_label.config(text=f"{alt:.1f} m")

    def _handle_attitude_update(self, data):
        """Handle attitude update event"""
        if hasattr(self, "attitude_label"):
            roll = data.get("roll_deg", 0.0)
            pitch = data.get("pitch_deg", 0.0)
            yaw = data.get("yaw_deg", 0.0)
            self.attitude_label.config(
                text=f"R:{roll:+.1f}, P:{pitch:+.1f}, Y:{yaw:+.1f} deg"
            )

    def _handle_flight_mode_update(self, mode):
        """Handle flight mode update event"""
        self.update_status_message(f"Flight mode: {mode}", "info")

    def _handle_connection_status(self, connected):
        """Handle connection status update event"""
        if connected and hasattr(self, "conn_status"):
            self.conn_status.config(text="● CONNECTED", fg=self.status_active_color)

    def _handle_armed_status(self, armed):
        """Handle armed status update event"""
        if armed:
            self.update_status_message("Drone armed", "info")

    def _handle_in_air_status(self, in_air):
        """Handle in-air status update event"""
        if in_air:
            self.update_status_message("Drone in air", "info")
        else:
            self.update_status_message("Drone landed", "info")

    def _handle_offboard_status(self, offboard):
        """Handle offboard status update event"""
        if offboard:
            self.update_status_message("Offboard mode active", "info")
        else:
            self.update_status_message("Offboard mode inactive", "info")

    def _handle_error_message(self, message):
        """Handle error message event"""
        self.update_status_message(f"Error: {message}", "error")

    def update_status_message(self, message: str, level: str = "info"):
        """Updates the status message label at the bottom."""
        if not self.root.winfo_exists():
            return
        color = "#333333"  # Default info color
        if level == "warning":
            color = self.status_warning_color
        elif level == "error":
            color = self.status_error_color
        self.status_message_label.config(text=message, fg=color)

    def create_header(self):
        """Create the header with title and connection status"""
        header_frame = tk.Frame(self.main_container, bg="#f0f0f0")
        header_frame.pack(fill="x", pady=(0, 10))
        title_label = tk.Label(
            header_frame,
            text="Drone Control System",
            font=self.title_font,
            fg=self.title_color,
            bg="#f0f0f0",
        )
        title_label.pack(side=tk.LEFT, pady=5)
        self.conn_status = tk.Label(
            header_frame,
            text="● INITIALIZING",
            font=self.text_font,
            fg=self.status_warning_color,
            bg="#f0f0f0",
        )
        self.conn_status.pack(side=tk.RIGHT, pady=5, padx=10)

    def create_main_layout(self):
        """Create the main layout with the controls section"""
        self.controls_frame = tk.Frame(self.main_container, bg="#f0f0f0")
        self.controls_frame.pack(fill="both", expand=True)

    def create_control_reference(self):
        """Create the control reference panel"""
        controls_frame = tk.LabelFrame(
            self.controls_frame,
            text="Control Reference",
            font=self.header_font,
            bg="#f0f0f0",
            fg=self.title_color,
            padx=10,
            pady=5,
        )
        controls_frame.pack(fill="x", pady=5)
        control_categories = {
            "Movement (Body Frame)": [
                ("W / S", "Forward / Backward", ("w", "s")),
                ("A / D", "Left / Right", ("a", "d")),
                ("R / F", "Up / Down", ("r", "f")),
                ("Q / E", "Yaw Left / Yaw Right", ("q", "e")),
            ],
            "Gimbal Control": [
                ("I / K", "Pitch Up / Down", ("i", "k")),
                ("J / L", "Yaw Left / Right", ("j", "l")),
                ("U / O", "Roll Left / Right", ("u", "o")),
            ],
            "Speed Control": [
                ("Z / X", "Increase / Decrease Multiplier", ("z", "x")),
                ("C", "Reset Multiplier (1.0x)", ("c",)),
            ],
            "System Control": [
                ("ESC / Close Window", "Emergency Stop & Land", ("escape",))
            ],
        }
        row = 0
        for category, controls in control_categories.items():
            category_label = tk.Label(
                controls_frame,
                text=category,
                font=("Arial", 10, "bold"),
                bg="#f0f0f0",
                fg=self.title_color,
                anchor="w",
            )
            category_label.grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(5, 0)
            )
            row += 1
            for keys_text, description, key_codes in controls:
                key_label = tk.Label(
                    controls_frame,
                    text=keys_text,
                    font=self.text_font,
                    bg="#e0e0e0",
                    width=18,
                    anchor="w",
                    relief=tk.GROOVE,
                    borderwidth=1,
                    padx=5,
                )
                key_label.grid(row=row, column=0, sticky="w", padx=2, pady=1)
                tk.Label(
                    controls_frame,
                    text=description,
                    font=self.text_font,
                    bg="#f0f0f0",
                    anchor="w",
                ).grid(row=row, column=1, columnspan=2, sticky="w", padx=5)
                for key_code in key_codes:
                    self.active_key_labels[key_code] = key_label
                row += 1
            tk.Label(controls_frame, text="", bg="#f0f0f0").grid(row=row, column=0)
            row += 1

    def create_status_section(self):
        """Create the status section with telemetry data"""
        status_frame = tk.LabelFrame(
            self.controls_frame,
            text="System Status",
            font=self.header_font,
            bg="#f0f0f0",
            fg=self.title_color,
            padx=10,
            pady=5,
        )
        status_frame.pack(fill="x", pady=5)

        def add_status_row(parent, label_text):
            row_frame = tk.Frame(parent, bg="#f0f0f0")
            row_frame.pack(fill="x", pady=2)
            tk.Label(
                row_frame,
                text=label_text,
                font=self.text_font,
                bg="#f0f0f0",
                width=15,
                anchor="w",
            ).pack(side=tk.LEFT)
            value_label = tk.Label(
                row_frame, text="N/A", font=self.text_font, bg="#f0f0f0", anchor="w"
            )
            value_label.pack(side=tk.LEFT, expand=True, fill="x")
            return value_label

        self.speed_label = add_status_row(status_frame, "Speed Multiplier:")
        self.velocity_label = add_status_row(status_frame, "Drone Velocity:")
        self.yaw_label = add_status_row(status_frame, "Yaw Speed:")
        self.gimbal_status_label = add_status_row(status_frame, "Gimbal Status:")
        self.gimbal_angle_label = add_status_row(status_frame, "Gimbal Angles:")
        self.altitude_label = add_status_row(status_frame, "Altitude (Rel):")
        self.attitude_label = add_status_row(status_frame, "Attitude:")

        # Set initial values
        self.speed_label.config(
            text=f"{self.state.speed_multiplier:.1f}x", fg="#0066CC"
        )
        self.velocity_label.config(text="F: +0.0, R: +0.0, D: +0.0")
        self.yaw_label.config(text="+0.0 deg/s")
        self.gimbal_status_label.config(
            text="Active", fg=self.status_active_color
        )
        self.gimbal_angle_label.config(
            text=f"P:{self.state.gimbal_pitch:+.1f}, R:{self.state.gimbal_roll:+.1f}, Y:{self.state.gimbal_yaw:+.1f}"
        )
        self.altitude_label.config(text="0.0 m")
        self.attitude_label.config(text="R: +0.0, P: +0.0, Y: +0.0")

    def create_visual_indicators(self):
        """Create visual indicators for drone and gimbal movement"""
        visual_frame = tk.LabelFrame(
            self.controls_frame,
            text="Visual Indicators",
            font=self.header_font,
            bg="#f0f0f0",
            fg=self.title_color,
            padx=10,
            pady=5,
        )
        visual_frame.pack(fill="both", expand=True, pady=5)

        # Movement indicator (arrows showing direction)
        movement_frame = tk.Frame(visual_frame, bg="#f0f0f0")
        movement_frame.pack(
            side=tk.LEFT, fill="both", expand=True, padx=5
        )  # Pack side by side
        tk.Label(
            movement_frame,
            text="Movement",
            font=self.text_font,
            bg="#f0f0f0",
            anchor="center",
        ).pack(anchor="n")
        self.movement_canvas = tk.Canvas(
            movement_frame,
            width=120,
            height=120,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc",
        )
        self.movement_canvas.pack(pady=5, expand=True)
        cx, cy = 60, 60  # Center
        self.movement_canvas.create_line(cx, 0, cx, 120, fill="#cccccc")  # Vertical
        self.movement_canvas.create_line(0, cy, 120, cy, fill="#cccccc")  # Horizontal
        self.movement_canvas.create_oval(
            cx - 3, cy - 3, cx + 3, cy + 3, fill="#555555", outline=""
        )  # Center dot
        self.movement_indicator = self.movement_canvas.create_polygon(
            cx, cy, cx, cy, cx, cy, fill="#4CAF50", outline="#4CAF50"
        )  # Placeholder triangle

        # Gimbal angle visualization
        gimbal_frame = tk.Frame(visual_frame, bg="#f0f0f0")
        gimbal_frame.pack(
            side=tk.RIGHT, fill="both", expand=True, padx=5
        )  # Pack side by side
        tk.Label(
            gimbal_frame,
            text="Gimbal",
            font=self.text_font,
            bg="#f0f0f0",
            anchor="center",
        ).pack(anchor="n")
        self.gimbal_canvas = tk.Canvas(
            gimbal_frame,
            width=120,
            height=80,
            bg="white",
            highlightthickness=1,
            highlightbackground="#cccccc",
        )
        self.gimbal_canvas.pack(pady=5, expand=True)
        gx, gy_base = 60, 70  # Gimbal base center-bottom
        self.gimbal_canvas.create_rectangle(
            gx - 15, gy_base - 10, gx + 15, gy_base, fill="#aaaaaa", outline=""
        )  # Base rectangle
        self.gimbal_line = self.gimbal_canvas.create_line(
            gx, gy_base - 5, gx, gy_base - 45, fill="#0066CC", width=3, arrow=tk.LAST
        )  # Initial line (pitch only)

    def create_control_buttons(self):
        """Create control buttons for common actions"""
        button_frame = tk.Frame(self.main_container, bg="#f0f0f0")
        button_frame.pack(fill="x", pady=5)

        self.stop_button = tk.Button(
            button_frame,
            text="STOP & LAND (ESC)",
            command=self.exit_control,
            bg="#E74C3C",
            fg="white",
            font=self.button_font,
            padx=10,
            pady=5,
            relief=tk.RAISED,
            borderwidth=2,
        )
        self.stop_button.pack(side=tk.LEFT, padx=5, fill="x", expand=True)

        self.reset_speed_button = tk.Button(
            button_frame,
            text="Reset Speed (C)",
            command=self.reset_speed,
            bg="#3498DB",
            fg="white",
            font=self.button_font,
            padx=10,
            pady=5,
        )
        self.reset_speed_button.pack(side=tk.LEFT, padx=5, fill="x", expand=True)

        self.center_gimbal_button = tk.Button(
            button_frame,
            text="Center Gimbal",
            command=self.center_gimbal_cmd,
            bg="#2ECC71",
            fg="white",
            font=self.button_font,
            padx=10,
            pady=5,
        )
        self.center_gimbal_button.pack(side=tk.LEFT, padx=5, fill="x", expand=True)

    def on_key_press(self, event):
        """Handle key press events"""
        if not self.root.winfo_exists():
            return
        key = event.keysym.lower()
        is_special_key = False

        if key == "escape":
            self.exit_control()
            return
        elif key == "z":
            self.increase_speed()
            is_special_key = True
        elif key == "x":
            self.decrease_speed()
            is_special_key = True
        elif key == "c":
            self.reset_speed()
            is_special_key = True

        if not is_special_key and key in self.active_key_labels:
            self.state.pressed_keys.add(key)
            # Update highlighting immediately
            self.active_key_labels[key].config(
                bg="#FFD700", relief=tk.SUNKEN
            )  # Highlight in gold, sunken

    def on_key_release(self, event):
        """Handle key release events"""
        if not self.root.winfo_exists():
            return
        key = event.keysym.lower()
        if key in self.state.pressed_keys:
            self.state.pressed_keys.remove(key)

        # Restore key highlighting only if it exists in our map
        if key in self.active_key_labels:
            self.active_key_labels[key].config(
                bg="#e0e0e0", relief=tk.GROOVE
            )  # Restore background, groove relief

    def center_gimbal_state(self):
        """Logic to center gimbal state variables."""
        self.state.gimbal_pitch = self.config.initial_pitch
        self.state.gimbal_yaw = 0.0
        self.state.gimbal_roll = 0.0
        # Also reset rates
        self.state.gimbal_pitch_rate = 0.0
        self.state.gimbal_yaw_rate = 0.0
        self.state.gimbal_roll_rate = 0.0
        LOGGER.info("Gimbal state centered.")

        # Notify via event bus that gimbal has been centered
        self.event_bus.publish(
            "gimbal_centered",
            {
                "pitch": self.state.gimbal_pitch,
                "roll": self.state.gimbal_roll,
                "yaw": self.state.gimbal_yaw,
            },
        )

    def center_gimbal_cmd(self):
        """Command to center the gimbal, updates state and UI."""
        self.center_gimbal_state()
        # Update UI immediately
        self.update_gimbal_indicator()
        self.gimbal_angle_label.config(
            text=f"P:{self.state.gimbal_pitch:+.1f}, R:{self.state.gimbal_roll:+.1f}, Y:{self.state.gimbal_yaw:+.1f}"
        )
        self.update_status_message("Gimbal centered.")

    def process_keys(self):
        """Process keyboard inputs held down and update control state. Called periodically."""
        if not self.root.winfo_exists() or self.state.exit_flag:
            return

        cfg = self.config
        state = self.state
        keys = state.pressed_keys  # Get currently pressed keys
        speed_mult = state.speed_multiplier
        accel = 1.0 - cfg.decay_factor  # Amount to move towards target per cycle
        decay = cfg.decay_factor  # Amount to retain from previous velocity per cycle

        # --- Target Velocities based on currently pressed keys ---
        target_forward = 0.0
        if "w" in keys:
            target_forward = cfg.max_speed * speed_mult
        elif "s" in keys:
            target_forward = -cfg.max_speed * speed_mult

        target_right = 0.0
        if "d" in keys:
            target_right = cfg.max_speed * speed_mult
        elif "a" in keys:
            target_right = -cfg.max_speed * speed_mult

        target_down = 0.0
        if "f" in keys:
            target_down = cfg.max_speed * speed_mult  # F = Down
        elif "r" in keys:
            target_down = -cfg.max_speed * speed_mult  # R = Up

        target_yawspeed = 0.0
        drone_yaw_command = None
        if "e" in keys:
            target_yawspeed = cfg.max_yaw_speed * speed_mult  # Clockwise
            drone_yaw_command = "right"
        elif "q" in keys:
            target_yawspeed = -cfg.max_yaw_speed * speed_mult  # Counter-Clockwise
            drone_yaw_command = "left"

        # --- Apply Smoothing  ---
        state.velocity_forward = state.velocity_forward * decay + target_forward * accel
        state.velocity_right = state.velocity_right * decay + target_right * accel
        state.velocity_down = state.velocity_down * decay + target_down * accel
        state.yawspeed = state.yawspeed * decay + target_yawspeed * accel

        # --- Apply Zero Threshold ---
        if target_forward == 0.0 and abs(state.velocity_forward) < cfg.zero_threshold:
            state.velocity_forward = 0.0
        if target_right == 0.0 and abs(state.velocity_right) < cfg.zero_threshold:
            state.velocity_right = 0.0
        if target_down == 0.0 and abs(state.velocity_down) < cfg.zero_threshold:
            state.velocity_down = 0.0
        if target_yawspeed == 0.0 and abs(state.yawspeed) < cfg.zero_threshold:
            state.yawspeed = 0.0

        # --- Gimbal Control ---
        pitch_change = 0.0
        yaw_change = 0.0
        roll_change = 0.0
        gimbal_increment = (
            cfg.gimbal_angle_increment * speed_mult
        )  # Scale gimbal speed too

        if "i" in keys:
            pitch_change = gimbal_increment  # Pitch Up
        elif "k" in keys:
            pitch_change = -gimbal_increment  # Pitch Down

        if drone_yaw_command == "right" or "l" in keys:
            yaw_change = gimbal_increment  # Yaw Right
        elif drone_yaw_command == "left" or "j" in keys:
            yaw_change = -gimbal_increment  # Yaw Left

        if "o" in keys:
            roll_change = gimbal_increment  # Roll Right
        elif "u" in keys:
            roll_change = -gimbal_increment  # Roll Left

        # Update angles directly
        state.gimbal_pitch += pitch_change
        state.gimbal_yaw += yaw_change
        state.gimbal_roll += roll_change

        # Clamp angles to limits
        state.gimbal_pitch = max(
            cfg.min_pitch_deg, min(cfg.max_pitch_deg, state.gimbal_pitch)
        )
        state.gimbal_roll = max(
            cfg.min_roll_deg, min(cfg.max_roll_deg, state.gimbal_roll)
        )
        # Wrap yaw angle between -180 and 180
        state.gimbal_yaw = (state.gimbal_yaw + 180.0) % 360.0 - 180.0

        gimbal_changed = pitch_change != 0.0 or yaw_change != 0.0 or roll_change != 0.0
        if gimbal_changed:
            LOGGER.debug(
                f"Gimbal changed to P:{state.gimbal_pitch:.1f}, R:{state.gimbal_roll:.1f}, Y:{state.gimbal_yaw:.1f}"
            )

            self.event_bus.publish(
                "gimbal_update",
                {
                    "pitch": state.gimbal_pitch,
                    "roll": state.gimbal_roll,
                    "yaw": state.gimbal_yaw,
                    "vehicle_name": self.config.connection.vehicle_name,
                    "camera_name": self.config.camera.camera_name,
                },
            )

        target_gimbal_pitch_rate = 0.0
        if "i" in keys:
            target_gimbal_pitch_rate = cfg.max_gimbal_rate
        elif "k" in keys:
            target_gimbal_pitch_rate = -cfg.max_gimbal_rate

        target_gimbal_yaw_rate = 0.0
        if "l" in keys:
            target_gimbal_yaw_rate = cfg.max_gimbal_rate
        elif "j" in keys:
            target_gimbal_yaw_rate = -cfg.max_gimbal_rate

        target_gimbal_roll_rate = 0.0
        if "o" in keys:
            target_gimbal_roll_rate = cfg.max_gimbal_rate
        elif "u" in keys:
            target_gimbal_roll_rate = -cfg.max_gimbal_rate

        # Apply smoothing to rates
        state.gimbal_pitch_rate = (
            state.gimbal_pitch_rate * decay + target_gimbal_pitch_rate * accel
        )
        state.gimbal_yaw_rate = (
            state.gimbal_yaw_rate * decay + target_gimbal_yaw_rate * accel
        )
        state.gimbal_roll_rate = (
            state.gimbal_roll_rate * decay + target_gimbal_roll_rate * accel
        )

        # Zero threshold for rates
        if (
            target_gimbal_pitch_rate == 0.0
            and abs(state.gimbal_pitch_rate) < cfg.zero_threshold
        ):
            state.gimbal_pitch_rate = 0.0
        if (
            target_gimbal_yaw_rate == 0.0
            and abs(state.gimbal_yaw_rate) < cfg.zero_threshold
        ):
            state.gimbal_yaw_rate = 0.0
        if (
            target_gimbal_roll_rate == 0.0
            and abs(state.gimbal_roll_rate) < cfg.zero_threshold
        ):
            state.gimbal_roll_rate = 0.0

        # --- Visual Indicators ---
        if hasattr(self, "update_movement_indicator"):
            self.update_movement_indicator()
        if hasattr(self, "update_gimbal_indicator"):
            self.update_gimbal_indicator()

    def update_movement_indicator(self):
        """Update the movement direction indicator on canvas"""
        if (
            not hasattr(self, "movement_canvas")
            or not self.movement_canvas.winfo_exists()
        ):
            return

        # Use current state velocities
        max_vel = self.config.max_speed  # Base max speed
        # Normalize based on potential max speed (ignoring multiplier for indicator range)
        forward_norm = np.clip(self.state.velocity_forward / (max_vel + 0.01), -1, 1)
        right_norm = np.clip(self.state.velocity_right / (max_vel + 0.01), -1, 1)

        cx, cy = 60, 60  # Canvas center
        scale = 45  # Max distance from center

        indicator_x = cx + right_norm * scale
        indicator_y = cy - forward_norm * scale  # Y is inverted

        # Update movement indicator (arrow shape)
        if abs(forward_norm) > 0.05 or abs(right_norm) > 0.05:
            angle_rad = np.arctan2(-forward_norm, right_norm)  # Angle of movement
            arrow_len = 15
            arrow_width = 10

            # Tip of the arrow
            x_tip = indicator_x
            y_tip = indicator_y

            # Base points (perpendicular to the direction vector)
            angle_left = angle_rad + np.pi / 2
            angle_right = angle_rad - np.pi / 2
            base_offset = arrow_len * 0.7  # Move base back from tip

            x_base_mid = x_tip - base_offset * np.cos(angle_rad)
            y_base_mid = y_tip - base_offset * np.sin(angle_rad)

            x_base_l = x_base_mid + arrow_width / 2 * np.cos(angle_left)
            y_base_l = y_base_mid + arrow_width / 2 * np.sin(angle_left)
            x_base_r = x_base_mid + arrow_width / 2 * np.cos(angle_right)
            y_base_r = y_base_mid + arrow_width / 2 * np.sin(angle_right)

            self.movement_canvas.coords(
                self.movement_indicator,
                x_tip,
                y_tip,
                x_base_l,
                y_base_l,
                x_base_r,
                y_base_r,
            )

            # Color based on speed multiplier or magnitude
            speed_magnitude = np.sqrt(
                self.state.velocity_forward**2 + self.state.velocity_right**2
            )
            norm_speed_mag = speed_magnitude / (max_vel + 0.01)

            if norm_speed_mag > 0.7:
                fill_color = "#E74C3C"  # Red
            elif norm_speed_mag > 0.3:
                fill_color = "#F39C12"  # Orange
            else:
                fill_color = "#2ECC71"  # Green

            self.movement_canvas.itemconfig(
                self.movement_indicator, fill=fill_color, outline=fill_color
            )
        else:
            self.movement_canvas.coords(self.movement_indicator, cx, cy, cx, cy, cx, cy)
            self.movement_canvas.itemconfig(
                self.movement_indicator, fill="", outline=""
            )

    def update_gimbal_indicator(self):
        """Update the gimbal orientation indicator on canvas"""
        if not hasattr(self, "gimbal_canvas") or not self.gimbal_canvas.winfo_exists():
            return

        pitch_rad = np.radians(self.state.gimbal_pitch)
        yaw_rad = np.radians(self.state.gimbal_yaw)

        gx, gy_base = 60, 70
        length = 40
        end_x = gx + length * np.sin(yaw_rad) * 0.5
        end_y = gy_base - length * np.sin(pitch_rad)

        self.gimbal_canvas.coords(self.gimbal_line, gx, gy_base - 5, end_x, end_y)

        is_moving = (
            abs(self.state.gimbal_pitch_rate) > 1.0
            or abs(self.state.gimbal_yaw_rate) > 1.0
            or abs(self.state.gimbal_roll_rate) > 1.0
        )

        fill_color = "#E74C3C" if is_moving else "#0066CC"
        line_width = 4 if is_moving else 3
        self.gimbal_canvas.itemconfig(
            self.gimbal_line, fill=fill_color, width=line_width
        )

    def update_gui_loop(self):
        """Periodic GUI update loop. Calls process_keys and updates labels."""
        if not self.root.winfo_exists() or self.state.exit_flag:
            LOGGER.info("GUI update loop stopping.")
            return

        current_time = time.monotonic()

        self.process_keys()

        if current_time - self.last_telemetry_update_time >= 0.1:
            self.last_telemetry_update_time = current_time

            self.speed_label.config(text=f"{self.state.speed_multiplier:.1f}x")
            self.velocity_label.config(
                text=f"F:{self.state.velocity_forward:+.1f}, R:{self.state.velocity_right:+.1f}, D:{self.state.velocity_down:+.1f} m/s"
            )
            self.yaw_label.config(text=f"{self.state.yawspeed:+.1f} deg/s")
            self.gimbal_angle_label.config(
                text=f"P:{self.state.gimbal_pitch:+.1f}, R:{self.state.gimbal_roll:+.1f}, Y:{self.state.gimbal_yaw:+.1f}"
            )

        self._update_gui_loop_id = self.root.after(20, self.update_gui_loop)

    def increase_speed(self):
        """Increase the speed multiplier"""
        self.state.speed_multiplier = min(3.0, self.state.speed_multiplier + 0.1)
        self.speed_label.config(text=f"{self.state.speed_multiplier:.1f}x")
        self.update_status_message(
            f"Speed multiplier: {self.state.speed_multiplier:.1f}x"
        )
        self.event_bus.publish("speed_changed", self.state.speed_multiplier)

    def decrease_speed(self):
        """Decrease the speed multiplier"""
        self.state.speed_multiplier = max(0.1, self.state.speed_multiplier - 0.1)
        self.speed_label.config(text=f"{self.state.speed_multiplier:.1f}x")
        self.update_status_message(
            f"Speed multiplier: {self.state.speed_multiplier:.1f}x"
        )
        self.event_bus.publish("speed_changed", self.state.speed_multiplier)

    def reset_speed(self):
        """Reset the speed multiplier to default"""
        self.state.speed_multiplier = 1.0
        self.speed_label.config(text=f"{self.state.speed_multiplier:.1f}x")
        self.update_status_message("Speed multiplier reset to 1.0x")
        self.event_bus.publish("speed_changed", self.state.speed_multiplier)

    def exit_control(self):
        """Initiates the shutdown sequence from the GUI."""
        if self.state.exit_flag:
            return

        LOGGER.info("Exit requested via GUI (Close button or ESC).")
        self.update_status_message("Exit requested. Initiating landing...", "warning")

        self.conn_status.config(text="● DISCONNECTING", fg=self.status_warning_color)
        self.stop_button.config(text="EXITING...", state=tk.DISABLED)
        self.reset_speed_button.config(state=tk.DISABLED)
        self.center_gimbal_button.config(state=tk.DISABLED)

        if hasattr(self, "_update_gui_loop_id"):
            try:
                self.root.after_cancel(self._update_gui_loop_id)
            except Exception as e:
                LOGGER.error(f"Error cancelling GUI update loop: {e}")

        self.state.exit_flag = True
        self.event_bus.publish("exit_requested", "User requested exit via GUI")

        if self.root.winfo_exists():
            try:
                self.root.update_idletasks()
                self.root.after(500, self._delayed_destroy)
            except Exception as e:
                LOGGER.error(f"Error updating GUI before exit: {e}")
                self._delayed_destroy()

    def _delayed_destroy(self):
        """Safely destroys the Tkinter window."""
        if self.root.winfo_exists():

            LOGGER.info("Closing GUI window.")
            try:
                for widget in self.root.winfo_children():
                    if hasattr(widget, "destroy"):
                        widget.destroy()

                self.root.quit()
                self.root.destroy()
            except Exception as e:
                LOGGER.error(f"Error quitting GUI: {e}")


# =============================================================================
# CONTROL APPLICATION ENTRY POINT
# =============================================================================

class ControlApplication:
    """
    Standalone control application that can be run independently of the display system.
    Handles drone connection, control loop, and GUI interface.
    """
    
    def __init__(self, config: DroneConfig, state: ControlState = None, event_bus_instance: EventBus = None):
        """Initialize the control application"""
        self.config = config
        self.state = state if state else ControlState(gimbal_pitch=config.initial_pitch)
        self.event_bus = event_bus_instance if event_bus_instance else event_bus
        
        # Components
        self.drone_controller = None
        self.gui_thread = None
        self.control_task = None
        
        # Event subscription
        self.event_bus.subscribe("exit_requested", self._handle_exit_request)
        self.event_bus.subscribe("gimbal_update", self._handle_gimbal_update)
    
    def _handle_exit_request(self, data):
        """Handle exit request events"""
        LOGGER.info(f"Exit requested: {data}")
        self.state.exit_flag = True
    
    def _handle_gimbal_update(self, pose_data):
        """Handle gimbal pose update events"""
        # Just forward this to the shared event bus
        # This will be picked up by the display application
        self.event_bus.publish("gimbal_update", pose_data)
    
    def start_gui(self):
        """Start the GUI in a separate thread"""
        gui_ready = threading.Event()
        gui_error = threading.Event()

        def gui_runner():
            try:
                root = tk.Tk()
                DroneControlGUI(root, self.config, self.state, self.event_bus)
                gui_ready.set()
                root.mainloop()
            except Exception as e:
                LOGGER.error(f"Error in GUI thread: {e}")
                gui_error.set()
                self.state.exit_flag = True
            finally:
                LOGGER.info("GUI thread finished.")
                self.state.exit_flag = True

        self.gui_thread = threading.Thread(
            target=gui_runner, name="GUIThread", daemon=True
        )
        self.gui_thread.start()

        if gui_error.is_set():
            LOGGER.error("GUI thread failed to initialize immediately.")
            return False

        if not gui_ready.wait(timeout=5.0):
            LOGGER.error("GUI thread timed out during initialization.")
            self.state.exit_flag = True
            return False

        LOGGER.info("GUI thread started successfully.")
        return True
    
    async def initialize_components(self):
        """Initialize all system components"""
        # Initialize drone controller
        LOGGER.info("Initializing Drone Controller...")
        self.drone_controller = DroneController(self.config, self.event_bus)
        
        # Try to connect to drone if MAVSDK is available
        if not await self.drone_controller.connect_drone():
                LOGGER.warning("Failed to connect to drone. Running in simulation mode.")
        else:
            LOGGER.warning("MAVSDK not available. Running in simulation mode.")

        # Start GUI Thread
        LOGGER.info("Starting GUI thread...")
        success = self.start_gui()
        if not success:
            LOGGER.error("Failed to start GUI. Cannot control drone.")
            self.state.exit_flag = True
            if self.drone_controller:
                await self.drone_controller.land_drone()
            return False

        return True
    
    async def run_control_loop(self):
        """Run the main control loop"""
        try:
            # Run the drone control loop
            await self.drone_controller.run_manual_control_loop(self.state)
            return True
        except Exception as e:
            LOGGER.error(f"Error in control loop: {e}")
            self.state.exit_flag = True
            return False
    
    async def run(self):
        """Run the control application"""
        try:
            if not await self.initialize_components():
                return 1
            
            LOGGER.info("Control application running.")
            
            # Run the control loop
            self.control_task = asyncio.create_task(self.run_control_loop())
            await self.control_task
            
            return 0
            
        except KeyboardInterrupt:
            LOGGER.info("Control application interrupted by user.")
            return 130
        except asyncio.CancelledError:
            LOGGER.info("Control task cancelled.")
            return 1
        except Exception as e:
            LOGGER.error(f"Error in control application: {e}")
            return 1
        finally:
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources before exit"""
        LOGGER.info("Cleaning up control application resources...")
        
        # Cancel control task if running
        if self.control_task and not self.control_task.done():
            self.control_task.cancel()
            try:
                await self.control_task
            except asyncio.CancelledError:
                pass
        
        # Land drone if connected and MAVSDK available
        if self.drone_controller:
            await self.drone_controller.land_drone()
        
        # Wait for GUI thread to finish
        if self.gui_thread and self.gui_thread.is_alive():
            LOGGER.info("Waiting for GUI thread to finish...")
            self.gui_thread.join(timeout=2.0)
            if self.gui_thread.is_alive():
                LOGGER.warning("GUI thread did not exit cleanly.")
        
        LOGGER.info("Control application cleanup complete.")


# Standalone execution
if __name__ == "__main__":
    import argparse
    
    def parse_arguments():
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description="AirSim Drone Control")
        
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
        
        # Enable verbose logging
        parser.add_argument(
            "--verbose", 
            action="store_true",
            help="Enable verbose logging"
        )
        
        return parser.parse_args()
    
    # Parse arguments
    args = parse_arguments()
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    else:
        LOGGER.setLevel(logging.INFO)
    
    # Create config
    config = DroneConfig()
    config.connection.ip_address = args.ip
    config.connection.vehicle_name = args.vehicle
    config.system_address = f"udp://:{args.mavsdk_port}"
    config.default_takeoff_altitude = args.takeoff_alt
    config.max_speed = args.max_speed
    
    # Run event loop
    loop = asyncio.get_event_loop()
    
    app = ControlApplication(config)
    
    try:
        exit_code = loop.run_until_complete(app.run())
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