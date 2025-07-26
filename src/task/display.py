"""
Drone display module for AirSim camera visualization using OpenCV.
This module handles fetching and displaying camera images from AirSim.
"""

import asyncio
import threading
import time
import cv2
import numpy as np
import concurrent.futures
import logging
import os
from typing import Optional, Dict, Any
from queue import Queue, Empty, Full
from abc import ABC, abstractmethod
import airsim  # type: ignore

from configs.airsim_config import DroneConfig, ControlState # type: ignore
from utils.util import EventBus # type: ignore
from utils.logger import LOGGER # type: ignore


# =============================================================================
# IMAGE PROCESSING STRATEGIES
# =============================================================================

class ImageProcessingStrategy(ABC):
    """Base class for image processing strategies"""

    @abstractmethod
    def process_image(self, img_data, height, width):
        pass


class DefaultImageProcessingStrategy(ImageProcessingStrategy):
    """Default image processing implementation"""

    def __init__(
        self,
        convert_to_rgb=True,
        resize_output=False,
        output_width=640,
        output_height=480,
    ):
        self.convert_to_rgb = convert_to_rgb
        self.resize_output = resize_output
        self.output_width = output_width
        self.output_height = output_height

    def process_image(self, img_data, height, width):
        """Process raw image data into displayable format"""
        if not img_data:
            return np.zeros(
                (
                    self.output_height if self.resize_output else height,
                    self.output_width if self.resize_output else width,
                    3,
                ),
                dtype=np.uint8,
            )
        try:
            img_bgr = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 3)
            img_display = img_bgr

            if self.resize_output:
                img_display = cv2.resize(
                    img_display,
                    (self.output_width, self.output_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            if self.convert_to_rgb:
                img_display = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)

            return img_display
        except Exception as e:
            LOGGER.error(f"Error processing image: {e}")
            return np.zeros(
                (
                    self.output_height if self.resize_output else height,
                    self.output_width if self.resize_output else width,
                    3,
                ),
                dtype=np.uint8,
            )


class HighPerformanceProcessingStrategy(ImageProcessingStrategy):
    """High performance image processing with minimal operations"""

    def __init__(
        self,
        convert_to_rgb=True,
        resize_output=False,
        output_width=1920,
        output_height=1080,
    ):
        self.convert_to_rgb = convert_to_rgb
        self.resize_output = resize_output
        self.output_width = output_width
        self.output_height = output_height

    def process_image(self, img_data, height, width):
        """Process image with minimal operations for maximum speed"""
        try:
            img = np.frombuffer(img_data, dtype=np.uint8).reshape(height, width, 3)
            
            if self.resize_output:
                if not hasattr(self, 'resized_buffer') or self.resized_buffer.shape != (self.output_height, self.output_width, 3):
                    self.resized_buffer = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
                
                cv2.resize(
                    img,
                    (self.output_width, self.output_height),
                    dst=self.resized_buffer,
                    interpolation=cv2.INTER_NEAREST
                )
                img = self.resized_buffer
            
            if self.convert_to_rgb:
                if not hasattr(self, 'rgb_buffer') or self.rgb_buffer.shape != img.shape:
                    self.rgb_buffer = np.zeros_like(img)
                
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=self.rgb_buffer)
                return self.rgb_buffer
            
            return img
        except Exception as e:
            LOGGER.error(f"Image processing error: {e}")
            return np.zeros((self.output_height if self.resize_output else height, 
                            self.output_width if self.resize_output else width, 3), 
                            dtype=np.uint8)


# =============================================================================
# DISPLAY MANAGER
# =============================================================================

class DisplayManager:
    """
    Manages OpenCV display windows and handles camera view display.
    This class is responsible for:
    1. Creating and managing OpenCV windows
    2. Displaying camera images
    3. Handling key events from display windows
    """

    def __init__(self, config: DroneConfig, state: ControlState, event_bus: EventBus):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        
        # Window settings
        self.windows = {}
        self.active = True
        
        # Create windows for each camera
        if config.multi_camera.enabled:
            for camera_name in config.multi_camera.cameras:
                window_name = f"AirSim - {camera_name}"
                self.windows[camera_name] = {
                    "window_name": window_name,
                    "last_image": None,
                    "last_update": 0,
                    "fps": 0,
                    "frame_count": 0
                }
        else:
            # Single camera mode
            window_name = config.display.window_name
            camera_name = config.camera.camera_name
            self.windows[camera_name] = {
                "window_name": window_name,
                "last_image": None,
                "last_update": 0,
                "fps": 0,
                "frame_count": 0
            }
        
        # Subscribe to image update events
        self.event_bus.subscribe("camera_image_updated", self._handle_image_update)
        
        # Create display thread
        self.display_thread = None
    
    def _handle_image_update(self, data):
        """Handle image update events from the event bus"""
        if not self.active or data.get("exit", False):
            return
            
        if "image" in data and data["image"] is not None:
            camera_name = data.get("camera_name", "front_center")
            
            if camera_name in self.windows:
                window = self.windows[camera_name]
                window["last_image"] = data["image"]
                window["last_update"] = time.time()
                window["frame_count"] += 1
    
    def start(self):
        """Start the display thread"""
        if self.display_thread and self.display_thread.is_alive():
            LOGGER.warning("Display thread already running")
            return
            
        self.active = True
        self.display_thread = threading.Thread(
            target=self._display_thread_worker,
            name="DisplayThread",
            daemon=True
        )
        self.display_thread.start()
        LOGGER.info("Display thread started")
    
    def stop(self):
        """Stop the display thread and close all windows"""
        self.active = False
        
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
            
        # Close all OpenCV windows
        try:
            cv2.destroyAllWindows()
            for _ in range(3):  # Sometimes needed to properly close windows
                cv2.waitKey(1)
        except Exception as e:
            LOGGER.error(f"Error closing OpenCV windows: {e}")
    
    def _display_thread_worker(self):
        """Worker function for the display thread"""
        try:
            # Create windows
            for camera_name, window in self.windows.items():
                cv2.namedWindow(window["window_name"], cv2.WINDOW_NORMAL)
                
                # Set initial window size
                width = self.config.display.output_width 
                height = self.config.display.output_height
                cv2.resizeWindow(window["window_name"], width, height)
            
            last_fps_update = time.time()
            
            # Main display loop
            while self.active and not self.state.exit_flag:
                any_update = False
                
                # Update each window
                for camera_name, window in self.windows.items():
                    if window["last_image"] is not None:
                        # Get the last image
                        image = window["last_image"]
                        
                        # Add overlay information
                        image = self._add_overlay(image, camera_name, window)
                        
                        # Display the image
                        cv2.imshow(window["window_name"], image)
                        any_update = True
                
                # Update FPS counters every second
                now = time.time()
                if now - last_fps_update >= 1.0:
                    for camera_name, window in self.windows.items():
                        elapsed = now - last_fps_update
                        fps = window["frame_count"] / elapsed if elapsed > 0 else 0
                        window["fps"] = fps
                        window["frame_count"] = 0
                    last_fps_update = now
                
                # Process key events
                key = cv2.waitKey(self.config.display.waitKey_delay) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or q
                    LOGGER.info("Exit requested from display window")
                    self.state.exit_flag = True
                    self.event_bus.publish("exit_requested", "User requested exit via OpenCV window")
                    break
                
                # Sleep a bit if no updates to reduce CPU usage
                if not any_update:
                    time.sleep(0.01)
        
        except Exception as e:
            LOGGER.error(f"Error in display thread: {e}")
            self.state.exit_flag = True
        finally:
            # Ensure windows are closed
            try:
                cv2.destroyAllWindows()
                for _ in range(3):
                    cv2.waitKey(1)
            except:
                pass
            
            LOGGER.info("Display thread stopped")
    
    def _add_overlay(self, image, camera_name, window):
        """Add information overlay to the image"""
        # Create a copy of the image to avoid modifying the original
        img_with_overlay = image.copy()
        
        # Add camera name and FPS
        fps_text = f"{camera_name} - {window['fps']:.1f} FPS"
        cv2.putText(
            img_with_overlay,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        # Add drone telemetry if available
        if self.state.last_position:
            altitude = self.state.last_position.get("relative_altitude_m", 0)
            alt_text = f"ALT: {altitude:.1f}m"
            cv2.putText(
                img_with_overlay,
                alt_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        
        if self.state.last_attitude:
            attitude = self.state.last_attitude
            attitude_text = f"ROLL: {attitude.get('roll_deg', 0):.1f}° PITCH: {attitude.get('pitch_deg', 0):.1f}° YAW: {attitude.get('yaw_deg', 0):.1f}°"
            cv2.putText(
                img_with_overlay,
                attitude_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        
        # Add gimbal info
        gimbal_text = f"GIMBAL - P: {self.state.gimbal_pitch:.1f}° R: {self.state.gimbal_roll:.1f}° Y: {self.state.gimbal_yaw:.1f}°"
        cv2.putText(
            img_with_overlay,
            gimbal_text,
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        return img_with_overlay


# =============================================================================
# AIRSIM CLIENT
# =============================================================================

class SimulationDroneClient:
    """AirSim client implementation for drone simulation and camera access"""

    def __init__(
        self,
        config: DroneConfig,
        state: ControlState,
        event_bus: EventBus,
        image_strategy: Optional[ImageProcessingStrategy] = None,
    ):
        self.config = config
        self.state = state
        self.event_bus = event_bus
        self.client: Optional[airsim.MultirotorClient] = None

        # Create queues through event_bus instead of using global variables
        self.camera_pose_queue = event_bus.create_queue("camera_pose_queue", maxsize=10)
        self.display_image_queue = event_bus.create_queue(
            "display_image_queue", maxsize=2
        )

        # Set up image processing strategy
        if image_strategy:
            self.image_strategy = image_strategy
        elif config.display.optimize_for_realtime:
            self.image_strategy = HighPerformanceProcessingStrategy(
                convert_to_rgb=config.display.convert_to_rgb,
                resize_output=config.display.resize_output,
                output_width=config.display.output_width,
                output_height=config.display.output_height,
            )
        else:
            self.image_strategy = DefaultImageProcessingStrategy(
                convert_to_rgb=config.display.convert_to_rgb,
                resize_output=config.display.resize_output,
                output_width=config.display.output_width,
                output_height=config.display.output_height,
            )

        self.running = False
        self.fetch_times: list = []
        self.raw_image_queue: Queue = Queue(maxsize=config.display.fetch_queue_size)
        self.fetch_thread = None
        self.thread_clients: list = []
        self.fetch_stats: Dict = {"fetch_times": [], "process_times": []}

        self.camera_configs = {
            "front_center": {
                "camera_name": "front_center",
                "image_type": airsim.ImageType.Scene,
                "pixels_as_float": False,
                "compress": False,
                "width": config.camera.width,
                "height": config.camera.height,
            },
            "side_left": {
                "camera_name": "side_left",
                "image_type": airsim.ImageType.Scene,
                "pixels_as_float": False,
                "compress": False,
                "width": config.camera.width,
                "height": config.camera.height,
            },
            "side_right": {
                "camera_name": "side_right",
                "image_type": airsim.ImageType.Scene,
                "pixels_as_float": False,
                "compress": False,
                "width": config.camera.width,
                "height": config.camera.height,
            }
        }

        self.process_pool = None
        if config.processing.enable_parallel_processing:
            self.process_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=os.cpu_count() or 2
            )

        self.opencv_display_thread = None
        self.camera_pose_thread = None

    def connect(self) -> bool:
        """Connect to AirSim server"""
        LOGGER.info(
            f"Attempting to connect to AirSim at: {self.config.connection.ip_address}"
        )
        result = {"success": False, "client": None, "error": None}
        connection_complete = threading.Event()

        def connect_thread():
            nonlocal result
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self.client = airsim.MultirotorClient(
                    ip=self.config.connection.ip_address
                )
                # self.client.confirmConnection()
                result["success"] = True
                result["client"] = self.client
                # Get additional info if needed
                try:
                    vehicles = self.client.listVehicles()
                    LOGGER.info(f"AirSim available vehicles: {vehicles}")
                    if self.config.connection.vehicle_name not in vehicles:
                        LOGGER.warning(
                            f"Vehicle '{self.config.connection.vehicle_name}' not found in AirSim list."
                        )
                except Exception as e_info:
                    LOGGER.warning(f"Could not list AirSim vehicles: {e_info}")

            except Exception as e:
                result["error"] = str(e)
                LOGGER.error(f"Failed to connect to AirSim in thread: {e}")
            finally:
                connection_complete.set()

        conn_thread = threading.Thread(target=connect_thread, daemon=True)
        conn_thread.start()

        if not connection_complete.wait(timeout=self.config.timeout):
            LOGGER.error(
                f"AirSim connection timeout after {self.config.timeout} seconds"
            )
            return False

        if not result["success"]:
            LOGGER.error(
                f"Failed to connect to AirSim: {result.get('error', 'Unknown error')}"
            )
            return False

        LOGGER.info(
            f"Connected to AirSim vehicle: {self.config.connection.vehicle_name}"
        )
        return True

    def get_image(self, client_to_use: airsim.MultirotorClient, camera_config=None) -> Optional[dict]:
        """Get a single raw image frame. Run by the fetcher thread."""
         
        # Use default config if none provided
        if camera_config is None:
            camera_config = {
                "camera_name": self.config.camera.camera_name,
                "image_type": self.config.camera.image_type,
                "pixels_as_float": self.config.camera.pixels_as_float,
                "compress": self.config.camera.compress,
                "width": self.config.camera.width,
                "height": self.config.camera.height,
            }

        try:
            start_time = time.time()

            # Handle gimbal pose for front camera
            if camera_config["camera_name"] == "front_center" and hasattr(self, "state") and (
                self.state.gimbal_pitch != 0
                or self.state.gimbal_roll != 0
                or self.state.gimbal_yaw != 0
            ):
                try:
                    quat = airsim.euler_to_quaternion(
                        self.state.gimbal_roll * np.pi / 180,
                        self.state.gimbal_pitch * np.pi / 180,
                        self.state.gimbal_yaw * np.pi / 180,
                    )
                    
                    client_to_use.simSetCameraPose(
                        camera_config["camera_name"],
                        airsim.Pose(airsim.Vector3r(0, 0, 0.2), quat),
                        vehicle_name=self.config.connection.vehicle_name,
                    )
                except Exception as e:
                    LOGGER.warning(f"Could not set camera pose: {e}")

            # Set image request
            request = airsim.ImageRequest(
                camera_name=camera_config["camera_name"],
                image_type=camera_config["image_type"],
                pixels_as_float=camera_config["pixels_as_float"],
                compress=camera_config["compress"],
            )

            if camera_config["width"] > 0 and camera_config["height"] > 0:
                request.width = camera_config["width"]
                request.height = camera_config["height"]

            # Get images
            responses = client_to_use.simGetImages(
                [request], vehicle_name=self.config.connection.vehicle_name
            )

            # Check if response is valid
            if not responses or not responses[0].image_data_uint8:
                return None

            response = responses[0]

            # Log fetch time
            fetch_time = time.time() - start_time
            if len(self.fetch_times) > 100:
                self.fetch_times.pop(0)
            self.fetch_times.append(fetch_time)

            # Return image info
            return {
                "data": response.image_data_uint8,
                "width": response.width,
                "height": response.height,
                "timestamp": time.time(),
                "fetch_time": fetch_time,
                "camera_name": camera_config["camera_name"]
            }
        except Exception as e:
            LOGGER.error(f"Failed to get image from {camera_config['camera_name']}: {e}")
            return None

    def _create_thread_client(self):
        """Create a thread-specific client instance"""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Create client instance with specific event loop
            thread_client = airsim.MultirotorClient(
                ip=self.config.connection.ip_address
            )
            thread_client.confirmConnection()
            self.thread_clients.append(thread_client)
            LOGGER.info(
                f"AirSim client created successfully for thread {threading.current_thread().name}"
            )
            return thread_client
        except Exception as e:
            LOGGER.error(f"Failed to connect thread client in {threading.current_thread().name}: {e}")
            return None

    def _fetch_images_worker(self):
        """Background thread for fetching raw images from all cameras."""
        thread_name = threading.current_thread().name
        LOGGER.info(f"Starting image fetch worker: {thread_name}")

        thread_client = self._create_thread_client()
        if not thread_client:
            LOGGER.error(
                f"Could not create AirSim client for {thread_name}, terminating worker."
            )
            return

        # Create dedicated threads for each camera
        def camera_fetcher(camera_name, camera_config):
            try:
                # Create a separate client instance for this thread
                camera_client = airsim.MultirotorClient(ip=self.config.connection.ip_address)
                camera_client.confirmConnection()
                
                while self.running and not self.state.exit_flag:
                    if not self.raw_image_queue.full():
                        try:
                            # Call get_image directly (synchronously)
                            image_info = self.get_image(camera_client, camera_config)
                            if image_info:
                                try:
                                    self.raw_image_queue.put_nowait(image_info)
                                except Full:
                                    try:
                                        # If queue is full, prioritize newer frames
                                        self.raw_image_queue.get_nowait()  # Discard oldest
                                        self.raw_image_queue.put_nowait(image_info)  # Put in newest
                                    except (Full, Empty):
                                        pass
                            else:
                                time.sleep(0.05)
                        except Exception as e:
                            LOGGER.error(f"Error fetching image from {camera_name}: {e}")
                            time.sleep(0.1)
                    else:
                        time.sleep(0.01)
                    
                    time.sleep(0.001)  # Prevent CPU overload
            except Exception as e:
                LOGGER.error(f"Fatal error in camera fetcher for {camera_name}: {e}")
        
        # Create a dedicated thread for each camera
        camera_threads = []
        for camera_name, camera_config in self.camera_configs.items():
            thread = threading.Thread(
                target=camera_fetcher, 
                args=(camera_name, camera_config),
                name=f"CameraFetch_{camera_name}",
                daemon=True
            )
            thread.start()
            camera_threads.append(thread)
            # Allow each thread time to initialize its event loop
            time.sleep(0.1)
        
        # Monitor thread status
        while self.running and not self.state.exit_flag:
            alive_threads = [t for t in camera_threads if t.is_alive()]
            if len(alive_threads) < len(camera_threads):
                LOGGER.warning(f"Some camera threads have died! Alive: {len(alive_threads)}/{len(camera_threads)}")
            time.sleep(1.0)
            
        LOGGER.info(f"Image fetch worker {thread_name} finished.")

    def start_image_fetcher(self):
        """Start the background image fetching thread."""
        if not self.config.display.use_threading:
            LOGGER.warning(
                "Threading is disabled, image fetching will be synchronous (not recommended)."
            )
            return

        if self.fetch_thread and self.fetch_thread.is_alive():
            LOGGER.warning("Image fetcher thread already running.")
            return

        self.running = True
        self.fetch_thread = threading.Thread(
            target=self._fetch_images_worker, name="ImageFetchThread", daemon=True
        )
        self.fetch_thread.start()
        LOGGER.info("Image fetcher thread started.")

    def start_camera_pose_thread(self):
        """Start the camera pose worker thread"""
        if self.camera_pose_thread and self.camera_pose_thread.is_alive():
            LOGGER.warning("Camera pose thread already running")
            return

        self.camera_pose_thread = threading.Thread(
            target=self.camera_pose_worker, name="CameraPoseWorker", daemon=True
        )
        self.camera_pose_thread.start()
        LOGGER.info("Camera pose worker thread started")

    def camera_pose_worker(self):
        """Worker thread to update camera poses from queue."""
        thread_name = threading.current_thread().name
        LOGGER.info(f"{thread_name} started.")

        pose_client = self._create_thread_client()
        if not pose_client:
            LOGGER.error(
                f"Failed to create AirSim client for {thread_name}. Worker exiting."
            )
            return

        try:
            while not self.state.exit_flag:
                try:
                    pose_data = self.camera_pose_queue.get(timeout=0.5)

                    if pose_data == "EXIT":
                        LOGGER.info(f"{thread_name} received exit signal.")
                        break

                    if isinstance(pose_data, dict):
                        try:
                            quat = airsim.euler_to_quaternion(
                                self.state.gimbal_roll * np.pi / 180,
                                self.state.gimbal_pitch * np.pi / 180,
                                self.state.gimbal_yaw * np.pi / 180,
                            )

                            pose_client.simSetCameraPose(
                                camera_name=pose_data.get(
                                    "camera_name", self.config.camera.camera_name
                                ),
                                pose=airsim.Pose(airsim.Vector3r(0, 0, 0.2), quat),
                                vehicle_name=pose_data.get(
                                    "vehicle_name", self.config.connection.vehicle_name
                                ),
                            )
                            LOGGER.debug(
                                f"Set camera pose: R={pose_data.get('roll', 0):.1f}, P={pose_data.get('pitch', 0):.1f}, Y={pose_data.get('yaw', 0):.1f}"
                            )
                        except Exception as e:
                            if "Connection" not in str(e) and "timeout" not in str(e):
                                LOGGER.error(f"Error setting camera pose: {e}")
                            time.sleep(0.5)

                except Empty:
                    continue
                except Exception as e:
                    LOGGER.error(f"Error in {thread_name}: {e}")
                    time.sleep(0.1)

        except Exception as e:
            LOGGER.error(f"Fatal error in {thread_name}: {e}")
        finally:
            LOGGER.info(f"{thread_name} exiting.")

    def _process_and_publish_image(self, image_info):
        """Process the image and publish to the event bus"""
        if not image_info or not isinstance(image_info, dict):
            LOGGER.warning("Invalid image information received")
            return None

        try:
            process_start_time = time.time()

            if "data" not in image_info or not image_info["data"]:
                LOGGER.warning("Image data is empty")
                return None

            if "height" not in image_info or "width" not in image_info:
                LOGGER.warning("Missing image dimension information")
                return None
                
            # Get the camera name from the image info
            camera_name = image_info.get("camera_name", "front_center")

            processed_img = self.image_strategy.process_image(
                image_info["data"], image_info["height"], image_info["width"]
            )
            process_time = time.time() - process_start_time

            if len(self.fetch_stats["process_times"]) > 100:
                self.fetch_stats["process_times"].pop(0)
            self.fetch_stats["process_times"].append(process_time)

            if processed_img is None:
                LOGGER.warning("Image processing failed, returned None")
                return None

            if processed_img.size == 0:
                LOGGER.warning("Processed image size is 0")
                return None

            LOGGER.debug(
                f"Processed {camera_name} image: shape={processed_img.shape}, process time={process_time*1000:.1f}ms"
            )

            self.event_bus.publish(
                "camera_image_updated",
                {
                    "image": processed_img,
                    "timestamp": image_info.get("timestamp", time.time()),
                    "process_time": process_time,
                    "camera_name": camera_name
                },
            )

            return processed_img
        except Exception as e:
            LOGGER.error(f"Error processing image: {e}")
            return None

    def _opencv_display_worker(self):
        """Thread for processing images."""
        thread_name = threading.current_thread().name
        LOGGER.info(f"Starting image processing worker: {thread_name}")

        try:
            while not self.state.exit_flag:
                processed_img = None
                try:
                    image_info = self.raw_image_queue.get(timeout=0.1)

                    if image_info == "STOP":
                        LOGGER.info("Display worker received STOP signal.")
                        break

                    if not isinstance(image_info, dict):
                        LOGGER.warning(
                            f"Unexpected item in raw image queue: {type(image_info)}"
                        )
                        continue

                    # Process and publish the image to event bus
                    processed_img = self._process_and_publish_image(image_info)

                except Empty:
                    time.sleep(0.005)
                    continue
                except Exception as q_err:
                    LOGGER.error(f"Error getting from raw image queue: {q_err}")
                    time.sleep(0.1)
                    continue

        except Exception as e:
            LOGGER.error(f"Fatal error in image processing worker: {e}")
            self.state.exit_flag = True
        finally:
            LOGGER.info(f"Image processing worker {thread_name} finished.")

    def start_display_thread(self):
        """Starts the image processing thread."""
        if self.opencv_display_thread and self.opencv_display_thread.is_alive():
            LOGGER.warning("Display thread already running.")
            return

        self.opencv_display_thread = threading.Thread(
            target=self._opencv_display_worker, name="DisplayThread", daemon=True
        )
        self.opencv_display_thread.start()
        LOGGER.info("Display thread started.")

    def stop_threads(self):
        """Stop all background threads."""
        LOGGER.info("Stopping SimulationDroneClient threads...")
        self.running = False  # Signal threads to stop

        self.event_bus.publish("camera_image_updated", {"exit": True})

        # Stop fetcher thread
        if self.fetch_thread and self.fetch_thread.is_alive():
            LOGGER.debug("Waiting for image fetcher thread to join...")
            self.fetch_thread.join(timeout=1.0)
            if self.fetch_thread.is_alive():
                LOGGER.warning("Image fetcher thread did not stop in time.")
            else:
                LOGGER.info("Image fetcher thread stopped.")
        self.fetch_thread = None

        # Stop display thread
        if self.opencv_display_thread and self.opencv_display_thread.is_alive():
            LOGGER.debug("Waiting for display thread to join...")
            try:
                self.raw_image_queue.put_nowait("STOP")  # Sentinel value
            except Full:
                LOGGER.warning("Display queue full, cannot add STOP sentinel.")

            self.opencv_display_thread.join(timeout=2.0)
            if self.opencv_display_thread.is_alive():
                LOGGER.warning("Display thread did not stop in time.")
            else:
                LOGGER.info("Display thread stopped.")
        self.opencv_display_thread = None

        # Stop camera pose thread
        if self.camera_pose_thread and self.camera_pose_thread.is_alive():
            LOGGER.debug("Waiting for camera pose thread to join...")
            try:
                self.camera_pose_queue.put_nowait("EXIT")  # Sentinel value
            except Full:
                LOGGER.warning("Camera pose queue full, cannot add EXIT sentinel.")

            self.camera_pose_thread.join(timeout=1.0)
            if self.camera_pose_thread.is_alive():
                LOGGER.warning("Camera pose thread did not stop in time.")
            else:
                LOGGER.info("Camera pose thread stopped.")
        self.camera_pose_thread = None

        # Shutdown parallel processing pool
        if self.process_pool:
            LOGGER.debug("Shutting down process pool...")
            self.process_pool.shutdown(wait=True)
            self.process_pool = None
            LOGGER.info("Process pool shut down.")

        # Clean up thread-specific clients
        self.thread_clients.clear()
        LOGGER.info("SimulationDroneClient threads stopped.")

    def get_performance_stats(self):
        """Get performance statistics"""
        stats = {}

        def avg(data):
            return sum(data) / len(data) if data else 0

        stats["avg_fetch_time_ms"] = avg(self.fetch_stats.get("fetch_times", [])) * 1000
        stats["avg_process_time_ms"] = (
            avg(self.fetch_stats.get("process_times", [])) * 1000
        )
        stats["raw_queue_size"] = self.raw_image_queue.qsize()
        stats["display_queue_size"] = (
            self.display_image_queue.qsize() if self.display_image_queue else 0
        )

        return stats


# =============================================================================
# DISPLAY APPLICATION ENTRY POINT
# =============================================================================

class DisplayApplication:
    """
    Standalone display application that can be run independently of the control system.
    Handles AirSim connection, image fetching, and OpenCV display.
    """
    
    def __init__(self, config: DroneConfig, state: ControlState = None, event_bus_instance: EventBus = None):
        """Initialize the display application"""
        self.config = config
        self.state = state if state else ControlState(gimbal_pitch=config.initial_pitch)
        self.event_bus = event_bus_instance if event_bus_instance else EventBus()
        
        # Components
        self.airsim_client = None
        self.display_manager = None
        
        # Event subscription
        self.event_bus.subscribe("exit_requested", self._handle_exit_request)
        self.event_bus.subscribe("gimbal_update", self._handle_gimbal_update)
    
    def _handle_exit_request(self, data):
        """Handle exit request events"""
        LOGGER.info(f"Exit requested: {data}")
        self.state.exit_flag = True
    
    def _handle_gimbal_update(self, pose_data):
        """Handle gimbal pose update events"""
        if hasattr(self.state, "gimbal_pitch"):
            self.state.gimbal_pitch = pose_data.get("pitch", self.state.gimbal_pitch)
            self.state.gimbal_roll = pose_data.get("roll", self.state.gimbal_roll)
            self.state.gimbal_yaw = pose_data.get("yaw", self.state.gimbal_yaw)
            
            # Forward to camera pose queue if AirSim client is available
            if self.airsim_client:
                try:
                    camera_pose_queue = self.event_bus.get_queue("camera_pose_queue")
                    if camera_pose_queue and not camera_pose_queue.full():
                        camera_pose_queue.put_nowait(pose_data)
                except Exception as e:
                    LOGGER.error(f"Error forwarding gimbal update: {e}")
    
    def initialize(self):
        """Initialize the display application components"""
        try:
            # Connect to AirSim
            LOGGER.info("Initializing AirSim connection...")
            self.airsim_client = SimulationDroneClient(
                self.config, self.state, self.event_bus
            )
            airsim_connected = self.airsim_client.connect()
            
            if not airsim_connected:
                LOGGER.error("Failed to connect to AirSim. Cannot start display.")
                return False
            
            # Start AirSim client threads
            self.airsim_client.start_image_fetcher()
            self.airsim_client.start_display_thread()
            self.airsim_client.start_camera_pose_thread()
            LOGGER.info("AirSim client threads started.")
            
            # Initialize display manager
            self.display_manager = DisplayManager(self.config, self.state, self.event_bus)
            self.display_manager.start()
            LOGGER.info("Display manager started.")
            
            return True
            
        except Exception as e:
            LOGGER.error(f"Error initializing display application: {e}")
            return False
    
    def run(self):
        """Run the display application main loop"""
        try:
            if not self.initialize():
                return 1
            
            LOGGER.info("Display application running. Press ESC or 'q' in any window to exit.")
            
            # Main loop - just wait for exit flag
            while not self.state.exit_flag:
                time.sleep(0.1)
            
            return 0
            
        except KeyboardInterrupt:
            LOGGER.info("Display application interrupted by user.")
            return 130
        except Exception as e:
            LOGGER.error(f"Error in display application: {e}")
            return 1
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources before exit"""
        LOGGER.info("Cleaning up display application resources...")
        
        # Stop display manager
        if self.display_manager:
            self.display_manager.stop()
        
        # Stop AirSim client threads
        if self.airsim_client:
            self.airsim_client.stop_threads()
        
        # Close all OpenCV windows
        try:
            cv2.destroyAllWindows()
            for _ in range(3):
                cv2.waitKey(1)
        except Exception as e:
            LOGGER.error(f"Error closing OpenCV windows: {e}")
        
        LOGGER.info("Display application cleanup complete.")


# Standalone execution
if __name__ == "__main__":
    import argparse
    
    def parse_arguments():
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description="AirSim Drone Display")
        
        # Connection parameters
        parser.add_argument(
            "--ip", 
            type=str, 
            default="172.19.160.1", 
            help="IP address of AirSim server"
        )
        parser.add_argument(
            "--vehicle", 
            type=str, 
            default="PX4", 
            help="Vehicle name in AirSim"
        )
        
        # Display parameters
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
        
        # Enable verbose logging
        parser.add_argument(
            "--verbose", 
            action="store_true",
            help="Enable verbose logging"
        )
        
        return parser.parse_args()
    
    # Parse arguments
    args = parse_arguments()
    
    # Set log level
    if args.verbose:
        LOGGER.setLevel(logging.DEBUG)
    
    # Create config
    config = DroneConfig()
    config.connection.ip_address = args.ip
    config.connection.vehicle_name = args.vehicle
    config.camera.width = args.width
    config.camera.height = args.height
    config.display.output_width = args.output_width
    config.display.output_height = args.output_height
    config.display.resize_output = not args.no_resize
    
    # Run application
    app = DisplayApplication(config)
    exit_code = app.run()
    
    # Exit with appropriate code
    import sys
    sys.exit(exit_code)