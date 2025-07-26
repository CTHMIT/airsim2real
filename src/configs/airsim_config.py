from typing import Set, Dict, Optional
from pydantic import BaseModel, Field


class ConnectionConfig(BaseModel):
    """Connection configuration parameters"""
    ip_address: str = Field(
        default="172.19.160.1", description="AirSim server IP address"
    )
    vehicle_name: str = Field(default="PX4", description="Vehicle name")


class CameraConfig(BaseModel):
    """Camera configuration parameters"""
    camera_name: str = Field(default="front_center", description="Camera name")
    image_type: int = Field(default=0, description="Image type")  # Will be replaced with airsim.ImageType.Scene
    pixels_as_float: bool = Field(default=False, description="Return pixels as float")
    compress: bool = Field(default=False, description="Compress image data")
    width: int = Field(default=960, description="Requested width (0 for default)")
    height: int = Field(default=540, description="Requested height (0 for default)")


class MultiCameraConfig(BaseModel):
    """Configuration for multiple cameras"""
    enabled: bool = Field(default=True, description="Enable multiple camera views")
    cameras: Dict[str, CameraConfig] = Field(
        default_factory=lambda: {
            "front_center": CameraConfig(camera_name="front_center"),
            "side_left": CameraConfig(camera_name="side_left"),
            "side_right": CameraConfig(camera_name="side_right")
        },
        description="Camera configurations by name"
    )
    active_camera: str = Field(default="front_center", description="Currently active camera")
    
    def get_camera_config(self, camera_name: str) -> Optional[CameraConfig]:
        """Get configuration for a specific camera"""
        return self.cameras.get(camera_name)


class DisplayConfig(BaseModel):
    """Display configuration parameters"""
    window_name: str = Field(
        default="AirSim Camera View", description="Display window name"
    )
    framerate_hz: float = Field(
        default=60.0,
        ge=10.0,
        le=240.0,
        description="Display framerate target in Hz (10-240)",
    )
    convert_to_rgb: bool = Field(default=False, description="Convert BGR to RGB")
    optimize_for_realtime: bool = Field(
        default=True, description="Optimize for realtime display"
    )
    use_threading: bool = Field(default=True, description="Use threaded image fetching")
    fetch_queue_size: int = Field(default=16, description="Raw image fetch queue size")
    thread_count: int = Field(
        default=3, ge=1, le=4, description="Number of background threads (1-6)"
    )
    queue_size: int = Field(default=1, description="Image queue size")
    waitKey_delay: int = Field(default=1, description="CV2 waitKey delay in ms (1-10)")
    resize_output: bool = Field(
        default=True, description="Resize output for performance"
    )
    output_width: int = Field(default=960, description="Output width if resizing")
    output_height: int = Field(default=540, description="Output height if resizing")

    @property
    def update_rate(self) -> float:
        """Convert framerate in Hz to update interval in seconds"""
        return 1.0 / self.framerate_hz


class ProcessingConfig(BaseModel):
    """Image processing configuration parameters"""
    enable_parallel_processing: bool = Field(
        default=True, description="Enable parallel image processing"
    )
    skip_frames_if_busy: bool = Field(
        default=True, description="Skip frames if processing is busy"
    )
    downscale_factor: float = Field(
        default=1.0,
        ge=0.1,
        le=1.0,
        description="Downscale factor for processing (0.1-1.0)",
    )


class DroneConfig(BaseModel):
    """Overall configuration parameters"""
    # AirSim Parameters
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    multi_camera: MultiCameraConfig = Field(default_factory=MultiCameraConfig) 
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    timeout: int = Field(default=10, description="Connection timeout (seconds)")

    # MAVSDK Connection
    system_address: str = Field("udp://:14540", description="MAVSDK connection URL")
    default_takeoff_altitude: float = Field(
        5.0, description="Default altitude for takeoff in meters"
    )
    connection_timeout: float = Field(
        10.0, description="Timeout for drone connection attempt in seconds"
    )

    # Control Parameters
    speed_increment: float = Field(0.5, description="Drone speed increment (m/s)")
    yaw_increment: float = Field(30.0, description="Drone yaw speed increment (deg/s)")
    max_speed: float = Field(5.0, description="Maximum forward/right/down speed (m/s)")
    max_yaw_speed: float = Field(45.0, description="Maximum yaw speed (deg/s)")
    max_gimbal_rate: float = Field(
        30.0, description="Maximum gimbal angular rate (deg/s)"
    )
    gimbal_angle_increment: float = Field(
        2.0, description="Gimbal angle increment per key press (degrees)"
    )
    acceleration_factor: float = Field(1.0, description="Control acceleration factor")
    decay_factor: float = Field(
        0.8, description="Control decay factor when keys are released"
    )
    zero_threshold: float = Field(0.05, description="Threshold to zero out values")

    initial_pitch: float = Field(
        -90.0, description="Initial gimbal pitch angle in degrees"
    )
    min_pitch_deg: float = Field(-90.0, description="Minimum gimbal pitch angle")
    max_pitch_deg: float = Field(30.0, description="Maximum gimbal pitch angle")
    min_roll_deg: float = Field(-45.0, description="Minimum gimbal roll angle")
    max_roll_deg: float = Field(45.0, description="Maximum gimbal roll angle")
    min_yaw_deg: float = Field(-180.0, description="Minimum gimbal yaw angle")
    max_yaw_deg: float = Field(180.0, description="Maximum gimbal yaw angle")


class ControlState(BaseModel):
    """Current control state of the drone and gimbal"""
    # Drone Velocity State
    velocity_forward: float = 0.0
    velocity_right: float = 0.0
    velocity_down: float = 0.0
    yawspeed: float = 0.0

    # Gimbal State
    gimbal_pitch: float = 0
    gimbal_roll: float = 0
    gimbal_yaw: float = 0

    gimbal_pitch_rate: float = 0.0
    gimbal_roll_rate: float = 0.0
    gimbal_yaw_rate: float = 0.0

    # System State
    exit_flag: bool = False
    speed_multiplier: float = 1.0
    pressed_keys: Set[str] = Field(default_factory=set)

    # Telemetry Data
    last_position: dict = Field(default_factory=dict)
    last_attitude: dict = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


