"""Configuration settings for the Posture Guard application."""

from dataclasses import dataclass


@dataclass
class CameraSettings:
    """Camera-related configuration settings."""

    default_cam_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    max_camera_scan: int = 10  # Maximum camera indices to scan


@dataclass
class PostureSettings:
    """Posture detection and alert configuration."""

    # Pose detection thresholds
    angle_threshold_deg: float = 15.0
    smoothing_buffer_size: int = 5
    bad_smoothing_buffer_size: int = 20  # ~2/3 sec at 30 FPS

    # Alert settings
    default_min_bad_ratio: float = 0.3  # Lower threshold - more sensitive
    default_voice_trigger_delay: float = 3.0  # Seconds of bad posture before voice
    speech_cooldown: float = 4.0  # Seconds between speech alerts

    # Pose angle thresholds for classification
    good_posture_angle_min: float = 165.0  # 165-180° is good
    mild_posture_angle_min: float = 150.0  # 150-165° is mild
    # < 150° is considered bad

    # Head forward position threshold
    head_forward_threshold: float = 0.15


@dataclass
class UISettings:
    """User interface configuration."""

    window_title: str = "Posture Guard — Side View"
    default_window_size: tuple = (1100, 750)
    video_label_min_size: tuple = (640, 360)
    log_max_blocks: int = 500

    # Status chip colors (CSS style)
    good_posture_color: str = "#2e7d32"
    bad_posture_color: str = "#c9342f"
    unknown_posture_color: str = "#444"


@dataclass
class AppLabels:
    """Labels used throughout the application."""

    def __post_init__(self):
        self.posture_labels = ["good", "bad"]


# Default configuration instances
CAMERA_CONFIG = CameraSettings()
POSTURE_CONFIG = PostureSettings()
UI_CONFIG = UISettings()
APP_LABELS = AppLabels()

# TTS Configuration
TTS_MESSAGES = {
    "bad_posture": "You have bad posture. Please straighten up.",
    "camera_started": "Camera started",
    "camera_stopped": "Camera stopped",
}

# Debug settings
DEBUG_SETTINGS = {
    "print_angle_info": True,
    "angle_print_interval": 30,  # Every 30 frames
    "print_posture_timing": True,
}
