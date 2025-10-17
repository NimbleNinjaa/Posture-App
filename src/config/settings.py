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
    smoothing_buffer_size: int = 5

    # Alert settings
    default_min_bad_ratio: float = (
        0.5  # Threshold for bad posture detection (0.5 = 50% of frames must be bad)
    )
    default_voice_trigger_delay: float = 3.0  # Seconds of bad posture before voice
    speech_cooldown: float = 4.0  # Seconds between speech alerts

    # Shoulder angle thresholds for classification
    # Note: Angle is measured at the shoulder point between ear→shoulder and hip→shoulder vectors
    # Smaller angles (<150°) indicate slouching/forward lean (bad posture)
    # Larger angles (165-180°) indicate upright alignment (good posture)
    shoulder_angle_good_min: float = 165.0  # 165-180° is good posture
    shoulder_angle_mild_min: float = 150.0  # 150-165° is mild posture
    # < 150° is considered bad posture

    # Head forward position threshold
    head_forward_threshold: float = 0.15

    # Side-view validation settings
    # Shoulder width ratio threshold: ratio of shoulder width to frame width
    # Lower values indicate side view (shoulders are aligned), higher values indicate frontal view
    side_view_shoulder_width_threshold: float = 0.15  # < 15% of frame width = side view
    side_view_confidence_threshold: float = (
        0.5  # Minimum landmark visibility for validation
    )

    # Screen overlay settings
    overlay_enabled: bool = True  # Enable/disable screen overlay


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

    # Screen overlay settings
    overlay_position: tuple = (20, 20)  # Top-left corner position (x, y)
    overlay_font_size: int = 24
    overlay_padding: int = 15
    overlay_opacity: float = 0.9


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
}

# Debug settings
DEBUG_SETTINGS = {
    "print_angle_info": True,
    "angle_print_interval": 30,  # Every 30 frames
}
