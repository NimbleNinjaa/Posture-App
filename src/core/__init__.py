"""Core functionality modules."""

from src.core.speech import SpeechManager
from src.core.utils import detect_available_cameras, draw_text
from src.core.video_thread import VideoThread

__all__ = [
    "SpeechManager",
    "VideoThread",
    "draw_text",
    "detect_available_cameras",
]
