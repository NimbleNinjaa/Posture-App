"""Base backend class for posture detection algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np


class BaseBackend(ABC):
    """Abstract base class for posture detection backends."""

    def __init__(self, labels: List[str]):
        """Initialize the backend with posture labels.

        Args:
            labels: List of posture labels (e.g., ["good", "bad"])
        """
        self.labels = labels

    @abstractmethod
    def inference(self, frame_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """Perform inference on a video frame.

        Args:
            frame_bgr: Input frame in BGR format

        Returns:
            Tuple containing:
            - status_label: Posture classification (e.g., "good", "bad")
            - confidence: Confidence score (0.0 to 1.0)
            - extras: Dictionary with additional inference data
        """
        raise NotImplementedError("Subclasses must implement inference method")

    def draw(self, frame_bgr: np.ndarray, extras: Dict[str, Any]) -> None:
        """Draw visualization overlays on the frame.

        Args:
            frame_bgr: Input frame in BGR format (modified in-place)
            extras: Additional data from inference method
        """
        # Default implementation does nothing
        # Subclasses can override to add visualizations
        del frame_bgr, extras  # Mark as intentionally unused
