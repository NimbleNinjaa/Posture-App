"""Video capture and processing thread."""

import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from PyQt6 import QtCore

from src.backend.base import BaseBackend
from src.config import CAMERA_CONFIG


class VideoThread(QtCore.QThread):
    """Thread for video capture and posture detection."""

    frame_ready = QtCore.pyqtSignal(np.ndarray, str, float, dict)

    def __init__(self, cam_index: int = 0, parent=None):
        """Initialize video thread.

        Args:
            cam_index: Camera index to use
            parent: Parent QObject
        """
        super().__init__(parent)
        self.cam_index = cam_index
        self.cap = None
        self.running = False
        self.backend: Optional[BaseBackend] = None
        self.bad_smoothing = deque(maxlen=CAMERA_CONFIG.frame_width // 64)  # ~20 frames
        self.camera_change_requested = False
        self.new_cam_index = 0

    def set_backend(self, backend: Optional[BaseBackend]) -> None:
        """Set the posture detection backend.

        Args:
            backend: Backend instance for posture detection
        """
        self.backend = backend

    def change_camera(self, new_cam_index: int) -> None:
        """Request camera change while thread is running.

        Args:
            new_cam_index: New camera index to switch to
        """
        self.new_cam_index = new_cam_index
        self.camera_change_requested = True

    def run(self) -> None:
        """Main thread execution loop."""
        self._init_camera()
        self.running = True

        while self.running:
            # Check for camera change request
            if self.camera_change_requested:
                self._change_camera_internal()
                self.camera_change_requested = False

            if self.cap is None or not self.cap.isOpened():
                time.sleep(0.01)
                continue

            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            if self.backend is not None:
                # Process frame with backend
                label, conf, extras = self.backend.inference(frame)

                # Update bad posture smoothing
                self.bad_smoothing.append(1 if label.lower().startswith("bad") else 0)
                bad_ratio = (
                    sum(self.bad_smoothing) / len(self.bad_smoothing)
                    if self.bad_smoothing
                    else 0.0
                )

                # Add guidance overlay
                frame = self._add_guidance_overlay(frame)

                # Let backend add annotations
                try:
                    self.backend.draw(frame, extras)
                except Exception:
                    pass  # Ignore drawing errors

                # Emit processed frame
                self.frame_ready.emit(
                    frame,
                    label,
                    float(conf if isinstance(conf, (int, float)) else 0.0),
                    {"bad_ratio": bad_ratio, "extras": extras},
                )
            else:
                # No backend - just emit raw frame
                self.frame_ready.emit(frame, "unknown", 0.0, {"bad_ratio": 0.0})

        if self.cap is not None:
            self.cap.release()

    def stop(self) -> None:
        """Stop the video thread."""
        self.running = False
        self.wait(500)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _init_camera(self) -> None:
        """Initialize camera capture."""
        try:
            if self.cap is not None:
                self.cap.release()

            self.cap = cv2.VideoCapture(
                self.cam_index, cv2.CAP_DSHOW if os.name == "nt" else 0
            )
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG.frame_height)

            # Test if camera is working
            if not self.cap.isOpened():
                print(f"Warning: Could not open camera {self.cam_index}")
        except Exception as e:
            print(f"Error initializing camera {self.cam_index}: {e}")

    def _change_camera_internal(self) -> None:
        """Change camera while thread is running."""
        print(f"Changing camera from {self.cam_index} to {self.new_cam_index}")
        self.cam_index = self.new_cam_index
        self._init_camera()
        # Clear smoothing buffer when changing cameras
        self.bad_smoothing.clear()

    def _add_guidance_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add guidance overlay to the frame.

        Args:
            frame: Input frame

        Returns:
            Frame with guidance overlay
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Guide line in the middle where ear-shoulder-hip should roughly align vertically
        cv2.line(overlay, (int(w * 0.33), 0), (int(w * 0.33), h), (100, 100, 100), 2)

        alpha = 0.15
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
