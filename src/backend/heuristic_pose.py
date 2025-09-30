"""Heuristic pose detection backend using MediaPipe."""

import math
from collections import deque
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from mediapipe.python.solutions.pose import Pose as mp_pose

from src.backend.base import BaseBackend
from src.config import DEBUG_SETTINGS, POSTURE_CONFIG
from src.core.utils import draw_text


class HeuristicPoseBackend(BaseBackend):
    """MediaPipe-based pose detection backend.

    Uses MediaPipe Pose to estimate angle between ear-shoulder-hip.
    Classifies posture based on angle thresholds defined in config.

    Angle Interpretation:
    --------------------
    The shoulder angle is measured at the shoulder point between two vectors:
    - Vector 1: ear → shoulder
    - Vector 2: hip → shoulder

    When sitting upright with good posture, these three points form a nearly
    straight line, resulting in an angle close to 180°.

    When slouching or leaning forward:
    - The ear moves forward relative to the shoulder
    - The angle at the shoulder decreases (becomes more acute)
    - Smaller angles (<150°) indicate bad posture
    - Larger angles (165-180°) indicate good posture
    """

    def __init__(self, labels: List[str]):
        """Initialize the heuristic pose backend.

        Args:
            labels: List of posture labels
        """
        super().__init__(labels)

        self.pose = mp_pose(
            static_image_mode=False, model_complexity=1, enable_segmentation=False
        )

        self.smoothing_buffer = deque(maxlen=POSTURE_CONFIG.smoothing_buffer_size)

        # Debug frame counter
        self._debug_frame_count = 0

    def inference(self, frame_bgr: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """Detect posture using MediaPipe pose landmarks.

        Args:
            frame_bgr: Input frame in BGR format

        Returns:
            Tuple of (label, confidence, extras)
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if (
            not hasattr(res, "pose_landmarks")
            or getattr(res, "pose_landmarks", None) is None
        ):
            return "unknown", 0.0, {}

        lm = getattr(res, "pose_landmarks").landmark

        # Check if user is positioned sideways (not frontal)
        is_side_view, side_view_info = self._validate_side_view(lm)

        if not is_side_view:
            # Return warning state when not properly positioned
            return (
                "unknown",
                0.0,
                {
                    "side_view_warning": True,
                    "side_view_info": side_view_info,
                    "landmarks": lm,
                },
            )

        # Right side landmarks (camera side). If mirrored, either side works.
        ear = np.array([lm[8].x, lm[8].y])  # Right ear
        shoulder = np.array([lm[12].x, lm[12].y])  # Right shoulder
        hip = np.array([lm[24].x, lm[24].y])  # Right hip

        # Calculate forward head posture using multiple metrics
        # 1. Angle at shoulder between ear-shoulder and hip-shoulder
        v1 = ear - shoulder
        v2 = hip - shoulder

        shoulder_angle = self._calculate_angle(v1, v2)

        # 2. Forward head position (ear x-coordinate relative to shoulder)
        head_forward_ratio = abs(ear[0] - shoulder[0]) / abs(
            hip[0] - shoulder[0] + 1e-6
        )

        # Classify posture based on angle and head position
        angle_bad = shoulder_angle < POSTURE_CONFIG.shoulder_angle_mild_min
        forward_bad = head_forward_ratio > POSTURE_CONFIG.head_forward_threshold

        # Use smoothing buffer for stability
        combined_bad_score = (angle_bad * 0.8) + (forward_bad * 0.2)
        self.smoothing_buffer.append(combined_bad_score)
        avg_bad_score = sum(self.smoothing_buffer) / len(self.smoothing_buffer)

        bad = avg_bad_score > 0.6  # Clear threshold for bad posture

        # Debug output
        self._debug_output(shoulder_angle, head_forward_ratio, avg_bad_score, bad)

        # Determine label and confidence
        label = "bad" if bad else "good"
        conf = min(1.0, avg_bad_score) if bad else max(0.7, 1.0 - avg_bad_score)

        extras = {
            "angle": shoulder_angle,
            "head_forward": head_forward_ratio,
            "landmarks": lm,
            "bad_score": avg_bad_score,
            "angle_bad": angle_bad,
            "forward_bad": forward_bad,
            "side_view_warning": False,
            "side_view_info": side_view_info,
        }

        return label, conf, extras

    def draw(self, frame_bgr: np.ndarray, extras: Dict[str, Any]) -> None:
        """Draw pose landmarks and angle information on the frame.

        Args:
            frame_bgr: Input frame (modified in-place)
            extras: Extra data from inference containing landmarks and angle
        """
        if not extras:
            return

        h, w = frame_bgr.shape[:2]
        lm = extras.get("landmarks")

        if not lm:
            return

        # Draw key points (ear, shoulder, hip)
        pts = [
            (int(landmark.x * w), int(landmark.y * h))
            for landmark in (lm[8], lm[12], lm[24])
        ]

        for p in pts:
            cv2.circle(frame_bgr, p, 6, (0, 255, 255), -1)

        # Draw lines connecting the points
        cv2.line(frame_bgr, pts[1], pts[0], (0, 255, 0), 2)  # shoulder to ear
        cv2.line(frame_bgr, pts[1], pts[2], (0, 255, 0), 2)  # shoulder to hip

        # Display angle information
        ang = extras.get("angle", 0.0)
        draw_text(frame_bgr, f"Shoulder angle: {ang:.1f}°", (10, 30))

    def _calculate_angle(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in degrees.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Angle in degrees
        """
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return math.degrees(math.acos(np.clip(cosang, -1, 1)))

    def _debug_output(
        self,
        shoulder_angle: float,
        head_forward_ratio: float,
        avg_bad_score: float,
        bad: bool,
    ) -> None:
        """Print debug information about posture analysis.

        Args:
            shoulder_angle: Calculated shoulder angle
            head_forward_ratio: Head forward position ratio
            avg_bad_score: Average bad posture score
            bad: Whether posture is classified as bad
        """
        if not DEBUG_SETTINGS["print_angle_info"]:
            return

        self._debug_frame_count += 1

        if self._debug_frame_count % DEBUG_SETTINGS["angle_print_interval"] == 0:
            posture_status = (
                "BAD"
                if bad
                else (
                    "MILD"
                    if shoulder_angle < POSTURE_CONFIG.shoulder_angle_good_min
                    else "GOOD"
                )
            )
            print(
                f"📐 Angle: {shoulder_angle:.1f}° ({posture_status}), "
                f"Forward: {head_forward_ratio:.3f}, "
                f"Score: {avg_bad_score:.3f}"
            )

    def _validate_side_view(self, landmarks) -> Tuple[bool, Dict[str, Any]]:
        """Validate if user is positioned sideways (not frontal).

        Uses shoulder width as indicator:
        - Narrow shoulder width (< threshold) = side view (good)
        - Wide shoulder width (> threshold) = frontal view (bad)

        Args:
            landmarks: MediaPipe pose landmarks

        Returns:
            Tuple of (is_side_view, info_dict)
        """
        # Get shoulder landmarks (11 = left shoulder, 12 = right shoulder)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]

        # Check landmark visibility/confidence
        if (
            left_shoulder.visibility < POSTURE_CONFIG.side_view_confidence_threshold
            or right_shoulder.visibility < POSTURE_CONFIG.side_view_confidence_threshold
        ):
            # If shoulders not visible enough, assume it's okay to avoid false warnings
            return True, {
                "shoulder_width_ratio": 0.0,
                "threshold": POSTURE_CONFIG.side_view_shoulder_width_threshold,
                "warning": "Low landmark visibility",
            }

        # Calculate shoulder width as horizontal distance
        shoulder_width = abs(left_shoulder.x - right_shoulder.x)

        # Shoulder width ratio (relative to frame width, which is normalized to 1.0)
        # In side view, shoulders are nearly aligned, so width should be small
        # In frontal view, shoulders are far apart, so width is large
        is_side_view = (
            shoulder_width < POSTURE_CONFIG.side_view_shoulder_width_threshold
        )

        info = {
            "shoulder_width_ratio": shoulder_width,
            "threshold": POSTURE_CONFIG.side_view_shoulder_width_threshold,
            "left_shoulder_visibility": left_shoulder.visibility,
            "right_shoulder_visibility": right_shoulder.visibility,
        }

        return is_side_view, info
