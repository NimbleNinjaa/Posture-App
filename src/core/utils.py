"""Utility functions for the Posture Guard application."""

import cv2
import numpy as np


def draw_text(
    img: np.ndarray,
    text: str,
    org: tuple,
    scale: float = 0.9,
    thickness: int = 2,
    color: tuple = (255, 255, 255),
    bg: tuple = (0, 0, 0),
) -> None:
    """Draw text with background on an image.

    Args:
        img: Input image
        text: Text to draw
        org: Text position (x, y)
        scale: Font scale
        thickness: Text thickness
        color: Text color (B, G, R)
        bg: Background color (B, G, R)
    """
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - h - baseline - 6), (x + w + 6, y + 6), bg, -1)
    cv2.putText(
        img,
        text,
        (x + 3, y - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def centered_text(
    img: np.ndarray,
    text: str,
    scale: float = 1.2,
    thickness: int = 3,
    color: tuple = (255, 255, 255),
    bg: tuple = (0, 0, 0),
) -> None:
    """Draw centered text with background on an image.

    Args:
        img: Input image
        text: Text to draw
        scale: Font scale
        thickness: Text thickness
        color: Text color (B, G, R)
        bg: Background color (B, G, R)
    """
    h, w = img.shape[:2]
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.rectangle(img, (x - 10, y - th - baseline - 10), (x + tw + 10, y + 10), bg, -1)
    cv2.putText(
        img,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def detect_available_cameras(max_cameras: int = 10) -> list:
    """Detect available camera indices.

    Args:
        max_cameras: Maximum number of camera indices to test

    Returns:
        List of available camera indices
    """
    available_cameras = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
        cap.release()

    return available_cameras
