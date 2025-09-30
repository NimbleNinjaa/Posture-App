"""Main application window for Posture Guard."""

import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from src.backend import HeuristicPoseBackend
from src.config import APP_LABELS, POSTURE_CONFIG, UI_CONFIG
from src.core import SpeechManager, VideoThread
from src.ui.helpers import MODERN_STYLE, hstack, stretch, vstack
from src.ui.settings_dialog import SettingsDialog


class PostureApp(QtWidgets.QMainWindow):
    """Main application window for posture monitoring."""

    def __init__(self):
        """Initialize the PostureApp main window."""
        super().__init__()
        self.setWindowTitle(UI_CONFIG.window_title)
        self.resize(*UI_CONFIG.default_window_size)

        # Initialize labels and backend
        self.labels = APP_LABELS.posture_labels
        self.backend = HeuristicPoseBackend(self.labels)

        # Initialize speech manager
        self.speech_manager = SpeechManager()

        # Application state
        self.voice_enabled = True
        self.min_bad_ratio = POSTURE_CONFIG.default_min_bad_ratio
        self.voice_trigger_delay = POSTURE_CONFIG.default_voice_trigger_delay
        self.bad_posture_start_time: Optional[float] = None
        self.bad_posture_accumulated_time = 0.0
        self.last_voice_trigger_time = 0.0

        # Build UI and initialize video
        self._build_ui()

        # Initialize video thread
        self.worker = VideoThread(cam_index=0)
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.set_backend(self.backend)

        self.log_msg("Using MediaPipe pose detection (default)")

    def _build_ui(self) -> None:
        """Build the main user interface."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # Video feed display
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(*UI_CONFIG.video_label_min_size)
        self.video_label.setStyleSheet("background: #0e0e10; border-radius: 16px;")

        # Status chip for displaying posture status
        self.status_chip = QtWidgets.QLabel("—")
        self.status_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_chip.setFixedHeight(44)
        self.status_chip.setStyleSheet(
            "border-radius: 22px; padding: 0 18px; font-weight: 600;"
            f"color: white; background: {UI_CONFIG.unknown_posture_color};"
        )

        # Control buttons
        self.btn_settings = QtWidgets.QPushButton("⚙️ Settings")
        self.btn_start = QtWidgets.QPushButton("Start Camera")
        self.btn_stop = QtWidgets.QPushButton("Stop")

        self.btn_settings.clicked.connect(self.open_settings)
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)

        buttons = hstack([self.btn_start, self.btn_stop, self.btn_settings])

        # Activity log
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(UI_CONFIG.log_max_blocks)
        self.log.setStyleSheet(
            "background:#0e0e10;color:#d7d7db;border-radius:12px;padding:8px;"
        )
        self.log.setPlaceholderText("Logs will appear here…")

        # Main layout
        layout_widget = vstack(
            [
                self.video_label,
                hstack([QtWidgets.QLabel("Status:"), self.status_chip, stretch()]),
                buttons,
                QtWidgets.QLabel("Activity log"),
                self.log,
                stretch(),
            ]
        )

        layout = layout_widget.layout()
        if layout is not None:
            layout.setContentsMargins(16, 16, 16, 16)
            central.setLayout(layout)

        self._build_menu()
        self._apply_modern_style()

    def _build_menu(self) -> None:
        """Build the application menu bar."""
        bar = self.menuBar()
        if bar is None:
            bar = QtWidgets.QMenuBar(self)
            self.setMenuBar(bar)

        # File menu
        m_file = QtWidgets.QMenu("File", self)
        bar.addMenu(m_file)

        act_quit = QtGui.QAction("Quit", self)
        m_file.addAction(act_quit)
        act_quit.triggered.connect(self.close)

        # Camera menu
        m_cam = QtWidgets.QMenu("Camera", self)
        bar.addMenu(m_cam)

        act_start = QtGui.QAction("Start", self)
        act_stop = QtGui.QAction("Stop", self)
        m_cam.addAction(act_start)
        m_cam.addAction(act_stop)

        act_start.triggered.connect(self.start_camera)
        act_stop.triggered.connect(self.stop_camera)

    def _apply_modern_style(self) -> None:
        """Apply the modern dark theme stylesheet."""
        self.setStyleSheet(MODERN_STYLE)

    def log_msg(self, msg: str) -> None:
        """Log a message with timestamp to the activity log.

        Args:
            msg: Message to log
        """
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {msg}")

    def open_settings(self) -> None:
        """Open the settings dialog."""
        settings_dialog = SettingsDialog(self)
        settings_dialog.exec()

    def start_camera(self) -> None:
        """Start the camera video feed."""
        if not self.worker.isRunning():
            self.worker.start()
            self.log_msg("Camera started")

    def stop_camera(self) -> None:
        """Stop the camera video feed."""
        if self.worker.isRunning():
            self.worker.stop()
            self.log_msg("Camera stopped")

    @QtCore.pyqtSlot(np.ndarray, str, float, dict)
    def on_frame(
        self, frame_bgr: np.ndarray, label: str, conf: float, info: Dict[str, Any]
    ) -> None:
        """Handle new frame from video thread.

        Args:
            frame_bgr: Video frame in BGR format
            label: Posture classification label
            conf: Confidence score
            info: Additional information from detection
        """
        bad_ratio = info.get("bad_ratio", 0.0)
        disp = frame_bgr.copy()

        # Check for side-view warning
        extras = info.get("extras", {})
        side_view_warning = extras.get("side_view_warning", False)

        if side_view_warning:
            # Draw warning overlay when not positioned sideways
            self._draw_side_view_warning(disp, extras.get("side_view_info", {}))

        # Update status chip
        self._update_status_chip(label, conf, bad_ratio, info)

        # Process posture timing and voice alerts (only if properly positioned)
        if not side_view_warning:
            self._process_posture_timing(bad_ratio)
        else:
            # Reset timing when not properly positioned
            self._reset_posture_timing()

        # Convert and display frame
        self._display_frame(disp)

    def _update_status_chip(
        self, label: str, conf: float, bad_ratio: float, info: Dict[str, Any]
    ) -> None:
        """Update the status chip with current posture information.

        Args:
            label: Posture classification label
            conf: Confidence score
            bad_ratio: Bad posture ratio
            info: Additional detection information
        """
        is_bad = label.lower().startswith("bad")
        angle = info.get("extras", {}).get("angle", 0)

        chip_text = (
            f"{label.upper()}  ({conf:.2f})  •  "
            f"angle: {angle:.1f}°  •  ratio: {bad_ratio:.2f}"
        )

        self.status_chip.setText(chip_text)

        if is_bad:
            color = UI_CONFIG.bad_posture_color
        else:
            color = UI_CONFIG.good_posture_color

        self.status_chip.setStyleSheet(
            f"border-radius:22px;padding:0 18px;font-weight:600;"
            f"color:white;background:{color};"
        )

    def _process_posture_timing(self, bad_ratio: float) -> None:
        """Process bad posture timing and trigger voice alerts.

        Args:
            bad_ratio: Current bad posture ratio
        """
        current_bad_posture = bad_ratio >= self.min_bad_ratio
        current_time = time.time()

        if current_bad_posture:
            # Start or continue tracking bad posture time
            if self.bad_posture_start_time is None:
                self.bad_posture_start_time = current_time
                if POSTURE_CONFIG.speech_cooldown:
                    print(
                        f"🔴 Bad posture detected! Starting timer... (ratio: {bad_ratio:.3f})"
                    )
            else:
                # Accumulate bad posture time
                self.bad_posture_accumulated_time += (
                    current_time - self.bad_posture_start_time
                )
                self.bad_posture_start_time = current_time

            # Debug output for timing
            self._debug_posture_timing()

            # Trigger voice alert if conditions are met
            self._check_voice_trigger(current_time)

        else:
            # Reset timing when posture is good
            self._reset_posture_timing()

    def _debug_posture_timing(self) -> None:
        """Print debug information about posture timing."""
        if not POSTURE_CONFIG.speech_cooldown:
            return

        # Print every 0.5 seconds
        if int(self.bad_posture_accumulated_time * 2) > int(
            (self.bad_posture_accumulated_time - 0.1) * 2
        ):
            print(
                f"⏱️  Bad posture time: {self.bad_posture_accumulated_time:.1f}s / "
                f"{self.voice_trigger_delay:.1f}s "
                f"(voice: {self.voice_enabled}, tts: {self.speech_manager.is_available})"
            )

    def _check_voice_trigger(self, current_time: float) -> None:
        """Check if voice alert should be triggered.

        Args:
            current_time: Current timestamp
        """
        if (
            self.voice_enabled
            and self.speech_manager.is_available
            and self.bad_posture_accumulated_time >= self.voice_trigger_delay
            and (current_time - self.last_voice_trigger_time)
            >= POSTURE_CONFIG.speech_cooldown
        ):
            print(
                f"🔊 TRIGGERING VOICE! Accumulated: {self.bad_posture_accumulated_time:.1f}s, "
                f"Threshold: {self.voice_trigger_delay:.1f}s"
            )

            self.speech_manager.speak_bad_posture_alert()
            self.last_voice_trigger_time = current_time
            # Reset accumulation after triggering
            self.bad_posture_accumulated_time = 0.0

    def _reset_posture_timing(self) -> None:
        """Reset posture timing when good posture is detected."""
        if self.bad_posture_start_time is not None:
            print(
                f"✅ Good posture restored! Reset timer "
                f"(was: {self.bad_posture_accumulated_time:.1f}s)"
            )
        self.bad_posture_start_time = None
        self.bad_posture_accumulated_time = 0.0

    def _draw_side_view_warning(
        self, frame: np.ndarray, side_view_info: Dict[str, Any]
    ) -> None:
        """Draw warning overlay when user is not positioned sideways.

        Args:
            frame: Video frame to draw on
            side_view_info: Information about side-view validation
        """
        h, w = frame.shape[:2]

        # Draw semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Warning text
        warning_text = "⚠️ POSITION YOURSELF SIDEWAYS"
        instruction_text = "Turn so your side faces the camera"

        # Get text size for centering
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2

        (text_width, text_height), _ = cv2.getTextSize(
            warning_text, font, font_scale, thickness
        )
        (inst_width, inst_height), _ = cv2.getTextSize(instruction_text, font, 0.8, 2)

        # Draw warning text (centered)
        warning_x = (w - text_width) // 2
        warning_y = (h // 2) - 40
        cv2.putText(
            frame,
            warning_text,
            (warning_x, warning_y),
            font,
            font_scale,
            (0, 140, 255),  # Orange color
            thickness,
            cv2.LINE_AA,
        )

        # Draw instruction text
        inst_x = (w - inst_width) // 2
        inst_y = warning_y + 50
        cv2.putText(
            frame,
            instruction_text,
            (inst_x, inst_y),
            font,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Draw visual indicator showing proper positioning
        self._draw_positioning_guide(frame, w, h)

        # Display shoulder width info for debugging
        shoulder_width_ratio = side_view_info.get("shoulder_width_ratio", 0.0)
        threshold = side_view_info.get("threshold", 0.15)
        debug_text = (
            f"Shoulder width: {shoulder_width_ratio:.2f} (need < {threshold:.2f})"
        )
        cv2.putText(
            frame,
            debug_text,
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    def _draw_positioning_guide(self, frame: np.ndarray, w: int, h: int) -> None:
        """Draw visual guide showing proper side-view positioning.

        Args:
            frame: Video frame to draw on
            w: Frame width
            h: Frame height
        """
        # Draw two stick figures: one showing wrong (frontal) and one showing correct (side) view
        guide_y = h - 180

        # Wrong position (frontal view) - red X
        wrong_x = w // 3
        cv2.putText(
            frame,
            "❌ Frontal",
            (wrong_x - 50, guide_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        # Simple frontal person representation
        cv2.circle(frame, (wrong_x, guide_y), 15, (0, 0, 255), 2)  # Head
        cv2.line(
            frame, (wrong_x, guide_y + 15), (wrong_x, guide_y + 60), (0, 0, 255), 2
        )  # Body
        cv2.line(
            frame,
            (wrong_x - 30, guide_y + 35),
            (wrong_x + 30, guide_y + 35),
            (0, 0, 255),
            2,
        )  # Arms

        # Correct position (side view) - green check
        correct_x = 2 * w // 3
        cv2.putText(
            frame,
            "✓ Side View",
            (correct_x - 50, guide_y - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        # Simple side view person representation
        cv2.circle(frame, (correct_x, guide_y), 15, (0, 255, 0), 2)  # Head
        cv2.line(
            frame, (correct_x, guide_y + 15), (correct_x, guide_y + 60), (0, 255, 0), 2
        )  # Body
        cv2.line(
            frame,
            (correct_x, guide_y + 35),
            (correct_x + 30, guide_y + 30),
            (0, 255, 0),
            2,
        )  # Arm

    def _display_frame(self, frame: np.ndarray) -> None:
        """Convert and display the video frame.

        Args:
            frame: Video frame to display
        """
        # Convert BGR to RGB for Qt
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)

        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.video_label.setPixmap(pix)

    def closeEvent(self, a0: Optional[QtGui.QCloseEvent]) -> None:
        """Handle application close event.

        Args:
            a0: Close event
        """
        try:
            self.stop_camera()
            self.speech_manager.cleanup()
        except Exception:
            pass

        if a0 is not None:
            super().closeEvent(a0)
