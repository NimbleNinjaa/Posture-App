from __future__ import annotations

import math
import os
import queue
import shutil
import sys
import threading
import time
from collections import deque
from typing import List, Optional

import cv2
import numpy as np
from mediapipe.python.solutions.pose import Pose as mp_pose
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

# Universal TTS with better voice quality
try:
    import tempfile

    import pygame
    from gtts import gTTS

    TTS_AVAILABLE = True
except Exception as e:
    print(f"TTS imports failed: {e}")
    gTTS = None
    pygame = None
    tempfile = None
    TTS_AVAILABLE = False


# ------------------------
# Utility helpers
# ------------------------


def draw_text(
    img, text, org, scale=0.9, thickness=2, color=(255, 255, 255), bg=(0, 0, 0)
):
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
    img, text, scale=1.2, thickness=3, color=(255, 255, 255), bg=(0, 0, 0)
):
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


# ------------------------
# Inference backends
# ------------------------


class BaseBackend:
    def __init__(self, labels: List[str]):
        self.labels = labels

    def inference(self, frame_bgr: np.ndarray):
        """Return (status_label, confidence, extras)."""
        raise NotImplementedError

    def draw(self, frame_bgr: np.ndarray, extras: dict) -> None:
        del frame_bgr, extras  # Mark as intentionally unused


class HeuristicPoseBackend(BaseBackend):
    """No model required – uses MediaPipe Pose to estimate angle between ear-shoulder-hip.
    If head/torso angle exceeds threshold, marks as bad.
    """

    def __init__(self, labels: List[str], angle_threshold_deg: float = 15.0):
        super().__init__(labels)
        if mp_pose is None:
            raise RuntimeError("mediapipe not installed. pip install mediapipe")
        self.pose = mp_pose(
            static_image_mode=False, model_complexity=1, enable_segmentation=False
        )
        self.angle_threshold = (
            angle_threshold_deg  # Increased from 25 to 35 for better sensitivity
        )
        self.last_landmarks = None
        self.smoothing_buffer = deque(maxlen=5)  # Add smoothing buffer for stability

    def inference(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if (
            not hasattr(res, "pose_landmarks")
            or getattr(res, "pose_landmarks", None) is None
        ):
            return "unknown", 0.0, {}
        lm = getattr(res, "pose_landmarks").landmark
        self.last_landmarks = lm
        # Right side landmarks (camera side). If mirrored, either side works.
        ear = np.array([lm[8].x, lm[8].y])  # Right ear
        shoulder = np.array([lm[12].x, lm[12].y])  # Right shoulder
        hip = np.array([lm[24].x, lm[24].y])  # Right hip
        # Calculate forward head posture using multiple metrics
        # 1. Angle at shoulder between ear-shoulder and hip-shoulder
        v1 = ear - shoulder
        v2 = hip - shoulder

        def angle(a, b):
            cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
            return math.degrees(math.acos(np.clip(cosang, -1, 1)))

        shoulder_angle = angle(v1, v2)

        # 2. Forward head position (ear x-coordinate relative to shoulder)
        head_forward_ratio = abs(ear[0] - shoulder[0]) / abs(
            hip[0] - shoulder[0] + 1e-6
        )

        # Proper posture guidelines:
        # Neutral alignment: 165-180°
        # Mild forward head: 150-165°
        # Bad posture: < 150°
        angle_bad = shoulder_angle < 150  # Bad posture when < 150°
        forward_bad = head_forward_ratio > 0.15  # Head significantly forward

        # Use smoothing buffer for stability
        combined_bad_score = (angle_bad * 0.8) + (
            forward_bad * 0.2
        )  # Angle primary, forward secondary
        self.smoothing_buffer.append(combined_bad_score)
        avg_bad_score = sum(self.smoothing_buffer) / len(self.smoothing_buffer)

        bad = avg_bad_score > 0.6  # Clear threshold for bad posture

        # Debug output to track angles
        if hasattr(self, "_debug_frame_count"):
            self._debug_frame_count += 1
        else:
            self._debug_frame_count = 0

        if self._debug_frame_count % 30 == 0:  # Every 30 frames (~1 second)
            posture_status = (
                "BAD" if bad else ("MILD" if shoulder_angle < 165 else "GOOD")
            )
            print(
                f"📐 Angle: {shoulder_angle:.1f}° ({posture_status}), Forward: {head_forward_ratio:.3f}, Score: {avg_bad_score:.3f}"
            )
        label = "bad" if bad else "good"
        conf = min(1.0, avg_bad_score) if bad else max(0.7, 1.0 - avg_bad_score)

        extras = {
            "angle": shoulder_angle,
            "head_forward": head_forward_ratio,
            "landmarks": lm,
            "bad_score": avg_bad_score,
            "angle_bad": angle_bad,
            "forward_bad": forward_bad,
        }
        return label, conf, extras

    def draw(self, frame_bgr: np.ndarray, extras: dict) -> None:
        if not extras:
            return
        h, w = frame_bgr.shape[:2]
        lm = extras.get("landmarks")
        if not lm:
            return
        # Draw key points
        pts = [
            (int(l.x * w), int(l.y * h)) for l in (lm[8], lm[12], lm[24])
        ]  # ear, shoulder, hip
        for p in pts:
            cv2.circle(frame_bgr, p, 6, (0, 255, 255), -1)
        # Lines
        cv2.line(frame_bgr, pts[1], pts[0], (0, 255, 0), 2)
        cv2.line(frame_bgr, pts[1], pts[2], (0, 255, 0), 2)
        ang = extras.get("angle", 0.0)
        draw_text(frame_bgr, f"Shoulder angle: {ang:.1f}°", (10, 30))


# ------------------------
# Video worker thread
# ------------------------


class VideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray, str, float, dict)

    def __init__(self, cam_index: int = 0, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self.cap = None
        self.running = False
        self.backend: Optional[BaseBackend] = None
        self.bad_smoothing = deque(maxlen=20)  # ~2/3 sec at 30 FPS
        self.camera_change_requested = False
        self.new_cam_index = 0

    def set_backend(self, backend: Optional[BaseBackend]):
        self.backend = backend

    def change_camera(self, new_cam_index: int):
        """Request camera change while thread is running"""
        self.new_cam_index = new_cam_index
        self.camera_change_requested = True

    def run(self):
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
                label, conf, extras = self.backend.inference(frame)
                self.bad_smoothing.append(1 if label.lower().startswith("bad") else 0)
                bad_ratio = (
                    sum(self.bad_smoothing) / len(self.bad_smoothing)
                    if self.bad_smoothing
                    else 0.0
                )
                # Draw guidance overlay (side-view guide)
                overlay = frame.copy()
                h, w = frame.shape[:2]
                # Guide line in the middle where ear-shoulder-hip should roughly align vertically
                cv2.line(
                    overlay, (int(w * 0.33), 0), (int(w * 0.33), h), (100, 100, 100), 2
                )
                alpha = 0.15
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                # Let backend add annotations
                try:
                    self.backend.draw(frame, extras)
                except Exception:
                    pass
                self.frame_ready.emit(
                    frame,
                    label,
                    float(conf if isinstance(conf, (int, float)) else 0.0),
                    {"bad_ratio": bad_ratio, "extras": extras},
                )
            else:
                self.frame_ready.emit(frame, "unknown", 0.0, {"bad_ratio": 0.0})
        if self.cap is not None:
            self.cap.release()

    def _init_camera(self):
        """Initialize camera capture"""
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(
                self.cam_index, cv2.CAP_DSHOW if os.name == "nt" else 0
            )
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Test if camera is working
            if not self.cap.isOpened():
                print(f"Warning: Could not open camera {self.cam_index}")
        except Exception as e:
            print(f"Error initializing camera {self.cam_index}: {e}")

    def _change_camera_internal(self):
        """Change camera while thread is running"""
        print(f"Changing camera from {self.cam_index} to {self.new_cam_index}")
        self.cam_index = self.new_cam_index
        self._init_camera()
        # Clear smoothing buffer when changing cameras
        self.bad_smoothing.clear()

    def stop(self):
        self.running = False
        self.wait(500)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ------------------------
# Settings Dialog
# ------------------------


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setFixedSize(400, 350)
        self.parent_app = parent

        # Store current settings
        if self.parent_app:
            self.original_cam_index = self.parent_app.worker.cam_index
            self.original_sensitivity = self.parent_app.min_bad_ratio
            self.original_alert_enabled = self.parent_app.alert_enabled
            self.original_voice_enabled = self.parent_app.voice_enabled
            self.original_voice_delay = self.parent_app.voice_trigger_delay

        self._build_ui()
        self._apply_style()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Camera selector with dynamic detection
        camera_group = QtWidgets.QGroupBox("Camera Settings")
        camera_layout = QtWidgets.QVBoxLayout(camera_group)

        self.camera_combo = QtWidgets.QComboBox()
        self._populate_cameras()
        if self.parent_app:
            current_cam = self.parent_app.worker.cam_index
            for i in range(self.camera_combo.count()):
                if self.camera_combo.itemData(i) == current_cam:
                    self.camera_combo.setCurrentIndex(i)
                    break
        # Remove direct connection - will apply on save

        camera_layout.addWidget(QtWidgets.QLabel("Select Camera:"))
        camera_layout.addWidget(self.camera_combo)

        # Alert settings
        alert_group = QtWidgets.QGroupBox("Alert Settings")
        alert_layout = QtWidgets.QVBoxLayout(alert_group)

        # Sensitivity slider
        self.sens_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(10, 90)
        if self.parent_app:
            self.sens_slider.setValue(int(self.parent_app.min_bad_ratio * 100))
        self.sens_slider.valueChanged.connect(self.update_sensitivity_label)

        sens_label = QtWidgets.QLabel("Sensitivity:")
        self.sens_value_label = QtWidgets.QLabel(f"{self.sens_slider.value()}%")
        sens_row = hstack([sens_label, self.sens_slider, self.sens_value_label])

        # Voice trigger delay setting
        voice_delay_label = QtWidgets.QLabel("Voice trigger delay:")
        self.voice_delay_spinbox = QtWidgets.QDoubleSpinBox()
        self.voice_delay_spinbox.setRange(1.0, 30.0)  # 1 to 30 seconds
        self.voice_delay_spinbox.setSingleStep(0.5)
        self.voice_delay_spinbox.setSuffix(" seconds")
        self.voice_delay_spinbox.setDecimals(1)
        if self.parent_app:
            self.voice_delay_spinbox.setValue(self.parent_app.voice_trigger_delay)
        else:
            self.voice_delay_spinbox.setValue(3.0)

        voice_delay_row = hstack([voice_delay_label, self.voice_delay_spinbox])

        # Toggle checkboxes
        self.chk_alert = QtWidgets.QCheckBox("Visual alerts")
        self.chk_voice = QtWidgets.QCheckBox("Voice alerts")
        if self.parent_app:
            self.chk_alert.setChecked(self.parent_app.alert_enabled)
            self.chk_voice.setChecked(self.parent_app.voice_enabled)
        # Remove direct connections - will apply on save

        alert_layout.addWidget(sens_row)
        alert_layout.addWidget(voice_delay_row)
        alert_layout.addWidget(self.chk_alert)
        alert_layout.addWidget(self.chk_voice)

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.save_btn = QtWidgets.QPushButton("Save")

        self.cancel_btn.clicked.connect(self.reject)
        self.save_btn.clicked.connect(self.save_and_close)

        button_layout.addStretch()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.save_btn)

        layout.addWidget(camera_group)
        layout.addWidget(alert_group)
        layout.addStretch()
        layout.addLayout(button_layout)

    def _apply_style(self):
        self.setStyleSheet("""
            QDialog { background: #0b0b0d; }
            QGroupBox { color: #e5e5e9; font-size: 14px; font-weight: 600; padding-top: 15px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
            QLabel, QCheckBox { color: #e5e5e9; font-size: 14px; }
            QSlider::groove:horizontal { height: 6px; background: #222; border-radius: 3px; }
            QSlider::handle:horizontal { width: 18px; background: #4b8bff; border-radius: 9px; margin: -6px 0; }
            QPushButton { background: #141418; color: #e5e5e9; border: 1px solid #26262b; padding: 10px 16px; border-radius: 12px; font-weight: 600; }
            QPushButton:hover { border-color: #3a3a42; }
            QPushButton:pressed { background: #0e0e10; }
            QComboBox { background: #141418; color: #e5e5e9; border: 1px solid #26262b; padding: 8px; border-radius: 8px; }
            QComboBox::drop-down { border: none; }
            QComboBox::down-arrow { image: none; }
        """)

    def _populate_cameras(self):
        """Dynamically detect available cameras"""
        available_cameras = []

        # Test camera indices 0-9 to find available cameras
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                    camera_name = f"Camera {i}"
                    if i == 0:
                        camera_name += " (default)"
                    self.camera_combo.addItem(camera_name, i)
            cap.release()

        if not available_cameras:
            self.camera_combo.addItem("No cameras detected", 0)
            print("Warning: No cameras detected")
        else:
            print(f"Detected cameras: {available_cameras}")

    def update_sensitivity_label(self, v: int):
        self.sens_value_label.setText(f"{v}%")

    def save_and_close(self):
        if not self.parent_app:
            self.accept()
            return

        # Apply camera change
        new_cam_index = self.camera_combo.itemData(self.camera_combo.currentIndex())
        if new_cam_index != self.original_cam_index:
            self.parent_app.log_msg(f"Camera changed to: {new_cam_index}")
            if self.parent_app.worker.isRunning():
                # Change camera while running
                self.parent_app.worker.change_camera(new_cam_index)
            else:
                # Just update the index if not running
                self.parent_app.worker.cam_index = new_cam_index

        # Apply sensitivity change
        new_sensitivity = self.sens_slider.value() / 100.0
        if new_sensitivity != self.original_sensitivity:
            self.parent_app.min_bad_ratio = new_sensitivity
            self.parent_app.log_msg(f"Sensitivity set: {new_sensitivity:.2f}")

        # Apply voice trigger delay change
        new_voice_delay = self.voice_delay_spinbox.value()
        if new_voice_delay != self.original_voice_delay:
            self.parent_app.voice_trigger_delay = new_voice_delay
            self.parent_app.log_msg(
                f"Voice trigger delay set: {new_voice_delay:.1f} seconds"
            )
            # Reset bad posture timing when delay changes
            self.parent_app.bad_posture_start_time = None
            self.parent_app.bad_posture_accumulated_time = 0.0

        # Apply alert settings
        self.parent_app.alert_enabled = self.chk_alert.isChecked()
        self.parent_app.voice_enabled = self.chk_voice.isChecked()

        self.accept()


# ------------------------
# Main UI
# ------------------------


class PostureApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Posture Guard — Side View")
        self.resize(1100, 750)
        self.labels: List[str] = ["good", "bad"]
        # Initialize MediaPipe backend by default
        self.backend = HeuristicPoseBackend(self.labels)

        # Initialize universal TTS system with Google TTS
        self.speech_queue = queue.Queue(maxsize=1)
        self.speech_worker_running = False
        self.tts_available = False
        self.last_speech_time = 0.0
        self.speech_cooldown = 4.0  # 4 seconds between speech
        if tempfile is not None:
            self.temp_dir = tempfile.mkdtemp()
        else:
            self.temp_dir = "/tmp"

        if TTS_AVAILABLE and pygame is not None:
            try:
                # Initialize pygame mixer for audio playback
                pygame.mixer.init()
                self.tts_available = True
                self._start_speech_worker()
                print("Voice alerts enabled with high-quality TTS")
            except Exception as e:
                print(f"TTS initialization failed: {e}")
                self.tts_available = False
        else:
            print("TTS libraries not available - voice alerts disabled")

        # State (must be set BEFORE building UI)
        self.alert_enabled = True
        self.voice_enabled = True
        self.min_bad_ratio = 0.3  # Lower threshold - more sensitive to bad posture
        self.voice_trigger_delay = 3.0  # Seconds of bad posture before voice triggers
        self.bad_posture_start_time = None  # When bad posture started
        self.bad_posture_accumulated_time = 0.0  # Total time in bad posture
        self.last_voice_trigger_time = 0.0  # When voice was last triggered

        self._build_ui()

        # Video thread
        self.worker = VideoThread(cam_index=0)
        self.worker.frame_ready.connect(self.on_frame)

        # Set up MediaPipe backend
        self.worker.set_backend(self.backend)
        self.log_msg("Using MediaPipe pose detection (default)")

    # ---------- UI ----------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        # Video feed
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background: #0e0e10; border-radius: 16px;")

        # Status chip
        self.status_chip = QtWidgets.QLabel("—")
        self.status_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_chip.setFixedHeight(44)
        self.status_chip.setStyleSheet(
            "border-radius: 22px; padding: 0 18px; font-weight: 600;"
            "color: white; background: #444;"
        )

        # Settings button
        self.btn_settings = QtWidgets.QPushButton("⚙️ Settings")
        self.btn_settings.clicked.connect(self.open_settings)

        # Control buttons
        self.btn_start = QtWidgets.QPushButton("Start Camera")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)

        buttons = hstack([self.btn_start, self.btn_stop, self.btn_settings])

        # Log area
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setStyleSheet(
            "background:#0e0e10;color:#d7d7db;border-radius:12px;padding:8px;"
        )
        self.log.setPlaceholderText("Logs will appear here…")

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

    def _build_menu(self):
        bar = self.menuBar()
        if bar is None:
            bar = QtWidgets.QMenuBar(self)
            self.setMenuBar(bar)
        m_file = QtWidgets.QMenu("File", self)
        bar.addMenu(m_file)
        act_quit = QtGui.QAction("Quit", self)
        m_file.addAction(act_quit)
        act_quit.triggered.connect(self.close)

        m_cam = QtWidgets.QMenu("Camera", self)
        bar.addMenu(m_cam)
        act_start = QtGui.QAction("Start", self)
        act_stop = QtGui.QAction("Stop", self)
        m_cam.addAction(act_start)
        m_cam.addAction(act_stop)
        act_start.triggered.connect(self.start_camera)
        act_stop.triggered.connect(self.stop_camera)

    def _apply_modern_style(self):
        self.setStyleSheet(
            """
            QMainWindow { background: #0b0b0d; }
            QLabel, QCheckBox, QMenuBar, QMenu, QPlainTextEdit { color: #e5e5e9; font-size: 14px; }
            QSlider::groove:horizontal { height: 6px; background: #222; border-radius: 3px; }
            QSlider::handle:horizontal { width: 18px; background: #4b8bff; border-radius: 9px; margin: -6px 0; }
            QPushButton { background: #141418; color: #e5e5e9; border: 1px solid #26262b; padding: 10px 16px; border-radius: 12px; font-weight: 600; }
            QPushButton:hover { border-color: #3a3a42; }
            QPushButton:pressed { background: #0e0e10; }
            QMenuBar { background: #0b0b0d; }
            QMenu { background: #141418; }
            """
        )

    # ---------- Actions ----------
    def log_msg(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {msg}")

    def open_settings(self):
        settings_dialog = SettingsDialog(self)
        settings_dialog.exec()

    def start_camera(self):
        if not self.worker.isRunning():
            self.worker.start()
            self.log_msg("Camera started")

    def stop_camera(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.log_msg("Camera stopped")

    def _start_speech_worker(self):
        """Start the speech worker thread"""
        if self.speech_worker_running:
            return

        def speech_worker():
            while self.speech_worker_running:
                try:
                    # Wait for speech requests with timeout
                    text = self.speech_queue.get(timeout=1.0)
                    if (
                        text
                        and self.tts_available
                        and gTTS is not None
                        and pygame is not None
                    ):
                        current_time = time.time()
                        # Enforce cooldown to prevent overlapping speech
                        if current_time - self.last_speech_time >= self.speech_cooldown:
                            try:
                                # Generate speech with Google TTS (natural female voice)
                                tts = gTTS(text=text, lang="en", slow=False)

                                # Create temporary file
                                temp_file = os.path.join(
                                    self.temp_dir, f"speech_{int(current_time)}.mp3"
                                )
                                tts.save(temp_file)

                                # Play the audio using pygame
                                pygame.mixer.music.load(temp_file)
                                pygame.mixer.music.play()

                                # Wait for playback to complete
                                while pygame.mixer.music.get_busy():
                                    time.sleep(0.1)

                                # Clean up the temporary file
                                try:
                                    os.remove(temp_file)
                                except OSError:
                                    pass

                                self.last_speech_time = current_time

                            except Exception as e:
                                print(f"Speech error: {e}")

                    self.speech_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Speech worker error: {e}")
                    try:
                        self.speech_queue.task_done()
                    except ValueError:
                        pass

        self.speech_worker_running = True
        worker_thread = threading.Thread(target=speech_worker, daemon=True)
        worker_thread.start()
        # Speech worker started (no output needed)

    def speak(self, text: str):
        if not self.tts_available or not self.speech_worker_running:
            return

        # Check cooldown before queuing
        current_time = time.time()
        if current_time - self.last_speech_time < self.speech_cooldown:
            return

        # Try to add speech request (non-blocking)
        try:
            # Clear any pending speech first
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                pass

            # Queue new speech
            self.speech_queue.put_nowait(text)
        except queue.Full:
            pass

    @QtCore.pyqtSlot(np.ndarray, str, float, dict)
    def on_frame(self, frame_bgr: np.ndarray, label: str, conf: float, info: dict):
        bad_ratio = info.get("bad_ratio", 0.0)
        disp = frame_bgr.copy()

        # Status banner with detailed info
        is_bad = label.lower().startswith("bad")
        angle = info.get("extras", {}).get("angle", 0)
        chip_text = f"{label.upper()}  ({conf:.2f})  •  angle: {angle:.1f}°  •  ratio: {bad_ratio:.2f}"
        self.status_chip.setText(chip_text)
        if is_bad:
            self.status_chip.setStyleSheet(
                "border-radius:22px;padding:0 18px;font-weight:600;color:white;background:#c9342f;"
            )
        else:
            self.status_chip.setStyleSheet(
                "border-radius:22px;padding:0 18px;font-weight:600;color:white;background:#2e7d32;"
            )

        # Track bad posture with time-based accumulation
        current_bad_posture = bad_ratio >= self.min_bad_ratio
        current_time = time.time()

        if current_bad_posture:
            # Start tracking bad posture time
            if self.bad_posture_start_time is None:
                self.bad_posture_start_time = current_time
                print(
                    f"🔴 Bad posture detected! Starting timer... (ratio: {bad_ratio:.3f})"
                )
            else:
                # Accumulate bad posture time
                self.bad_posture_accumulated_time += (
                    current_time - self.bad_posture_start_time
                )
                self.bad_posture_start_time = current_time

            # Debug output every 0.5 seconds
            if int(self.bad_posture_accumulated_time * 2) > int(
                (self.bad_posture_accumulated_time - 0.1) * 2
            ):
                print(
                    f"⏱️  Bad posture time: {self.bad_posture_accumulated_time:.1f}s / {self.voice_trigger_delay:.1f}s (voice: {self.voice_enabled}, tts: {self.tts_available})"
                )

            # Trigger voice if accumulated time exceeds threshold and enough time passed since last trigger
            if (
                self.voice_enabled
                and self.tts_available
                and self.bad_posture_accumulated_time >= self.voice_trigger_delay
                and (current_time - self.last_voice_trigger_time)
                >= self.speech_cooldown
            ):
                print(
                    f"🔊 TRIGGERING VOICE! Accumulated: {self.bad_posture_accumulated_time:.1f}s, Threshold: {self.voice_trigger_delay:.1f}s"
                )
                self.speak("You have bad posture. Please straighten up.")
                self.last_voice_trigger_time = current_time
                # Reset accumulation after triggering
                self.bad_posture_accumulated_time = 0.0
        else:
            # Reset timing when posture is good
            if self.bad_posture_start_time is not None:
                print(
                    f"✅ Good posture restored! Reset timer (was: {self.bad_posture_accumulated_time:.1f}s)"
                )
            self.bad_posture_start_time = None
            self.bad_posture_accumulated_time = 0.0

        # Convert to QImage and paint
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    def closeEvent(self, a0: Optional[QtGui.QCloseEvent]) -> None:
        try:
            self.stop_camera()
            # Stop speech worker
            self.speech_worker_running = False
            # Clean up pygame and temp files
            if (
                hasattr(self, "tts_available")
                and self.tts_available
                and pygame is not None
            ):
                pygame.mixer.quit()
            if hasattr(self, "temp_dir"):
                try:
                    shutil.rmtree(self.temp_dir)
                except OSError:
                    pass
        except Exception:
            pass
        if a0 is not None:
            return super().closeEvent(a0)


# ------------------------
# Small UI layout helpers
# ------------------------
def hstack(widgets: List[QtWidgets.QWidget]) -> QtWidgets.QWidget:
    box = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(12)
    for w in widgets:
        lay.addWidget(w)
    return box


def vstack(widgets: List[QtWidgets.QWidget]) -> QtWidgets.QWidget:
    box = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(box)
    lay.setSpacing(12)
    for w in widgets:
        if isinstance(w, QtWidgets.QLayout):
            wrap = QtWidgets.QWidget()
            wrap.setLayout(w)
            lay.addWidget(wrap)
        else:
            lay.addWidget(w)
    box.setLayout(lay)
    return box


def stretch():
    s = QtWidgets.QWidget()
    s.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred
    )
    return s


# ------------------------
# Main
# ------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PostureApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
