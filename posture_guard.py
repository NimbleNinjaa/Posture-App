from __future__ import annotations
"""
Side‑View Posture Guard — Local Desktop App

Modern PyQt6 desktop app that runs entirely on-device to watch your posture
from a SIDE camera view. If it detects a slouch/bad posture, it shows a big
warning, plays a sound, and (optionally) speaks a reminder.

Backends supported:
  1) ONNX classification model (e.g., outputs [good, bad])
  2) Ultralytics YOLOv8/v5 .pt detector (pip install ultralytics)
  3) OpenCV DNN YOLO (.cfg + .weights) detector (your case — YOLOv3)
  4) Fallback heuristic (no model): uses MediaPipe Pose to estimate angles

Everything runs locally.
"""
import os
import sys
import time
import math
from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import cv2

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

# Optional deps guarded by try/except
try:
    import onnxruntime as ort  # For ONNX models
except Exception:
    ort = None

try:
    import pyttsx3  # Offline TTS
except Exception:
    pyttsx3 = None

# Optional: Ultralytics YOLO (for .pt models)
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None

# Optional: MediaPipe heuristic fallback
try:
    import mediapipe as mp  # type: ignore
    from mediapipe.python.solutions.pose import Pose as mp_pose
except Exception:
    mp_pose = None


# ------------------------
# Utility helpers
# ------------------------

def load_names(names_path: str) -> List[str]:
    labels: List[str] = []
    try:
        with open(names_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
    except Exception as e:
        print(f"Failed to read names file: {e}")
    return labels or ["good", "bad"]


def draw_text(img, text, org, scale=0.9, thickness=2, color=(255, 255, 255), bg=(0, 0, 0)):
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - h - baseline - 6), (x + w + 6, y + 6), bg, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def centered_text(img, text, scale=1.2, thickness=3, color=(255, 255, 255), bg=(0, 0, 0)):
    h, w = img.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.rectangle(img, (x - 10, y - th - baseline - 10), (x + tw + 10, y + 10), bg, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


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
        pass


class OnnxClassifierBackend(BaseBackend):
    def __init__(self, labels: List[str], onnx_path: str, input_size: Tuple[int, int] = (224, 224)):
        super().__init__(labels)
        if ort is None:
            raise RuntimeError("onnxruntime not installed. pip install onnxruntime")
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # local CPU
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        self.input_size = input_size

    def preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(img_bgr, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img = np.transpose(img, (2, 0, 1))[None, ...]  # NCHW
        return img

    def inference(self, frame_bgr: np.ndarray):
        x = self.preprocess(frame_bgr)
        logits = self.sess.run([self.output_name], {self.input_name: x})[0]
        logits = np.array(logits)
        probs = softmax(logits[0])
        idx = int(np.argmax(probs))
        label = self.labels[idx] if idx < len(self.labels) else str(idx)
        conf = float(probs[idx])
        return label, conf, {}


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z)
    e = np.exp(z)
    return e / (np.sum(e) + 1e-9)


def _is_bad_label(name: str) -> bool:
    n = name.lower()
    return (n.endswith("_bad") or
            n in ("bad", "bad_posture", "slouch", "incorrect"))


class UltralyticsYOLOBackend(BaseBackend):
    def __init__(self, labels: List[str], pt_path: str):
        super().__init__(labels)
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. pip install ultralytics")
        self.model = YOLO(pt_path)
        self.model.fuse()

    def inference(self, frame_bgr: np.ndarray):
        results = self.model.predict(source=frame_bgr, verbose=False, imgsz=640)
        bad_score = 0.0
        boxes = []
        for r in results:
            for b in r.boxes:
                cls_idx = int(b.cls[0])
                conf = float(b.conf[0])
                name = self.model.names.get(cls_idx, str(cls_idx))
                xyxy = b.xyxy[0].tolist()
                boxes.append((xyxy, name, conf))
                if _is_bad_label(name):
                    bad_score = max(bad_score, conf)
        label = "bad" if bad_score > 0 else "good"
        conf = bad_score if label == "bad" else 1.0 - bad_score
        return label, conf, {"boxes": boxes}

    def draw(self, frame_bgr: np.ndarray, extras: dict) -> None:
        for xyxy, name, conf in extras.get("boxes", []):
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (50, 200, 50), 2)
            draw_text(frame_bgr, f"{name} {conf:.2f}", (x1, y1 - 8))


class OpenCVYoloBackend(BaseBackend):
    def __init__(self, labels: List[str], cfg_path: str, weights_path: str):
        super().__init__(labels)
        # Parse cfg to pick correct network input size and class count
        self.net_w, self.net_h = self._read_net_size(cfg_path)
        self.cfg_classes = self._read_classes(cfg_path)
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.out_names = self.net.getUnconnectedOutLayersNames()

    def _read_net_size(self, cfg_path: str) -> Tuple[int, int]:
        w, h = 608, 608  # sensible default
        try:
            with open(cfg_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    s = line.strip().split('=')
                    if len(s) == 2:
                        k, v = s[0].strip().lower(), s[1].strip()
                        if k == 'width':
                            w = int(v)
                        elif k == 'height':
                            h = int(v)
        except Exception:
            pass
        # make sure multiples of 32
        w = max(32, (w // 32) * 32)
        h = max(32, (h // 32) * 32)
        return w, h

    def _read_classes(self, cfg_path: str) -> Optional[int]:
        try:
            with open(cfg_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.strip().lower().startswith('classes='):
                        return int(line.split('=')[1])
        except Exception:
            pass
        return None

    def inference(self, frame_bgr: np.ndarray):
        blob = cv2.dnn.blobFromImage(frame_bgr, 1/255.0, (self.net_w, self.net_h), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)
        H, W = frame_bgr.shape[:2]
        boxes = []
        bad_score = 0.0
        for out in outs:
            for det in out:
                det = np.array(det)
                scores = det[5:]
                cls = int(np.argmax(scores))
                conf = float(scores[cls])
                if conf < 0.4:
                    continue
                cx, cy, w, h = det[0:4]
                x = int((cx - w/2) * W)
                y = int((cy - h/2) * H)
                w = int(w * W)
                h = int(h * H)
                name = self.labels[cls] if cls < len(self.labels) else str(cls)
                boxes.append(((x, y, w, h), name, conf))
                if _is_bad_label(name):
                    bad_score = max(bad_score, conf)
        label = "bad" if bad_score > 0 else "good"
        conf = bad_score if label == "bad" else 1.0 - bad_score
        return label, conf, {"boxes": boxes}

    def draw(self, frame_bgr: np.ndarray, extras: dict) -> None:
        for (x, y, w, h), name, conf in extras.get("boxes", []):
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (50, 200, 50), 2)
            draw_text(frame_bgr, f"{name} {conf:.2f}", (x, max(20, y - 8)))


class HeuristicPoseBackend(BaseBackend):
    """No model required – uses MediaPipe Pose to estimate angle between ear-shoulder-hip.
    If head/torso angle exceeds threshold, marks as bad.
    """
    def __init__(self, labels: List[str], angle_threshold_deg: float = 25.0):
        super().__init__(labels)
        if mp_pose is None:
            raise RuntimeError("mediapipe not installed. pip install mediapipe")
        self.pose = mp_pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
        self.angle_threshold = angle_threshold_deg
        self.last_landmarks = None

    def inference(self, frame_bgr: np.ndarray):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        print('MediaPipe result attributes:', dir(res))  # DEBUG
        if not hasattr(res, 'pose_landmarks') or getattr(res, 'pose_landmarks', None) is None:
            return "unknown", 0.0, {}
        lm = getattr(res, 'pose_landmarks').landmark
        self.last_landmarks = lm
        # Right side landmarks (camera side). If mirrored, either side works.
        ear = np.array([lm[8].x, lm[8].y])   # Right ear
        shoulder = np.array([lm[12].x, lm[12].y])  # Right shoulder
        hip = np.array([lm[24].x, lm[24].y])  # Right hip
        # Angle at shoulder between ear-shoulder and hip-shoulder
        v1 = ear - shoulder
        v2 = hip - shoulder
        def angle(a, b):
            cosang = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
            return math.degrees(math.acos(np.clip(cosang, -1, 1)))
        ang = angle(v1, v2)
        bad = ang < (180 - self.angle_threshold)
        label = "bad" if bad else "good"
        conf = min(1.0, abs((180 - ang) / self.angle_threshold)) if bad else 0.8
        return label, conf, {"angle": ang, "landmarks": lm}

    def draw(self, frame_bgr: np.ndarray, extras: dict) -> None:
        if not extras:
            return
        h, w = frame_bgr.shape[:2]
        lm = extras.get("landmarks")
        if not lm:
            return
        # Draw key points
        pts = [(int(l.x * w), int(l.y * h)) for l in (lm[8], lm[12], lm[24])]  # ear, shoulder, hip
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

    def set_backend(self, backend: BaseBackend):
        self.backend = backend

    def run(self):
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW if os.name == 'nt' else 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True
        while self.running and self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            if self.backend is not None:
                label, conf, extras = self.backend.inference(frame)
                self.bad_smoothing.append(1 if label.lower().startswith("bad") else 0)
                bad_ratio = sum(self.bad_smoothing) / len(self.bad_smoothing) if self.bad_smoothing else 0.0
                # Draw guidance overlay (side-view guide)
                overlay = frame.copy()
                h, w = frame.shape[:2]
                # Guide line in the middle where ear-shoulder-hip should roughly align vertically
                cv2.line(overlay, (int(w*0.33), 0), (int(w*0.33), h), (100, 100, 100), 2)
                alpha = 0.15
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                # Let backend add annotations
                try:
                    self.backend.draw(frame, extras)
                except Exception:
                    pass
                self.frame_ready.emit(frame, label, float(conf if isinstance(conf, (int, float)) else 0.0), {"bad_ratio": bad_ratio})
            else:
                self.frame_ready.emit(frame, "unknown", 0.0, {"bad_ratio": 0.0})
        if self.cap is not None:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait(500)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ------------------------
# Main UI
# ------------------------

class PostureApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Posture Guard — Side View")
        self.resize(1100, 750)
        self.labels: List[str] = ["good", "bad"]
        self.backend: Optional[BaseBackend] = None
        self.tts_engine = None
        if pyttsx3 is not None:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 175)
            except Exception:
                self.tts_engine = None

        # State (must be set BEFORE building UI)
        self.alert_enabled = True
        self.voice_enabled = True
        self.min_bad_ratio = 0.5  # how much recent history must be bad to alert
        self.last_alert_ts = 0.0
        self.alert_cooldown = 6.0  # seconds

        self._build_ui()

        # Video thread
        self.worker = VideoThread(cam_index=0)
        self.worker.frame_ready.connect(self.on_frame)

        # State
        self.alert_enabled = True
        self.voice_enabled = True
        self.min_bad_ratio = 0.5  # how much recent history must be bad to alert
        self.last_alert_ts = 0.0
        self.alert_cooldown = 6.0  # seconds

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

        # Sensitivity slider
        self.sens_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.sens_slider.setRange(10, 90)
        self.sens_slider.setValue(int(self.min_bad_ratio * 100))
        self.sens_slider.valueChanged.connect(self.on_sensitivity)

        sens_row = hstack([QtWidgets.QLabel("Alert sensitivity"), self.sens_slider])

        # Control buttons
        self.btn_start = QtWidgets.QPushButton("Start Camera")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)

        buttons = hstack([self.btn_start, self.btn_stop])

        # Toggles
        self.chk_alert = QtWidgets.QCheckBox("Visual alerts")
        self.chk_alert.setChecked(True)
        self.chk_alert.toggled.connect(lambda v: setattr(self, 'alert_enabled', v))
        self.chk_voice = QtWidgets.QCheckBox("Voice")
        self.chk_voice.setChecked(True)
        self.chk_voice.toggled.connect(lambda v: setattr(self, 'voice_enabled', v))
        toggles = hstack([self.chk_alert, self.chk_voice])

        # Log area
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setStyleSheet("background:#0e0e10;color:#d7d7db;border-radius:12px;padding:8px;")
        self.log.setPlaceholderText("Logs will appear here…")

        layout_widget = vstack([
            self.video_label,
            hstack([QtWidgets.QLabel("Status:"), self.status_chip, stretch()]),
            sens_row,
            toggles,
            buttons,
            QtWidgets.QLabel("Activity log"),
            self.log,
            stretch(),
        ])
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

        m_model = QtWidgets.QMenu("Model", self)
        bar.addMenu(m_model)
        self.backend_group = QtGui.QActionGroup(self)
        self.backend_group.setExclusive(True)

        self.act_backend_onnx = QtGui.QAction("Backend: ONNX (classification)", self)
        self.act_backend_yolo_pt = QtGui.QAction("Backend: YOLO .pt (Ultralytics)", self)
        self.act_backend_yolo_cv = QtGui.QAction("Backend: YOLO .cfg + .weights", self)
        self.act_backend_heur = QtGui.QAction("Backend: Heuristic (MediaPipe)", self)
        for a in (self.act_backend_onnx, self.act_backend_yolo_pt, self.act_backend_yolo_cv, self.act_backend_heur):
            a.setCheckable(True)
            m_model.addAction(a)
            self.backend_group.addAction(a)
        self.act_backend_yolo_cv.setChecked(True)

        m_model.addSeparator()
        act_load_names = QtGui.QAction("Load labels (.names)", self)
        act_load_model = QtGui.QAction("Load model file…", self)
        act_load_yolo_pair = QtGui.QAction("Load YOLO cfg + weights…", self)
        m_model.addAction(act_load_names)
        m_model.addAction(act_load_model)
        m_model.addAction(act_load_yolo_pair)

        act_load_names.triggered.connect(self.load_names_file)
        act_load_model.triggered.connect(self.load_model_file)
        act_load_yolo_pair.triggered.connect(self.load_yolo_pair)

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

    def on_sensitivity(self, v: int):
        self.min_bad_ratio = v / 100.0
        self.log_msg(f"Sensitivity set: {self.min_bad_ratio:.2f}")

    def load_names_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load labels (.names)", "", "Names (*.names);;All files (*)")
        if not path:
            return
        self.labels = load_names(path)
        self.log_msg(f"Loaded labels: {self.labels}")

    def load_model_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load model", "", "Model (*.onnx *.pt);;All files (*)")
        if not path:
            return
        try:
            if path.lower().endswith('.onnx'):
                self.backend = OnnxClassifierBackend(self.labels, path)
                self.worker.set_backend(self.backend)
                self.act_backend_onnx.setChecked(True)
                self.log_msg(f"ONNX model loaded: {os.path.basename(path)}")
            elif path.lower().endswith('.pt'):
                self.backend = UltralyticsYOLOBackend(self.labels, path)
                self.worker.set_backend(self.backend)
                self.act_backend_yolo_pt.setChecked(True)
                self.log_msg(f"YOLO .pt model loaded: {os.path.basename(path)}")
            else:
                QtWidgets.QMessageBox.warning(self, "Unsupported", "Please choose an .onnx or .pt file.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Model load error", str(e))
            self.log_msg(f"Model load failed: {e}")

    def load_yolo_pair(self):
        cfg, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select YOLO cfg", "", "Config (*.cfg)")
        if not cfg:
            return
        weights, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select YOLO weights", "", "Weights (*.weights)")
        if not weights:
            return
        try:
            self.backend = OpenCVYoloBackend(self.labels, cfg, weights)
            self.worker.set_backend(self.backend)
            self.act_backend_yolo_cv.setChecked(True)
            self.log_msg(f"YOLO cfg+weights loaded: {os.path.basename(cfg)}, {os.path.basename(weights)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Model load error", str(e))
            self.log_msg(f"Model load failed: {e}")

    def start_camera(self):
        if not self.worker.isRunning():
            if self.backend is None and mp_pose is not None:
                # default to heuristic so user sees something immediately
                try:
                    self.backend = HeuristicPoseBackend(self.labels)
                    self.worker.set_backend(self.backend)
                    self.act_backend_heur.setChecked(True)
                    self.log_msg("Using heuristic backend (MediaPipe)")
                except Exception as e:
                    self.log_msg(f"Heuristic init failed: {e}")
            self.worker.start()
            self.log_msg("Camera started")

    def stop_camera(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.log_msg("Camera stopped")

    def speak(self, text: str):
        if self.tts_engine is None:
            return
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception:
            pass

    @QtCore.pyqtSlot(np.ndarray, str, float, dict)
    def on_frame(self, frame_bgr: np.ndarray, label: str, conf: float, info: dict):
        bad_ratio = info.get("bad_ratio", 0.0)
        disp = frame_bgr.copy()

        # Status banner
        is_bad = label.lower().startswith("bad")
        chip_text = f"{label.upper()}  ({conf:.2f})  •  bad ratio {bad_ratio:.2f}"
        self.status_chip.setText(chip_text)
        if is_bad:
            self.status_chip.setStyleSheet("border-radius:22px;padding:0 18px;font-weight:600;color:white;background:#c9342f;")
        else:
            self.status_chip.setStyleSheet("border-radius:22px;padding:0 18px;font-weight:600;color:white;background:#2e7d32;")

        # BIG overlay if bad
        if self.alert_enabled and bad_ratio >= self.min_bad_ratio:
            centered_text(disp, "Bad posture — please straighten up", scale=1.1, thickness=3, color=(255,255,255), bg=(50,0,0))
            now = time.time()
            if self.voice_enabled and (now - self.last_alert_ts) > self.alert_cooldown:
                self.last_alert_ts = now
                self.speak("You have bad posture. Please straighten up.")

        # Convert to QImage and paint
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def closeEvent(self, a0: Optional[QtGui.QCloseEvent]) -> None:
        try:
            self.stop_camera()
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
    s.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
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
