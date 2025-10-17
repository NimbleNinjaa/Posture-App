"""Settings dialog for configuring application preferences."""

from typing import Any, Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from src.config import CAMERA_CONFIG
from src.core.utils import detect_available_cameras
from src.ui.helpers import SETTINGS_DIALOG_STYLE, hstack


class SettingsDialog(QtWidgets.QDialog):
    """Dialog for configuring application settings."""

    def __init__(self, parent: Optional[Any] = None):
        """Initialize the settings dialog.

        Args:
            parent: Parent PostureApp instance
        """
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setFixedSize(400, 350)
        self.parent_app = parent

        # Store current settings for comparison
        if self.parent_app:
            self.original_cam_index = self.parent_app.worker.cam_index
            self.original_sensitivity = self.parent_app.min_bad_ratio
            self.original_voice_enabled = self.parent_app.voice_enabled
            self.original_voice_delay = self.parent_app.voice_trigger_delay
            self.original_overlay_enabled = self.parent_app.overlay_enabled

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        """Build the settings dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Camera settings group
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

        camera_layout.addWidget(QtWidgets.QLabel("Select Camera:"))
        camera_layout.addWidget(self.camera_combo)

        # Alert settings group
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

        # Voice alerts checkbox
        self.chk_voice = QtWidgets.QCheckBox("Voice alerts")

        if self.parent_app:
            self.chk_voice.setChecked(self.parent_app.voice_enabled)

        # Screen overlay checkbox
        self.chk_overlay = QtWidgets.QCheckBox("Screen overlay")
        self.chk_overlay.setToolTip("Show posture status overlay on screen (stays visible even when app is minimized)")

        if self.parent_app:
            self.chk_overlay.setChecked(self.parent_app.overlay_enabled)

        alert_layout.addWidget(sens_row)
        alert_layout.addWidget(voice_delay_row)
        alert_layout.addWidget(self.chk_voice)
        alert_layout.addWidget(self.chk_overlay)

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

    def _apply_style(self) -> None:
        """Apply the dark theme stylesheet."""
        self.setStyleSheet(SETTINGS_DIALOG_STYLE)

    def _populate_cameras(self) -> None:
        """Dynamically detect and populate available cameras."""
        available_cameras = detect_available_cameras(CAMERA_CONFIG.max_camera_scan)

        if available_cameras:
            for cam_idx in available_cameras:
                camera_name = f"Camera {cam_idx}"
                if cam_idx == 0:
                    camera_name += " (default)"
                self.camera_combo.addItem(camera_name, cam_idx)
            print(f"Detected cameras: {available_cameras}")
        else:
            self.camera_combo.addItem("No cameras detected", 0)
            print("Warning: No cameras detected")

    def update_sensitivity_label(self, v: int) -> None:
        """Update the sensitivity label when slider changes.

        Args:
            v: New slider value (percentage)
        """
        self.sens_value_label.setText(f"{v}%")

    def save_and_close(self) -> None:
        """Save settings and close the dialog."""
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

        # Apply voice alert setting
        self.parent_app.voice_enabled = self.chk_voice.isChecked()

        # Apply overlay setting
        new_overlay_enabled = self.chk_overlay.isChecked()
        if new_overlay_enabled != self.original_overlay_enabled:
            self.parent_app.overlay_enabled = new_overlay_enabled
            if new_overlay_enabled:
                self.parent_app.overlay.show()
                self.parent_app.log_msg("Screen overlay enabled")
            else:
                self.parent_app.overlay.hide()
                self.parent_app.log_msg("Screen overlay disabled")

        self.accept()
