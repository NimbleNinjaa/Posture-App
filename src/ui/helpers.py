"""UI layout helpers and utilities."""

from typing import List, Union

from PyQt6 import QtWidgets


def hstack(widgets: List[QtWidgets.QWidget]) -> QtWidgets.QWidget:
    """Create a horizontal stack of widgets.

    Args:
        widgets: List of widgets to stack horizontally

    Returns:
        Widget container with horizontal layout
    """
    box = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(12)

    for w in widgets:
        lay.addWidget(w)

    return box


def vstack(
    widgets: List[Union[QtWidgets.QWidget, QtWidgets.QLayout]],
) -> QtWidgets.QWidget:
    """Create a vertical stack of widgets.

    Args:
        widgets: List of widgets or layouts to stack vertically

    Returns:
        Widget container with vertical layout
    """
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


def stretch() -> QtWidgets.QWidget:
    """Create a stretchable spacer widget.

    Returns:
        Widget that expands to fill available space
    """
    s = QtWidgets.QWidget()
    s.setSizePolicy(
        QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred
    )
    return s


# Modern dark theme stylesheet constants
MODERN_STYLE = """
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

SETTINGS_DIALOG_STYLE = """
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
"""
