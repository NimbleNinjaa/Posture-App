"""Screen overlay window for posture status display."""

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt

from src.config import UI_CONFIG


class PostureOverlay(QtWidgets.QWidget):
    """Transparent overlay window that stays on top to display posture status."""

    def __init__(self):
        """Initialize the posture overlay window."""
        super().__init__()
        
        # Configure window to be frameless and stay on top
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # Don't show in taskbar
        )
        
        # Make window transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Set initial position
        x, y = UI_CONFIG.overlay_position
        self.move(x, y)
        
        # Initialize status
        self.current_status = "Good"
        self.current_color = QtGui.QColor(46, 125, 50)  # Green
        
        # Setup font for size calculations
        self._display_font = QtGui.QFont()
        self._display_font.setPixelSize(UI_CONFIG.overlay_font_size)
        self._display_font.setBold(True)
        
        # Calculate initial size
        self._update_size()
        
    def update_status(self, label: str, is_bad: bool) -> None:
        """Update the overlay with new posture status.
        
        Args:
            label: Posture label ("good", "bad", "unknown")
            is_bad: Whether the posture is bad
        """
        if label.lower() == "unknown":
            self.current_status = "Unknown"
            self.current_color = QtGui.QColor(150, 150, 150)
        elif is_bad or label.lower().startswith("bad"):
            self.current_status = "Bad"
            self.current_color = QtGui.QColor(201, 52, 47)  # Red
        else:
            self.current_status = "Good"
            self.current_color = QtGui.QColor(46, 125, 50)  # Green
        
        self._update_size()
        self.update()  # Trigger repaint
    
    def _update_size(self) -> None:
        """Update window size based on current text."""
        metrics = QtGui.QFontMetrics(self._display_font)
        text_width = metrics.horizontalAdvance(self.current_status)
        text_height = metrics.height()
        
        # Set window size with small padding
        self.setFixedSize(text_width + 10, text_height + 5)
    
    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Paint the overlay content.
        
        Args:
            event: Paint event
        """
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        
        # Draw only colored text - no background
        painter.setPen(self.current_color)
        painter.setFont(self._display_font)
        
        # Draw text
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            self.current_status
        )
    
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse press to enable dragging.
        
        Args:
            event: Mouse event
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
    
    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        """Handle mouse move to drag the overlay.
        
        Args:
            event: Mouse event
        """
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()