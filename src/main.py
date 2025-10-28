"""Main entry point for Posture Guard application."""

import sys

from PyQt6 import QtWidgets

from src.ui import PostureApp


def main():
    """Main application entry point."""
    app = QtWidgets.QApplication(sys.argv)
    win = PostureApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
