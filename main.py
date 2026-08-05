"""
Optical Flow Motion Analyzer — Entry Point

A real-time desktop application for analyzing motion patterns
between consecutive video frames using optical flow techniques.
"""

import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui.app_window import AppWindow


def main():
    # Enable high-DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("Optical Flow Motion Analyzer")
    app.setOrganizationName("TAE1")

    window = AppWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
