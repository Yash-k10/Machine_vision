"""
Video canvas widget — displays OpenCV frames inside a QLabel.
"""

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy


class VideoCanvas(QLabel):
    """Custom QLabel that renders OpenCV BGR frames with aspect-ratio preservation."""

    # Emitted when the user clicks on the canvas (x, y in frame coords)
    clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #11111b; border-radius: 8px;")

        self._frame_size = (0, 0)   # (w, h) of the last displayed frame
        self._display_rect = None   # QRect of the rendered image within the label

        # Placeholder text
        self.setText("No video source\n\nFile → Open Video  or  Open Webcam")
        self.setStyleSheet(
            "background-color: #11111b; border-radius: 8px; "
            "color: #585b70; font-size: 16px; font-weight: 500;"
        )

    def update_frame(self, frame: np.ndarray):
        """Display a new OpenCV BGR frame."""
        if frame is None or frame.size == 0:
            return

        self.setText("")  # Clear placeholder

        h, w, ch = frame.shape
        self._frame_size = (w, h)

        # Convert BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * w
        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale to fit label while preserving aspect ratio
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)

        # Store display rect for click mapping
        lw, lh = self.width(), self.height()
        sw, sh = scaled.width(), scaled.height()
        x_off = (lw - sw) // 2
        y_off = (lh - sh) // 2
        self._display_rect = (x_off, y_off, sw, sh)

    def mousePressEvent(self, event):
        """Map click position to frame coordinates and emit signal."""
        if self._display_rect is None or self._frame_size == (0, 0):
            return

        x_off, y_off, sw, sh = self._display_rect
        fw, fh = self._frame_size

        # Click position relative to the displayed image
        cx = event.x() - x_off
        cy = event.y() - y_off

        if 0 <= cx < sw and 0 <= cy < sh:
            # Map to original frame coordinates
            fx = int(cx * fw / sw)
            fy = int(cy * fh / sh)
            self.clicked.emit(fx, fy)
