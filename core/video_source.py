"""
Unified video source abstraction for webcam and video file input.
"""

import cv2
import numpy as np


class VideoSource:
    """Abstracts webcam and video file capture into a single interface."""

    def __init__(self):
        self._cap = None
        self._is_webcam = False
        self._path = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_webcam(self, device_id: int = 0) -> bool:
        """Open a webcam device.  Returns True on success."""
        self.release()
        self._cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback without DirectShow
            self._cap = cv2.VideoCapture(device_id)
        self._is_webcam = True
        self._path = f"Webcam {device_id}"
        return self._cap.isOpened()

    def open_file(self, path: str) -> bool:
        """Open a video file.  Returns True on success."""
        self.release()
        self._cap = cv2.VideoCapture(path)
        self._is_webcam = False
        self._path = path
        return self._cap.isOpened()

    def read_frame(self):
        """Read the next frame.  Returns (success: bool, frame: np.ndarray | None)."""
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        return ret, frame

    def seek_to(self, frame_number: int):
        """Seek to a specific frame (video files only)."""
        if self._cap and not self._is_webcam:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def release(self):
        """Release the underlying capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def is_webcam(self) -> bool:
        return self._is_webcam

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 30.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    @property
    def frame_count(self) -> int:
        if self._cap is None or self._is_webcam:
            return -1
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def current_frame(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

    @property
    def width(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        if self._cap is None:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def path(self) -> str:
        return self._path or ""
