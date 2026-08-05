"""
Cumulative motion heatmap generator.

Accumulates frame-to-frame differences with temporal decay and
visualises the result as a coloured heatmap overlay.
"""

import cv2
import numpy as np


class MotionHeatmap:
    """Builds and renders a motion heatmap from frame differences."""

    # Available OpenCV colormaps
    COLORMAPS = {
        "JET": cv2.COLORMAP_JET,
        "INFERNO": cv2.COLORMAP_INFERNO,
        "HOT": cv2.COLORMAP_HOT,
        "MAGMA": cv2.COLORMAP_MAGMA,
        "TURBO": cv2.COLORMAP_TURBO,
        "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    }

    def __init__(self):
        self._prev_gray = None
        self._accumulator = None

        # Configurable parameters
        self.decay = 0.95           # temporal decay factor (0–1)
        self.threshold = 15         # motion threshold (pixel intensity diff)
        self.overlay_alpha = 0.55   # blending alpha for overlay
        self.colormap_name = "INFERNO"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear accumulated heatmap."""
        self._prev_gray = None
        self._accumulator = None

    def process(self, frame: np.ndarray):
        """
        Process a new frame and return the heatmap overlay.

        Returns
        -------
        vis : np.ndarray
            Original frame with coloured heatmap overlay.
        magnitude_map : np.ndarray | None
            2-D float array of current motion magnitudes.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._accumulator = np.zeros(gray.shape, dtype=np.float64)
            return frame.copy(), None

        # Frame difference
        diff = cv2.absdiff(gray, self._prev_gray).astype(np.float64)

        # Apply threshold — ignore minor noise
        diff[diff < self.threshold] = 0

        # Accumulate with decay
        self._accumulator = self._accumulator * self.decay + diff

        # Normalise to 0-255 for colormap
        max_val = self._accumulator.max()
        if max_val > 0:
            norm = (self._accumulator / max_val * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(gray, dtype=np.uint8)

        # Apply colormap
        cmap = self.COLORMAPS.get(self.colormap_name, cv2.COLORMAP_INFERNO)
        heatmap = cv2.applyColorMap(norm, cmap)

        # Create a mask: only overlay where there is significant accumulated motion
        mask = (norm > 10).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)[..., np.newaxis]

        # Blend
        vis = frame.astype(np.float32) * (1 - mask * self.overlay_alpha) + \
              heatmap.astype(np.float32) * mask * self.overlay_alpha
        vis = np.clip(vis, 0, 255).astype(np.uint8)

        self._prev_gray = gray
        return vis, diff.astype(np.float32)

    # ------------------------------------------------------------------
    # Parameter setters
    # ------------------------------------------------------------------

    def set_decay(self, val: float):
        self.decay = max(0.0, min(1.0, val))

    def set_threshold(self, val: int):
        self.threshold = max(0, min(100, val))

    def set_overlay_alpha(self, val: float):
        self.overlay_alpha = max(0.0, min(1.0, val))

    def set_colormap(self, name: str):
        if name in self.COLORMAPS:
            self.colormap_name = name
