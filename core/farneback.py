"""
Farnebäck dense optical flow analyser.

Computes per-pixel motion vectors and visualises them as
HSV colour-coded flow and optional arrow overlays.
"""

import cv2
import numpy as np


class FarnebackAnalyzer:
    """Dense optical flow using Farnebäck's algorithm."""

    def __init__(self):
        self._prev_gray = None

        # Configurable parameters (updated via GUI sliders)
        self.pyr_scale = 0.5
        self.levels = 3
        self.winsize = 15
        self.iterations = 3
        self.poly_n = 5
        self.poly_sigma = 1.2
        self.show_arrows = True
        self.arrow_step = 16        # pixel spacing between arrows
        self.arrow_scale = 2.0      # length multiplier

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear internal state."""
        self._prev_gray = None

    def process(self, frame: np.ndarray):
        """
        Process a new frame and return the HSV flow visualisation.

        Returns
        -------
        vis : np.ndarray
            BGR frame with HSV-coded optical flow.
        flow : np.ndarray | None
            Raw flow field (h, w, 2) — dx, dy per pixel.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return frame.copy(), None

        # Compute dense flow
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray,
            gray,
            None,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=0,
        )

        # HSV visualisation
        vis = self._flow_to_hsv(flow, frame)

        # Optional arrow overlay
        if self.show_arrows:
            self._draw_arrows(vis, flow)

        self._prev_gray = gray
        return vis, flow

    # ------------------------------------------------------------------
    # Parameter setters (for GUI sliders)
    # ------------------------------------------------------------------

    def set_pyr_scale(self, val: float):
        self.pyr_scale = max(0.1, min(0.9, val))

    def set_levels(self, val: int):
        self.levels = max(1, min(10, val))

    def set_winsize(self, val: int):
        self.winsize = max(5, val) | 1  # must be odd

    def set_iterations(self, val: int):
        self.iterations = max(1, min(20, val))

    def set_poly_n(self, val: int):
        self.poly_n = max(3, val)

    def set_poly_sigma(self, val: float):
        self.poly_sigma = max(0.1, val)

    def set_arrow_step(self, val: int):
        self.arrow_step = max(4, val)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _flow_to_hsv(flow: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Convert a flow field to an HSV-coded BGR image blended with the original."""
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2          # Hue = direction
        hsv[..., 1] = 255                              # Full saturation
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # Value = magnitude
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Blend with original for context
        vis = cv2.addWeighted(original, 0.4, flow_bgr, 0.6, 0)
        return vis

    def _draw_arrows(self, vis: np.ndarray, flow: np.ndarray):
        """Draw quiver-style arrows on the visualisation frame."""
        h, w = flow.shape[:2]
        step = self.arrow_step
        y, x = np.mgrid[step // 2:h:step, step // 2:w:step]
        fx = flow[y, x, 0] * self.arrow_scale
        fy = flow[y, x, 1] * self.arrow_scale

        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                pt1 = (int(x[i, j]), int(y[i, j]))
                pt2 = (int(x[i, j] + fx[i, j]), int(y[i, j] + fy[i, j]))
                mag = np.sqrt(fx[i, j] ** 2 + fy[i, j] ** 2)
                if mag > 1.0:  # skip negligible motion
                    cv2.arrowedLine(vis, pt1, pt2, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
