"""
Motion statistics calculator.

Computes real-time metrics from optical flow fields:
average/max magnitude, dominant direction, motion area %,
and an angular direction histogram.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MotionMetrics:
    """Container for per-frame motion statistics."""
    avg_magnitude: float = 0.0
    max_magnitude: float = 0.0
    dominant_direction_deg: float = 0.0
    motion_area_pct: float = 0.0
    direction_histogram: List[float] = field(default_factory=lambda: [0.0] * 8)
    total_points: int = 0


class MotionStatsCalculator:
    """Calculates motion statistics from flow data."""

    NUM_BINS = 8  # Angular bins for direction histogram

    # ------------------------------------------------------------------
    # From dense flow (Farnebäck / heatmap)
    # ------------------------------------------------------------------

    @classmethod
    def from_dense_flow(cls, flow: np.ndarray, threshold: float = 1.0) -> MotionMetrics:
        """
        Compute metrics from a dense flow field (h, w, 2).

        Parameters
        ----------
        flow : np.ndarray
            Dense flow field from Farnebäck.
        threshold : float
            Minimum magnitude to count as "moving".
        """
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

        # Mask significant motion
        mask = mag > threshold
        total_pixels = mag.size
        motion_pixels = int(np.sum(mask))

        if motion_pixels == 0:
            return MotionMetrics()

        moving_mag = mag[mask]
        moving_ang = ang[mask]

        # Direction histogram (8 bins × 45°)
        hist = np.zeros(cls.NUM_BINS, dtype=np.float64)
        bin_edges = np.linspace(0, 360, cls.NUM_BINS + 1)
        for i in range(cls.NUM_BINS):
            low, high = bin_edges[i], bin_edges[i + 1]
            hist[i] = np.sum((moving_ang >= low) & (moving_ang < high))
        total = hist.sum()
        if total > 0:
            hist = hist / total  # Normalise to proportions

        # Dominant direction
        dominant_bin = int(np.argmax(hist))
        dominant_deg = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2

        return MotionMetrics(
            avg_magnitude=float(np.mean(moving_mag)),
            max_magnitude=float(np.max(moving_mag)),
            dominant_direction_deg=dominant_deg,
            motion_area_pct=motion_pixels / total_pixels * 100,
            direction_histogram=hist.tolist(),
            total_points=motion_pixels,
        )

    # ------------------------------------------------------------------
    # From sparse flow (Lucas-Kanade vectors)
    # ------------------------------------------------------------------

    @classmethod
    def from_sparse_vectors(
        cls, vectors: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> MotionMetrics:
        """
        Compute metrics from a list of (old_pt, new_pt) vectors.
        """
        if not vectors:
            return MotionMetrics()

        dxs = []
        dys = []
        mags = []
        for (ox, oy), (nx, ny) in vectors:
            dx = nx - ox
            dy = ny - oy
            dxs.append(dx)
            dys.append(dy)
            mags.append(math.sqrt(dx * dx + dy * dy))

        mags = np.array(mags)
        angs = np.degrees(np.arctan2(dys, dxs)) % 360

        # Direction histogram
        hist = np.zeros(cls.NUM_BINS, dtype=np.float64)
        bin_edges = np.linspace(0, 360, cls.NUM_BINS + 1)
        for i in range(cls.NUM_BINS):
            low, high = bin_edges[i], bin_edges[i + 1]
            hist[i] = np.sum((angs >= low) & (angs < high))
        total = hist.sum()
        if total > 0:
            hist = hist / total

        dominant_bin = int(np.argmax(hist))
        dominant_deg = (bin_edges[dominant_bin] + bin_edges[dominant_bin + 1]) / 2

        return MotionMetrics(
            avg_magnitude=float(np.mean(mags)),
            max_magnitude=float(np.max(mags)),
            dominant_direction_deg=dominant_deg,
            motion_area_pct=0.0,  # Not meaningful for sparse
            direction_histogram=hist.tolist(),
            total_points=len(vectors),
        )
