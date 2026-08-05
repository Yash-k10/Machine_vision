"""
Lucas-Kanade sparse optical flow tracker.

Detects Shi-Tomasi corners and tracks them across consecutive frames,
drawing coloured motion trails.
"""

import cv2
import numpy as np


class LucasKanadeTracker:
    """Sparse optical flow using pyramidal Lucas-Kanade."""

    # Default Shi-Tomasi corner detection parameters
    DEFAULT_FEATURE_PARAMS = dict(
        maxCorners=200,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7,
    )

    # Default LK parameters
    DEFAULT_LK_PARAMS = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    def __init__(self):
        self._prev_gray = None
        self._points = None
        self._tracks = []          # list of point trails
        self._colors = None
        self._trail_length = 25    # max trail length

        # Configurable parameters (will be updated via sliders)
        self.feature_params = dict(self.DEFAULT_FEATURE_PARAMS)
        self.lk_params = dict(self.DEFAULT_LK_PARAMS)
        self._min_points = 30      # re-detect when below this count

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all state."""
        self._prev_gray = None
        self._points = None
        self._tracks = []
        self._colors = None

    def process(self, frame: np.ndarray):
        """
        Process a new frame and return the visualization overlay.

        Returns
        -------
        vis : np.ndarray
            Frame with motion trails drawn.
        flow_vectors : list[tuple[tuple, tuple]]
            List of (old_pt, new_pt) for statistics calculation.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = frame.copy()
        flow_vectors = []

        if self._prev_gray is None:
            # First frame — detect initial features
            self._prev_gray = gray
            self._detect_features(gray)
            return vis, flow_vectors

        if self._points is None or len(self._points) == 0:
            self._detect_features(gray)
            self._prev_gray = gray
            return vis, flow_vectors

        # Calculate optical flow
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._points, None, **self.lk_params
        )

        if new_pts is None:
            self._detect_features(gray)
            self._prev_gray = gray
            return vis, flow_vectors

        # Select good points
        good_new = new_pts[status.ravel() == 1]
        good_old = self._points[status.ravel() == 1]

        # Filter colors to match surviving points
        good_mask = status.ravel() == 1
        surviving_indices = np.where(good_mask)[0]

        # Update trails
        new_tracks = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            flow_vectors.append(((float(c), float(d)), (float(a), float(b))))

            # Build trail
            if i < len(self._tracks):
                trail = self._tracks[surviving_indices[i]] if surviving_indices[i] < len(self._tracks) else [(float(c), float(d))]
            else:
                trail = [(float(c), float(d))]
            trail.append((float(a), float(b)))
            if len(trail) > self._trail_length:
                trail = trail[-self._trail_length:]
            new_tracks.append(trail)

        self._tracks = new_tracks

        # Generate stable colours
        if self._colors is None or len(self._colors) < len(good_new):
            self._colors = np.random.randint(0, 255, (max(len(good_new), 300), 3)).tolist()

        # Draw trails and points
        for i, trail in enumerate(self._tracks):
            color = tuple(self._colors[i % len(self._colors)])
            for j in range(1, len(trail)):
                pt1 = (int(trail[j - 1][0]), int(trail[j - 1][1]))
                pt2 = (int(trail[j][0]), int(trail[j][1]))
                thickness = max(1, int(1 + 1.5 * j / len(trail)))
                cv2.line(vis, pt1, pt2, color, thickness, cv2.LINE_AA)
            # Draw current position
            cx, cy = int(trail[-1][0]), int(trail[-1][1])
            cv2.circle(vis, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(vis, (cx, cy), 5, (255, 255, 255), 1, cv2.LINE_AA)

        # Update state
        self._points = good_new.reshape(-1, 1, 2)
        self._prev_gray = gray

        # Re-detect if too few points
        if len(self._points) < self._min_points:
            self._detect_features(gray)

        return vis, flow_vectors

    # ------------------------------------------------------------------
    # Parameter setters (for GUI sliders)
    # ------------------------------------------------------------------

    def set_max_corners(self, val: int):
        self.feature_params["maxCorners"] = max(10, val)

    def set_quality_level(self, val: float):
        self.feature_params["qualityLevel"] = max(0.01, min(1.0, val))

    def set_min_distance(self, val: int):
        self.feature_params["minDistance"] = max(1, val)

    def set_win_size(self, val: int):
        val = max(5, val)
        self.lk_params["winSize"] = (val, val)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_features(self, gray: np.ndarray):
        """Detect Shi-Tomasi corners."""
        pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
        if pts is not None:
            self._points = pts
            self._tracks = [[(float(p[0][0]), float(p[0][1]))] for p in pts]
            self._colors = np.random.randint(0, 255, (len(pts), 3)).tolist()
        else:
            self._points = None
            self._tracks = []
