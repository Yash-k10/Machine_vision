"""
Motion-based object detector.

Uses background subtraction and contour detection to find,
bound, and label moving objects in the video frame.
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class DetectedObject:
    """A single detected moving object."""
    obj_id: int
    bbox: Tuple[int, int, int, int]   # (x, y, w, h)
    center: Tuple[int, int]
    area: int
    speed: float                       # pixels/frame
    direction_deg: float               # 0-360
    direction_label: str               # "→", "↑", etc.


class MotionDetector:
    """Detects and labels moving objects using background subtraction."""

    DIRECTION_ARROWS = {
        "Right": "→",
        "Up-Right": "↗",
        "Up": "↑",
        "Up-Left": "↖",
        "Left": "←",
        "Down-Left": "↙",
        "Down": "↓",
        "Down-Right": "↘",
    }

    def __init__(self):
        # Background subtractor
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=50, detectShadows=True
        )

        # Previous frame centroids for speed/direction calculation
        self._prev_centroids: List[Tuple[int, int]] = []
        self._object_counter = 0

        # Configurable parameters
        self.min_area = 800          # Minimum contour area to consider
        self.max_objects = 20        # Max objects to track
        self.blur_size = 5           # Gaussian blur kernel size
        self.dilate_iterations = 3   # Dilation iterations for filling gaps
        self.show_boxes = True
        self.show_labels = True
        self.show_trails = True

        # Object tracking (simple centroid-based)
        self._tracked_objects: dict = {}  # id -> list of centroids (trail)
        self._next_id = 1
        self._max_trail = 30

    def reset(self):
        """Clear all state."""
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=50, detectShadows=True
        )
        self._prev_centroids = []
        self._tracked_objects = {}
        self._next_id = 1

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectedObject]]:
        """
        Detect moving objects in the frame.

        Returns
        -------
        vis : np.ndarray
            Frame with bounding boxes, labels, and trails drawn.
        objects : list[DetectedObject]
            List of detected objects with metadata.
        """
        vis = frame.copy()
        h, w = frame.shape[:2]

        # Apply background subtraction
        fg_mask = self._bg_sub.apply(frame)

        # Remove shadows (shadow pixels = 127 in MOG2)
        fg_mask[fg_mask == 127] = 0

        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=self.dilate_iterations)
        fg_mask = cv2.GaussianBlur(fg_mask, (self.blur_size | 1, self.blur_size | 1), 0)
        _, fg_mask = cv2.threshold(fg_mask, 128, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter and sort by area (largest first)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= self.min_area]
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        valid_contours = valid_contours[:self.max_objects]

        # Extract current centroids
        current_centroids = []
        current_bboxes = []
        for contour in valid_contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            cx, cy = x + bw // 2, y + bh // 2
            current_centroids.append((cx, cy))
            current_bboxes.append((x, y, bw, bh))

        # Match with tracked objects (simple nearest-centroid matching)
        matched_ids = self._match_objects(current_centroids)

        # Build detected objects list
        detected: List[DetectedObject] = []
        for i, ((cx, cy), (x, y, bw, bh), obj_id) in enumerate(
            zip(current_centroids, current_bboxes, matched_ids)
        ):
            # Calculate speed and direction from trail
            speed = 0.0
            direction_deg = 0.0
            direction_label = ""

            trail = self._tracked_objects.get(obj_id, [])
            if len(trail) >= 2:
                prev = trail[-2]
                dx = cx - prev[0]
                dy = cy - prev[1]
                speed = math.sqrt(dx * dx + dy * dy)
                direction_deg = math.degrees(math.atan2(-dy, dx)) % 360
                direction_label = self._deg_to_label(direction_deg)

            obj = DetectedObject(
                obj_id=obj_id,
                bbox=(x, y, bw, bh),
                center=(cx, cy),
                area=bw * bh,
                speed=speed,
                direction_deg=direction_deg,
                direction_label=direction_label,
            )
            detected.append(obj)

            # Draw on visualization
            self._draw_object(vis, obj)

        # Draw object count
        count_text = f"Objects Detected: {len(detected)}"
        cv2.rectangle(vis, (w - 260, h - 40), (w - 5, h - 5), (0, 0, 0), -1)
        cv2.rectangle(vis, (w - 260, h - 40), (w - 5, h - 5), (137, 180, 250), 1)
        cv2.putText(vis, count_text, (w - 250, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (166, 227, 161), 2, cv2.LINE_AA)

        self._prev_centroids = current_centroids
        return vis, detected

    def _match_objects(self, current_centroids: List[Tuple[int, int]]) -> List[int]:
        """Simple nearest-neighbour centroid matching for tracking."""
        max_dist = 120  # Maximum distance to consider same object
        assigned_ids = []
        used_ids = set()

        for cx, cy in current_centroids:
            best_id = None
            best_dist = float("inf")

            for obj_id, trail in self._tracked_objects.items():
                if obj_id in used_ids:
                    continue
                last = trail[-1]
                dist = math.sqrt((cx - last[0]) ** 2 + (cy - last[1]) ** 2)
                if dist < best_dist and dist < max_dist:
                    best_dist = dist
                    best_id = obj_id

            if best_id is not None:
                assigned_ids.append(best_id)
                used_ids.add(best_id)
                self._tracked_objects[best_id].append((cx, cy))
                if len(self._tracked_objects[best_id]) > self._max_trail:
                    self._tracked_objects[best_id] = \
                        self._tracked_objects[best_id][-self._max_trail:]
            else:
                # New object
                new_id = self._next_id
                self._next_id += 1
                assigned_ids.append(new_id)
                self._tracked_objects[new_id] = [(cx, cy)]

        # Remove stale objects (not seen for a while)
        active_ids = set(assigned_ids)
        stale = [oid for oid in self._tracked_objects if oid not in active_ids]
        for oid in stale:
            # Keep for a few frames before removing
            trail = self._tracked_objects[oid]
            if len(trail) > 0:
                self._tracked_objects[oid] = trail  # Will be cleaned next cycle
            # Actually remove if stale for too long
            # Simple approach: just remove immediately for now
            del self._tracked_objects[oid]

        return assigned_ids

    def _draw_object(self, vis: np.ndarray, obj: DetectedObject):
        """Draw bounding box, label, and trail for a detected object."""
        x, y, bw, bh = obj.bbox
        cx, cy = obj.center

        # Colour based on object ID (cycle through palette)
        colors = [
            (250, 180, 137),  # Blue
            (161, 227, 166),  # Green
            (168, 186, 243),  # Red/pink
            (175, 226, 249),  # Yellow
            (247, 166, 203),  # Purple
            (235, 220, 137),  # Cyan
            (167, 195, 250),  # Orange
            (205, 214, 244),  # White-ish
        ]
        color = colors[obj.obj_id % len(colors)]

        if self.show_boxes:
            # Bounding box with rounded corners effect
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2, cv2.LINE_AA)

            # Corner accents (thicker corner lines for premium look)
            corner_len = min(20, bw // 4, bh // 4)
            thick = 3
            # Top-left
            cv2.line(vis, (x, y), (x + corner_len, y), color, thick, cv2.LINE_AA)
            cv2.line(vis, (x, y), (x, y + corner_len), color, thick, cv2.LINE_AA)
            # Top-right
            cv2.line(vis, (x + bw, y), (x + bw - corner_len, y), color, thick, cv2.LINE_AA)
            cv2.line(vis, (x + bw, y), (x + bw, y + corner_len), color, thick, cv2.LINE_AA)
            # Bottom-left
            cv2.line(vis, (x, y + bh), (x + corner_len, y + bh), color, thick, cv2.LINE_AA)
            cv2.line(vis, (x, y + bh), (x, y + bh - corner_len), color, thick, cv2.LINE_AA)
            # Bottom-right
            cv2.line(vis, (x + bw, y + bh), (x + bw - corner_len, y + bh), color, thick, cv2.LINE_AA)
            cv2.line(vis, (x + bw, y + bh), (x + bw, y + bh - corner_len), color, thick, cv2.LINE_AA)

            # Center dot
            cv2.circle(vis, (cx, cy), 4, color, -1, cv2.LINE_AA)

        if self.show_labels:
            # Label background
            label = f"ID:{obj.obj_id}"
            if obj.speed > 1.0:
                arrow = obj.direction_label
                label += f" {arrow} {obj.speed:.0f}px/f"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y - 8, th + 4)
            cv2.rectangle(vis, (x, label_y - th - 4), (x + tw + 8, label_y + 4),
                          (0, 0, 0), -1)
            cv2.rectangle(vis, (x, label_y - th - 4), (x + tw + 8, label_y + 4),
                          color, 1, cv2.LINE_AA)
            cv2.putText(vis, label, (x + 4, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        if self.show_trails:
            # Draw motion trail
            trail = self._tracked_objects.get(obj.obj_id, [])
            if len(trail) >= 2:
                for j in range(1, len(trail)):
                    alpha = j / len(trail)
                    thickness = max(1, int(alpha * 3))
                    pt1 = trail[j - 1]
                    pt2 = trail[j]
                    # Fade the trail color
                    faded = tuple(int(c * alpha) for c in color)
                    cv2.line(vis, pt1, pt2, faded, thickness, cv2.LINE_AA)

    @staticmethod
    def _deg_to_label(deg: float) -> str:
        """Convert angle in degrees to a compass direction arrow."""
        # 0° = Right, 90° = Up, 180° = Left, 270° = Down
        directions = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
        idx = int((deg + 22.5) / 45) % 8
        return directions[idx]

    def set_min_area(self, val: int):
        self.min_area = max(100, val)

    def set_max_objects(self, val: int):
        self.max_objects = max(1, min(50, val))
