"""
Smart Parking Slot Detection & Classification Engine.

Implements Computer Vision algorithms using OpenCV to classify parking slots as
Occupied 🔴 or Vacant 🟢 based on edge & thresholded pixel density analysis.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class ParkingDetector:
    """
    OpenCV-based Parking Slot Occupancy Classifier.
    """

    def __init__(self, pixel_threshold: int = 800):
        """
        :param pixel_threshold: Number of non-zero thresholded pixels in an ROI to consider a slot OCCUPIED.
        """
        self.pixel_threshold = pixel_threshold

    def preprocess_frame(
        self,
        image: np.ndarray,
        blur_kernel: int = 3,
        block_size: int = 25,
        c_val: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses parking frame using Grayscale, Gaussian Blur, Adaptive Thresholding, and Median Blur.
        Returns (grayscale_img, binary_threshold_img).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Gaussian Blur to remove noise
        blur_kernel = max(3, blur_kernel | 1)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Adaptive Thresholding to highlight car edges, shadows, and textures
        block_size = max(3, block_size | 1)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_val
        )

        # Median filter to eliminate small speckle noise
        binary = cv2.medianBlur(binary, 5)

        # Morphological dilation to accentuate vehicle contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)

        return gray, binary

    def process_slots(
        self,
        image: np.ndarray,
        slots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyzes pixel density in each slot ROI and overlays status bounding boxes.

        :param image: Input parking lot BGR image matrix.
        :param slots: List of slot dicts [{'id': 1, 'bbox': [x, y, w, h]}, ...]
        :return: Dict containing processed overlay image, stats, and individual slot details.
        """
        gray, binary = self.preprocess_frame(image)
        overlay = image.copy()

        total_slots = len(slots)
        occupied_count = 0
        vacant_count = 0
        slot_results = []

        for slot in slots:
            slot_id = slot.get("id", len(slot_results) + 1)
            x, y, w, h = slot["bbox"]

            # Crop slot Region of Interest (ROI) from binary image
            roi = binary[y:y + h, x:x + w]
            non_zero_count = cv2.countNonZero(roi)

            # Classification logic
            is_occupied = non_zero_count > self.pixel_threshold
            if is_occupied:
                occupied_count += 1
                status_str = "OCCUPIED"
                color = (0, 0, 220)       # Red for Occupied 🔴
                badge_bg = (0, 0, 180)
            else:
                vacant_count += 1
                status_str = "VACANT"
                color = (0, 200, 0)       # Green for Vacant 🟢
                badge_bg = (0, 150, 0)

            # Draw bounding box on overlay image
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

            # Draw slot ID tag header
            label = f"P{slot_id}"
            cv2.rectangle(overlay, (x, y - 18), (x + 35, y), badge_bg, -1)
            cv2.putText(overlay, label, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Draw pixel count on slot bottom
            count_label = f"{non_zero_count}"
            cv2.putText(overlay, count_label, (x + 4, y + h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            slot_results.append({
                "id": slot_id,
                "bbox": [x, y, w, h],
                "non_zero_pixels": non_zero_count,
                "occupied": is_occupied,
                "status": status_str
            })

        # Calculate metrics
        occupancy_rate = round((occupied_count / total_slots * 100), 1) if total_slots > 0 else 0.0

        # Draw HUD Summary Banner on top left of image
        banner_h, banner_w = 40, max(320, total_slots * 20)
        hud_bg = overlay[10:10 + banner_h, 10:10 + banner_w].copy()
        cv2.rectangle(overlay, (10, 10), (10 + banner_w, 10 + banner_h), (30, 30, 30), -1)
        cv2.rectangle(overlay, (10, 10), (10 + banner_w, 10 + banner_h), (85, 107, 47), 2)

        hud_text = f"Total: {total_slots} | Free: {vacant_count} | Busy: {occupied_count} ({occupancy_rate}%)"
        cv2.putText(overlay, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return {
            "overlay_image": overlay,
            "binary_image": binary,
            "gray_image": gray,
            "total_slots": total_slots,
            "occupied_slots": occupied_count,
            "available_slots": vacant_count,
            "occupancy_rate": occupancy_rate,
            "slots": slot_results
        }
