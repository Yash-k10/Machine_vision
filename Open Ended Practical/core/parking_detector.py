"""
Advanced Multi-Feature Smart Parking Occupancy Detection Engine.

Uses Computer Vision feature fusion (Inner ROI Edge Density, Texture Variance,
Canny Contour Analysis, and Color Standard Deviation) to classify parking slots as
Occupied 🔴 or Vacant 🟢 with high accuracy.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class ParkingDetector:
    """
    Multi-Feature OpenCV Parking Slot Occupancy Classifier.
    """

    def __init__(self, sensitivity: float = 0.5):
        """
        :param sensitivity: Sensitivity slider [0.1 to 1.0]. Higher values increase occupancy detection sensitivity.
        """
        self.sensitivity = max(0.1, min(1.0, sensitivity))

    def preprocess_frame(
        self,
        image: np.ndarray,
        blur_kernel: int = 3,
        block_size: int = 25,
        c_val: int = 16
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocesses frame into Grayscale, Gaussian Blur, Adaptive Binarization, and Dilation.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Gaussian Blur to remove sensor noise and asphalt grain
        blur_kernel = max(3, blur_kernel | 1)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Adaptive Thresholding for illumination invariant edge extraction
        block_size = max(3, block_size | 1)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_val
        )

        # Median filter to eliminate small noise specs
        binary = cv2.medianBlur(binary, 5)

        # Morphological dilation to connect vehicle structural lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)

        return gray, binary

    def analyze_slot_roi(
        self,
        gray_img: np.ndarray,
        binary_img: np.ndarray,
        bbox: List[int]
    ) -> Dict[str, Any]:
        """
        Multi-Feature classification for a single parking slot ROI:
        1. Inner Center ROI Cropping (Ignores white divider line paint)
        2. Binarized Edge Density
        3. Canny Contour Edge Ratio
        4. Grayscale Texture Variance (Std Dev)
        """
        x, y, w, h = bbox

        # Crop inner 80% ROI to ignore outer white parking line markers
        pad_w = int(w * 0.10)
        pad_h = int(h * 0.10)

        crop_x = max(0, x + pad_w)
        crop_y = max(0, y + pad_h)
        crop_w = max(1, w - 2 * pad_w)
        crop_h = max(1, h - 2 * pad_h)

        roi_binary = binary_img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        roi_gray = gray_img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        if roi_binary.size == 0 or roi_gray.size == 0:
            return {
                "occupied": False,
                "confidence": 0.0,
                "non_zero_pixels": 0,
                "binary_ratio": 0.0,
                "texture_std_dev": 0.0
            }

        total_pixels = float(roi_binary.size)

        # Feature 1: Adaptive Binarized Pixel Density
        non_zero = cv2.countNonZero(roi_binary)
        binary_ratio = non_zero / total_pixels

        # Feature 2: Texture Standard Deviation (Asphalt < 15, Vehicles > 25)
        _, std_dev = cv2.meanStdDev(roi_gray)
        std_val = float(std_dev[0][0])

        # Feature 3: Canny Edge Ratio
        canny_img = cv2.Canny(roi_gray, 50, 150)
        canny_ratio = cv2.countNonZero(canny_img) / total_pixels

        # Multi-Feature Weighted Confidence Scoring (0.0 to 1.0)
        score_binary = min(1.0, binary_ratio * 4.0)
        score_canny = min(1.0, canny_ratio * 8.0)
        score_texture = min(1.0, std_val / 42.0)

        composite_score = (0.45 * score_binary) + (0.35 * score_canny) + (0.20 * score_texture)

        # Decision Threshold calculated based on sensitivity slider
        decision_threshold = 0.32 * (1.2 - self.sensitivity)
        is_occupied = composite_score >= decision_threshold

        return {
            "occupied": is_occupied,
            "confidence": round(composite_score * 100, 1),
            "non_zero_pixels": non_zero,
            "binary_ratio": round(binary_ratio * 100, 1),
            "texture_std_dev": round(std_val, 1)
        }

    def process_slots(
        self,
        image: np.ndarray,
        slots: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyzes all parking slot ROIs and renders color-coded occupancy overlays.
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

            roi_stats = self.analyze_slot_roi(gray, binary, slot["bbox"])
            is_occupied = roi_stats.get("occupied", False)

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

            # Draw slot ID header tag
            label = f"P{slot_id}"
            cv2.rectangle(overlay, (x, y - 18), (x + 38, y), badge_bg, -1)
            cv2.putText(overlay, label, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Draw status icon & confidence score label at bottom of slot
            conf_val = roi_stats.get("confidence", 0.0)
            conf_str = f"{int(conf_val)}%"
            cv2.rectangle(overlay, (x, y + h - 18), (x + w, y + h), (20, 20, 20), -1)
            cv2.putText(overlay, f"{status_str} ({conf_str})", (x + 4, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

            slot_results.append({
                "id": slot_id,
                "bbox": [x, y, w, h],
                "occupied": is_occupied,
                "status": status_str,
                "confidence": conf_val,
                "non_zero_pixels": roi_stats.get("non_zero_pixels", 0),
                "texture_std_dev": roi_stats.get("texture_std_dev", 0.0)
            })

        # Calculate occupancy percentage
        occupancy_rate = round((occupied_count / total_slots * 100), 1) if total_slots > 0 else 0.0

        # Draw HUD Summary Banner at top left of image
        banner_h, banner_w = 42, max(360, total_slots * 22)
        cv2.rectangle(overlay, (10, 10), (10 + banner_w, 10 + banner_h), (30, 30, 30), -1)
        cv2.rectangle(overlay, (10, 10), (10 + banner_w, 10 + banner_h), (85, 107, 47), 2)

        hud_text = f"Total: {total_slots} | Available: {vacant_count} | Occupied: {occupied_count} ({occupancy_rate}%)"
        cv2.putText(overlay, hud_text, (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

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
