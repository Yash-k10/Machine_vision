"""
Advanced Smart Parking Occupancy & Slot Detection Engine.

Implements:
1. Automatic Parking Bay Outline Detection via Contour & Aspect Ratio Filtering.
2. Multi-Feature Vision Classification (Inner ROI Edge Density, Grayscale Variance, and HSV Saturation/Value Contrast).
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class ParkingDetector:
    """
    Multi-Feature OpenCV Parking Slot Detector & Occupancy Classifier.
    """

    def __init__(self, sensitivity: float = 0.5):
        """
        :param sensitivity: Sensitivity scalar [0.1 to 1.0].
        """
        self.sensitivity = max(0.1, min(1.0, sensitivity))

    def detect_automatic_slots(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Automatically detects parking bay rectangular outlines from image
        using Canny edge analysis and aspect-ratio geometry filtering.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect white/light slot boundary lines
        _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_slots = []
        img_h, img_w = image.shape[:2]

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(h) / max(1, w)

            # Filter for parking slot geometry (width: 35-250px, height: 70-350px, aspect ratio: 1.1-3.5)
            if (35 <= w <= int(img_w * 0.35)) and (70 <= h <= int(img_h * 0.5)):
                if 1.1 <= aspect_ratio <= 3.2 or 0.3 <= aspect_ratio <= 0.9:
                    detected_slots.append({
                        "id": len(detected_slots) + 1,
                        "bbox": [x, y, w, h]
                    })

        # Sort slots top-to-bottom, left-to-right
        if len(detected_slots) >= 4:
            detected_slots = sorted(detected_slots, key=lambda s: (s["bbox"][1] // 60, s["bbox"][0]))
            for i, s in enumerate(detected_slots):
                s["id"] = i + 1
            return detected_slots

        # If automatic detection yields few slots, return empty to fallback to grid
        return []

    def preprocess_frame(
        self,
        image: np.ndarray,
        blur_kernel: int = 3,
        block_size: int = 25,
        c_val: int = 16
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocesses frame into Grayscale, HSV, Adaptive Binarization, and Morphological Dilation.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            gray = image.copy()
            hsv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

        # Gaussian Blur to suppress asphalt texture noise
        blur_kernel = max(3, blur_kernel | 1)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Adaptive Thresholding for illumination invariant edge extraction
        block_size = max(3, block_size | 1)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_val
        )

        # Median filter & Dilation
        binary = cv2.medianBlur(binary, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=1)

        return gray, hsv, binary

    def analyze_slot_roi(
        self,
        gray_img: np.ndarray,
        hsv_img: np.ndarray,
        binary_img: np.ndarray,
        bbox: List[int]
    ) -> Dict[str, Any]:
        """
        Multi-Feature classification for a single parking slot ROI:
        1. Inner Center ROI Cropping (Ignores outer white parking lines)
        2. Edge & Contour Density (Canny + Adaptive Binary)
        3. Grayscale Texture Standard Deviation (Smooth Asphalt < 15 vs Vehicle Paint/Glass > 24)
        4. HSV Color Saturation Variance
        """
        x, y, w, h = bbox

        # Crop inner 82% ROI to strictly isolate vehicle body from boundary lines
        pad_w = int(w * 0.09)
        pad_h = int(h * 0.09)

        crop_x = max(0, x + pad_w)
        crop_y = max(0, y + pad_h)
        crop_w = max(1, w - 2 * pad_w)
        crop_h = max(1, h - 2 * pad_h)

        roi_binary = binary_img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        roi_gray = gray_img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        roi_hsv = hsv_img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

        if roi_binary.size == 0 or roi_gray.size == 0:
            return {
                "occupied": False,
                "confidence": 0.0,
                "non_zero_pixels": 0,
                "texture_std_dev": 0.0
            }

        total_pixels = float(roi_binary.size)

        # Feature 1: Adaptive Binarized Pixel Density
        non_zero = cv2.countNonZero(roi_binary)
        binary_ratio = non_zero / total_pixels

        # Feature 2: Grayscale Texture Standard Deviation (Asphalt < 15, Vehicles > 22)
        _, std_dev = cv2.meanStdDev(roi_gray)
        std_val = float(std_dev[0][0])

        # Feature 3: HSV Saturation & Value Variance
        if len(roi_hsv.shape) == 3 and roi_hsv.shape[2] == 3:
            sat_channel = roi_hsv[:, :, 1]
            val_channel = roi_hsv[:, :, 2]
            _, sat_std = cv2.meanStdDev(sat_channel)
            _, val_std = cv2.meanStdDev(val_channel)
            color_variance = float((sat_std[0][0] + val_std[0][0]) / 2.0)
        else:
            color_variance = std_val

        # Feature 4: Canny Edge Ratio
        canny_img = cv2.Canny(roi_gray, 40, 140)
        canny_ratio = cv2.countNonZero(canny_img) / total_pixels

        # Multi-Feature Fusion Confidence Score [0.0 to 1.0]
        score_binary = min(1.0, binary_ratio * 3.8)
        score_canny = min(1.0, canny_ratio * 7.5)
        score_texture = min(1.0, std_val / 38.0)
        score_color = min(1.0, color_variance / 45.0)

        composite_score = (0.35 * score_binary) + (0.30 * score_canny) + (0.20 * score_texture) + (0.15 * score_color)

        # Dynamic Threshold based on sensitivity slider
        decision_threshold = 0.28 * (1.25 - self.sensitivity)
        is_occupied = composite_score >= decision_threshold

        return {
            "occupied": is_occupied,
            "confidence": round(composite_score * 100, 1),
            "non_zero_pixels": non_zero,
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
        gray, hsv, binary = self.preprocess_frame(image)
        overlay = image.copy()

        total_slots = len(slots)
        occupied_count = 0
        vacant_count = 0
        slot_results = []

        for slot in slots:
            slot_id = slot.get("id", len(slot_results) + 1)
            x, y, w, h = slot["bbox"]

            roi_stats = self.analyze_slot_roi(gray, hsv, binary, slot["bbox"])
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
