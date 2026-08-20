"""
Real-World Multi-Feature Smart Parking Occupancy & Slot Detection Engine.

Features:
1. Multi-Color (White & Yellow) Line Detection & Hough Transform Slot Outline Extractor.
2. Background Contrast Differential + HSV Saturation & Texture Variance Classifier.
3. Robust Fallback & Custom Bounding Box ROI Support.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


class ParkingDetector:
    """
    Real-World Computer Vision Parking Slot Detector & Occupancy Classifier.
    """

    def __init__(self, sensitivity: float = 0.5):
        """
        :param sensitivity: Sensitivity slider [0.1 to 1.0].
        """
        self.sensitivity = max(0.1, min(1.0, sensitivity))

    def detect_automatic_slots(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects parking bay outlines from real-life images using HSV White & Yellow line masks,
        Morphological kernel filtering, and Hough Line segment analysis.
        """
        img_h, img_w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # 1. White & Yellow Line Color Masks (Common parking line paint)
        if hsv is not None:
            # White mask
            lower_white = np.array([0, 0, 180], dtype=np.uint8)
            upper_white = np.array([180, 50, 255], dtype=np.uint8)
            mask_white = cv2.inRange(hsv, lower_white, upper_white)

            # Yellow mask
            lower_yellow = np.array([15, 80, 140], dtype=np.uint8)
            upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            line_mask = cv2.bitwise_or(mask_white, mask_yellow)
        else:
            _, line_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # 2. Morphological dilation to connect broken line segments
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        
        dilated_v = cv2.dilate(line_mask, kernel_v, iterations=2)
        dilated_h = cv2.dilate(line_mask, kernel_h, iterations=2)
        combined_lines = cv2.bitwise_or(dilated_v, dilated_h)

        # 3. Contour Detection on Line Mask
        contours, _ = cv2.findContours(combined_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_slots = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            aspect_ratio = float(h) / max(1, w)

            # Filter contours by size and aspect ratio suitable for parking slots
            min_area = (img_w * img_h) * 0.008
            max_area = (img_w * img_h) * 0.20

            if min_area <= area <= max_area:
                if 0.3 <= aspect_ratio <= 3.5:
                    detected_slots.append({
                        "id": len(detected_slots) + 1,
                        "bbox": [x, y, w, h]
                    })

        # Non-Maximum Suppression to remove overlapping bounding boxes
        if len(detected_slots) > 1:
            detected_slots = self._suppress_overlaps(detected_slots)

        # Sort slots top-to-bottom, left-to-right
        if len(detected_slots) >= 2:
            detected_slots = sorted(detected_slots, key=lambda s: (s["bbox"][1] // 50, s["bbox"][0]))
            for i, s in enumerate(detected_slots):
                s["id"] = i + 1
            return detected_slots

        return []

    def _suppress_overlaps(self, slots: List[Dict[str, Any]], iou_thresh: float = 0.35) -> List[Dict[str, Any]]:
        """Non-Maximum Suppression (NMS) for overlapping slot bounding boxes."""
        boxes = np.array([s["bbox"] for s in slots])
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]

        order = areas.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(slots[i])

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]

        return keep

    def preprocess_frame(
        self,
        image: np.ndarray,
        blur_kernel: int = 3,
        block_size: int = 25,
        c_val: int = 16
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Preprocesses frame into Grayscale, HSV, and Adaptive Threshold maps."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            gray = image.copy()
            hsv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

        # Gaussian Blur to remove asphalt grain
        blur_kernel = max(3, blur_kernel | 1)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

        # Adaptive Thresholding for illumination invariant edge extraction
        block_size = max(3, block_size | 1)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c_val
        )

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
        1. Inner Center ROI Cropping (Ignores outer painted boundary lines)
        2. Edge Density & Canny Contour Ratio
        3. Grayscale Texture Variance (Std Dev)
        4. HSV Color Saturation Variance
        5. Background Contrast Differential
        """
        x, y, w, h = bbox

        # Crop inner 82% ROI to isolate vehicle body
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

        # Feature 1: Edge Pixel Density
        non_zero = cv2.countNonZero(roi_binary)
        binary_ratio = non_zero / total_pixels

        # Feature 2: Texture Standard Deviation (Asphalt < 15, Vehicles > 22)
        _, std_dev = cv2.meanStdDev(roi_gray)
        std_val = float(std_dev[0][0])

        # Feature 3: HSV Color Saturation Variance
        if len(roi_hsv.shape) == 3 and roi_hsv.shape[2] == 3:
            sat_channel = roi_hsv[:, :, 1]
            val_channel = roi_hsv[:, :, 2]
            _, sat_std = cv2.meanStdDev(sat_channel)
            _, val_std = cv2.meanStdDev(val_channel)
            color_variance = float((sat_std[0][0] + val_std[0][0]) / 2.0)
        else:
            color_variance = std_val

        # Feature 4: Canny Contour Edge Ratio
        canny_img = cv2.Canny(roi_gray, 40, 140)
        canny_ratio = cv2.countNonZero(canny_img) / total_pixels

        # Multi-Feature Fusion Confidence Score [0.0 to 1.0]
        score_binary = min(1.0, binary_ratio * 3.8)
        score_canny = min(1.0, canny_ratio * 7.5)
        score_texture = min(1.0, std_val / 38.0)
        score_color = min(1.0, color_variance / 45.0)

        composite_score = (0.35 * score_binary) + (0.30 * score_canny) + (0.20 * score_texture) + (0.15 * score_color)

        # Dynamic Decision Threshold based on sensitivity slider
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
        """Analyzes all parking slot ROIs and renders color-coded occupancy overlays."""
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
