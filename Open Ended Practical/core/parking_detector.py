"""
AI/ML Expert Grade Smart Parking Occupancy Classification Engine.

Implements Calibrated Multi-Feature Ensemble Classifier:
1. Center 60% x 55% ROI Isolation (Ignores "SLOT N" text and white boundary lines).
2. Asphalt Baseline Noise Floor Subtraction.
3. Feature Fusion: Edge Density + Texture StdDev + HSV Color Saturation + HOG Gradient.
4. Fixed Decision Boundary — Robust across ALL sensitivity values.
Made by Yash Kapse.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


class ParkingDetector:
    """Expert AI/ML Computer Vision Parking Slot Occupancy Classifier."""

    def __init__(self, sensitivity: float = 0.5):
        self.sensitivity = max(0.1, min(1.0, sensitivity))

    # ------------------------------------------------------------------ #
    #  Automatic Slot Outline Detector (Real-World Images)               #
    # ------------------------------------------------------------------ #
    def detect_automatic_slots(self, image: np.ndarray) -> List[Dict[str, Any]]:
        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)

        thresh_gauss = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12)
        _, thresh_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        combined = cv2.bitwise_or(thresh_gauss, thresh_otsu)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            ar = float(w) / max(1, h)
            min_w, max_w = int(img_w * 0.06), int(img_w * 0.40)
            min_h, max_h = int(img_h * 0.10), int(img_h * 0.55)
            if (min_w <= w <= max_w) and (min_h <= h <= max_h) and (0.3 <= ar <= 2.8):
                candidates.append({"id": 0, "bbox": [x, y, w, h]})

        if len(candidates) >= 2:
            candidates = self._suppress_overlaps(candidates)
            candidates.sort(key=lambda s: (s["bbox"][1] // 60, s["bbox"][0]))
            for i, s in enumerate(candidates):
                s["id"] = i + 1
            return candidates
        return []

    def _suppress_overlaps(self, slots, iou_thresh=0.30):
        if not slots:
            return []
        boxes = np.array([s["bbox"] for s in slots])
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
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
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[np.where(ovr <= iou_thresh)[0] + 1]
        return keep

    # ------------------------------------------------------------------ #
    #  Feature Extraction Helpers                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hog_energy(roi_gray: np.ndarray) -> float:
        if roi_gray.shape[0] < 16 or roi_gray.shape[1] < 16:
            return 0.0
        resized = cv2.resize(roi_gray, (64, 64))
        gx = cv2.Sobel(resized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(resized, cv2.CV_32F, 0, 1, ksize=3)
        mag, _ = cv2.cartToPolar(gx, gy)
        return float(np.mean(mag))

    # ------------------------------------------------------------------ #
    #  Per-Slot Classification                                           #
    # ------------------------------------------------------------------ #
    def analyze_slot_roi(self, gray, hsv, binary, bbox):
        x, y, w, h = bbox

        # ---- Center 60 % width × 55 % height crop ----
        cx = x + int(w * 0.20)
        cy = y + int(h * 0.25)
        cw = max(1, int(w * 0.60))
        ch = max(1, int(h * 0.55))

        img_h, img_w = gray.shape[:2]
        cx, cy = max(0, min(img_w - 1, cx)), max(0, min(img_h - 1, cy))
        cw, ch = min(img_w - cx, cw), min(img_h - cy, ch)

        roi_bin = binary[cy:cy + ch, cx:cx + cw]
        roi_gry = gray[cy:cy + ch, cx:cx + cw]
        roi_hsv = hsv[cy:cy + ch, cx:cx + cw]

        if roi_bin.size == 0 or roi_gry.size == 0:
            return {"occupied": False, "confidence": 0.0, "non_zero_pixels": 0, "texture_std_dev": 0.0}

        total = float(roi_bin.size)

        # --- Raw features ---
        non_zero   = cv2.countNonZero(roi_bin)
        edge_ratio = non_zero / total

        _, sd = cv2.meanStdDev(roi_gry)
        std_val = float(sd[0][0])

        hog = self._hog_energy(roi_gry)

        # HSV saturation std
        if len(roi_hsv.shape) == 3 and roi_hsv.shape[2] == 3:
            _, ss = cv2.meanStdDev(roi_hsv[:, :, 1])
            sat_std = float(ss[0][0])
        else:
            sat_std = 0.0

        # --- Baseline-subtracted & clamped normalisation ---
        # Measured baselines on empty synthetic asphalt:
        #   edge_ratio ≈ 0.00,  std_val ≈ 3.1,  hog ≈ 9.8,  sat_std ≈ 4.0
        # Measured occupied (car present):
        #   edge_ratio ≈ 0.005–0.25,  std_val ≈ 12–56,  hog ≈ 9–62,  sat_std ≈ 30–90
        #
        # We pick generous thresholds well above asphalt noise:
        EDGE_FLOOR  = 0.003          # asphalt noise is ~0.0
        EDGE_CEIL   = 0.15
        TEX_FLOOR   = 8.0            # asphalt ~3.1 → anything above 8 = car
        TEX_CEIL    = 40.0
        HOG_FLOOR   = 12.0           # asphalt ~9.8
        HOG_CEIL    = 45.0
        SAT_FLOOR   = 15.0           # asphalt ~4–12
        SAT_CEIL    = 70.0

        ne = np.clip((edge_ratio - EDGE_FLOOR) / (EDGE_CEIL - EDGE_FLOOR), 0.0, 1.0)
        nt = np.clip((std_val    - TEX_FLOOR)  / (TEX_CEIL  - TEX_FLOOR),  0.0, 1.0)
        nh = np.clip((hog        - HOG_FLOOR)  / (HOG_CEIL  - HOG_FLOOR),  0.0, 1.0)
        ns = np.clip((sat_std    - SAT_FLOOR)  / (SAT_CEIL  - SAT_FLOOR),  0.0, 1.0)

        # Weighted fusion
        score = 0.30 * ne + 0.30 * nt + 0.15 * nh + 0.25 * ns

        # --- Fixed decision boundary with small sensitivity shift ---
        # At sensitivity=0.5 → threshold = 0.20
        # At sensitivity=0.1 → threshold = 0.28  (more conservative)
        # At sensitivity=1.0 → threshold = 0.12  (more aggressive)
        threshold = 0.20 + 0.20 * (0.5 - self.sensitivity)
        is_occupied = bool(score >= threshold)

        return {
            "occupied": is_occupied,
            "confidence": round(float(score) * 100, 1),
            "non_zero_pixels": int(non_zero),
            "texture_std_dev": round(float(std_val), 1),
        }

    # ------------------------------------------------------------------ #
    #  Full-Frame Processing                                             #
    # ------------------------------------------------------------------ #
    def process_slots(self, image, slots):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  if len(image.shape) == 3 else cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary  = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)

        overlay = image.copy()
        total_slots = len(slots)
        occ = 0; vac = 0; results = []

        for slot in slots:
            sid = slot.get("id", len(results) + 1)
            x, y, w, h = slot["bbox"]
            stats = self.analyze_slot_roi(gray, hsv, binary, slot["bbox"])
            occupied = stats["occupied"]

            if occupied:
                occ += 1; label = "OCCUPIED"; col = (0, 0, 220); bg = (0, 0, 180)
            else:
                vac += 1; label = "VACANT";   col = (0, 200, 0); bg = (0, 150, 0)

            cv2.rectangle(overlay, (x, y), (x + w, y + h), col, 2)
            cv2.rectangle(overlay, (x, y - 18), (x + 38, y), bg, -1)
            cv2.putText(overlay, f"P{sid}", (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            conf = stats.get("confidence", 0.0)
            cv2.rectangle(overlay, (x, y + h - 18), (x + w, y + h), (20, 20, 20), -1)
            cv2.putText(overlay, f"{label} ({int(conf)}%)", (x + 4, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

            results.append({
                "id": sid, "bbox": [x, y, w, h], "occupied": occupied,
                "status": label, "confidence": conf,
                "non_zero_pixels": stats.get("non_zero_pixels", 0),
                "texture_std_dev": stats.get("texture_std_dev", 0.0),
            })

        rate = round(occ / total_slots * 100, 1) if total_slots > 0 else 0.0
        bw = max(360, total_slots * 22)
        cv2.rectangle(overlay, (10, 10), (10 + bw, 52), (30, 30, 30), -1)
        cv2.rectangle(overlay, (10, 10), (10 + bw, 52), (85, 107, 47), 2)
        cv2.putText(overlay, f"Total: {total_slots} | Available: {vac} | Occupied: {occ} ({rate}%)", (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        return {
            "overlay_image": overlay, "binary_image": binary, "gray_image": gray,
            "total_slots": total_slots, "occupied_slots": occ,
            "available_slots": vac, "occupancy_rate": rate, "slots": results,
        }
