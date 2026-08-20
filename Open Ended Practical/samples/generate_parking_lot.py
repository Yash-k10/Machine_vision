"""
Synthetic Parking Lot Graphic Generator for Open Ended Practical.

Generates realistic top-down aerial parking lot images complete with asphalt textures,
white divider lines, numbered bays, and parked cars for offline testing and viva demos.
"""

import cv2
import numpy as np
import os
import random
from typing import Tuple, List, Dict, Any


def draw_car(img: np.ndarray, x: int, y: int, w: int, h: int, color_bgr: Tuple[int, int, int]):
    """Draws a top-down vehicle graphic inside a parking slot ROI."""
    # Car Body Shape
    margin = 8
    cx, cy, cw, ch = x + margin, y + margin, w - (2 * margin), h - (2 * margin)

    # Main Body Base
    cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), color_bgr, -1)
    cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), (30, 30, 30), 2)

    # Roof & Windshield Overlay
    roof_margin_x = int(cw * 0.15)
    roof_margin_y = int(ch * 0.25)
    rx = cx + roof_margin_x
    ry = cy + roof_margin_y
    rw = cw - (2 * roof_margin_x)
    rh = ch - (2 * roof_margin_y)

    # Windshields (Front & Rear)
    cv2.rectangle(img, (rx, ry), (rx + rw, ry + int(rh * 0.2)), (60, 60, 60), -1)  # Front windshield
    cv2.rectangle(img, (rx, ry + int(rh * 0.8)), (rx + rw, ry + rh), (60, 60, 60), -1)  # Rear windshield

    # Roof
    cv2.rectangle(img, (rx, ry + int(rh * 0.2)), (rx + rw, ry + int(rh * 0.8)), (
        max(0, color_bgr[0] - 30),
        max(0, color_bgr[1] - 30),
        max(0, color_bgr[2] - 30)
    ), -1)

    # Side Mirrors
    cv2.rectangle(img, (cx - 3, cy + int(ch * 0.25)), (cx, cy + int(ch * 0.35)), color_bgr, -1)
    cv2.rectangle(img, (cx + cw, cy + int(ch * 0.25)), (cx + cw + 3, cy + int(ch * 0.35)), color_bgr, -1)


def generate_parking_lot_image(
    rows: int = 2,
    cols: int = 6,
    occupied_indices: List[int] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Generates a synthetic parking lot image with occupied and vacant slots.
    """
    if occupied_indices is None:
        occupied_indices = [0, 2, 3, 5, 7, 10]  # Default 6 occupied out of 12

    slot_w, slot_h = 100, 160
    gap_x, gap_y = 20, 60
    start_x, start_y = 60, 90

    img_w = start_x * 2 + cols * slot_w + (cols - 1) * gap_x
    img_h = start_y * 2 + rows * slot_h + (rows - 1) * gap_y

    # Dark Asphalt Background (#2b2b2b)
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 45

    # Add subtle asphalt texture noise
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Drive Lane Marking Lines
    lane_y = start_y + slot_h + gap_y // 2
    for lx in range(20, img_w - 20, 40):
        cv2.line(img, (lx, lane_y), (lx + 20, lane_y), (200, 200, 200), 2)

    slots = []
    slot_id = 1
    car_colors = [
        (40, 40, 220),   # Crimson Red
        (220, 120, 40),  # Ocean Blue
        (40, 180, 40),   # Emerald Green
        (200, 200, 200), # Metallic Silver
        (30, 160, 220),  # Amber Yellow
        (180, 40, 180),  # Purple
        (50, 50, 50)     # Dark Graphite
    ]

    for r in range(rows):
        for c in range(cols):
            x = start_x + c * (slot_w + gap_x)
            y = start_y + r * (slot_h + gap_y)

            # Draw Slot White Divider Lines
            cv2.rectangle(img, (x, y), (x + slot_w, y + slot_h), (240, 240, 240), 2)

            # Slot Number Header
            cv2.putText(
                img, f"SLOT {slot_id}", (x + 18, y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1
            )

            is_occupied = (slot_id - 1) in occupied_indices
            if is_occupied:
                car_color = car_colors[(slot_id - 1) % len(car_colors)]
                draw_car(img, x, y, slot_w, slot_h, car_color)

            slots.append({
                "id": slot_id,
                "bbox": [x, y, slot_w, slot_h],
                "ground_truth": "OCCUPIED" if is_occupied else "VACANT"
            })
            slot_id += 1

    metadata = {
        "total_slots": len(slots),
        "occupied_count": len(occupied_indices),
        "vacant_count": len(slots) - len(occupied_indices),
        "occupancy_rate": round((len(occupied_indices) / len(slots) * 100), 1)
    }

    return img, slots, metadata


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    img, slots, meta = generate_parking_lot_image()
    save_path = os.path.join(out_dir, "sample_parking_lot.png")
    cv2.imwrite(save_path, img)
    print(f"Sample parking lot image generated at: {save_path}")
    print(f"Stats: Total {meta['total_slots']} slots | Occupied: {meta['occupied_count']} | Free: {meta['vacant_count']}")
