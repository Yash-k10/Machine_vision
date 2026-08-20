"""Test the full API process flow end-to-end to find crash."""
import sys, os, base64, json, random, traceback
sys.path.append('.')

import cv2
import numpy as np
from core.parking_detector import ParkingDetector
from core.slot_picker import SlotManager
from samples.generate_parking_lot import generate_parking_lot_image

try:
    # 1. Generate image
    occupied_indices = random.sample(range(12), 6)
    img, slots_gen, meta = generate_parking_lot_image(rows=2, cols=6, occupied_indices=occupied_indices)
    print(f"Step 1 OK: Generated image {img.shape}")

    # 2. Encode to base64 and decode back (simulates web UI round-trip)
    _, buf = cv2.imencode('.png', img)
    b64_str = "data:image/png;base64," + base64.b64encode(buf).decode('utf-8')
    clean_b64 = b64_str.split(",")[1]
    img_bytes = base64.b64decode(clean_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(f"Step 2 OK: Decoded image {raw_img.shape}")

    # 3. Load slots and scale
    sm = SlotManager()
    base_slots = sm.load_slots()
    img_h, img_w = raw_img.shape[:2]
    scale_x = img_w / 780.0
    scale_y = img_h / 530.0
    slots = []
    for s in base_slots:
        bx, by, bw, bh = s["bbox"]
        sx = max(0, int(bx * scale_x))
        sy = max(0, int(by * scale_y))
        sw = max(10, min(img_w - sx, int(bw * scale_x)))
        sh = max(10, min(img_h - sy, int(bh * scale_y)))
        slots.append({"id": s["id"], "bbox": [sx, sy, sw, sh]})
    print(f"Step 3 OK: Loaded {len(slots)} slots")

    # 4. Run detector
    detector = ParkingDetector(sensitivity=0.5)
    res = detector.process_slots(raw_img, slots)
    print(f"Step 4 OK: Occupied={res['occupied_slots']}, Available={res['available_slots']}")

    # 5. Encode overlay to base64
    overlay = res["overlay_image"]
    _, obuf = cv2.imencode('.png', overlay)
    overlay_b64 = "data:image/png;base64," + base64.b64encode(obuf).decode('utf-8')
    print(f"Step 5 OK: Overlay encoded ({len(overlay_b64)} chars)")

    print("\nALL STEPS PASSED - No crash!")

except Exception as e:
    print(f"\nCRASH at: {e}")
    traceback.print_exc()
