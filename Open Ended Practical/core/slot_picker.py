"""
Parking Slot Coordinate Manager & Layout Configuration Tool.

Manages parking bay bounding box coordinates [x, y, w, h] and provides
JSON layout persistence for machine vision experiments.
"""

import json
import os
from typing import List, Dict, Any


class SlotManager:
    """
    Manages slot bounding box coordinates and JSON layout files.
    """

    def __init__(self, config_path: str = "parking_slots.json"):
        self.config_path = config_path

    def generate_grid_slots(
        self,
        rows: int = 2,
        cols: int = 6,
        start_x: int = 60,
        start_y: int = 90,
        slot_w: int = 100,
        slot_h: int = 160,
        gap_x: int = 20,
        gap_y: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Generates a structured grid array of parking slot bounding boxes.
        """
        slots = []
        slot_id = 1

        for r in range(rows):
            for c in range(cols):
                x = start_x + c * (slot_w + gap_x)
                y = start_y + r * (slot_h + gap_y)
                slots.append({
                    "id": slot_id,
                    "bbox": [x, y, slot_w, slot_h]
                })
                slot_id += 1

        return slots

    def save_slots(self, slots: List[Dict[str, Any]], filepath: str = None):
        """Save slot configuration to JSON file."""
        target = filepath or self.config_path
        with open(target, "w") as f:
            json.dump(slots, f, indent=2)
        return target

    def load_slots(self, filepath: str = None) -> List[Dict[str, Any]]:
        """Load slot configuration from JSON file, generating defaults if absent."""
        target = filepath or self.config_path
        if os.path.exists(target):
            with open(target, "r") as f:
                return json.load(f)
        else:
            # Fallback to default 12-slot layout
            default_slots = self.generate_grid_slots()
            self.save_slots(default_slots, target)
            return default_slots
