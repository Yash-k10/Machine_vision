"""
Frame utility helpers for colour conversion, overlays, and drawing.
"""

import cv2
import numpy as np


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to RGB."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resize_with_aspect(
    frame: np.ndarray, max_width: int, max_height: int
) -> np.ndarray:
    """Resize a frame to fit within max dimensions while preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return frame

    scale = min(max_width / w, max_height / h)
    if scale >= 1.0:
        return frame

    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def overlay_transparent(
    background: np.ndarray, overlay: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """Blend two frames with a given alpha."""
    return cv2.addWeighted(background, 1 - alpha, overlay, alpha, 0)


def draw_info_box(
    frame: np.ndarray, lines: list, position: str = "top-left"
) -> np.ndarray:
    """
    Draw a semi-transparent info box with text lines on the frame.

    Parameters
    ----------
    frame : np.ndarray
        The frame to draw on (modified in-place and returned).
    lines : list[str]
        Lines of text to render.
    position : str
        One of 'top-left', 'top-right', 'bottom-left', 'bottom-right'.
    """
    if not lines:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 22
    padding = 10

    # Calculate box dimensions
    max_text_w = 0
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_text_w = max(max_text_w, tw)

    box_w = max_text_w + 2 * padding
    box_h = len(lines) * line_height + 2 * padding
    h, w = frame.shape[:2]

    # Position
    if position == "top-left":
        x1, y1 = 10, 10
    elif position == "top-right":
        x1, y1 = w - box_w - 10, 10
    elif position == "bottom-left":
        x1, y1 = 10, h - box_h - 10
    else:  # bottom-right
        x1, y1 = w - box_w - 10, h - box_h - 10

    x2, y2 = x1 + box_w, y1 + box_h

    # Draw semi-transparent background
    sub = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    dark = (sub * 0.3).astype(np.uint8)
    frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = dark

    # Draw text
    for i, line in enumerate(lines):
        ty = y1 + padding + (i + 1) * line_height - 4
        cv2.putText(frame, line, (x1 + padding, ty), font, font_scale,
                    (220, 240, 255), thickness, cv2.LINE_AA)

    return frame
