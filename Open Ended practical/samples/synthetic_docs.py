"""
Synthetic Government Document Generator for Practical 9.

Generates realistic scanned government documents, identity certificates,
and official forms with skew, noise, stains, and stamps for testing OCR pipelines.
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random
from typing import Tuple, Dict, Any


def create_synthetic_government_doc(
    skew_angle: float = 4.5,
    noise_level: float = 12.0,
    add_stamp: bool = True
) -> Tuple[np.ndarray, str, Dict[str, str]]:
    """
    Generates a synthetic government verification certificate image and ground truth text.
    """
    width, height = 850, 1100
    img = Image.new('RGB', (width, height), color=(250, 249, 246)) # Off-white paper background
    draw = ImageDraw.Draw(img)

    # Try loading default font
    try:
        font_title = ImageFont.truetype("arial.ttf", 26)
        font_header = ImageFont.truetype("arial.ttf", 18)
        font_bold = ImageFont.truetype("arialbd.ttf", 15)
        font_text = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font_title = font_header = font_bold = font_text = ImageFont.load_default()

    # Outer Document Border
    draw.rectangle([(30, 30), (width - 30, height - 30)], outline=(40, 40, 40), width=3)
    draw.rectangle([(36, 36), (width - 36, height - 36)], outline=(100, 100, 100), width=1)

    # Government Emblem Header
    draw.text((width // 2 - 190, 60), "GOVERNMENT OF INDIA", fill=(10, 25, 47), font=font_title)
    draw.text((width // 2 - 170, 95), "DEPARTMENT OF DOCUMENTATION", fill=(40, 60, 90), font=font_header)
    draw.line([(50, 130), (width - 50, 130)], fill=(120, 120, 120), width=2)

    # Document Title
    draw.text((width // 2 - 160, 150), "OFFICIAL VERIFICATION CERTIFICATE", fill=(180, 0, 0), font=font_header)

    # Metadata & Fields
    fields = {
        "Certificate No": "GOV-2026-89412",
        "Issue Date": "15/08/2026",
        "Applicant Name": "RAHUL KUMAR SHARMA",
        "Father's Name": "SURESH SHARMA",
        "DOB": "24/05/1992",
        "Gender": "MALE",
        "Aadhaar No": "4512 8901 3467",
        "PAN No": "ABCDE1234F",
        "Phone": "+91 9876543210",
        "Email": "rahul.sharma@gov.in",
        "Pincode": "110001",
        "Issuing Office": "NEW DELHI CENTRAL REGISTRY"
    }

    y_offset = 210
    ground_truth_lines = [
        "GOVERNMENT OF INDIA",
        "DEPARTMENT OF DOCUMENTATION",
        "OFFICIAL VERIFICATION CERTIFICATE"
    ]

    for key, val in fields.items():
        draw.text((80, y_offset), f"{key}:", fill=(20, 20, 20), font=font_bold)
        draw.text((260, y_offset), val, fill=(0, 0, 0), font=font_text)
        ground_truth_lines.append(f"{key}: {val}")
        y_offset += 38

    # Declaration Paragraph
    y_offset += 20
    draw.line([(50, y_offset), (width - 50, y_offset)], fill=(180, 180, 180), width=1)
    y_offset += 25

    declaration = (
        "This is to certify that the details mentioned above have been verified\n"
        "against the central database records. Any unauthorized alteration or tampering\n"
        "of this document is a punishable offense under government regulations."
    )
    draw.text((80, y_offset), declaration, fill=(50, 50, 50), font=font_text)
    ground_truth_lines.append(declaration)

    # Official Seal / Stamp Simulation
    if add_stamp:
        stamp_center = (width - 180, height - 200)
        draw.ellipse(
            [stamp_center[0] - 60, stamp_center[1] - 60, stamp_center[0] + 60, stamp_center[1] + 60],
            outline=(0, 51, 153), width=4
        )
        draw.text((stamp_center[0] - 45, stamp_center[1] - 12), "SEAL APPROVED", fill=(0, 51, 153), font=font_bold)

    # Signatures
    draw.line([(100, height - 140), (280, height - 140)], fill=(0, 0, 0), width=2)
    draw.text((120, height - 130), "Authorized Signatory", fill=(30, 30, 30), font=font_bold)

    draw.line([(width - 260, height - 140), (width - 80, height - 140)], fill=(0, 0, 0), width=2)
    draw.text((width - 240, height - 130), "Registrar Seal", fill=(30, 30, 30), font=font_bold)

    # Convert PIL Image to OpenCV BGR
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Apply Skew Rotation
    if abs(skew_angle) > 0.01:
        (h, w) = cv_img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        cv_img = cv2.warpAffine(cv_img, M, (w, h), borderValue=(255, 255, 255))

    # Add Gaussian Noise & Scan Gradient Overlay
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, cv_img.shape).astype(np.float32)
        noisy_img = np.clip(cv_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        cv_img = noisy_img

    ground_truth_text = "\n".join(ground_truth_lines)
    return cv_img, ground_truth_text, fields


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    img, gt_text, metadata = create_synthetic_government_doc(skew_angle=5.0, noise_level=10.0)
    save_path = os.path.join(out_dir, "sample_government_doc.png")
    cv2.imwrite(save_path, img)
    print(f"Sample government document generated at: {save_path}")
