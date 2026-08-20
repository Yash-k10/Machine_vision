"""
Document Image Preprocessing Engine for Practical 9.

Provides deskewing, noise reduction, adaptive thresholding,
morphological refinement, and perspective unwarping for scanned documents.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional


class DocumentPreprocessor:
    """
    Advanced Document Image Preprocessing pipeline using OpenCV.
    """

    def __init__(self, debug: bool = False):
        self.debug = debug

    def load_image(self, input_path_or_array) -> np.ndarray:
        """Load image from path or return numpy array if already loaded."""
        if isinstance(input_path_or_array, str):
            image = cv2.imread(input_path_or_array)
            if image is None:
                raise ValueError(f"Failed to read image from path: {input_path_or_array}")
            return image
        elif isinstance(input_path_or_array, np.ndarray):
            return input_path_or_array.copy()
        else:
            raise TypeError("Input must be a file path string or numpy array.")

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert color image to grayscale."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        return image.copy()

    def denoise(self, gray: np.ndarray, method: str = "gaussian", kernel_size: int = 5) -> np.ndarray:
        """Apply noise reduction filters."""
        kernel_size = max(3, kernel_size | 1)  # Ensure odd kernel size
        if method == "gaussian":
            return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        elif method == "median":
            return cv2.medianBlur(gray, kernel_size)
        elif method == "bilateral":
            return cv2.bilateralFilter(gray, d=kernel_size, sigmaColor=75, sigmaSpace=75)
        return gray.copy()

    def apply_threshold(self, gray: np.ndarray, method: str = "otsu", block_size: int = 15, c: int = 4) -> np.ndarray:
        """Apply binarization thresholding."""
        if method == "otsu":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return thresh
        elif method == "adaptive_gaussian":
            block_size = max(3, block_size | 1)
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c
            )
        elif method == "adaptive_mean":
            block_size = max(3, block_size | 1)
            return cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c
            )
        elif method == "binary_inv_otsu":
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            return thresh
        return gray.copy()

    def apply_morphology(self, binary: np.ndarray, operation: str = "close", kernel_size: int = 3) -> np.ndarray:
        """Apply morphological operations to connect broken text lines or remove noise artifacts."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        if operation == "close":
            return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif operation == "open":
            return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif operation == "dilate":
            return cv2.dilate(binary, kernel, iterations=1)
        elif operation == "erode":
            return cv2.erode(binary, kernel, iterations=1)
        return binary.copy()

    def detect_skew_angle(self, gray: np.ndarray) -> float:
        """
        Detect text skew angle using minimum area bounding box on thresholded text contours
        or Hough line angles.
        """
        # Invert image: text becomes white, background black
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find non-zero points (text pixels)
        pts = np.column_stack(np.where(thresh > 0))
        if len(pts) == 0:
            return 0.0

        # Calculate minimum bounding box
        rect = cv2.minAreaRect(pts)
        angle = rect[-1]

        # Standardize angle range [-45, 45]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        # Cap angle to realistic document skew bounds [-30, 30]
        if abs(angle) > 30:
            angle = 0.0

        return float(angle)

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle with white background padding."""
        if abs(angle) < 0.1:
            return image.copy()

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions to avoid cropping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Adjust matrix translation
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        border_value = (255, 255, 255) if len(image.shape) == 3 else 255
        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderValue=border_value)
        return rotated

    def unwarp_document(self, image: np.ndarray) -> np.ndarray:
        """Detect document page quadrangle contour and apply perspective transform."""
        gray = self.to_grayscale(image)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break

        if doc_contour is None:
            return image.copy()

        # Perspective Transform
        pts = doc_contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-Left
        rect[2] = pts[np.argmax(s)]  # Bottom-Right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-Right
        rect[3] = pts[np.argmax(diff)]  # Bottom-Left

        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def process(self, input_image, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute full document preprocessing pipeline.
        Returns a dictionary of all intermediate images and parameters.
        """
        cfg = {
            "auto_deskew": True,
            "denoise_method": "gaussian",
            "denoise_kernel": 5,
            "threshold_method": "otsu",
            "adaptive_block_size": 15,
            "adaptive_c": 4,
            "apply_morphology": False,
            "morph_op": "close",
            "unwarp": False
        }
        if config:
            cfg.update(config)

        original = self.load_image(input_image)

        # Unwarp if requested
        working_img = self.unwarp_document(original) if cfg["unwarp"] else original.copy()

        # Deskewing
        gray_init = self.to_grayscale(working_img)
        skew_angle = 0.0
        if cfg["auto_deskew"]:
            skew_angle = self.detect_skew_angle(gray_init)
            working_img = self.rotate_image(working_img, skew_angle)

        gray = self.to_grayscale(working_img)
        denoised = self.denoise(gray, method=cfg["denoise_method"], kernel_size=cfg["denoise_kernel"])
        binary = self.apply_threshold(
            denoised,
            method=cfg["threshold_method"],
            block_size=cfg["adaptive_block_size"],
            c=cfg["adaptive_c"]
        )

        final_processed = binary
        if cfg["apply_morphology"]:
            final_processed = self.apply_morphology(binary, operation=cfg["morph_op"])

        return {
            "original": original,
            "gray": gray,
            "denoised": denoised,
            "binary": binary,
            "final": final_processed,
            "skew_angle": skew_angle,
            "config": cfg
        }
