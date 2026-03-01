import cv2
import numpy as np
from typing import Tuple, List


class MathImageProcessor:
    """
    Preprocesses handwritten math equation images
    using classical Computer Vision techniques.
    """


    def __init__(self, target_size: Tuple[int, int] = (256, 64)):
        self.target_size = target_size


    def preprocess(self, image_path: str) -> np.ndarray:
        "Full Preprocessing Pipeline"
        img = cv2.imread(image_path)
        gray = self.to_grayscale(img)
        denoised = self.denoise(gray)
        binary = self.adaptive_thershold(denoised)
        deskewed = self.deskew(binary)
        normalized = self.normalize_and_resize(deskewed)
        return normalized


    def to_grayscale(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    def denoise(self, img: np.ndarray) -> np.ndarray:
        "Apply Gaussian blur + bilateral filter for noise removal"
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        denoised = cv2.bilateralFilter(blurred, 9, 75, 75)
        return denoised


    def adaptive_threshold(self, img: np.ndarray) -> np.ndarray:
        "Binarise using adaptive thresholding"
        binary = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=2
        )
        return binary


    def deskew(self, img: np.ndarray) -> np.ndarray:
        """
        Correct skew using image moments.

        Math: Uses second-order central moments (μ20, μ02, μ11)
        Skew angle = 0.5 * arctan(2 * μ11 / (μ20 - μ02))
        """
        moments = cv2.moments(img)
        if abs(moments['mu02']) < 1e-2:
            return img

        skew = moments['mu11'] / moments['mu02']
        M = np.float32([
            [1, skew, -0.5 * img.shape[0] * skew],
            [0, 1, 0]
        ])
        deskewed = cv2.warpAffine(
            img, M,
            (img.shape[1], img.shape[0]),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR
        )
        return deskewed


    def normalize_and_resize(self, img: np.ndarray) -> np.ndarray:
        """Resize and normalize pixel values to [0, 1]."""
        resized = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized


    def segment_symbols(self, binary_img: np.ndarray) -> List[np.ndarray]:
        """
        Segment individual symbols using connected components.
        Uses contour detection + bounding box extraction.
        """
        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours left-to-right
        bounding_boxes = [cv2.boundingRect(c) for c in contours]
        sorted_boxes = sorted(bounding_boxes, key=lambda b: b[0])

        symbols = []
        for (x, y, w, h) in sorted_boxes:
            if w * h > 50:  # Filter noise
                symbol = binary_img[y:y + h, x:x + w]
                # Pad to square
                symbol = self._pad_to_square(symbol)
                symbol = cv2.resize(symbol, (45, 45))
                symbols.append(symbol)

        return symbols


    def _pad_to_square(self, img: np.ndarray) -> np.ndarray:
        """Pad image to make it square (preserves aspect ratio)."""
        h, w = img.shape
        max_dim = max(h, w)
        padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        padded[y_offset:y_offset + h, x_offset:x_offset + w] = img
        return padded


















































