# src/preprocessing/segmentation.py

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SegmentedSymbol:
    """Represents a single segmented symbol from an equation image."""
    image: np.ndarray             # Cropped symbol image
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    center: Tuple[int, int]       # Center point
    area: int                     # Contour area
    position_index: int           # Left-to-right order


class EquationSegmenter:
    """
    Segments a preprocessed equation image into
    individual symbol images.

    Pipeline:
      Binary image
        → Contour detection
        → Bounding box extraction
        → Overlap merging
        → Left-to-right sorting
        → Individual symbol crops
    """

    def __init__(self, min_area: int = 50,
                 merge_threshold: int = 5,
                 target_size: Tuple[int, int] = (45, 45)):
        self.min_area = min_area
        self.merge_threshold = merge_threshold
        self.target_size = target_size

    def segment(self, binary_image: np.ndarray) -> List[SegmentedSymbol]:
        """
        Main segmentation pipeline.

        Args:
            binary_image: Preprocessed binary image (white symbols on black)

        Returns:
            List of SegmentedSymbol objects, sorted left-to-right
        """
        # Step 1: Find contours
        contours = self._find_contours(binary_image)

        # Step 2: Get bounding boxes
        bounding_boxes = self._get_bounding_boxes(contours)

        # Step 3: Filter noise (too small)
        bounding_boxes = self._filter_noise(bounding_boxes)

        # Step 4: Merge overlapping boxes (handles 'i', '=', ':' etc.)
        bounding_boxes = self._merge_overlapping(bounding_boxes)

        # Step 5: Sort left-to-right
        bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

        # Step 6: Crop and pad each symbol
        symbols = []
        for idx, (x, y, w, h) in enumerate(bounding_boxes):
            cropped = binary_image[y:y+h, x:x+w]
            padded = self._pad_to_square(cropped)
            resized = cv2.resize(padded, self.target_size,
                                 interpolation=cv2.INTER_AREA)

            symbol = SegmentedSymbol(
                image=resized,
                bounding_box=(x, y, w, h),
                center=(x + w // 2, y + h // 2),
                area=w * h,
                position_index=idx
            )
            symbols.append(symbol)

        return symbols

    def _find_contours(self, binary_image: np.ndarray) -> list:
        """Find external contours in binary image."""
        contours, hierarchy = cv2.findContours(
            binary_image.copy(),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def _get_bounding_boxes(self, contours) -> List[Tuple[int, int, int, int]]:
        """Extract bounding rectangles from contours."""
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
        return boxes

    def _filter_noise(self, boxes: List[Tuple]) -> List[Tuple]:
        """Remove bounding boxes that are too small (noise)."""
        filtered = []
        for (x, y, w, h) in boxes:
            area = w * h
            if area >= self.min_area:
                filtered.append((x, y, w, h))
        return filtered

    def _merge_overlapping(self, boxes: List[Tuple]) -> List[Tuple]:
        """
        Merge vertically overlapping bounding boxes.

        This handles symbols like:
          'i'  → dot + vertical stroke (2 contours)
          '='  → two horizontal lines (2 contours)
          ':'  → two dots (2 contours)
          '!'  → vertical stroke + dot

        Two boxes are merged if their x-ranges overlap
        significantly (> 50% of the smaller width).
        """
        if not boxes:
            return boxes

        boxes = sorted(boxes, key=lambda b: b[0])
        merged = [boxes[0]]

        for current in boxes[1:]:
            last = merged[-1]

            # Check horizontal overlap
            overlap = self._x_overlap(last, current)
            min_width = min(last[2], current[2])

            if overlap > min_width * 0.5:
                # Merge: create bounding box that covers both
                x1 = min(last[0], current[0])
                y1 = min(last[1], current[1])
                x2 = max(last[0] + last[2], current[0] + current[2])
                y2 = max(last[1] + last[3], current[1] + current[3])
                merged[-1] = (x1, y1, x2 - x1, y2 - y1)
            else:
                merged.append(current)

        return merged

    def _x_overlap(self, box1: Tuple, box2: Tuple) -> int:
        """Calculate horizontal overlap between two boxes."""
        x1_start, _, w1, _ = box1
        x2_start, _, w2, _ = box2

        x1_end = x1_start + w1
        x2_end = x2_start + w2

        overlap_start = max(x1_start, x2_start)
        overlap_end = min(x1_end, x2_end)

        return max(0, overlap_end - overlap_start)

    def _pad_to_square(self, img: np.ndarray,
                       padding: int = 10) -> np.ndarray:
        """Pad image to square with border padding."""
        h, w = img.shape[:2]
        max_dim = max(h, w) + 2 * padding

        padded = np.zeros((max_dim, max_dim), dtype=np.uint8)
        y_offset = (max_dim - h) // 2
        x_offset = (max_dim - w) // 2
        padded[y_offset:y_offset+h, x_offset:x_offset+w] = img

        return padded

    def visualize_segmentation(self, original_image: np.ndarray,
                                symbols: List[SegmentedSymbol]) -> np.ndarray:
        """Draw bounding boxes on the original image for visualization."""
        if len(original_image.shape) == 2:
            vis_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = original_image.copy()

        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255)
        ]

        for i, symbol in enumerate(symbols):
            x, y, w, h = symbol.bounding_box
            color = colors[i % len(colors)]
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(vis_image, str(i), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return vis_image


    def detect_spatial_relations(self,
                                  symbols: List[SegmentedSymbol]) -> List[dict]:
        """
        Detect spatial relationships between symbols.
        Important for: superscripts (^), subscripts (_), fractions (/)

        Uses relative vertical position:
          - If symbol center is above the baseline → superscript
          - If symbol center is below the baseline → subscript
          - If horizontal line detected → fraction
        """
        if not symbols:
            return []

        # Calculate baseline (median of all symbol centers' y-coordinates)
        y_centers = [s.center[1] for s in symbols]
        baseline_y = int(np.median(y_centers))
        heights = [s.bounding_box[3] for s in symbols]
        median_height = int(np.median(heights))

        relations = []
        for symbol in symbols:
            cx, cy = symbol.center
            _, _, _, h = symbol.bounding_box

            relation = 'inline'  # default

            # Superscript: center is significantly above baseline
            if cy < baseline_y - median_height * 0.3 and h < median_height * 0.7:
                relation = 'superscript'

            # Subscript: center is significantly below baseline
            elif cy > baseline_y + median_height * 0.3 and h < median_height * 0.7:
                relation = 'subscript'

            relations.append({
                'symbol_index': symbol.position_index,
                'relation': relation,
                'center': (cx, cy),
                'baseline_y': baseline_y,
            })

        return relations