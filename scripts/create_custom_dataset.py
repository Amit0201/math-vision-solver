# scripts/create_custom_dataset.py

import cv2
import numpy as np
import os
from pathlib import Path
import random
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from src.utils.helpers import setup_logger

logger = setup_logger('create_dataset')


class DatasetCreator:
    """
    Generates synthetic handwritten-style math symbols
    using font rendering + augmentation.

    This creates training data when real handwriting
    datasets are unavailable or insufficient.
    """

    SYMBOL_MAP = {
        # digits
        '0': '0', '1': '1', '2': '2', '3': '3', '4': '4',
        '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
        # operators
        '+': 'plus', '-': 'minus', '*': 'multiply',
        '/': 'divide', '=': 'equals', '^': 'power',
        # brackets
        '(': 'lparen', ')': 'rparen',
        # variables
        'x': 'x', 'y': 'y', 'z': 'z',
        # special
        '.': 'decimal', 'pi': 'pi', 'sqrt': 'sqrt', 'frac': 'frac',
    }

    def __init__(self, output_dir: str = 'data/datasets/custom_synthetic'):
        self.output_dir = Path(output_dir)
        self._create_directories()

    def _create_directories(self):
        """Create train/val/test folders for each symbol class."""
        for symbol, folder_name in self.SYMBOL_MAP.items():
            for split in ['train', 'val', 'test']:
                (self.output_dir / split / folder_name).mkdir(
                    parents=True, exist_ok=True
                )

    def generate_synthetic(self, samples_per_class: int = 1000):
        """Generate synthetic symbol images using font rendering."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.error("PIL not installed. Run: pip install Pillow")
            return

        # Try to find available fonts
        fonts = self._load_fonts()

        if not fonts:
            logger.warning("No fonts found. Using basic CV rendering.")
            self._generate_cv_fallback(samples_per_class)
            return

        total_generated = 0

        for symbol, folder_name in self.SYMBOL_MAP.items():
            display_char = symbol if len(symbol) == 1 else symbol

            for i in range(samples_per_class):
                # Determine split
                if i < int(samples_per_class * 0.8):
                    split = 'train'
                elif i < int(samples_per_class * 0.9):
                    split = 'val'
                else:
                    split = 'test'

                # Create image
                img = self._render_symbol(display_char, fonts)

                # Apply augmentations
                img = self._augment(img)

                # Resize to 45×45
                img = cv2.resize(img, (45, 45))

                # Save
                save_path = (self.output_dir / split / folder_name /
                             f"{folder_name}_{i:05d}.png")
                cv2.imwrite(str(save_path), img)
                total_generated += 1

            logger.info(f"  Generated {samples_per_class} samples "
                       f"for '{symbol}' ({folder_name})")

        logger.info(f"\n✅ Total generated: {total_generated} images")
        logger.info(f"   Location: {self.output_dir}")

    def _load_fonts(self):
        """Try to load handwriting-style fonts."""
        from PIL import ImageFont

        font_paths = [
            # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
            # macOS
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
            # Windows
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/times.ttf",
        ]

        fonts = []
        for path in font_paths:
            if os.path.exists(path):
                for size in [24, 28, 32, 36, 40]:
                    try:
                        font = ImageFont.truetype(path, size)
                        fonts.append(font)
                    except Exception:
                        pass

        if not fonts:
            # Use default font
            try:
                for size in [24, 28, 32, 36, 40]:
                    fonts.append(ImageFont.load_default())
            except Exception:
                pass

        return fonts

    def _render_symbol(self, symbol: str, fonts) -> np.ndarray:
        """Render a single symbol using PIL."""
        from PIL import Image, ImageDraw

        img = Image.new('L', (64, 64), color=255)
        draw = ImageDraw.Draw(img)
        font = random.choice(fonts)

        # Get text size and center it
        try:
            bbox = draw.textbbox((0, 0), symbol, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
        except Exception:
            w, h = 20, 30

        x = (64 - w) // 2 + random.randint(-6, 6)
        y = (64 - h) // 2 + random.randint(-6, 6)

        gray_val = random.randint(0, 60)
        draw.text((x, y), symbol, fill=gray_val, font=font)

        return np.array(img)

    def _generate_cv_fallback(self, samples_per_class: int):
        """Fallback: Generate symbols using OpenCV putText."""
        total = 0

        for symbol, folder_name in self.SYMBOL_MAP.items():
            display = symbol if len(symbol) == 1 else symbol[0].upper()

            for i in range(samples_per_class):
                split = 'train' if i < samples_per_class * 0.8 else \
                        'val' if i < samples_per_class * 0.9 else 'test'

                img = np.ones((64, 64), dtype=np.uint8) * 255

                font_scale = random.uniform(1.0, 2.0)
                thickness = random.randint(1, 3)
                font = random.choice([
                    cv2.FONT_HERSHEY_SIMPLEX,
                    cv2.FONT_HERSHEY_DUPLEX,
                    cv2.FONT_HERSHEY_PLAIN,
                ])

                text_size = cv2.getTextSize(display, font,
                                            font_scale, thickness)[0]
                x = (64 - text_size[0]) // 2 + random.randint(-5, 5)
                y = (64 + text_size[1]) // 2 + random.randint(-5, 5)

                cv2.putText(img, display, (max(0, x), max(10, y)),
                           font, font_scale, 0, thickness)

                img = self._augment(img)
                img = cv2.resize(img, (45, 45))

                save_path = (self.output_dir / split / folder_name /
                             f"{folder_name}_{i:05d}.png")
                cv2.imwrite(str(save_path), img)
                total += 1

        logger.info(f"✅ Generated {total} images (CV fallback)")

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        # Random rotation
        angle = random.uniform(-15, 15)
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderValue=255)

        # Random noise
        if random.random() > 0.5:
            noise = np.random.normal(0, 8, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Random brightness/contrast
        alpha = random.uniform(0.8, 1.2)
        beta = random.randint(-15, 15)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # Random erosion/dilation (thickness variation)
        if random.random() > 0.6:
            kernel = np.ones((2, 2), np.uint8)
            if random.random() > 0.5:
                img = cv2.dilate(img, kernel, iterations=1)
            else:
                img = cv2.erode(img, kernel, iterations=1)

        return img


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/datasets/custom_synthetic')
    parser.add_argument('--samples', type=int, default=1000)
    args = parser.parse_args()

    creator = DatasetCreator(args.output)
    creator.generate_synthetic(args.samples)