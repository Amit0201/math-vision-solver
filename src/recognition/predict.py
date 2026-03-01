# src/recognition/predict.py

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.recognition.model import MathSymbolCNN
from src.preprocessing.image_processor import MathImageProcessor
from src.preprocessing.segmentation import EquationSegmenter, SegmentedSymbol
from src.math_engine.parser import MathParser
from src.utils.helpers import load_config, get_device, setup_logger

logger = setup_logger('predict')


class MathEquationPredictor:
    """
    End-to-end prediction pipeline:
      Image → Preprocessing → Segmentation → Recognition → Parsed Expression

    This is the BRIDGE between Computer Vision and the Math Solver.
    """

    SYMBOL_CLASSES = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '+', '-', '*', '/', '=', '(', ')', '^',
        'x', 'y', 'z', 'sqrt', 'pi', '.', 'frac'
    ]

    def __init__(self, model_path: Optional[str] = None,
                 config_path: str = "config.yaml"):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to saved model weights (.pth file)
            config_path: Path to config.yaml
        """
        self.config = load_config(config_path)
        self.device = get_device(self.config['training']['device'])

        # Initialize components
        self.processor = MathImageProcessor(
            target_size=tuple(self.config['preprocessing']['input_size'])
        )
        self.segmenter = EquationSegmenter(
            min_area=self.config['preprocessing']['min_contour_area'],
            target_size=tuple(self.config['preprocessing']['symbol_size'])
        )
        self.parser = MathParser()

        # Load model
        self.model = MathSymbolCNN(
            num_classes=self.config['model']['symbol_cnn']['num_classes']
        )

        if model_path and Path(model_path).exists():
            self._load_model(model_path)
            logger.info(f"✅ Model loaded from {model_path}")
        else:
            logger.warning(
                "⚠️  No trained model found. "
                "Predictions will be random until you train the model.\n"
                "   Run: python src/recognition/train.py"
            )

        self.model.to(self.device)
        self.model.eval()

    def _load_model(self, model_path: str):
        """Load trained model weights."""
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

    def predict_from_image(self, image_path: str) -> dict:
        """
        Full pipeline: Image file → Parsed equation string.

        Args:
            image_path: Path to equation image

        Returns:
            {
                'symbols': ['x', '^', '2', '+', '3'],
                'confidences': [0.98, 0.95, 0.99, ...],
                'expression': 'x**2 + 3',
                'problem_type': 'expression',
                'num_symbols': 5,
                'avg_confidence': 0.96,
            }
        """
        # Step 1: Preprocess
        processed = self.processor.preprocess(image_path)
        binary = (processed * 255).astype(np.uint8)

        # Step 2: Segment into individual symbols
        segmented_symbols = self.segmenter.segment(binary)

        if not segmented_symbols:
            return {
                'symbols': [],
                'confidences': [],
                'expression': '',
                'problem_type': 'unknown',
                'num_symbols': 0,
                'avg_confidence': 0.0,
                'error': 'No symbols detected in image'
            }

        # Step 3: Classify each symbol
        symbols = []
        confidences = []

        for seg_symbol in segmented_symbols:
            symbol, confidence = self._classify_symbol(seg_symbol.image)
            symbols.append(symbol)
            confidences.append(confidence)

        # Step 4: Parse into expression
        expression = self.parser.symbols_to_expression(symbols)
        problem_type = self.parser.detect_problem_type(symbols)

        result = {
            'symbols': symbols,
            'confidences': confidences,
            'expression': expression,
            'problem_type': problem_type,
            'num_symbols': len(symbols),
            'avg_confidence': float(np.mean(confidences)),
        }

        logger.info(f"Recognized: {expression} "
                    f"(confidence: {result['avg_confidence']:.2f})")

        return result

    def predict_from_array(self, image_array: np.ndarray) -> dict:
        """
        Same as predict_from_image but takes numpy array directly.
        Used by the web app when image comes from upload/camera.
        """
        # Preprocess
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array

        denoised = self.processor.denoise(gray)
        binary = self.processor.adaptive_threshold(denoised)
        deskewed = self.processor.deskew(binary)

        # Segment
        segmented_symbols = self.segmenter.segment(deskewed)

        if not segmented_symbols:
            return {
                'symbols': [], 'confidences': [],
                'expression': '', 'problem_type': 'unknown',
                'num_symbols': 0, 'avg_confidence': 0.0,
            }

        # Classify
        symbols = []
        confidences = []
        for seg_symbol in segmented_symbols:
            symbol, confidence = self._classify_symbol(seg_symbol.image)
            symbols.append(symbol)
            confidences.append(confidence)

        # Parse
        expression = self.parser.symbols_to_expression(symbols)
        problem_type = self.parser.detect_problem_type(symbols)

        return {
            'symbols': symbols,
            'confidences': confidences,
            'expression': expression,
            'problem_type': problem_type,
            'num_symbols': len(symbols),
            'avg_confidence': float(np.mean(confidences)),
        }

    def _classify_symbol(self, symbol_image: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single symbol image using the CNN model.

        Args:
            symbol_image: Grayscale image of one symbol (45×45)

        Returns:
            (predicted_symbol, confidence_score)
        """
        # Prepare tensor
        img = symbol_image.astype(np.float32) / 255.0
        tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)  # (1, 1, 45, 45)
        tensor = tensor.to(self.device)

        # Predict
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)

        symbol = self.SYMBOL_CLASSES[predicted_idx.item()]
        conf = confidence.item()

        return symbol, conf

    def predict_with_alternatives(self, symbol_image: np.ndarray,
                                   top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top-k predictions for a symbol (useful for ambiguous cases).

        Returns:
            [(symbol_1, conf_1), (symbol_2, conf_2), ...]
        """
        img = symbol_image.astype(np.float32) / 255.0
        tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)

        alternatives = []
        for i in range(top_k):
            symbol = self.SYMBOL_CLASSES[top_indices[0][i].item()]
            conf = top_probs[0][i].item()
            alternatives.append((symbol, conf))

        return alternatives


# ──────────────────────────────────────────────
#  Standalone Usage
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(
        description="Predict math equation from image"
    )
    arg_parser.add_argument('image', type=str,
                            help='Path to equation image')
    arg_parser.add_argument('--model', type=str,
                            default='models/best_model.pth',
                            help='Path to trained model')
    args = arg_parser.parse_args()

    predictor = MathEquationPredictor(model_path=args.model)
    result = predictor.predict_from_image(args.image)

    print(f"\n{'='*50}")
    print(f"📷 Image      : {args.image}")
    print(f"🔍 Symbols    : {result['symbols']}")
    print(f"📝 Expression : {result['expression']}")
    print(f"📊 Type       : {result['problem_type']}")
    print(f"🎯 Confidence : {result['avg_confidence']:.2%}")
    print(f"{'='*50}")