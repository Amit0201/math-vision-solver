# src/utils/helpers.py

import yaml
import os
import sys
import logging
import time
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, Optional
from functools import wraps


# ──────────────────────────────────────────────
#  Configuration Loader
# ──────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(config_path)

    if not config_path.exists():
        # Search parent directories
        for parent in config_path.resolve().parents:
            candidate = parent / "config.yaml"
            if candidate.exists():
                config_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"config.yaml not found. Searched from {Path.cwd()}"
            )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# ──────────────────────────────────────────────
#  Logger Setup
# ──────────────────────────────────────────────

def setup_logger(name: str, log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ──────────────────────────────────────────────
#  Device Selection
# ──────────────────────────────────────────────

def get_device(preference: str = "auto") -> torch.device:
    """
    Select computation device.

    Args:
        preference: 'auto', 'cuda', 'mps', 'cpu'

    Returns:
        torch.device
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preference)

    return device


# ──────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
#  Directory Management
# ──────────────────────────────────────────────

def ensure_directories(config: Dict):
    """Create all required directories from config."""
    dirs_to_create = [
        config['paths']['data_raw'],
        config['paths']['data_processed'],
        config['paths']['datasets'],
        config['paths']['models'],
        config['paths'].get('logs', 'logs'),
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

    print("✅ All directories verified/created")


# ──────────────────────────────────────────────
#  Timer Decorator
# ──────────────────────────────────────────────

def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start

        if elapsed < 60:
            print(f"⏱️  {func.__name__} took {elapsed:.2f}s")
        else:
            mins = int(elapsed // 60)
            secs = elapsed % 60
            print(f"⏱️  {func.__name__} took {mins}m {secs:.1f}s")

        return result
    return wrapper


# ──────────────────────────────────────────────
#  Model Utilities
# ──────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
    }


def save_checkpoint(model, optimizer, epoch, loss, accuracy, path):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path, device='cpu'):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']


# ──────────────────────────────────────────────
#  Symbol Mapping
# ──────────────────────────────────────────────

SYMBOL_TO_SAFE_NAME = {
    '+': 'plus', '-': 'minus', '*': 'multiply',
    '/': 'divide', '=': 'equals', '^': 'power',
    '(': 'lparen', ')': 'rparen', '.': 'decimal',
}

SAFE_NAME_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_SAFE_NAME.items()}


def symbol_to_folder(symbol: str) -> str:
    """Convert symbol to safe folder name."""
    return SYMBOL_TO_SAFE_NAME.get(symbol, symbol)


def folder_to_symbol(folder: str) -> str:
    """Convert folder name back to symbol."""
    return SAFE_NAME_TO_SYMBOL.get(folder, folder)