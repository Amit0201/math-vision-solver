# src/data/data_pipeline.py

import torch
from torch.utils.data import (
    DataLoader, Dataset, ConcatDataset,
    WeightedRandomSampler, random_split
)
from torchvision import transforms, datasets
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.helpers import load_config, setup_logger

logger = setup_logger('data_pipeline')


class MathSymbolDataset(Dataset):
    """
    Custom dataset that loads symbol images from folder structure:
        root/
          class_name_1/
            img_001.png
            img_002.png
          class_name_2/
            ...
    """

    # Maps folder names back to symbol class indices
    FOLDER_TO_CLASS = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
        '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'plus': 10, 'minus': 11, 'multiply': 12,
        'divide': 13, 'equals': 14,
        'lparen': 15, 'rparen': 16, 'power': 17,
        'x': 18, 'y': 19, 'z': 20,
        'sqrt': 21, 'pi': 22, 'decimal': 23, 'frac': 24,
    }

    def __init__(self, root_dir: str, transform=None, img_size: int = 45):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.img_size = img_size
        self.samples = []  # (image_path, label)
        self.targets = []  # Just labels (for sampler)

        self._load_samples()

    def _load_samples(self):
        """Scan directory and collect all image paths with labels."""
        if not self.root_dir.exists():
            logger.warning(f"Dataset directory not found: {self.root_dir}")
            return

        for class_dir in sorted(self.root_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name

            if class_name in self.FOLDER_TO_CLASS:
                label = self.FOLDER_TO_CLASS[class_name]
            elif class_name.isdigit() and int(class_name) < 25:
                label = int(class_name)
            else:
                logger.debug(f"Skipping unknown class: {class_name}")
                continue

            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                    self.samples.append((str(img_path), label))
                    self.targets.append(label)

        logger.info(f"Loaded {len(self.samples)} samples from {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Return blank image if loading fails
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.transform:
            from PIL import Image
            img_pil = Image.fromarray(img)
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.FloatTensor(img).unsqueeze(0) / 255.0

        label_tensor = torch.LongTensor([label]).squeeze()
        return img_tensor, label_tensor


class DataPipeline:
    """
    Complete data pipeline that:
      1. Loads data from multiple sources
      2. Applies augmentation
      3. Handles class imbalance
      4. Creates DataLoaders
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.img_size = self.config['preprocessing']['symbol_size'][0]
        self.batch_size = self.config['training']['batch_size']
        self.data_dir = Path(self.config['paths']['datasets'])

    def get_transforms(self, split: str = 'train') -> transforms.Compose:
        """Get augmentation transforms based on config."""
        aug = self.config['augmentation']

        if split == 'train':
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomRotation(aug['rotation_degrees']),
                transforms.RandomAffine(
                    degrees=0,
                    translate=tuple(aug['translate']),
                    scale=tuple(aug['scale']),
                    shear=aug['shear']
                ),
                transforms.RandomPerspective(
                    distortion_scale=aug['perspective_distortion'],
                    p=0.3
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])
        else:
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ])

    def load_all_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and combine all enabled data sources."""
        train_datasets = []
        val_datasets = []
        test_datasets = []

        train_transform = self.get_transforms('train')
        val_transform = self.get_transforms('val')

        for source in self.config['data']['sources']:
            if not source['enabled']:
                continue

            name = source['name']

            if name == 'custom_synthetic':
                base_path = Path(source.get('path',
                                 self.data_dir / 'custom_synthetic'))

                if (base_path / 'train').exists():
                    train_ds = MathSymbolDataset(
                        str(base_path / 'train'),
                        transform=train_transform
                    )
                    train_datasets.append(train_ds)

                if (base_path / 'val').exists():
                    val_ds = MathSymbolDataset(
                        str(base_path / 'val'),
                        transform=val_transform
                    )
                    val_datasets.append(val_ds)

                if (base_path / 'test').exists():
                    test_ds = MathSymbolDataset(
                        str(base_path / 'test'),
                        transform=val_transform
                    )
                    test_datasets.append(test_ds)

            elif name == 'emnist':
                try:
                    from torchvision.datasets import EMNIST
                    emnist_path = self.data_dir / 'emnist'

                    if emnist_path.exists():
                        full_train = EMNIST(
                            root=str(emnist_path), split='balanced',
                            train=True, download=False,
                            transform=train_transform
                        )
                        # Only take a subset (digits + some letters)
                        # EMNIST balanced has 47 classes, we need subset
                        train_datasets.append(full_train)
                        logger.info(f"Loaded EMNIST: {len(full_train)} samples")
                except Exception as e:
                    logger.warning(f"Could not load EMNIST: {e}")

            elif name == 'hasyv2':
                hasy_path = self.data_dir / 'HASYv2'
                if (hasy_path / 'hasy-data').exists():
                    # HASYv2 needs special loading via CSV
                    logger.info("HASYv2 detected — use custom loader")

            elif name == 'kaggle_math_symbols':
                kaggle_path = Path(source.get('path',
                                   self.data_dir / 'kaggle_math_symbols'))
                if kaggle_path.exists():
                    kaggle_train = MathSymbolDataset(
                        str(kaggle_path / 'train') if (kaggle_path / 'train').exists()
                        else str(kaggle_path),
                        transform=train_transform
                    )
                    train_datasets.append(kaggle_train)

        # Combine
        if train_datasets:
            combined_train = ConcatDataset(train_datasets)
        else:
            logger.error("❌ No training data found! Run download_data.py first.")
            combined_train = MathSymbolDataset(
                "data/datasets/custom_synthetic/train",
                transform=self.get_transforms('train')
            )

        combined_val = ConcatDataset(val_datasets) if val_datasets else \
                       MathSymbolDataset(
                           "data/datasets/custom_synthetic/val",
                           transform=self.get_transforms('val')
                       )

        combined_test = ConcatDataset(test_datasets) if test_datasets else \
                        combined_val

        # Print summary
        print(f"\n📊 Data Pipeline Summary:")
        print(f"   Train : {len(combined_train):>8,} samples")
        print(f"   Val   : {len(combined_val):>8,} samples")
        print(f"   Test  : {len(combined_test):>8,} samples")
        print(f"   Total : {len(combined_train) + len(combined_val) + len(combined_test):>8,} samples")

        return combined_train, combined_val, combined_test

    def create_weighted_sampler(self, dataset) -> WeightedRandomSampler:
        """Handle class imbalance with weighted sampling."""
        targets = []

        if hasattr(dataset, 'targets'):
            targets = dataset.targets
        elif hasattr(dataset, 'datasets'):
            for d in dataset.datasets:
                if hasattr(d, 'targets'):
                    targets.extend(d.targets)

        if not targets:
            return None

        targets = np.array(targets)
        class_counts = np.bincount(targets, minlength=25)
        class_counts = np.maximum(class_counts, 1)  # Avoid division by zero
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[targets]

        return WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(sample_weights),
            replacement=True
        )

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get ready-to-use DataLoaders."""
        train_ds, val_ds, test_ds = self.load_all_datasets()

        sampler = self.create_weighted_sampler(train_ds)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    pipeline = DataPipeline()
    train_loader, val_loader, test_loader = pipeline.get_dataloaders()

    # Verify
    batch = next(iter(train_loader))
    images, labels = batch
    print(f"\n✅ Batch loaded successfully:")
    print(f"   Images shape : {images.shape}")
    print(f"   Labels shape : {labels.shape}")
    print(f"   Pixel range  : [{images.min():.2f}, {images.max():.2f}]")