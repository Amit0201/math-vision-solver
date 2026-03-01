# scripts/download_data.py

import os
import sys
import subprocess
import shutil
import tarfile
import zipfile
from pathlib import Path
import urllib.request

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from src.utils.helpers import load_config, setup_logger, timer

logger = setup_logger('download_data')


class DataDownloader:
    """Downloads and organizes all training datasets."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.data_dir = Path(self.config['paths']['datasets'])
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @timer
    def download_all(self):
        """Download all enabled datasets."""
        print("=" * 60)
        print("📥 DATASET DOWNLOAD MANAGER")
        print("=" * 60)

        for source in self.config['data']['sources']:
            if source['enabled']:
                print(f"\n{'─'*60}")
                print(f"📦 {source['name']}")
                print(f"{'─'*60}")

                if source['name'] == 'emnist':
                    self.download_emnist()
                elif source['name'] == 'hasyv2':
                    self.download_hasyv2()
                elif source['name'] == 'kaggle_math_symbols':
                    self.download_kaggle()
                elif source['name'] == 'custom_synthetic':
                    self.generate_synthetic(source.get('samples_per_class', 1000))

        print(f"\n{'='*60}")
        print("✅ All datasets ready!")
        self.print_summary()

    def download_emnist(self):
        """Download EMNIST via torchvision (automatic)."""
        try:
            from torchvision.datasets import EMNIST

            emnist_dir = self.data_dir / 'emnist'
            emnist_dir.mkdir(exist_ok=True)

            logger.info("Downloading EMNIST (balanced split)...")
            train_data = EMNIST(
                root=str(emnist_dir),
                split='balanced',
                train=True,
                download=True
            )
            test_data = EMNIST(
                root=str(emnist_dir),
                split='balanced',
                train=False,
                download=True
            )

            logger.info(f"✅ EMNIST downloaded: "
                       f"{len(train_data)} train, {len(test_data)} test")

        except Exception as e:
            logger.error(f"❌ EMNIST download failed: {e}")

    def download_hasyv2(self):
        """Download HASYv2 dataset."""
        hasy_dir = self.data_dir / 'HASYv2'
        hasy_dir.mkdir(exist_ok=True)

        archive_path = hasy_dir / 'HASYv2.tar.bz2'

        if (hasy_dir / 'hasy-data').exists():
            logger.info("✅ HASYv2 already exists, skipping download")
            return

        url = "https://zenodo.org/record/259444/files/HASYv2.tar.bz2"
        logger.info(f"Downloading from {url}...")

        try:
            urllib.request.urlretrieve(url, str(archive_path),
                                        reporthook=self._download_progress)
            print()  # newline after progress bar

            logger.info("Extracting...")
            with tarfile.open(str(archive_path), 'r:bz2') as tar:
                tar.extractall(str(hasy_dir))

            # Clean up archive
            archive_path.unlink()
            logger.info("✅ HASYv2 downloaded and extracted")

        except Exception as e:
            logger.error(f"❌ HASYv2 download failed: {e}")
            logger.info("Manual download:")
            logger.info(f"  1. Go to: {url}")
            logger.info(f"  2. Extract to: {hasy_dir}")

    def download_kaggle(self):
        """Download Kaggle math symbols dataset."""
        kaggle_dir = self.data_dir / 'kaggle_math_symbols'
        kaggle_dir.mkdir(exist_ok=True)

        logger.info("Kaggle dataset requires manual setup:")
        logger.info("  1. Install: pip install kaggle")
        logger.info("  2. Set up API key: ~/.kaggle/kaggle.json")
        logger.info("  3. Run:")
        logger.info("     kaggle datasets download "
                    "-d sagyamthapa/handwritten-math-symbols")
        logger.info(f"  4. Extract to: {kaggle_dir}")

        # Try automatic download if kaggle CLI is available
        try:
            result = subprocess.run(
                ['kaggle', 'datasets', 'download',
                 '-d', 'sagyamthapa/handwritten-math-symbols',
                 '-p', str(kaggle_dir)],
                capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                # Extract zip
                for zip_file in kaggle_dir.glob('*.zip'):
                    with zipfile.ZipFile(str(zip_file), 'r') as z:
                        z.extractall(str(kaggle_dir))
                    zip_file.unlink()
                logger.info("✅ Kaggle dataset downloaded")
            else:
                logger.warning(f"Kaggle CLI failed: {result.stderr}")

        except FileNotFoundError:
            logger.warning("Kaggle CLI not found. Please download manually.")
        except subprocess.TimeoutExpired:
            logger.warning("Download timed out. Please download manually.")

    def generate_synthetic(self, samples_per_class: int = 1000):
        """Generate synthetic handwritten math symbols."""
        sys.path.append(str(Path(__file__).parent))

        from create_custom_dataset import DatasetCreator

        output_dir = str(self.data_dir / 'custom_synthetic')
        creator = DatasetCreator(output_dir)
        creator.generate_synthetic(samples_per_class=samples_per_class)
        logger.info("✅ Synthetic dataset generated")

    def _download_progress(self, count, block_size, total_size):
        """Display download progress bar."""
        percent = int(count * block_size * 100 / total_size)
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\r  Progress: |{bar}| {percent}%', end='', flush=True)

    def print_summary(self):
        """Print summary of all downloaded datasets."""
        print(f"\n{'='*60}")
        print("📊 DATASET SUMMARY")
        print(f"{'='*60}")

        total_files = 0
        for source_dir in self.data_dir.iterdir():
            if source_dir.is_dir():
                count = sum(1 for _ in source_dir.rglob('*.png'))
                count += sum(1 for _ in source_dir.rglob('*.jpg'))
                total_files += count
                print(f"  {source_dir.name:30s} : {count:>8,} images")

        print(f"  {'─'*48}")
        print(f"  {'TOTAL':30s} : {total_files:>8,} images")


if __name__ == "__main__":
    downloader = DataDownloader()
    downloader.download_all()