# Save this as check_status.py in project root
# Run: python check_status.py

import os
from pathlib import Path

print("=" * 60)
print("📋 PROJECT STATUS CHECK")
print("=" * 60)

checks = {
    ".venv exists": Path(".venv").exists(),
    "config.yaml exists": Path("config.yaml").exists(),
    "requirements.txt exists": Path("requirements.txt").exists(),
    "": None,  # separator
    "Synthetic data generated": Path("data/raw/synthetic/train/0").exists(),
    "EMNIST downloaded": Path("data/raw/emnist").exists(),
    "HASYv2 downloaded": Path("data/raw/hasyv2/hasy-data").exists(),
    " ": None,  # separator
    "Combined data ready": Path("data/processed/train/0").exists(),
    "Dataset stats saved": Path("data/processed/dataset_stats.json").exists(),
    "  ": None,  # separator
    "Model trained": Path("models/best_model.pth").exists(),
    "Training history saved": Path("models/training_history.json").exists(),
}

for check_name, result in checks.items():
    if result is None:
        print(f"{'─'*60}")
        continue
    status = "✅" if result else "❌"
    print(f"  {status} {check_name}")

# Count data
processed = Path("data/processed")
if processed.exists():
    for split in ['train', 'val', 'test']:
        split_dir = processed / split
        if split_dir.exists():
            count = sum(1 for _ in split_dir.rglob("*.png"))
            print(f"\n  📊 {split}: {count:,} images")

print(f"\n{'='*60}")

# Next step recommendation
if not Path("data/raw/synthetic/train/0").exists():
    print("👉 NEXT: python scripts/create_custom_dataset.py --output data/raw/synthetic --samples 1000")
elif not Path("data/processed/train/0").exists():
    print("👉 NEXT: python scripts/combine_datasets.py --sources synthetic")
elif not Path("models/best_model.pth").exists():
    print("👉 NEXT: python src/recognition/train.py")
else:
    print("👉 NEXT: streamlit run src/api/app.py")