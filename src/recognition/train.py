import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path

# import from project, not in-built library
from model import MathSymbolCNN


# ──────────────────────────────────────────────
#  Custom Dataset
# ──────────────────────────────────────────────


class MathSymbolDataset(Dataset):
    """Dataset for handwritten math symbols"""


    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        self.labels = []

        for label_idx, class_dir in enumerate(
            sorted(self.data_dir.iterdir())
        ):
            if class_dir.is_dir():
                for img_path in class_dir.glob("*.png"):
                    self.samples.append(str(img_path))
                    self.labels.append(label_idx)


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        import cv2
        img = cv2.imread(self.samples[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (45, 45))
        img = img.astype(np.float32) / 255.0

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.FloatTensor(img).unsqueeze(0)  # (1, 45, 45)

        label = torch.LongTensor([self.labels[idx]]).squeeze()
        return img, label

    # ──────────────────────────────────────────────
    #  Training Loop
    # ──────────────────────────────────────────────


class Trainer:
    """Handles model training with metrics tracking"""

    def __init__(self, model, train_loader, val_loader,
                 lr=1e-3, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr,
                                    weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )

    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.train_loader)

        return {'loss': avg_loss, 'accuracy': accuracy}

    def validate(self) -> dict:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.val_loader)

        return {'loss': avg_loss, 'accuracy': accuracy}

    def fit(self, epochs: int = 50):
        best_val_acc = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            self.scheduler.step(val_metrics['loss'])

            print(f"Epoch {epoch:3d}/{epochs} │ "
                  f"Train Loss: {train_metrics['loss']:.4f} │ "
                  f"Train Acc: {train_metrics['accuracy']:.2f}% │ "
                  f"Val Loss: {val_metrics['loss']:.4f} │ "
                  f"Val Acc: {val_metrics['accuracy']:.2f}%")

            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(self.model.state_dict(),
                           'models/best_model.pth')
                print(f" Saved best model (Val Acc: {best_val_acc:.2f}%)")


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Data augmentation for training robustness

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
    ])

    train_dataset = MathSymbolDataset("data/processed/train",
                                      transform=train_transform)
    val_dataset = MathSymbolDataset("data/processed/val")

    train_loader = DataLoader(train_dataset, batch_size=64,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64,
                            shuffle=False, num_workers=4)

    model = MathSymbolCNN(num_classes=25)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = Trainer(model, train_loader, val_loader,
                      lr=1e-3, device=device)
    trainer.fit(epochs=50)
