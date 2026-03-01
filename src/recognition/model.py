# src/recognition/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
#  CNN-based Symbol Classifier
# ──────────────────────────────────────────────


class MathSymbolCNN(nn.Module):
    """
    CNN for recognizing individual math symbols.

    Architecture:
      Input (1, 45, 45)
        → Conv layers with BatchNorm
        → Global Average Pooling
        → Fully Connected → Softmax

    Supports: digits (0-9), operators (+, -, ×, ÷, =),
              variables (x, y, z), brackets, powers, etc.
    """

    SYMBOL_CLASSES = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        '+', '-', '*', '/', '=', '(', ')', '^',
        'x', 'y', 'z', 'sqrt', 'pi', '.', 'frac'
    ]

    def __init__(self, num_classes: int = 25):
        super().__init__()
        self.num_classes = num_classes

        # Feature Extractor
        self.features = nn.Sequential(
            # Block 1: (1, 45, 45) → (32, 22, 22)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 2: (32, 22, 22) → (64, 11, 11)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),

            # Block 3: (64, 11, 11) → (128, 5, 5)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

    def predict(self, x: torch.Tensor) -> str:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x.unsqueeze(0))
            probs = F.softmax(logits, dim=1)
            idx = torch.argmax(probs, dim=1).item()
        return self.SYMBOL_CLASSES[idx], probs[0][idx].item()


# ──────────────────────────────────────────────
#  Full Equation Recognizer (Encoder-Decoder)
# ──────────────────────────────────────────────

class EquationEncoder(nn.Module):
    """
    CNN encoder that extracts feature maps from
    the entire equation image.
    """

    def __init__(self, encoded_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, encoded_dim, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 32)),
        )

    def forward(self, x):
        features = self.encoder(x)  # (B, 256, 8, 32)
        B, C, H, W = features.shape
        features = features.permute(0, 3, 2, 1)  # (B, W, H, C)
        features = features.reshape(B, W, H * C)  # (B, 32, 2048)
        return features


class AttentionDecoder(nn.Module):
    """
    LSTM decoder with Bahdanau Attention for
    sequence-to-sequence equation recognition.

    Math (Attention Mechanism):
        e_ij = v^T · tanh(W_h · h_i + W_s · s_{j-1})
        α_ij = softmax(e_ij)
        context_j = Σ α_ij · h_i
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, encoder_dim: int = 2048):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Attention layers
        self.W_h = nn.Linear(encoder_dim, hidden_dim, bias=False)
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

        # Decoder LSTM
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def attention(self, encoder_out, decoder_hidden):
        """
        Compute attention weights.
        encoder_out: (B, seq_len, encoder_dim)
        decoder_hidden: (B, hidden_dim)
        """
        src_proj = self.W_h(encoder_out)  # (B, seq, hidden)
        tgt_proj = self.W_s(decoder_hidden).unsqueeze(1)  # (B, 1, hidden)

        energy = torch.tanh(src_proj + tgt_proj)  # (B, seq, hidden)
        scores = self.v(energy).squeeze(2)  # (B, seq)

        alpha = F.softmax(scores, dim=1)  # (B, seq)
        context = (alpha.unsqueeze(2) * encoder_out).sum(1)  # (B, encoder_dim)

        return context, alpha

    def forward(self, encoder_out, targets, max_len=50):
        batch_size = encoder_out.size(0)
        device = encoder_out.device

        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim).to(device)
        c = torch.zeros(batch_size, self.hidden_dim).to(device)

        outputs = []
        # Start token (index 0)
        input_token = torch.zeros(batch_size, dtype=torch.long).to(device)

        for t in range(max_len):
            embed = self.embedding(input_token)  # (B, embed_dim)
            context, alpha = self.attention(encoder_out, h)
            lstm_input = torch.cat([embed, context], dim=1)

            h, c = self.lstm(lstm_input, (h, c))
            output = self.fc_out(self.dropout(h))  # (B, vocab_size)
            outputs.append(output)

            if targets is not None:
                input_token = targets[:, t]  # Teacher forcing
            else:
                input_token = output.argmax(dim=1)

        return torch.stack(outputs, dim=1)  # (B, max_len, vocab_size)