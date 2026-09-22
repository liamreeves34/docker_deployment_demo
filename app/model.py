"""Model definition for the MNIST hand-written digit classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# MNIST images are 1x28x28 grayscale, labelled 0-9.
INPUT_SHAPE = (1, 28, 28)
NUM_CLASSES = 10


class DigitNet(nn.Module):
    """A small convolutional net for 28x28 grayscale digits.

    Two conv blocks (each conv -> ReLU -> max-pool) take the image down to
    64x7x7, which a two-layer classifier head maps to one logit per digit.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of images (N, 1, 28, 28) to logits (N, num_classes)."""
        x = self.pool(F.relu(self.conv1(x)))  # -> (N, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (N, 64, 7, 7)
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return the predicted digit for each image in the batch."""
        self.eval()
        return self.forward(x).argmax(dim=1)


def build_model(num_classes: int = NUM_CLASSES) -> DigitNet:
    """Create an untrained DigitNet. Entry point for training and serving."""
    return DigitNet(num_classes=num_classes)


if __name__ == "__main__":
    model = build_model()
    batch = torch.randn(4, *INPUT_SHAPE)
    print(model)
    print("logits:", model(batch).shape)
    print("predictions:", model.predict(batch).tolist())
    print("parameters:", sum(p.numel() for p in model.parameters()))
