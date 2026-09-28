"""Train the digit classifier on MNIST, test it, and save what it learned.

Run:  python train.py

The first run downloads MNIST into ./data (about 12 MB); later runs reuse it.
"""
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.simple_model import Digit_Classifier

# Settings worth experimenting with once everything works.
BATCH_SIZE = 64       # images the model sees per weight update
EPOCHS = 5
torch.manual_seed(0)  # same random starting weights and shuffle order on every run

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
# download=True fetches the four MNIST files, checks them, and unpacks them into
# data/MNIST/raw/ (skipped when they're already there). ToTensor converts each
# image's pixels from 0-255 to 0.0-1.0, shaped (1, 28, 28). predict.py must
# prepare images exactly the same way, or the model will be looking at
# numbers it never saw in training.
to_tensor = transforms.ToTensor()
train_data = datasets.MNIST("data", train=True, download=True, transform=to_tensor)
test_data = datasets.MNIST("data", train=False, download=True, transform=to_tensor)

# A DataLoader deals the data out in batches of (64 images, 64 labels),
# reshuffled every epoch so the batches are a fresh mix each time.
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

# ---------------------------------------------------------------------------
# 2. Model, loss, optimizer
# ---------------------------------------------------------------------------
model = Digit_Classifier()
print(f"before training: {model.evaluate(test_loader):.1%} test accuracy (random guessing is 10%)")

# # ---------------------------------------------------------------------------
# # 3. Training loop
# # ---------------------------------------------------------------------------
model.fit(train_loader, EPOCHS)

# # ---------------------------------------------------------------------------
# # 4. Test on images the model has never seen
# # ---------------------------------------------------------------------------
print(f"after training:  {model.evaluate(test_loader):.2%} test accuracy on 10,000 unseen images")

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
# state_dict() is only the learned numbers, not the code. That's why predict.py
# rebuilds DigitClassifier from model.py and then loads these numbers into it.
torch.save(model.state_dict(), "weights/digit_classifier.pt")
print("saved weights to digit_classifier.pt")

# Save one real test image of each digit as a PNG, so you have files to try
# with predict.py now and with your API later. The file name is the right answer.
Path("samples").mkdir(exist_ok=True)
test_labels = test_data.targets.tolist()
for digit in range(10):
    index = test_labels.index(digit)  # position of the first test image of this digit
    Image.fromarray(test_data.data[index].numpy()).save(f"samples/digit_{digit}.png")
print("saved 10 sample images to samples/")
