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

from model import DigitClassifier

# Settings worth experimenting with once everything works.
EPOCHS = 5            # full passes over all 60,000 training images
BATCH_SIZE = 64       # images the model sees per weight update
LEARNING_RATE = 1e-3  # update size: too big and training goes haywire, too small and it crawls

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
test_loader = DataLoader(test_data, batch_size=1000)

# ---------------------------------------------------------------------------
# 2. Model, loss, optimizer
# ---------------------------------------------------------------------------
model = DigitClassifier()
loss_fn = nn.CrossEntropyLoss()  # (scores, correct answers) -> one "how wrong" number
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  # applies the updates


def test_accuracy(model):
    """Fraction of the 10,000 test images the model labels correctly."""
    model.eval()              # prediction mode (see the note in the training loop)
    correct = 0
    with torch.no_grad():  # only measuring, so skip the bookkeeping learning needs
        for images, labels in test_loader:
            guesses = model(images).argmax(dim=1)  # position of the highest score = the digit
            correct += (guesses == labels).sum().item()
    return correct / len(test_data)


print(f"before training: {test_accuracy(model):.1%} test accuracy (random guessing is 10%)")

# ---------------------------------------------------------------------------
# 3. Training loop
# ---------------------------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    # train() and eval() switch layers like dropout between their training and
    # prediction behavior. This model has none, so here they change nothing,
    # but always setting the mode explicitly is a habit that prevents bugs later.
    model.train()
    total_loss, correct = 0.0, 0

    for images, labels in train_loader:  # images: (64, 1, 28, 28)   labels: (64,)
        scores = model(images)           # forward pass -> (64, 10)
        loss = loss_fn(scores, labels)   # how wrong were these 64 guesses?

        optimizer.zero_grad()  # clear what was computed for the previous batch
        loss.backward()        # backpropagation: which way should each weight move?
        optimizer.step()       # move every weight a small step that way

        total_loss += loss.item()
        correct += (scores.argmax(dim=1) == labels).sum().item()

    print(
        f"epoch {epoch}/{EPOCHS}: "
        f"loss {total_loss / len(train_loader):.4f}, "
        f"train accuracy {correct / len(train_data):.2%}"
    )

# ---------------------------------------------------------------------------
# 4. Test on images the model has never seen
# ---------------------------------------------------------------------------
print(f"after training: {test_accuracy(model):.2%} test accuracy on 10,000 unseen images")

# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------
# state_dict() is only the learned numbers, not the code. That's why predict.py
# rebuilds DigitClassifier from model.py and then loads these numbers into it.
torch.save(model.state_dict(), "digit_classifier.pt")
print("saved weights to digit_classifier.pt")

# Save one real test image of each digit as a PNG, so you have files to try
# with predict.py now and with your API later. The file name is the right answer.
Path("samples").mkdir(exist_ok=True)
test_labels = test_data.targets.tolist()
for digit in range(10):
    index = test_labels.index(digit)  # position of the first test image of this digit
    Image.fromarray(test_data.data[index].numpy()).save(f"samples/digit_{digit}.png")
print("saved 10 sample images to samples/")
