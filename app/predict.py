"""Classify digit images with the trained model.

Run:  python predict.py samples             (every PNG in a folder)
      python predict.py my_digit.png        (one or more image files)

load_model(), preprocess() and predict() are the entire machine-learning side
of your future API: load the model once when the server starts, then
preprocess and predict for each request.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import DigitClassifier


def load_model(weights_path="digit_classifier.pt"):
    """Rebuild the network and fill it with the weights train.py saved."""
    model = DigitClassifier()  # same shape as in training, random weights for now
    # map_location="cpu" lets the weights load on a machine without a GPU (like
    # most containers). weights_only=True loads numbers only, never code.
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)  # swap the random weights for the learned ones
    model.eval()
    return model


def preprocess(image):
    """Turn any PIL image into exactly what the model saw in training.

    Takes an image rather than a file path so an API can hand it an upload.
    """
    image = image.convert("L").resize((28, 28))           # grayscale, 28x28 pixels
    pixels = np.asarray(image, dtype=np.float32) / 255.0  # 0-255 -> 0.0-1.0, same as ToTensor
    # MNIST digits are white on black. A mostly light image is probably a dark
    # digit on white paper, so flip it to match what the model knows.
    if pixels.mean() > 0.5:
        pixels = 1.0 - pixels
    return torch.from_numpy(pixels).reshape(1, 1, 28, 28)  # a batch of one image


def predict(model, image):
    """Return (digit, confidence) for one PIL image."""
    with torch.no_grad():
        scores = model(preprocess(image))        # shape (1, 10)
        probs = torch.softmax(scores, dim=1)[0]  # 10 probabilities that add up to 1
    digit = int(probs.argmax())
    return digit, float(probs[digit])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python predict.py IMAGE_OR_FOLDER [...]")

    # A folder expands to the PNG files inside it.
    paths = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        paths += sorted(path.glob("*.png")) if path.is_dir() else [path]

    model = load_model()
    for path in paths:
        digit, confidence = predict(model, Image.open(path))
        print(f"{path}: {digit}  ({confidence:.1%} confident)")
