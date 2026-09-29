"""Classify digit images with the trained model.

Run:  python -m models.predict samples        (every PNG in a folder)
      python -m models.predict my_digit.png   (one or more image files)

DigitPredictor is the entire machine-learning side of the API: construct it
once when the server starts, then call preprocess() and predict_tensor() for
each request.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from models.simple_model import Digit_Classifier


class DigitPredictor:
    """Loads the trained network once, then classifies images with it."""

    def __init__(self, weights_path="weights/digit_classifier.pt"):
        """Rebuild the network and fill it with the weights train.py saved."""
        self.model = Digit_Classifier()  # same shape as in training, random weights for now
        # map_location="cpu" lets the weights load on a machine without a GPU (like
        # most containers). weights_only=True loads numbers only, never code.
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights)  # swap the random weights for the learned ones
        self.model.eval()

    def preprocess(self, image):
        """Turn any PIL image into what the model saw in training.

        MNIST digits are white on black, about 20 px tall, and centered by their
        center of mass in a 28x28 frame. This makes any image look like that.
        A real MNIST image passes through practically unchanged, so the model's
        test accuracy is the same with or without these steps.
        Takes an image rather than a file path so an API can hand it an upload.
        Returns an all-zero tensor when the image holds no readable digit.
        """
        pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0  # grayscale, 0.0-1.0

        # 1. White digit on black. A mostly light image is dark ink on paper, so flip it.
        if pixels.mean() > 0.5:
            pixels = 1.0 - pixels

        # 2. Remove the background. A digital canvas is already pure black, but a
        #    photo of paper is gray, unevenly lit, and grainy. Subtract a heavily
        #    blurred copy (the lighting), then drop the faint haze that's left.
        if np.median(pixels) > 0.02:
            blur = max(pixels.shape) // 8
            lighting = Image.fromarray((pixels * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
            pixels = np.clip(pixels - np.asarray(lighting, dtype=np.float32) / 255.0, 0.0, None)
            pixels[pixels < 0.3 * pixels.max()] = 0.0
        if pixels.max() == 0:
            return torch.zeros(1, 1, 28, 28)  # blank image, nothing to read
        pixels /= pixels.max()                # the strongest stroke becomes fully white

        # 3. Crop to the digit, so empty canvas doesn't eat the 28x28 frame.
        rows = np.flatnonzero(pixels.max(axis=1) > 0.2)
        cols = np.flatnonzero(pixels.max(axis=0) > 0.2)
        digit = Image.fromarray((pixels[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] * 255).astype(np.uint8))

        # 4. Thicken the stroke by about one final pixel's worth, so a thin pen line
        #    doesn't fade to nothing when shrunk. On a 28x28 MNIST image this is a no-op.
        width, height = digit.size
        grow = round(max(width, height) / 20) | 1  # filter size must be odd
        if grow > 1:
            digit = digit.filter(ImageFilter.MaxFilter(grow))

        # 5. Shrink so the longer side is 20 px, keeping the digit's proportions.
        scale = 20 / max(width, height)
        digit = digit.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)
        small = np.asarray(digit, dtype=np.float32) / 255.0

        # 6. Paste it into a black 28x28 frame so its center of mass sits at the
        #    middle, the same way MNIST centered its digits.
        ys, xs = np.indices(small.shape)
        total = small.sum()
        top = min(max(round(14 - (ys * small).sum() / total), 0), 28 - small.shape[0])
        left = min(max(round(14 - (xs * small).sum() / total), 0), 28 - small.shape[1])
        canvas = np.zeros((28, 28), dtype=np.float32)
        canvas[top:top + small.shape[0], left:left + small.shape[1]] = small
        return torch.from_numpy(canvas).reshape(1, 1, 28, 28)  # a batch of one image

    def predict_tensor(self, tensor):
        """Return (digit, confidence) for an already-preprocessed tensor."""
        with torch.no_grad():
            scores = self.model(tensor)              # shape (1, 10)
            probs = torch.softmax(scores, dim=1)[0]  # 10 probabilities that add up to 1
        digit = int(probs.argmax())
        return digit, float(probs[digit])

    def predict(self, image):
        """Return (digit, confidence) for one PIL image."""
        return self.predict_tensor(self.preprocess(image))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m models.predict IMAGE_OR_FOLDER [...]")

    # A folder expands to the PNG files inside it.
    paths = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        paths += sorted(path.glob("*.png")) if path.is_dir() else [path]

    predictor = DigitPredictor()
    for path in paths:
        digit, confidence = predictor.predict(Image.open(path))
        print(f"{path}: {digit}  ({confidence:.1%} confident)")