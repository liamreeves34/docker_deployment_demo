"""The neural network itself.

train.py and predict.py both import DigitClassifier from here. Saved weights
only fit a network with exactly this shape, so keeping the definition in one
shared file guarantees that training and prediction always agree.
"""
import torch
import torch.nn as nn


class DigitClassifier(nn.Module):
    """784 pixels in -> 128 hidden values -> 10 scores out, one per digit.

    It holds 101,770 adjustable numbers (weights and biases). They start out
    random; training is the process of tuning them.
    """

    def __init__(self):
        super().__init__()  # standard PyTorch setup; every model starts with this
        # A Linear layer computes each of its outputs as a weighted sum of all of
        # its inputs, plus a bias. Those weights and biases are what training tunes.
        self.hidden = nn.Linear(28 * 28, 128)  # 784 inputs -> 128 outputs
        self.output = nn.Linear(128, 10)       # 128 inputs -> 10 outputs

    def forward(self, x):
        """Run a batch of images through the network. model(images) calls this."""
        # x arrives as (batch, 1, 28, 28): a stack of one-channel 28x28 images.
        x = x.flatten(start_dim=1)  # -> (batch, 784): each image as one long row of pixels
        x = self.hidden(x)          # -> (batch, 128)
        # ReLU turns negative values into 0. Without this bend, two Linear layers
        # in a row are mathematically the same as one, and the extra layer is wasted.
        x = torch.relu(x)
        return self.output(x)       # -> (batch, 10): the highest score is the guess

