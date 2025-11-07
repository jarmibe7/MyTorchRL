"""

A generic module in a neural network for deep learning, modeled after PyTorch.

Author: Jared Berry

"""
import numpy as np

class Module():
    """
    A base neural network module.
    """
    def __init__(self):
        pass

    def forward(self, x):
        pass

    def backward(self, dx):
        pass

    def __call__(self, x):
        return self.forward(x)