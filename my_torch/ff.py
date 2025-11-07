"""
A feed forward neural network class

Author: Jared Berry
"""
import numpy as np

from module import Module
from linear import Linear

class FeedForward(Module):
    """
    A feed forward neural network

    Args:
        arch: A list of triplets of length n, where each index i corresponds to a layer
              and arch[i] is a triplet with that layer's input size, output size, and activation function.
    """
    def __init__(self, arch):
        # Get first input size and final output size
        self.input_size = arch[0][0]
        self.output_size = arch[-1][0]

        # Create layers
        self.layers = []
        for input_size, output_size, activation in arch:
            self.layers.append(Linear(input_size, output_size, activation))

    def forward(self, x):
        # Iterate through all layers
        xi = x.copy()
        for layer in self.layers:
            xi = layer(xi)

        return xi
    
    def backward(self):
        # Iterate over layers and collect gradients
        self.gradients = [] # List of (dW, db) tuples
        for layer in self.layers:
            self.gradients.append(layer.backward())

        return self.gradients   # Return if using external optimizer