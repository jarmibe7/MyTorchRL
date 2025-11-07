"""

A linear layer for a neural network.

Author: Jared Berry

https://medium.com/data-science/lets-code-a-neural-network-in-plain-numpy-ae7e74410795

"""
import numpy as np

from module import Module
from activation import Sigmoid, ReLU

class Linear(Module):
    """
    A simple linear layer in a neural network
    """
    def __init__(self, in_features, out_features, activation_class):
        # Initialize dims and weights
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.uniform(-1.0, 1.0, (self.out_features, self.in_features))
        self.b = np.random.uniform(-1.0, 1.0)

        # Create activation function
        if activation_class == 'sigmoid':
            self.activation = Sigmoid()
        elif activation_class == 'relu':
            self.activation = ReLU()
        else:
            raise NotImplementedError(f'{activation_class} is not a supported activation function!')

    def forward(self, x):
        # Dimension check
        assert x.shape[0] == self.W.shape[1], f'Expected dim 0 of x to be {self.W.shape[1]}, received {x.shape[0]}'

        # Given a_i-1, calculate z_i
        self.z = (self.W @ x) + self.b
        self.a = self.activation(self.z)

        return self.a

    def backward(self):
        # Calculate dLdW_i and dLdb_i
        delta_i = self.activation.backward()
        dLdW = delta_i @ self.a.T
        dLdb = delta_i

        return dLdW, dLdb