"""

A linear layer for a neural network.

Author: Jared Berry

https://medium.com/data-science/lets-code-a-neural-network-in-plain-numpy-ae7e74410795

"""
import numpy as np

from my_torch.module import Module
from my_torch.activation import Sigmoid, ReLU, Dummy, Softmax

class Linear(Module):
    """
    A simple linear layer in a neural network
    """
    def __init__(self, in_features, out_features, activation_class):
        # Initialize dims and weights
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_limit = 0.5
        self.W = np.random.uniform(-init_limit, init_limit, (self.out_features, self.in_features))    # (n_i, n_i-1)
        self.b = np.random.uniform(-init_limit, init_limit, (1, self.out_features))   # (n_i, 1)

        # Create activation function
        if activation_class == 'sigmoid':
            self.activation = Sigmoid()
        elif activation_class == 'relu':
            self.activation = ReLU()
        elif activation_class == 'softmax':
            self.activation = Softmax()
        elif activation_class == 'dummy':
            self.activation = Dummy()
        else:
            raise NotImplementedError(f'{activation_class} is not a supported activation function!')

    def forward(self, x):
        # Dimension check
        assert x.shape[1] == self.W.shape[1], f'Expected dim 1 of x to be {self.W.shape[1]}, received {x.shape[1]}'
        self.x = x

        # Given a_i-1, calculate z_i
        self.z = (x @ self.W.T) + self.b
        self.a = self.activation(self.z)

        return self.a

    def backward(self, dLda):
        # Calculate dLdW_i and dLdb_i
        delta_i = self.activation.backward(dLda)
        dLdW = delta_i.T @ self.x
        dLdb = np.sum(delta_i, axis=0, keepdims=True)
        dLdx = delta_i @ self.W

        return dLdx, dLdW, dLdb