"""
Activation functions for neural network.

Author: Jared Berry

"""
import numpy as np

from my_torch.module import Module

class Dummy(Module):
    """
    Dummy activation function for when an activation isn't wanted.
    """
    def forward(self, z):
        return z

    def backward(self, dLda):
        return dLda

class Sigmoid(Module):
    """
    Sigmoid activation function
    """
    def __init__(self):
        super().__init__()
        pass

    def forward(self, z):
        # Given z_i, calculate a_i
        self.a = 1 / (1 + np.exp(-z))
        return self.a
    
    def backward(self, dLda):
        # Calculate dL/dz_i
        dadz = (self.a*(1 - self.a))
        return dLda*dadz
    
class ReLU(Module):
    """
    ReLU activation function
    """
    def __init__(self):
        super().__init__()
        pass

    def forward(self, z):
        # Given z_i, calculate a_i
        self.a = np.maximum(0.0, z)
        return self.a
    
    def backward(self, dLda):
        # Calculate dL/dz_i
        dadz = np.ones_like(self.a)
        dadz[self.a <= 0] = 0.0
        return dLda*dadz
    
class Softmax(Module):
    """
    Softmax activation function

    https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    https://mmuratarat.github.io/2019-01-27/derivation-of-softmax-function
    """
    def __init__(self):
        super().__init__()
        pass

    def forward(self, z):
        # Given z_i, calculate a_i
        expo = np.exp(z - np.max(z, axis=1, keepdims=True))    # Better numerical stability
        self.a = expo / np.sum(expo, axis=1, keepdims=True)
        return self.a
    
    def backward(self, dLda):
        # Calculate dL/dz_i
        # Formula:
        #   da_i/dz_j = a(del - Sj)
        #   Where del = 1 if i==j | del = 0 otherwise
        # TODO: Vectorize
        dLdz = np.zeros_like(self.a)
        for i in range(self.a.shape[0]):
            # Compute softmax derivative for every item in batch
            ai = self.a[i]
            Ji = np.diag(ai) - np.outer(ai, ai)
            dLdz[i] = Ji @ dLda[i]  # Matrix mult instead of element-wise because softmax enforces outputs to sum to 1
        return dLdz