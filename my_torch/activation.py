"""
Activation functions for neural network.

Author: Jared Berry

"""
import numpy as np

from my_torch.module import Module

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
    
    def backward(self):
        # Calculate da_i
        da = (self.a*(1 - self.a))
        return da
    
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
    
    def backward(self):
        # Calculate da_i
        da = np.ones(self.a.shape)
        da[self.a <= 0] = 0.0
        return da