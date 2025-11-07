"""
Activation functions for neural network.

Author: Jared Berry

"""
import numpy as np

from module import Module

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
    
    def backward(self, dz):
        # Given W_i+1.T @ delta_i+1, calculate delta_i
        dA = (self.a*(1 - self.a))
        return dz*dA
    
class ReLU(Module):
    """
    ReLU activation function
    """
    def __init__(self):
        super().__init__()
        pass

    def forward(self, z):
        # Given z_i, calculate a_i
        self.a = np.max(0.0, z)
        return self.a
    
    def backward(self, dz):
        # Given W_i+1.T @ delta_i+1, calculate delta_i
        delta = dz.copy()
        delta[self.a <= 0] = 0.0
        return delta