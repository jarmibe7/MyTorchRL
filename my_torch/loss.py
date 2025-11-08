"""
Loss functions for training a neural network.

Author: Jared Berry
"""
import numpy as np
from my_torch.module import Module

class BCELoss(Module):
    """
    Binary Cross-Entropy Loss
    """
    def __init__(self):
        super().__init__()
        self.eps = 1e-7

    def forward(self, predicted, actual):
        # Save predicted and actual for backwards pass
        self.predicted = predicted
        self.actual = actual
        pred = np.clip(predicted, self.eps, 1 - self.eps)   # Avoid infinity in log calc
        bce_vec = actual*np.log(pred) + (1 - actual)*np.log(1 - pred)
        return -np.sum(bce_vec, axis=0) / len(bce_vec)
    
    def backward(self):
        # return (self.predicted - self.actual) / (self.predicted*(1 - self.predicted))
        return self.predicted - self.actual # If final layer is sigmoid derivative is simplified

class MSELoss(Module):
    """
    Mean Squared Error Loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, predicted, actual):
        self.error = predicted - actual
        return 0.5*np.mean(self.error**2)
    
    def backward(self):
        return self.error / self.error.shape[0] # Normalize by batch dimension