"""
A feed forward neural network class

Author: Jared Berry
"""
import numpy as np

from my_torch.module import Module
from my_torch.linear import Linear
from my_torch.loss import MSELoss, BCELoss

class FeedForward(Module):
    """
    A feed forward neural network

    Args:
        arch: A list of triplets of length n, where each index i corresponds to a layer
              and arch[i] is a triplet with that layer's input size, output size, and activation function.
        alpha: Learning rate
    """
    def __init__(self, arch, loss_type, alpha, conv_thresh):
        super().__init__()
        # Get first input size and final output size
        self.input_size = arch[0][0]
        self.output_size = arch[-1][1]

        # Learning rate and convergence threshold
        self.alpha = alpha
        self.conv_thresh = conv_thresh

        # Create layers
        self.layers = []
        for input_size, output_size, activation in arch:
            self.layers.append(Linear(input_size, output_size, activation))
        
        # Loss function
        if loss_type == 'mse': self.criterion = MSELoss()
        elif loss_type == 'bce': self.criterion = BCELoss()

    def forward(self, x):
        # Iterate through all layers
        xi = x.copy()
        for layer in self.layers:
            xi = layer(xi)

        return xi
    
    def backward(self):
        # Zero gradients, iterate over layers, and collect gradients
        dLda = self.criterion.backward()
        self.gradients = [] # List of (dW, db) tuple
        for layer in reversed(self.layers):
            dLda, dW, db = layer.backward(dLda)
            self.gradients.append((dW, db))

        return self.gradients   # Return if using external optimizer
    
    def optimize(self):
        # Optimize weights with simple gradient descent
        max_grad = 0.0
        for layer, grads in zip(reversed(self.layers), self.gradients):
            dW, db = grads[0], grads[1]
            layer.W -= self.alpha*dW
            layer.b -= self.alpha*db

            max_grad = max(max_grad, np.max(np.abs(dW)), np.max(np.abs(db)))

        return max_grad < self.conv_thresh
    
    def predict(self, x, classify=False):
        pred = self.forward(x)
        if classify: pred = np.round(pred)
        return pred