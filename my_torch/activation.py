"""
Neural network architecture for Deep RL

https://medium.com/data-science/lets-code-a-neural-network-in-plain-numpy-ae7e74410795

"""
import numpy as np

class Activation():
    """
    A base vectorized activation function
    """
    def __init__(self):
        pass

    def forward(self, Z):
        pass

    def backward(self, dLdA, Z):
        pass

    def __call__(self, Z):
        return self.forward(Z)

class Sigmoid(Activation):
    """
    Sigmoid activation function

    # TODO: Derive by hand
    """
    def __init__(self):
        super.__init__()
        pass

    def forward(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def backward(self, dLdA, Z):
        sig = self.forward(Z)
        return dLdA*sig*(1 - sig)
    
class ReLU(Activation):
    """
    ReLU activation function

    # TODO: Derive by hand
    """
    def __init__(self):
        super.__init__()
        pass

    def forward(self, Z):
        return np.max(0.0, Z)
    
    def backward(self, dLdA, Z):
        relu = self.forward()