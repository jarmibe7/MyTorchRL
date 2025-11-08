"""
Script for testing neural network
"""
import numpy as np
import os

from my_torch.ff import FeedForward
from utils import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(SCRIPT_DIR, "test_data")
TEST_DATA_PATH = os.path.normpath(TEST_DATA_PATH)

def main():
    print("*** STARTING ***\n")

    # Load data
    filename = 'circles.csv'
    filepath = os.path.join(TEST_DATA_PATH, filename)
    dataset = np.loadtxt(filepath, delimiter=',')
    X_train = dataset[:, :-1]
    y_train = dataset[:, -1].reshape(-1, 1)

    # Create model
    arch = [
        (X_train.shape[1], 64, 'relu'),
        (64, 32, 'relu'),
        (32, 8, 'relu'),
        (8, 1, 'sigmoid')
    ]
    alpha = 1e-3
    model = FeedForward(arch, 'bce', alpha, conv_thresh=1e-5)
    
    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward pass
        y_pred = model.forward(X_train)

        # Compute loss
        current_mse = model.criterion(y_pred, y_train)

        # Backward pass and optimize
        gradients = model.backward()
        converged = model.optimize()
        if converged and epoch > 10: break

    # Determine overall accuracy
    final_pred = model.predict(X_train, classify=True)
    accuracy = accuracy_score(final_pred, y_train)
    print(f'Final accuracy: {accuracy*100}%')
    print("\n*** DONE ***")

if __name__ == '__main__':
    main()