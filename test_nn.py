"""
Script for testing neural network
"""
import numpy as np
import os
import matplotlib.pyplot as plt

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

    # Plot dataset
    plt.figure(figsize=(6,6))
    plt.scatter(X_train[y_train[:,0]==0, 0], X_train[y_train[:,0]==0, 1], c='red', label='Class 0', alpha=0.6)
    plt.scatter(X_train[y_train[:,0]==1, 0], X_train[y_train[:,0]==1, 1], c='blue', label='Class 1', alpha=0.6)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title(filename)
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create model
    arch = [
        (X_train.shape[1], 64, 'relu'),
        (64, 32, 'relu'),
        (32, 8, 'relu'),
        (8, 1, 'sigmoid')
    ]
    alpha = 1e-3
    model = FeedForward(arch, 'bce', alpha, conv_thresh=1e-5, weight_init='basic')
    
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