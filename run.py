"""
run.py

Main script for HW2 of ME 469 at Northwestern University.

Author: Jared Berry
Date: 11/05/2025
""" 
import numpy as np

from train import train

# np.random.seed(42)

def main():
    print("*** STARTING ***\n")

    train(100)
    
    print("\n*** DONE ***")
    return

if __name__ == "__main__":
    main()