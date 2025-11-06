"""
Hold common utility functions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpl_path
import os
import json
import time

PLOT_PATH = os.path.join(__file__, "..\\figures")
DATA_PATH = os.path.join(__file__, "..\\data")
METRICS_PATH = os.path.join(__file__, "..\\metrics")

#
# --- Grid Representation ---
#
def pos_to_grid(pos, res):
    """
    Convert from orig units to internal integer representation
    """
    return tuple(np.floor(np.array(pos) / res).astype(int))

def grid_to_pos(grid, res):
    """
    Convert from integer rep back to orig units
    """
    return np.round(np.array(grid)*res, 1)

def round_to_res(n, res):
    """
    Given a number or np.ndarray of numbers, round to a given resolution.
    """
    if isinstance(n, tuple): n_arr = np.array(n)
    else: n_arr = n
    return np.round(np.floor(n_arr / res)*res, 1)   # TODO: Better way of eliminating floating point

def inflate_obstacles(bounds, res, obstacles, inflate):
    """
    Inflate a given set of obstacles by a specified amount
    """
    # Plot obstacles
    obstacles_rounded = set()  # Set of obstacles
    # Inflate obstacle by inflate number of cells
    for l in obstacles:
        # Cover full square of size (inflate)
        for dx in range(-inflate, inflate + 1):
            for dy in range(-inflate, inflate + 1):
                x, y = (l[0] + dx * res), (l[1] + dy * res)

                l_inf = round_to_res(np.array([x,y]), res)

                # Check bounds
                if bounds[0][0] <= l_inf[0] < bounds[0][1] and bounds[1][0] <= l_inf[1] < bounds[1][1]:
                    obstacles_rounded.add(tuple(l_inf))

    return obstacles_rounded

def get_obstacles(bounds, res, inflate=0):
    """
    Inflate is the number of cells to inflate in each direction
    """
    # Read ground truth obstacle data
    landmarks_truth_data_path = os.path.join(DATA_PATH, 'ds1_Landmark_Groundtruth.dat')
    landmarks_truth = pd.read_csv(landmarks_truth_data_path, sep=r"\s+", comment="#", header=None, names=["subject", "x", "y", "x_sig", "y_sig"])
    landmarks = landmarks_truth.to_numpy()[:, 1:3]
     
    obstacles = inflate_obstacles(bounds, res, landmarks, inflate=inflate)

    return obstacles