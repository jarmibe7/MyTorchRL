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
# --- Evaluation ---
#
def t_match(traj, num_samples):
    """
    Resample a trajectory to have a certain number of samples
    """
    old_path_idx = np.linspace(0, 1, traj.shape[0])
    new_path_idx = np.linspace(0, 1, num_samples)

    traj_resamp = np.column_stack([
        np.interp(new_path_idx, old_path_idx, traj[:, i]) for i in range(traj.shape[1])
    ])

    return traj_resamp

def accuracy_score(predicted, actual):
    """
    Given two 1D numpy arrays containing two possible classes, compute the accuray
    """
    return len(actual[predicted == actual]) / len(actual)

def mse(predicted, actual, angle=False):
    """
    Given two 1D numpy arrays of the same length, compute Mean Squared Error
    between them.
    """
    if angle: error = np.unwrap(actual - predicted)
    else: error = actual - predicted
    return np.mean(error**2)

def rmse(predicted, actual, angle=False):
    """
    Given two 1D numpy arrays of the same length, compute Root Mean Squared Error
    between them.
    """
    if angle: error = np.unwrap(actual - predicted)
    else: error = np.linalg.norm(actual - predicted)
    return np.sqrt(error)

def compute_traj_statistics(predicted, actual):
    """
    Given a trajectory, compute various statistics about it from a ground truth.
    """
    stats = {}
    stats['rmse_x'] = rmse(predicted[:, 0], actual[:, 0])
    stats['rmse_y'] = rmse(predicted[:, 1], actual[:, 1])
    stats['rmse_theta'] = rmse(predicted[:, 2], actual[:, 2])
    stats['corr_x'] = np.corrcoef(predicted[:, 0], actual[:, 0])[0, 1]
    stats['corr_y'] = np.corrcoef(predicted[:, 1], actual[:, 1])[0, 1]
    stats['corr_theta'] = np.corrcoef(predicted[:, 2], actual[:, 2])[0, 1]

    return stats

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