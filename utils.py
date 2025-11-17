"""
Hold common utility functions.
"""
import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(SCRIPT_DIR, "metrics")
METRICS_PATH = os.path.normpath(METRICS_PATH)
DATA_PATH = os.path.join(SCRIPT_DIR, "data")
DATA_PATH = os.path.normpath(DATA_PATH)
PLOT_PATH = os.path.join(SCRIPT_DIR, "figures")
PLOT_PATH = os.path.normpath(PLOT_PATH)

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
    return (error**2)

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
    # Floor maps world coords to grid indices
    arr = np.array(pos)
    return tuple(np.floor(arr / res).astype(int))

def grid_to_pos(grid, res):
    """
    Convert from integer rep back to orig units
    """
    return np.array(grid, dtype=float) * res

def round_to_grid(n, res):
    """
    Given a number or np.ndarray of numbers, round to a given grid resolution.
    """
    # Return integer grid indices
    if isinstance(n, tuple) or isinstance(n, list): n_arr = np.array(n)
    else: n_arr = np.array(n)
    return np.floor(n_arr / res).astype(int)

def round_to_res(n, res):
    """
    Given a number or np.ndarray of numbers, round to a given resolution.
    """
    # Convert numeric input to the nearest grid-aligned world coordinate
    if isinstance(n, tuple) or isinstance(n, list): n_arr = np.array(n)
    else: n_arr = np.array(n)
    return np.round(n_arr / res) * res

def inflate_obstacles(bounds, res, obstacles, inflate):
    """
    Inflate a given set of obstacles by a specified amount
    """
    # Get borders in grid rep coords
    grid_min = np.array(
            [int(np.floor(bounds[0][0]/res)), 
             int(np.floor(bounds[1][0]/res))], dtype=int)
    grid_max = np.array([
        int(np.floor(bounds[0][1]/res)), 
        int(np.floor(bounds[1][1]/res))], dtype=int)
    obstacles_rounded = set()  # Set of obstacles
    
    # Inflate obstacle by inflate number of cells
    for l in obstacles:
        # Cover full square of size (inflate)
        for dx in range(-inflate, inflate + 1):
            for dy in range(-inflate, inflate + 1):
                x, y = (l[0] + dx), (l[1] + dy)

                l_inf = np.array([x,y])

                # Check bounds
                if grid_min[0] <= l_inf[0] < grid_max[0] and grid_min[1] <= l_inf[1] < grid_max[1]:
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
     
    # Convert landmarks to integer grid tuples for internal representation
    landmarks_grid = [round_to_grid(l, res) for l in landmarks]
    obstacles = inflate_obstacles(bounds, res, landmarks_grid, inflate=inflate)

    return obstacles

#
# --- Plotting ---
#
def plot_model_path(model, title, filename):
    env = model.env
    state, _ = env.reset()
    done = False
    step_count = 0
    episode_reward = 0.0
    success = False
    
    path = [state]
    while not done and step_count < 500:
        # Take greedy action
        action = model.predict(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Record progress
        episode_reward += reward
        
        state = next_state
        path.append(grid_to_pos(state, env.res))
        step_count += 1

    plot_search(grid_to_pos(env.start, env.res), grid_to_pos(env.goal, env.res), path, env.bounds, env.res, env.obstacles, title, filename)
        
    return


def plot_search(start, goal, path, bounds, res, obstacles, title, filename, traj=None, display_robot=True):
    fig, ax = plot_grid(bounds, res, obstacles, title)

    # Plot path
    for cell in path:
        rect = patches.Rectangle(
            (cell[0], cell[1]), res, res,
            facecolor='red', edgecolor='black'
        )
        ax.add_patch(rect)

    # Plot start and goal
    start_rect = patches.Rectangle(
        (start[0], start[1]), res, res,
        facecolor='blue', edgecolor='black'
    )
    ax.add_patch(start_rect)
    goal_rect = patches.Rectangle(
        (goal[0], goal[1]), res, res,
        facecolor='green', edgecolor='black'
    )
    ax.add_patch(goal_rect)

    fig_path = os.path.join(PLOT_PATH, filename)
    plt.savefig(fig_path)
    plt.close()

def plot_grid(bounds, res, obstacles, title=None):
    # Create figure
    length = int(abs(bounds[0][1] - bounds[0][0]))
    height = int(abs(bounds[1][1] - bounds[1][0]))
    fig, ax = plt.subplots(figsize=(length, height))

    # Set up grid
    x_range = np.arange(bounds[0][0], bounds[0][1] + 1e-9, step=res)    # Add small
    y_range = np.arange(bounds[1][0], bounds[1][1] + 1e-9, step=res)

    # Plot landmarks
    if obstacles is not None:
        for o in obstacles:
            world_o = grid_to_pos(o, res)  # Convert back to float
            # Plot obstacles as rectanagles
            rect = patches.Rectangle(
                (world_o[0], world_o[1]), res, res,
                facecolor='gray', edgecolor='black'
            )
            ax.add_patch(rect)

    # Set up grid
    ax.set_xticks(x_range)
    ax.set_yticks(y_range)
    ax.grid(color='black', linewidth=0.4)

    # Set up axis labels
    if res >= 0.5:
        ax.set_xticklabels(x_range, fontsize=14)
        ax.set_yticklabels(y_range, fontsize=14)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.set_xlabel('X Coordinate', fontsize=16)
    ax.set_ylabel('Y Coordinate', fontsize=16)
    ax.set_aspect('equal')
    
    # Set axis limits
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])

    if title is None:
        ax.set_title('A* Gridworld', fontsize=16)
        fig_path = os.path.join(PLOT_PATH, 'q1.png')
        plt.savefig(fig_path)
    else:
        ax.set_title(title, fontsize=18)

    return fig, ax