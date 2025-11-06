"""
Hold training functionality for Deep RL

Author: Jared Berry
"""
import numpy as np

from gym.env import GridEnv
from utils import get_obstacles

def train(num_timesteps):
    # Define world bounds and grid resolution
    bounds = np.array([
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ])
    res = 1.0
    obstacles = get_obstacles(bounds, res, inflate=0)

    # Create and initialize environment
    env = GridEnv(bounds, res, obstacles, render_mode='no_vis')
    obs, info = env.reset()
    
    # Training loop
    for t in range(num_timesteps):
        action = env.sample_action()
        obs, reward, terminated, truncated, info = env.step(action)
        print(obs)

        env.render()

        if terminated or truncated:
            print('Reached goal')
            obs, info = env.reset()

    return
        