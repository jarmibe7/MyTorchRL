"""
run.py

Main script for HW2 of ME 469 at Northwestern University.

Author: Jared Berry
Date: 11/05/2025
""" 
import numpy as np

from gym.env import GridEnv
from utils import get_obstacles

from a2c import A2C

# np.random.seed(42)

def main():
    print("*** STARTING ***\n")
    # Define world bounds and grid resolution
    bounds = np.array([
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ])
    res = 1.0
    obstacles = get_obstacles(bounds, res, inflate=0)

    # Create and initialize environment
    env = GridEnv(bounds, res, obstacles, use_shaped=True, render_mode='human')
    obs, info = env.reset()

    # Initialize model
    critic_arch = [
        (env.state_dim, 64, 'relu'),
        (64, 1, 'dummy')
    ]
    actor_arch = [
        (env.state_dim, 64, 'relu'),
        (64, env.action_dim, 'softmax')
    ]
    alpha_actor = 1e-3
    alpha_critic = 1e-3
    gamma = 0.99
    episode_limit = 1000
    step_limit = 100
    conv_thresh = 1e-5
    model = A2C(env, critic_arch, actor_arch, alpha_actor, alpha_critic, gamma, episode_limit, step_limit, conv_thresh, save_model=False)
    
    # Training loop
    # for t in range(100):
    #     action = env.sample_action()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     print(obs)

    #     env.render()

    #     if terminated or truncated:
    #         print('Reached goal')
    #         obs, info = env.reset()
    model.train()
    print("\n*** DONE ***")
    return

if __name__ == "__main__":
    main()