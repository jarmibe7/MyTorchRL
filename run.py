"""
run.py

Main script for HW2 of ME 469 at Northwestern University.

Author: Jared Berry
Date: 11/05/2025

Ideas:
    - Imitation learning with A* trajectories
    - Remove goal position and see if it can be more efficient than random, learning search rather
      than goal position.
    - Use multiple actors, do some swarm stuff
    - Use double critic network
    - Try non-deep version
    - n-step returns in A2C
    - Don't subtract mean when normalizing advantage
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
    env = GridEnv(bounds, res, obstacles=None, use_shaped=True, wrap_arena=True, render_mode='no_vis')
    obs, info = env.reset()

    # Initialize model
    critic_arch = [
        (env.state_dim, 32, 'relu'),
        (32, 64, 'relu'),
        (64, 1, 'dummy')
    ]
    actor_arch = [
        (env.state_dim, 32, 'relu'),
        (32, 64, 'relu'),
        (64, env.action_dim, 'softmax')
    ]
    alpha_actor = 1e-4
    alpha_critic = 1e-4
    gamma = 0.95
    exp_prob = 0.0
    rollout_limit = 10
    episode_limit = 10000
    step_limit = 100
    conv_thresh = 1e-5
    model = A2C(
        env, 
        critic_arch, 
        actor_arch, 
        alpha_actor, 
        alpha_critic, 
        gamma,
        exp_prob,
        rollout_limit,
        episode_limit, 
        step_limit, 
        conv_thresh, 
        save_model=False
    )
    
    # Training loop
    # for t in range(100):
    #     action = env.sample_action()
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     print(obs)

    #     env.render()

    #     if terminated or truncated:
    #         print('Reached goal')
    #         obs, info = env.reset()
    model.learn()
    print("\n*** DONE ***")
    return

if __name__ == "__main__":
    main()