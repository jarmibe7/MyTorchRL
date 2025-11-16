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

from gym.env import DeepRLGridEnv, QLGridEnv
from utils import get_obstacles

from a2c import A2C
from q_learning import VanillaQL

# np.random.seed(42)

def main():
    print("*** STARTING ***\n")
    # Define world bounds and grid resolution
    bounds = np.array([
        [-2, 5],    # x bounds
        [-6, 6]     # y bounds
    ])
    res = 0.1
    obstacles = get_obstacles(bounds, res, inflate=0)

    # Whether to use a vanilla or deep formulation for RL
    use_deep = False

    # Initialize model
    if use_deep:
        # Create deep RL A2C agent
        env = DeepRLGridEnv(bounds, res, obstacles=None, use_shaped=True, wrap_arena=False, render_mode='no_vis', randomize_start=False, randomize_goal=False)
        critic_arch = [
            (env.state_dim, 16, 'relu'),
            (16, 32, 'relu'),
            (32, 1, 'dummy'),
        ]
        actor_arch = [
            (env.state_dim, 16, 'relu'),
            (16, 16, 'relu'),
            (16, env.action_dim, 'softmax')
        ]
        model = A2C(
            env, 
            critic_arch, 
            actor_arch, 
            alpha_actor=1e-5, 
            alpha_critic=1e-5, 
            gamma=0.99,
            exp_prob=0.1,
            rollout_limit=25,
            episode_limit=50000, 
            step_limit=100, 
            conv_thresh=1e-5, 
        )
    else:
        # Create vanilla Q-learning agent
        env = QLGridEnv(bounds, res, obstacles=None, render_mode='no_vis', randomize_start=False, randomize_goal=False)
        model = VanillaQL(
            env,
            alpha=0.01,
            gamma=0.99,
            epsilon=0.0,
            episode_limit=30000,
            step_limit=500,
            conv_thresh=-1
        )
    
    model.learn()

    model.env.render_mode = 'no_vis'
    results = model.test(save=True)

    # Final summary
    print('\n----------------------------')
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"Success Rate: {results['success_rate']*100:.1f}%")
    print(f"Avg Reward: {results['avg_reward']:.3f}")
    print(f"Avg Episode Length: {results['avg_length']:.1f}")
    print('----------------------------\n')

    print("\n*** DONE ***")
    return

if __name__ == "__main__":
    main()