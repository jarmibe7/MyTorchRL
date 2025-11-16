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
import os
import json
from itertools import product

from gym.env import DeepRLGridEnv, QLGridEnv
from utils import get_obstacles

from a2c import A2C
from q_learning import VanillaQL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(SCRIPT_DIR, "metrics")
METRICS_PATH = os.path.normpath(METRICS_PATH)
# np.random.seed(42)

def main():
    print("*** STARTING ***\n")
    alg = 'ql'
    res_types = ['coarse', 'fine']
    obs_types = ['open', 'obs']
    rand_types = ['det', 'start']
    num_trials = 10
    for res_type, obs, rand in product(res_types, obs_types, rand_types):
        print(f'\n TESTING COMBO: {res_type}   {obs}   {rand}')
        total_timesteps = []
        success_rates = []
        avg_episode_lengths = []
        std_episode_lengths = []
        for trial in range(num_trials):
            print(f'\n TRIAL {trial + 1}')
            # Define world bounds and grid resolution
            bounds = np.array([
                [-2, 5],    # x bounds
                [-6, 6]     # y bounds
            ])
            if res_type == 'coarse': 
                res = 1.0
                step_lim = 100
                inf = 0
            else: 
                res = 0.1
                step_lim = 500
                inf = 3

            if obs == 'obs': obstacles = get_obstacles(bounds, res, inflate=inf)
            else: obstacles = None
            
            if rand == 'start': rand_start = True
            else: rand_start = False

            if obs == 'obs' or rand_start:
                eps = 0.25
            else:
                eps = 0.1

            # Initialize model
            if alg == 'a2c':
                # Create deep RL A2C agent
                env = DeepRLGridEnv(bounds, res, obstacles=obstacles, use_shaped=True, wrap_arena=False, render_mode='no_vis', randomize_start=rand_start, randomize_goal=False)
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
                    alpha_actor=1e-4, 
                    alpha_critic=1e-4, 
                    gamma=0.99,
                    exp_prob=0.025,
                    rollout_limit=25,
                    episode_limit=50000, 
                    step_limit=step_lim, 
                    conv_thresh=1e-5, 
                )
            else:
                # Create vanilla Q-learning agent
                env = QLGridEnv(bounds, res, obstacles=obstacles, render_mode='no_vis', randomize_start=rand_start, randomize_goal=False)
                model = VanillaQL(
                    env,
                    alpha=0.1,
                    gamma=0.99,
                    epsilon=eps,
                    episode_limit=25000,
                    step_limit=step_lim
                )
            
            model.learn()

            model.env.render_mode = 'no_vis'
            results = model.test(save=False, step_limit=500)

            total_timesteps.append(results['total_timesteps'])
            success_rates.append(results['success_rate'])
            avg_episode_lengths.append(results['avg_length'])
            std_episode_lengths.append(results['std_length'])

            # Final summary
            print('\n----------------------------')
            print(f"TRIAL {trial + 1} SUMMARY")
            print(f"Success Rate: {results['success_rate']*100:.1f}%")
            print(f"Avg Reward: {results['avg_reward']:.3f}")
            print(f"Avg Episode Length: {results['avg_length']:.1f}")
            print('----------------------------\n')

        metrics_dict = {
            'avg_total_timesteps': np.mean(total_timesteps),
            'avg_success_rate': np.mean(success_rates),
            'avg_episode_length': np.mean(avg_episode_lengths),
            'std_episode_length': np.std(std_episode_lengths)
        }
        filepath = os.path.join(METRICS_PATH, f'{alg}_{res_type}_{obs}_{rand}.json')
        with open(filepath, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        print(f'\nOutput to {filepath}')

    print("\n*** DONE ***")
    return

if __name__ == "__main__":
    main()