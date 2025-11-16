"""
test_gym.py

Script for testing Gymnasium functionality.

Author: Jared Berry
Date: 11/05/2025
""" 
import numpy as np
import os
import json

from gym.env import DeepRLGridEnv, QLGridEnv
from utils import get_obstacles

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(SCRIPT_DIR, "metrics")
METRICS_PATH = os.path.normpath(METRICS_PATH)

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

    # Whether to use a vanilla or deep formulation for RL
    use_deep = False

    # Initialize env
    env = QLGridEnv(bounds, res, obstacles=obstacles, render_mode='no_vis', randomize_start=True, randomize_goal=False)
    test_rewards = []
    test_lengths = []
    test_successes = []
    num_episodes = 100
    step_limit = 100
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        step_count = 0
        episode_reward = 0.0
        success = False
        
        while not done and step_count < step_limit:
            # Take greedy action
            action = env.sample_action()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Record progress
            episode_reward += reward
            if terminated:
                success = True
            
            if env.render_mode == 'human': env.render()
            step_count += 1
            
        test_rewards.append(episode_reward)
        test_lengths.append(step_count)
        test_successes.append(1 if success else 0)
        
    # Calculate metrics
    avg_reward = np.mean(test_rewards)
    avg_length = np.mean(test_lengths)
    std_length = np.std(test_lengths)
    success_rate = np.mean(test_successes)

    # Save metrics
    results = {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'std_length': std_length,
    }
    filepath = os.path.join(METRICS_PATH, f'baseline_metrics.json')
    with open(filepath, "w") as f:
        json.dump(results, f, indent=4)

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