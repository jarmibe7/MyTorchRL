"""
Vanilla tabular Q-Learning in a discretized gridworld.

Author: Jared Berry
"""
import numpy as np
import os
import json
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(SCRIPT_DIR, "metrics")
METRICS_PATH = os.path.normpath(METRICS_PATH)

class VanillaQL:
    """
    Tabular Q-Learning agent for discrete state-action spaces.
    
    Args:
        env: GridEnv environment
        learning_rate: Learning rate
        discount_factor: Gamma parameter for future discounting
        epsilon: Epsilon-greedy exploration rate
        state_bins: Number of bins per dimension for state discretization
        episode_limit: How many episodes to train for
        step_limit: How many transitions per episode
    """
    loaded = False
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, episode_limit=5000, step_limit=100, conv_thresh=1e-5):
        # Initialize params
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode_limit = episode_limit
        self.step_limit = step_limit
        self.conv_thresh = conv_thresh
        self.converged = False
        self.total_timesteps = 0
        
        # Discretize state space based on resolution
        self.x_range = int(env.bounds[0][1] - env.bounds[0][0])
        self.y_range = int(env.bounds[1][1] - env.bounds[1][0])
        
        # Number of bins per dimension
        self.num_pos_x_states = int(np.ceil(self.x_range / env.res))
        self.num_pos_y_states = int(np.ceil(self.y_range / env.res))
        
        # Relative position can range from -bounds to +bounds
        self.num_rel_x_states = int(np.ceil((self.x_range * 2) / env.res))
        self.num_rel_y_states = int(np.ceil((self.y_range * 2) / env.res))
        self.res = env.res
        self.bounds = env.bounds
        
        # Initialize Q-table: Q[pos_x_bin, pos_y_bin, rel_x_bin, rel_y_bin, action]
        self.Q = np.zeros((self.num_pos_x_states, self.num_pos_y_states, 
                           self.num_rel_x_states, self.num_rel_y_states, 
                           env.action_dim))
        
        # Statistics and convergence tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_max_q_deltas = []  # Track max Q-change per episode
        self.window_size = 100  # Episodes per evaluation window
        self.convergence_counter = 0
        self.convergence_threshold_windows = 3  # Converged if stable for 3 windows
    
    def discretize_state(self, state):
        """
        Convert continuous 4D state [pos_x, pos_y, rel_x, rel_y] to discrete grid indices.
        
        Args:
            state: 4D numpy array [pos_x, pos_y, rel_x, rel_y]
        """
        # Extract absolute and relative parts
        pos_x, pos_y = state[0], state[1]
        rel_x, rel_y = state[2], state[3]
        
        # Discretize absolute position [relative to bounds]
        pos_x_bin = int((pos_x - self.bounds[0][0]) / self.res)
        pos_y_bin = int((pos_y - self.bounds[1][0]) / self.res)
        
        # Discretize positionss
        rel_x_offset = self.x_range / 2.0
        rel_y_offset = self.y_range / 2.0
        rel_x_bin = int((rel_x + rel_x_offset) / self.res)
        rel_y_bin = int((rel_y + rel_y_offset) / self.res)
        
        # Clamp to valid range
        pos_x_bin = np.clip(pos_x_bin, 0, self.num_pos_x_states - 1)
        pos_y_bin = np.clip(pos_y_bin, 0, self.num_pos_y_states - 1)
        rel_x_bin = np.clip(rel_x_bin, 0, self.num_rel_x_states - 1)
        rel_y_bin = np.clip(rel_y_bin, 0, self.num_rel_y_states - 1)
        
        return (pos_x_bin, pos_y_bin, rel_x_bin, rel_y_bin)
    
    def get_action_epsilon_greedy(self, state_bin):
        """
        Select action using epsilon-greedy with self.epsilon change to explore
        """
        if np.random.rand() < self.epsilon:
            # Random action
            return self.env.sample_action()
        else:
            # Greedy action
            q_vals = self.Q[state_bin]
            return np.argmax(q_vals)
    
    def update_q_table(self, state_bin, action, reward, next_state_bin, done):
        """
        Perform standard Q-Learning update
        """
        # Get current q value
        current_q = self.Q[state_bin][action]
        
        if done:
            # If done, no future rewards
            max_next_q = 0.0
        else:
            # If not done use max Q-value of next state for greedy policy update
            max_next_q = np.max(self.Q[next_state_bin])
        
        # Compute target and update
        target = reward + self.gamma * max_next_q
        error = target - current_q
        self.Q[state_bin][action] += self.alpha * error
        if np.abs(self.alpha*error) <= self.conv_thresh: 
            self.converged = True
    
    def train(self):
        """
        Train vanilla Q-learning agent
        """
        print(f"Starting vanilla Q-Learning training for {self.episode_limit} episodes...")
        # Iterate over episodes
        for episode in range(self.episode_limit):
            if self.converged: break
            state, _ = self.env.reset()
            state_bin = self.discretize_state(state)
            
            done = False
            step_count = 0
            episode_reward = 0.0
            success = False
            
            # Iterate over transitions
            while not done and step_count < self.step_limit:
                # Select and take action
                action = self.get_action_epsilon_greedy(state_bin)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Discretize next state
                next_state_bin = self.discretize_state(next_state)
                
                # Update Q-table
                self.update_q_table(state_bin, action, reward, next_state_bin, done)
                
                # Track episode stats
                episode_reward += reward
                if terminated:
                    success = True
                
                # Move to next state
                if self.env.render_mode == 'human': self.env.render()
                state_bin = next_state_bin
                step_count += 1
                self.total_timesteps += 1
            
            # Log episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step_count)
            self.episode_successes.append(1 if success else 0)
            
            # Print progress of last 500 episodes
            if (episode) % 500 == 0:
                avg_reward = np.mean(self.episode_rewards[-500:])
                avg_length = np.mean(self.episode_lengths[-500:])
                success_rate = np.mean(self.episode_successes[-500:])
                print(f"Episode {episode+1}/{self.episode_limit} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Avg Length: {avg_length:.1f} | "
                      f"Success Rate: {success_rate*100:.1f}%")
                
        print("\nTraining complete!\n")
    
    def test(self, num_episodes=100, step_limit=100, save=True):
        """
        Test the learned policy with no exploration
        """
        print(f"Running eval on trained QL for {num_episodes} episodes...\n")
        try:
            # Iterate over episodes
            test_rewards = []
            test_lengths = []
            test_successes = []
            for episode in range(num_episodes):
                state, _ = self.env.reset()
                state_bin = self.discretize_state(state)
                done = False
                step_count = 0
                episode_reward = 0.0
                success = False
                
                while not done and step_count < step_limit:
                    # Take greedy action
                    q_vals = self.Q[state_bin]
                    action = np.argmax(q_vals)
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    next_state_bin = self.discretize_state(next_state)
                    
                    # Record progress
                    episode_reward += reward
                    if terminated:
                        success = True
                    
                    if self.env.render_mode == 'human': self.env.render()
                    state_bin = next_state_bin
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
            metrics_dict = {
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'avg_length': avg_length,
                'std_length': std_length,
                'total_timesteps': self.total_timesteps
            }
            if save:
                filepath = os.path.join(METRICS_PATH, f'ql_{time.time()}.json')
                with open(filepath, "w") as f:
                    json.dump(metrics_dict, f, indent=4)

            return metrics_dict
        except KeyboardInterrupt:
            self.env.close()
            print('\nManual interrupt, stopping training!')
    
    def learn(self):
        """
        Wrapper function to match deep RL structure
        """
        try:
            self.train()
        except KeyboardInterrupt:
            print('\nManual interrupt, stopping training!')

