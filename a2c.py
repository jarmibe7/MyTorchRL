"""
Deep Advantage Actor-Critic (A2C) algorithm.

A2C Paper: https://arxiv.org/abs/1602.01783
Advantage Normalization: https://github.com/openai/baselines/issues/362

Author: Jared Berry
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from my_torch.module import Module
from my_torch.ff import FeedForward
from my_torch.loss import Loss, MSELoss

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_PATH = os.path.join(SCRIPT_DIR, "figures")
PLOT_PATH = os.path.normpath(PLOT_PATH)

class A2CActorLoss(Loss):
    """
    Actor loss for the A2C algorithm
    """
    def __init__(self):
        super().__init__()

    def forward(self, actions, action_probs, advantages):
        # Negative log action probability of selected action * advantage
        self.actions = actions.flatten()
        self.action_probs = action_probs
        self.advantages = advantages.flatten()
        self.batch_indices = np.arange(action_probs.shape[0])
        self.selected_probs = self.action_probs[self.batch_indices, self.actions]
        return -np.log(self.selected_probs + 1e-8)*self.advantages
    
    def backward(self):
        # Only calculate gradient for selected action
        dLda = np.zeros_like(self.action_probs)
        # dL/dp_selected = - advantage / p_selected  (from d(-log p * A)/dp)
        dLda[self.batch_indices, self.actions] = -self.advantages / (self.selected_probs + 1e-8)
        return dLda

class Actor(Module):
    """
    A simple actor for A2C.
    
    Args:
        arch: A list of triplets of length n, where each index i corresponds to a layer
              and arch[i] is a triplet with that layer's input size, output size, and activation function.
        alpha: Learning rate
        conv_thresh: If the maximum gradient magnitude is less than this threshold, optimization should end.
    """
    def __init__(self, arch, alpha, conv_thresh):
        super().__init__()
        self.input_size = arch[0][0]
        self.num_actions = arch[-1][1]
        self.ff = FeedForward(arch, loss_type=None, alpha=alpha, conv_thresh=conv_thresh)
        self.criterion = A2CActorLoss()

    def forward(self, x):
        out = self.ff(x)
        if np.isnan(out).any():
            print('\nNaN detected in actor output!\n')
            out = 0.25*np.ones_like(out)
        return out
    
    def backward(self):
        dLdy = self.criterion.backward()
        return self.ff.backward(dLdy)
    
    def optimize(self):
        return self.ff.optimize()
    
class Critic(Module):
    """
    A simple critic for A2C.
    
    Args:
        arch: A list of triplets of length n, where each index i corresponds to a layer
              and arch[i] is a triplet with that layer's input size, output size, and activation function.
        alpha: Learning rate
        conv_thresh: If the maximum gradient magnitude is less than this threshold, optimization should end.
    """
    def __init__(self, arch, alpha, conv_thresh):
        super().__init__()
        self.input_size = arch[0][0]
        assert arch[-1][1] == 1, f'Critic output must be scalar, received {arch[-1][1]}!'
        self.ff = FeedForward(arch, loss_type=None, alpha=alpha, conv_thresh=conv_thresh)
        self.criterion = MSELoss()

    def forward(self, x):
        return self.ff(x)
    
    def backward(self):
        dLdy = self.criterion.backward()
        return self.ff.backward(dLdy)
    
    def optimize(self):
        return self.ff.optimize()
    
class A2C():
    """
    An implementation of the Advantage Actor-Critic (A2C) algorithm.

    Args:
        env: Gymnasium environment
        critic_arch: Critic network architecture
        actor_arch: Actor network architecture
        alpha_actor: Actor learning rate
        alpha_critic: Critic learning rate
        gamma: Discount factor
        exp_prob: Probability of random exploration
        rollout_limit: Number of episodes to collect in each rollout (batch_size = rollout_limit*step_limit)
        episode_limit: Limit number of episodes for training
        step_limit: Limit number of timesteps per episode
        conv_thresh: Gradient convergence threshold
        save_model: Whether to save model weights
    """
    def __init__(self, 
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
                 save_model):
        # Assign environment
        self.env = env

        # Create actor and critic
        self.critic = Critic(critic_arch, alpha=alpha_critic, conv_thresh=conv_thresh)
        self.actor = Actor(actor_arch, alpha=alpha_actor, conv_thresh=conv_thresh)

        # Algorithm params
        self.gamma = gamma
        self.exp_prob = exp_prob
        self.rollout_limit = rollout_limit
        self.num_rollouts = 0
        self.episode_limit = episode_limit
        self.step_limit = step_limit
        self.save_model = save_model

        # Logging and visualization
        self.render = self.env.render_mode == 'human'
        self.plot_freq = 1000
        self.__init_logging__()
        self.__init_plot__()

    def __init_plot__(self):
        # Initalize figure for plotting
        self.plot_history = {"reward": [], "actor_loss": [], "critic_loss": [], "advantage": []}
        self.fig, self.axs = plt.subplots(4, 1, figsize=(8, 10))
        titles = [
            "Average Episode Reward",
            "Actor Loss",
            "Critic Loss",
            "Advantage"
        ]
        for ax, title in zip(self.axs, titles):
            ax.set_title(title)
            ax.set_xlabel("Episode")
            ax.grid(True)
        plt.tight_layout()
        if self.render: 
            plt.ion()
            plt.show()
        else:
            plt.ioff()

    def __init_logging__(self):
        # Reset episode statistics logger
        self.logger = {
            'episode': 0,
            'critic_loss': [],
            'critic_max_grad': [],
            'reward': [],
            'actor_loss': [],
            'actor_max_grad': [],
            'advantage': [],
        }

    def __init_batch__(self):
        # Reset batch data structure
        self.batch = {
            'states': [],
            'actions': [],
            'action_probs': [],
            'rewards': [],
            'next_states': [],
            'dones': [],
        }

    def update_plot(self, avg_critic_loss, avg_reward, avg_actor_loss, avg_advantage):
        # Update plot history arrays
        self.plot_history["reward"].append(avg_reward)
        self.plot_history["actor_loss"].append(avg_actor_loss)
        self.plot_history["critic_loss"].append(avg_critic_loss)
        self.plot_history["advantage"].append(avg_advantage)

        # Clear and replot
        if self.num_rollouts % self.plot_freq == 0:
            self.axs[0].cla(); self.axs[0].plot(self.plot_history["reward"], label="Reward", color='blue'); self.axs[0].legend(); self.axs[0].grid(True)
            self.axs[1].cla(); self.axs[1].plot(self.plot_history["actor_loss"], label="Actor Loss", color='orange'); self.axs[1].legend(); self.axs[1].grid(True)
            self.axs[2].cla(); self.axs[2].plot(self.plot_history["critic_loss"], label="Critic Loss", color='green'); self.axs[2].legend(); self.axs[2].grid(True)
            self.axs[3].cla(); self.axs[3].plot(self.plot_history["advantage"], label="Advantage", color='red'); self.axs[3].legend(); self.axs[3].grid(True)

            plt.tight_layout()
            if self.render: plt.pause(0.001)

    def save_plot(self):
        # Save training plot
        filepath = os.path.join(PLOT_PATH, 'a2c_0.png')
        self.fig.savefig(filepath)

    def display_episode(self):
        # Store running means
        avg_critic_loss = np.mean(np.array(self.logger["critic_loss"])) if len(self.logger["critic_loss"]) > 0 else 0
        avg_reward = np.mean(np.array(self.logger["reward"])) if len(self.logger["reward"]) > 0 else 0
        avg_actor_loss = np.mean(np.array(self.logger["actor_loss"])) if len(self.logger["actor_loss"]) > 0 else 0
        avg_advantage = np.mean(np.array(self.logger["advantage"])) if len(self.logger["advantage"]) > 0 else 0

        # Display episode statistics
        print('\n----------------------------')
        print(f'Episode {self.logger['episode']}:')
        print(f'Avg. Critic Loss: {avg_critic_loss}')
        print(f'Avg. Max Critic Gradient: {np.mean(np.array(self.logger['critic_max_grad']))}')
        print(f'Avg. Reward: {avg_reward}')
        print(f'Avg. Actor Loss: {avg_actor_loss}')
        print(f'Avg. Max Actor Gradient: {np.mean(np.array(self.logger['actor_max_grad']))}')
        print(f'Avg. Advantage: {avg_advantage}')
        print('----------------------------\n')
        self.update_plot(avg_critic_loss, avg_reward, avg_actor_loss, avg_advantage)
        self.__init_logging__()
        return
    
    def collect_rollouts(self):
        self.__init_batch__()
        # Collect batch of rollouts
        for rollout in range(self.rollout_limit):
            state, _ = self.env.reset()
            done = False
            step_count = 0
            # self.logger['episode'] = episode

            while not done and step_count < self.step_limit:
                # Take action
                action_probs = self.actor(state)
                action = self.env.sample_action(probs=action_probs[0])

                # To explore, take an unlikely action
                exp_chance = np.random.uniform(0.0, 1.0)
                if exp_chance < self.exp_prob:
                    inverse_probs = (1.0 - action_probs)
                    action_probs = inverse_probs / np.sum(inverse_probs)
                    action = self.env.sample_action(probs=action_probs[0])
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                # Save for batch
                self.batch['states'].append(state)
                self.batch['actions'].append(action)
                self.batch['action_probs'].append(action_probs)
                self.batch['rewards'].append(reward)
                self.batch['next_states'].append(next_state)
                self.batch['dones'].append(terminated or truncated)

                state = next_state
                done = terminated or truncated
                step_count += 1
                if self.num_rollouts % 1000 == 0: self.env.render()
            self.num_rollouts += 1

        # Convert to numpy arrays
        self.batch['states'] = np.vstack(self.batch['states'])
        self.batch['actions'] = np.array(self.batch['actions'])[:, np.newaxis]
        self.batch['action_probs'] = np.vstack(self.batch['action_probs'])
        self.batch['rewards'] = np.array(self.batch['rewards'])[:, np.newaxis]
        self.batch['next_states'] = np.vstack(self.batch['next_states'])
        self.batch['dones'] = np.array(self.batch['dones'])[:, np.newaxis]

        return

    def batch_train(self):
        # Collect batch of training data
        for step in range(self.episode_limit):
            self.collect_rollouts()
            self.logger['episode'] = step

            # Unpack batched variables
            states = self.batch['states']
            actions = self.batch['actions']
            action_probs = self.batch['action_probs']
            rewards = self.batch['rewards']
            next_states = self.batch['next_states']
            dones = self.batch['dones']

            # Estimate value and next value with critic
            value = self.critic(states)
            value = np.clip(value, -10.0, 10.0)
            next_value = self.critic(next_states)
            next_value = np.clip(next_value, -10.0, 10.0)

            # Compute critic loss and update
            # If next state is terminal no need to incorporate future rewards
            target = rewards + (self.gamma*next_value)*(1 - (dones))
            critic_loss = self.critic.criterion(value, target)
            critic_gradients = self.critic.backward()
            critic_converged = self.critic.optimize()

            # Compute actor loss and update
            advantage = target - value
            advantage = (advantage - np.mean(advantage)) / (np.std(advantage) + 1e-8) # Normalize the advantage
            actor_loss = self.actor.criterion(actions, action_probs, advantage)
            actor_gradients = self.actor.backward()
            actor_converged = self.actor.optimize()

            # Logging
            self.logger['critic_loss'].append(critic_loss)
            self.logger['critic_max_grad'].append(max(grad[0].max() for grad in critic_gradients))
            self.logger['reward'].extend(list(rewards))
            self.logger['actor_loss'].append(actor_loss)
            self.logger['actor_max_grad'].append(max(grad[0].max() for grad in actor_gradients))
            self.logger['advantage'].extend(list(advantage))
            self.display_episode()
        
        self.save_plot()
        return

    def train(self):
        # Main A2C training loop
        advantages = np.zeros((self.episode_limit*self.step_limit))
        for episode in range(self.episode_limit):
            state, _ = self.env.reset()
            done = False
            step_count = 0
            self.logger['episode'] = episode

            while not done and step_count < self.step_limit:
                # Get action probabilities from actor and select action
                action_probs = self.actor(state)
                action = self.env.sample_action(probs=action_probs[0])

                # To explore, take an unlikely action
                exp_chance = np.random.uniform(0.0, 1.0)
                if exp_chance < self.exp_prob:
                    inverse_probs = (1.0 - action_probs)
                    action_probs = inverse_probs / np.sum(inverse_probs)
                    action = self.env.sample_action(probs=action_probs[0])

                # Take action and receive next action, reward, and dones
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Estimate value and next value with critic
                value = self.critic(state)
                value = np.clip(value, -10.0, 10.0)
                next_value = self.critic(next_state)
                next_value = np.clip(next_value, -10.0, 10.0)

                # Compute critic loss and update
                # If next state is terminal no need to incorporate future rewards
                target = reward + (self.gamma*next_value)*(1 - (terminated or truncated))
                critic_loss = self.critic.criterion(value, target)
                critic_gradients = self.critic.backward()
                critic_converged = self.critic.optimize()

                # Compute actor loss and update
                advantage = target - value
                # advantage = np.clip(advantage, -1.0, 1.0)
                # advantages[int(episode*self.step_limit + step_count)] = advantage
                advantage = advantage / (np.std(advantage) + 1e-8)
                actor_loss = self.actor.criterion(action, action_probs, advantage)
                actor_gradients = self.actor.backward()
                actor_converged = self.actor.optimize()

                # Logging
                self.logger['critic_loss'].append(critic_loss)
                self.logger['critic_max_grad'].append(max(grad[0].max() for grad in critic_gradients))
                self.logger['reward'].append(reward)
                self.logger['actor_loss'].append(actor_loss)
                self.logger['actor_max_grad'].append(max(grad[0].max() for grad in actor_gradients))
                self.logger['advantage'].append(advantage)

                # Update current state and step count
                state = next_state
                done = terminated or truncated
                if episode % 100 == 0: self.env.render()
                step_count += 1

            self.display_episode()

    def learn(self, batch=True):
        try:
            if batch: self.batch_train()
            else: self.train()
        except KeyboardInterrupt:
            print('\nManual interrupt, saving plot!')
            self.save_plot()
        except Exception as e:
            print(f'\n{e}')
            self.save_plot()

        return

