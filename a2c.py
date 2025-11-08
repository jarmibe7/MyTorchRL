"""
Deep Advantage Actor-Critic (A2C) algorithm.

https://medium.com/@dixitaniket76/advantage-actor-critic-a2c-algorithm-explained-and-implemented-in-pytorch-dc3354b60b50

Paper: https://arxiv.org/abs/1602.01783

Author: Jared Berry
"""
import numpy as np

from my_torch.module import Module
from my_torch.ff import FeedForward
from my_torch.loss import Loss, MSELoss

class A2CActorLoss(Loss):
    """
    Actor loss for the A2C algorithm
    """
    def __init__(self):
        super().__init__()

    def forward(self, action, action_probs, advantage):
        # Negative log action probability of selected action * advantage
        self.action = action
        self.action_probs = action_probs
        self.advantage = advantage
        return -np.log(action_probs[action])*advantage
    
    def backward(self):
        # Only calculate gradient for selected action
        dLda = np.zeros_like(self.action_probs)
        dLda[self.action] = self.advantage / self.action_probs[self.action]
        return dLda

class Actor(Module):
    """
    A simple actor for A2C.
    
    Args:
        arch: A list of triplets of length n, where each index i corresponds to a layer
              and arch[i] is a triplet with that layer's input size, output size, and activation function.
        loss_type: A string specifying what type of loss function to use.
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
        return self.ff(x)
    
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
        loss_type: A string specifying what type of loss function to use.
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
        alpha: Learning rate
        conv_thresh: Gradient convergence threshold
    """
    def __init__(self, env, critic_arch, actor_arch, alpha_actor, alpha_critic, gamma, episode_limit, step_limit, conv_thresh, save_model):
        # Assign environment
        self.env = env

        # Create actor and critic
        self.critic = Critic(critic_arch, loss_type=None, alpha=alpha_critic, conv_thresh=conv_thresh)
        self.actor = Actor(actor_arch, loss_type=None, alpha=alpha_actor, conv_thresh=conv_thresh)

        # Algorithm params
        self.gamma = gamma
        self.episode_limit = episode_limit
        self.step_limit = step_limit
        self.save_model = save_model

        # Logging
        self.__init_logging__()

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

    def display_episode(self):
        # Display episode statistics
        print('\n----------------------------')
        print(f'Episode {self.logger['episode']}:')
        print(f'Avg. Critic Loss: {np.mean(np.array(self.logger['critic_loss']))}')
        print(f'Avg. Max Critic Gradient: {np.mean(np.array(self.logger['critic_max_grad']))}')
        print(f'Avg. Reward: {np.mean(np.array(self.logger['reward']))}')
        print(f'Avg. Actor Loss: {np.mean(np.array(self.logger['actor_loss']))}')
        print(f'Avg. Max Actor Gradient: {np.mean(np.array(self.logger['actor_max_grad']))}')
        print(f'Avg. Advantage: {np.mean(np.array(self.logger['advantage']))}')
        print('----------------------------\n')
        return

    def train(self):
        # Main A2C training loop
        for episode in range(self.episode_limit):
            state, _ = self.env.reset()
            done = False
            step_count = 0
            self.__init_logging__()
            self.logger['episode'] = episode

            while not done and step_count < self.step_limit:
                # Get action probabilities from actor and select action
                action_probs = self.actor(state)[0]
                action = self.env.sample_action(probs=action_probs)

                # Take action and receive next action, reward, and dones
                next_state, reward, terminated, truncated, info = self.env.step(action)

                # Estimate value and next value with critic
                value = self.critic(state)
                next_value = self.critic(next_state)

                # Compute critic loss and update
                # If next state is terminal no need to incorporate future rewards
                target = reward + (self.gamma*next_value)*(1 - (terminated or truncated))
                critic_loss = self.critic.criterion(value, target)
                critic_gradients = self.critic.backward()
                critic_converged = self.critic.optimize()

                # Compute actor loss and update
                advantage = target - value
                actor_loss = self.actor.criterion(action, action_probs, advantage)
                actor_gradients = self.actor.backward()
                actor_converged = self.actor.optimize()

                # Logging
                self.logger['critic_loss'] = critic_loss
                self.logger['critic_max_grad'] = np.max(critic_gradients)
                self.logger['reward'] = reward
                self.logger['actor_loss'] = actor_loss
                self.logger['actor_max_grad'] = np.max(actor_gradients)
                self.logger['advantage'] = advantage

                # Update current state and step count
                state = next_state
                step_count += 1

            # Display episode results
            print(f'Episode: {episode}  Actor Los')

