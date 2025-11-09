"""
Simple gymnasium-like environment for RL

Author: Jared Berry
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

from utils import round_to_res

class Env():
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

class GridEnv(Env):
    def __init__(self, bounds, res, obstacles=None, use_shaped=False, wrap_arena=False, render_mode='human'):
        # Environment
        self.bounds = bounds
        self.max_dist = abs(bounds[0][1] - bounds[0][0]) + abs(bounds[1][1] - bounds[1][0])
        self.res = res
        self.obstacles = obstacles
        self.wrap_arena = wrap_arena

        # Reward value
        self.reward = 10.0
        self.punishment = -0.0
        self.use_shaped = use_shaped
        self.shaped_mult = 0.1
        self.time_penalty = -0.01

        # Action and state space
        self.state_dim = 4
        self.action_space = np.array([0, 1, 2, 3])
        self.action_dim = len(self.action_space)
        self.action_map = {
            0: np.array([self.res, 0]),
            1: np.array([0, -self.res]),
            2: np.array([-self.res, 0]),
            3: np.array([0, self.res])
        }

        # Visualization
        self.fig = None
        self.ax = None
        self.robot_patch = None
        self.render_mode = render_mode

    def __init_render__(self):
        if self.fig is None and self.render_mode == 'human':
            # Create figure
            length = int(abs(self.bounds[0][1] - self.bounds[0][0]))
            height = int(abs(self.bounds[1][1] - self.bounds[1][0]))
            self.fig, self.ax = plt.subplots(figsize=(length, height))

            # Set up grid
            x_range = np.arange(self.bounds[0][0], self.bounds[0][1] + 1e-9, step=self.res)
            y_range = np.arange(self.bounds[1][0], self.bounds[1][1] + 1e-9, step=self.res)

            # Plot obstacles
            if self.obstacles is not None:
                for o in self.obstacles:
                    # Plot obstacles as rectanagles
                    rect = patches.Rectangle(
                        (o[0], o[1]), self.res, self.res,
                        facecolor='gray', edgecolor='black'
                    )
                    self.ax.add_patch(rect)

            # Set up grid
            self.ax.set_xticks(x_range)
            self.ax.set_yticks(y_range)
            self.ax.grid(color='black', linewidth=0.4)

            # Set up axis labels
            if self.res >= 0.5:
                self.ax.set_xticklabels(x_range, fontsize=12)
                self.ax.set_yticklabels(y_range, fontsize=12)
            else:
                self.ax.set_xticklabels([])
                self.ax.set_yticklabels([])
            self.ax.set_xlabel('X Coordinate', fontsize=14)
            self.ax.set_ylabel('Y Coordinate', fontsize=14)
            self.ax.set_aspect('equal')
            
            # Set axis limits
            self.ax.set_xlim(self.bounds[0])
            self.ax.set_ylim(self.bounds[1])

            # Plot start and goal
            self.start_patch = patches.Rectangle(
                (self.start[0], self.start[1]), self.res, self.res,
                facecolor='blue', edgecolor='black'
            )
            self.ax.add_patch(self.start_patch)
            self.goal_patch = patches.Rectangle(
                (self.goal[0], self.goal[1]), self.res, self.res,
                facecolor='green', edgecolor='black'
            )
            self.ax.add_patch(self.goal_patch)

            # Draw robot
            self.robot_patch = patches.Rectangle(
                self.pos, self.res, self.res,
                facecolor='red', edgecolor='black'
            )
            self.ax.add_patch(self.robot_patch)

            self.ax.set_title('Gridworld', fontsize=16)
            plt.ion() 
            plt.show()

    def sample_action(self, probs=None):
        # Sample random action from the action space
        return np.random.choice(self.action_space, p=probs)
    
    def out_of_bounds(self, pos):
        # Determine whether a given position is out of bounds
        if self.bounds[0][0] <= pos[0] < self.bounds[0][1] and self.bounds[1][0] <= pos[1] < self.bounds[1][1]:
            return False
        else:
            return True

    def reset(self):
        # Initialize starting and goal positions
        initialized = np.zeros((3,))
        while not initialized.all():
            self.start = round_to_res(np.random.uniform(self.bounds[:, 0], self.bounds[:, 1]), self.res)
            self.goal = round_to_res(np.random.uniform(self.bounds[:, 0], self.bounds[:, 1]), self.res)
            self.pos = round_to_res(self.start.copy(), self.res)
            if self.obstacles is not None:
                initialized[0] = tuple(self.start) not in self.obstacles    # Don't start on obstacle
                initialized[1] = tuple(self.goal) not in self.obstacles     # Don't end on obstacle
            else:
                initialized[0] = 1
                initialized[1] = 1
            initialized[2] = tuple(self.start) != tuple(self.goal)      # Don't start on goal
            if self.fig is None:
                self.__init_render__()
            else:
                # Update start and goal positions
                self.start_patch.set_xy(self.start)
                self.goal_patch.set_xy(self.goal)
                self.robot_patch.set_xy(self.pos)

        # Add batch dimension to observation
        obs = np.concatenate([self.pos, self.goal])
        return obs[np.newaxis, ...], {}
    
    def wrap(self, pos):
        # Wrap x
        new_pos = pos.copy()
        if pos[0] < self.bounds[0][0]: new_pos[0] = self.bounds[0][1] - 1
        elif pos[0] >= self.bounds[0][1]: new_pos[0] = self.bounds[0][0]

        # Wrap y
        if pos[1] < self.bounds[1][0]: new_pos[1] = self.bounds[1][1] - 1
        elif pos[1] >= self.bounds[1][1]: new_pos[1] = self.bounds[1][0]

        return new_pos

    def step(self, action_index):
        # Get physical action corresponding to action index
        action = self.action_map[action_index]
        next_state = self.pos + action

        # Wrap around arena edges
        if self.wrap_arena: next_state = self.wrap(next_state)
        
        # Initialize done flags
        terminated, truncated = False, False

        # Check for rewards or punishments
        reward = self.time_penalty
        obs_collision = (self.obstacles is not None) and (tuple(next_state) in self.obstacles)
        if obs_collision or self.out_of_bounds(next_state):
            # Hit obstacle or out of bounds
            reward += self.punishment
            truncated = True
        elif (next_state == self.goal).all():
            # Found goal
            reward += self.reward
            self.pos = round_to_res(next_state, self.res)
            terminated = True
        else:
            self.pos = round_to_res(next_state, self.res)

        # Shaped reward uses inverse distance
        if self.use_shaped: 
            dist = np.linalg.norm(self.pos - self.goal)
            reward += self.shaped_mult*(1 - dist / self.max_dist)

        obs = np.concatenate([self.pos, self.goal])
        return obs[np.newaxis, ...], reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == 'human':
            # Update position and draw robot
            self.robot_patch.set_xy(self.pos)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        elif self.render_mode == 'no_vis':
            return