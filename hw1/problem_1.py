""" Start with 1 agent and 1 target. Devise a learning algorithm to reach T1 """

import numpy as np
import matplotlib.pyplot as plt

from grid_world import GridWorld
from train import train_learners
from learners import QLearner

agents_start_pos = np.array([
    [3,2]
], dtype=int)
targets_pos = np.array([
    [1,1]
], dtype=int)

# Alpha is learning rate. Gamma is discount on future rewards. Epsilon is probability agent randomly chooses non-greedy action.
world = GridWorld(agents_start_pos, targets_pos)
learner = QLearner(grid_world=world, alpha=0.4, gamma=0.4, epsilon=0.2)

train_learners(world=world, learners=[learner], num_steps=5, num_episodes=100)

learner.plot_q_table(plt.subplots()[1], show=True, worst=False)
learner.plot_q_table(plt.subplots()[1], show=True, worst=True)
