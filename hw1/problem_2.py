""" Introduce second agent and second target. Agents start at the same location. Each one gets an individual reward. One agent moves after the other."""

import numpy as np
import matplotlib.pyplot as plt

from grid_world import GridWorld
from train import train_learners
from learners import QLearner

agents_start_pos = np.array([
    [3,2],
    [3,2]
], dtype=int)
targets_pos = np.array([
    [1,1],
    [9,4]
], dtype=int)

# Alpha is learning rate. Gamma is discount on future rewards. Epsilon is probability agent randomly chooses non-greedy action.
world = GridWorld(agents_start_pos, targets_pos)
learners = [QLearner(grid_world=world, alpha=0.4, gamma=0.4, epsilon=0.05) for _ in range(len(agents_start_pos))]

train_learners(world=world, learners=learners, num_steps=15, num_episodes=100)

for learner in learners:
    learner.plot_q_table(plt.subplots()[1], show=True, worst=False)
    learner.plot_q_table(plt.subplots()[1], show=True, worst=True)
