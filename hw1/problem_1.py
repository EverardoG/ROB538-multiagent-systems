""" Start with 1 agent and 1 target. Devise a learning algorithm to reach T1 """

import numpy as np
import matplotlib.pyplot as plt

from grid_world import GridWorld
from learners import QLearner

agents_start_pos = np.array([
    [3,2]
])
targets_pos = np.array([
    [1,1]
])

# Alpha is learning rate. Gamma is discount on future rewards. Epsilon is probability agent randomly chooses non-greedy action.
world = GridWorld(agents_start_pos, targets_pos)
# learner = QLearner(alpha=0.1, gamma=0.1, epsilon=0.1)

# print(world.get_cli_state())

fig, ax = plt.subplots()
world.plot_state(ax, show=True)
plt.show()