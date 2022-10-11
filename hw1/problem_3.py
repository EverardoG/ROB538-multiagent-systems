""" Now both agents recieve a collective team reward. One agent moves after the other."""

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
world = GridWorld(agents_start_pos, targets_pos, team_reward=True)
learners = [QLearner(grid_world=world, alpha=0.4, gamma=0.4, epsilon=0.05) for _ in range(len(agents_start_pos))]

all_rewards, _, _, q_tables = train_learners(world=world, learners=learners, num_steps=15, num_episodes=1000)

# Data wrangling
rewards_arr = np.sum(all_rewards, axis=1)

# Total rewards in an episode
total_rewards = np.sum(rewards_arr, axis=1)

fig, ax = plt.subplots()
ax.plot(total_rewards)
ax.set_title("Rewards Earned by System")
ax.set_xlabel("Episode Number")
ax.set_ylabel("Total Rewards in Episode")
plt.show()

for ind, learner in enumerate(learners):
    fig, ax = plt.subplots()
    learner.plot_q_table(ax, show=False, worst=False)
    ax.set_title("Max Q Value w. System Reward | Agent "+str(ind+1))
    plt.show()
