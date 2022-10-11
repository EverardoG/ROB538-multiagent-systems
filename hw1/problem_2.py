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

all_rewards, _, _, q_tables = train_learners(world=world, learners=learners, num_steps=15, num_episodes=1000)

# Data wrangling
rewards_arr = np.array(all_rewards)
print(rewards_arr)

# Total rewards in an episode
total_rewards = np.sum(rewards_arr, axis=2)
print(total_rewards)

fig, ax = plt.subplots()
ax.plot(total_rewards)
ax.set_title("Rewards Earned by Each Agent")
ax.set_xlabel("Episode Number")
ax.set_ylabel("Total Rewards in Episode")
ax.legend(["Agent 1", "Agent 2"])
plt.show()


for ind, learner in enumerate(learners):
    fig, ax = plt.subplots()
    learner.plot_q_table(ax, show=False, worst=False)
    ax.set_title("Max Q Value 2 Agents, 2 Targets | Agent "+str(ind+1))
    plt.show()
    # learner.plot_q_table(plt.subplots()[1], show=False, worst=True)
