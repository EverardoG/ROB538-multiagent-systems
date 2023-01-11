from enum import IntEnum
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Agent():
    def __init__(self, K: int, epsilon: float, alpha: float) -> None:
        self.estimates = np.ones(K)
        self.epsilon = epsilon # Probability of taking a random action
        self.alpha = alpha # Learning rate

    def randomAction(self):
        return np.random.choice(self.estimates.size)

    def greedyAction(self):
        # Note this will take the lowest index if there are any ties
        return np.argmax(self.estimates)

    def epsilonGreedyAction(self):
        if np.random.uniform(0,1) < self.epsilon:
            return self.randomAction()
        else:
            return self.greedyAction()

    def updateEstimate(self, action: int, reward: float):
        self.estimates[action] = self.alpha * (reward - self.estimates[action])
        return None

def calculateGlobal(state, b):
    nightly_rewards = [night_attend*np.exp(-night_attend/b) for night_attend in state]
    global_reward = np.sum(nightly_rewards)
    return global_reward

def calculateRewards(actions, state, b, reward_type):
    nightly_rewards = [night_attend*np.exp(-night_attend/b) for night_attend in state]
    global_reward = np.sum(nightly_rewards)

    if reward_type == RewardType.Global:
        return global_reward*np.ones(len(actions))

    elif reward_type == RewardType.Local:
        local_rewards = np.zeros(len(actions))
        for agent_ind, action in enumerate(actions):
            local_rewards[agent_ind] = nightly_rewards[action]
        return local_rewards

    elif reward_type == RewardType.Difference:
        difference_rewards = np.zeros(len(actions))
        for agent_ind, action in enumerate(actions):
            counterfactual_state = state.copy()
            counterfactual_state[action] -= 1
            counterfactual_global_reward = calculateGlobal(counterfactual_state, b)
            difference_rewards[agent_ind] = global_reward - counterfactual_global_reward
        return difference_rewards

class RewardType(IntEnum):
    Global = 0
    Local = 1
    Difference = 2

def saveRewardsPlot(global_rewards_train: List[float], global_rewards_test: List[float], filename: Optional[str] = None):
    fig, ax = plt.subplots()
    ax.set_title("Rewards Plot")
    ax.set_xlabel("Week")
    ax.set_ylabel("Reward")
    ax.plot(global_rewards_train)
    ax.plot(global_rewards_test)
    ax.legend(["Training", "Testing"])

    if filename is None:
        filename = "rewards_plot"
    plt.savefig("hw3/figures/"+filename+'.svg', format='svg', dpi=1200)
    plt.savefig("hw3/figures/"+filename+'.png', format='png', dpi=1200)

# def saveDistributionPlot(nightly_distributions_train: List[np.ndarray], nightly_distributions_test: List[np.ndarray]):
#     fig, ax = plt.subplots()
#     ax.set_title("Distrubtion accross nights")
#     ax.set_ylabel("Number agents")
#     ax.set_xlabel("Week")
#     n_train = np.array(nightly_distributions_train)
#     num_agents = n_train.shape[1]

def saveHistogram(state, filename):
    fix, ax = plt.subplots()
    ax.set_title("Histogram Final Policies")
    ax.set_xlabel("Night")
    ax.set_ylabel("Number of Agents Attending")
    # counts = state
    # bins = len(state)
    # print(bins, counts)
    # ax.stairs(bins, counts)
    # ax.hist(state[:-1], state, weights=state[:-1])
    # print(state)
    # counts, bins = np.histogram(state)
    # ax.stairs(bins=len(state), values=state)
    # ax.hist(bins=state[:-1], state,  )
    # ax.bar(state)

    # bins = len(state)
    # x = state
    # print(state, len(state))
    # import sys; sys.exit()

    # This maybe works????
    # plt.hist(x=state, bins=len(state))

    # plt.hist(state[:1], state, weights=)

    # trying again w. numpy hist
    # counts, bins = np.histogram(state)
    # print(counts, bins)
    # plt.stairs(bins, counts)
    plt.bar(x=np.arange(len(state)), height=state)

    if filename is None:
        filename = "histogram_plot"
    plt.savefig("hw3/figures/"+filename+".svg", format="svg", dpi=1200)
    plt.savefig("hw3/figures/"+filename+".png", format="png", dpi=1200)

def runBarProblem(num_agents, K_nights, num_episodes, epsilon_off, epsilon, alpha, b, reward_type, filename):
    """
    num_agents: number of agents going to the bar
    K_nights: number of nights that agents are going to bar
    num_episodes: number of episodes to run for agents to learn
    epsilon: probability that an agent will take random action during training
    alpha: learning rate for an agent
    b: optimal number of agents to be in the bar in a night
    reward_type: what type of reward to give agents as a learning signal
    """
    # Track global rewards and nightly distribution of agents throughout training
    global_rewards_train = []   # with epsilon
    global_rewards_test = []    # without epsilon
    nightly_distributions_train = []    # List for each episode representing how any agents went on each night (w. epsilon)
    nightly_distributions_test = []     # (w. out epsilon)

    agents: List[Agent] = [Agent(K=K_nights, epsilon=epsilon, alpha=alpha) for _ in range(num_agents)]
    for n in tqdm(range(num_episodes)):
        # Each agent picks an action
        if epsilon_off < n:
            for agent in agents:
                agent.epsilon *= 0.9
            actions = [agent.greedyAction() for agent in agents]
        else:
            actions = [agent.epsilonGreedyAction() for agent in agents]
        # Sort agents into nights based on actions
        state = np.zeros(K_nights)
        for action in actions:
            state[action] += 1
        # Calculate the reward for the agents
        rewards = calculateRewards(actions, state, b, reward_type)

        # Store the global reward
        global_reward = calculateRewards(actions, state, b, RewardType.Global)[0]
        global_rewards_train.append(global_reward)
        nightly_distributions_train.append(state)

        # Find the global reward if each agent went their highest valued night
        test_actions = [agent.greedyAction() for agent in agents]
        test_state = np.zeros(K_nights)
        for t_action in test_actions:
            test_state[t_action] += 1
        global_reward_test = calculateRewards(test_actions, test_state, b, RewardType.Global)[0]
        global_rewards_test.append(global_reward_test)
        nightly_distributions_test.append(test_state)

        # Update the value estimates for each agent
        for agent, action, reward in zip(agents, actions, rewards):
            agent.updateEstimate(action, reward)

    saveRewardsPlot(global_rewards_train, global_rewards_test, filename)
    saveHistogram(test_state, "Histogram"+filename)
