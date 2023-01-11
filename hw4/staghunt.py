from enum import IntEnum
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Agent():
    def __init__(self, epsilon: float, alpha: float) -> None:
        # Each player can chose from 2 actions. Either go to McMenamin's or go to Clod's
        self.estimates = np.ones(2)
        self.epsilon = epsilon # Probability of taking a random action
        self.alpha = alpha # Learning rate

    def randomAction(self):
        # 0 is Clod's. 1 is McMenamin's
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
        # if np.any(self.estimates[0]<0):
        #     print("FLAG")
        return None

def calculatePayoffs(state):
    # Look at the state. First value is which option Player 1 chose. Second value is which option Player 2 chose.
    # 0 is Clod's. 1 is McMenamin's

    # Returns payoffs for each player. 0th index goes to Player 1. 1st index goes to Player 2.

    # Both Players go to Clod's
    if state[0] == 0 and state[1] == 0:
        return np.array([2,5])

    # Both players go to McMenamin's
    elif state[0] == 1 and state[1] == 1:
        return np.array([5,2])

    # Each player goes to a different bar
    else:
        return np.zeros(2)

def runChoosingProblem(num_episodes, epsilon_off, epsilon, alpha, filename):
    """
    num_episodes: number of episodes to run for agents to learn
    epsilon: probability that an agent will take random action during training
    alpha: learning rate for an agent

    Player 1 runs a mixed strategy and Player 2 does their best to learn what to do
    """
    # Track global rewards and nightly distribution of agents throughout training
    # global_rewards_train = []   # with epsilon
    # global_rewards_test = []    # without epsilon
    # nightly_distributions_train = []    # List for each episode representing how any agents went on each night (w. epsilon)
    # nightly_distributions_test = []     # (w. out epsilon)
    player_1_payoffs = []
    player_2_payoffs = []
    action_1_values = []
    action_2_values = []
    player_1_actions = []
    player_2_actions = []

    # agents: List[Agent] = [Agent(epsilon=epsilon, alpha=alpha) for _ in range(2)]
    agent = Agent(epsilon=epsilon, alpha=alpha)
    for n in tqdm(range(num_episodes)):
        # Player 1 picks an action w. random
        player_1_action = np.random.choice([0,1])

        # Player 1 picks an action w. epsilon greedy
        if epsilon_off < n:
            agent.epsilon *= 0.99
            # print(n, " | ", agent.epsilon)
            action = agent.epsilonGreedyAction()
        else:
            action = agent.epsilonGreedyAction()

        # Aggregate actions into the state
        state = np.array([player_1_action, action])

        # Calculate payoff for Player 2
        payoffs = calculatePayoffs(state)
        player_1_payoff = payoffs[0]
        player_2_payoff = payoffs[1]

        # Track different variables
        player_1_payoffs.append(player_1_payoff)
        player_2_payoffs.append(player_2_payoff)
        action_1_values.append(agent.estimates[0])
        action_2_values.append(agent.estimates[1])
        player_1_actions.append(player_1_action)
        player_2_actions.append(action)

        # Update the value estimates for Player 2
        agent.updateEstimate(action, player_2_payoff)

    print(np.sum(player_1_actions))
    print(np.sum(player_2_actions))

    savePayoffsPlot(player_1_payoffs, player_2_payoffs, filename)
    saveEstimatesPlot(action_1_values, action_2_values, filename)
    saveActionsPlot(player_1_actions, player_2_actions, filename)

def savePayoffsPlot(player_1_payoffs, player_2_payoffs, filename: Optional[str] = None):
    fig, ax = plt.subplots()
    ax.set_title("Player Payoffs")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Payoff")
    ax.plot(player_1_payoffs)
    ax.plot(player_2_payoffs)
    ax.legend(["Player 1", "Player 2"])

    if filename is None:
        filename = "Payoffs Plot"
    plt.savefig("hw4/figures/"+filename+'.png', format='png', dpi=1200)

def saveEstimatesPlot(action_1_values, action_2_values, filename: Optional[str] = None):
    fig, ax = plt.subplots()
    ax.set_title("Value Estimates Throughout Learning")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Value")
    ax.plot(action_1_values)
    ax.plot(action_2_values)
    ax.legend(["Clod's", "McMenamin's"])

    if filename is None:
        filename = "Value Estimates Plot"
    plt.savefig("hw4/figures/"+filename+'.png', format='png', dpi=1200)

def saveActionsPlot(player_1_actions, player_2_actions, filename: Optional[str] = None):
    fig, ax = plt.subplots()
    ax.set_title("Player Choices")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Action | 0 Clod's | 1 McMenamin's")
    ax.plot(player_1_actions)
    ax.plot(player_2_actions)
    ax.legend(["Player 1", "Player 2"])

    if filename is None:
        filename = "Choices Plot"
    plt.savefig("hw4/figures/"+filename+'.png', format='png', dpi=1200)