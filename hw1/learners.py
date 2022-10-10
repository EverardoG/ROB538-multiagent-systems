from typing import List
from copy import deepcopy

import numpy as np

from base_utils import Action
from grid_world import GridWorld

class QLearner():
    def __init__(self, grid_world: GridWorld, alpha, gamma, epsilon):
        self.q_table = np.zeros( ((grid_world.bounds+1)[0], (grid_world.bounds+1)[1], 5))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def max_q_value(self, adj_states: List[np.ndarray])->float:
        # Given adjacent states, Get the maximum Q value from all of them
        return np.max([ np.max(self.q_table[ state[0], state[1] ]) for state in adj_states ])

    def update_q_table(self, agent_pos: np.ndarray, last_action: Action, current_reward: int, adj_states: List[Action])->None:
        self.q_table[agent_pos[0], agent_pos[1], last_action.value] += \
            self.alpha * ( current_reward + self.gamma * self.max_q_value(adj_states) - self.q_table[agent_pos[0], agent_pos[1], last_action.value])
        return None

    def greedy_pick_next_action(self, adj_states: List[np.ndarray], actions: List[Action])->Action:
        best_value = np.max(self.q_table[adj_states[0][0], adj_states[0][1] ])
        best_action = actions[0]
        for adj_state, action in zip(adj_states[1:], actions[1:]):
            if np.max(self.q_table[adj_state[0], adj_state[1]]) > best_value:
                best_value = np.max(self.q_table[adj_state[0], adj_state[1]])
                best_action = action
        return best_action

        # # print("greedy_pick_next_action")
        # # print(self.q_table[state[0], state[1]])
        # action_int = np.argmax(self.q_table[state[0], state[1]])
        # # print(action_int)
        # return IntToAction[action_int]

    def epsilon_pick_next_action(self, adj_states: List[np.ndarray], actions: List[Action])->Action:
        best_action = self.greedy_pick_next_action(adj_states, actions)
        if 1-self.epsilon > np.random.uniform():
            return best_action
        # Otherwise, randomly pick from remaining actions
        else:
            possible_actions = deepcopy(actions)
            for action in possible_actions:
                if action.value == best_action.value:
                    possible_actions.remove(action)
                    break
            return np.random.choice(possible_actions)
