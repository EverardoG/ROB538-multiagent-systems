from typing import Optional, List, Tuple
import numpy as np
from base_utils import Action

class GridWorld():
    def __init__(self, agent_start_pos: Optional[np.ndarray] = None, random_restart: bool = True, random_goal: bool = False) -> None:
        self.barrier_pos_list = [
            np.array([7,0]),
            np.array([7,1]),
            np.array([7,2])
        ]
        self.goal_pos = np.array([9,1])
        self.bounds = np.array([9, 4])
        if agent_start_pos is not None:
            self.start_pos = agent_start_pos
            # self.agent_pos = agent_start_pos
        else:
            self.start_pos = self.generate_random_starting_pos()
            # self.agent_pos = self.generate_random_starting_pos()
        self.agent_pos = np.copy(self.start_pos)
        self.ActionToCoordChange = {
            Action.LEFT.value: [-1,0],
            Action.RIGHT.value: [1,0],
            Action.DOWN.value: [0,-1],
            Action.UP.value: [0,1]
        }
        self.possible_actions = [Action.LEFT, Action.RIGHT, Action.DOWN, Action.UP]
        self.random_restart = random_restart
        self.random_goal = random_goal
        self.adj_coord_moves = [
            np.array([0,1]),
            np.array([0,-1]),
            np.array([1,0]),
            np.array([-1,0])
        ]

    def reset(self)->None:
        if self.random_restart:
            self.agent_pos = self.generate_random_starting_pos()
        else:
            self.agent_pos = np.copy(self.start_pos)

    def generate_random_starting_pos(self)->np.ndarray:
        # Generate a random position within bounds
        start_pos = np.random.randint([0,0], self.bounds)
        # Nudge position left or right if it's a barrier position
        for barrier_pos in self.barrier_pos_list:
            if np.allclose(start_pos, barrier_pos):
                start_pos[0] += np.random.choice([-1,1], 1)
        return start_pos

    def step(self, action: Action)->None:
        """Update simulation state based on action of the agent."""
        if self.random_goal:
            self.goal_pos = self.choose_random_adj_coord(self.goal_pos)
        # Generate a new position assuming no constraints
        new_pos = self.agent_pos + self.ActionToCoordChange[action.value]
        # Don't udpate agent position if new position would put it in barrier
        for barrier_pos in self.barrier_pos_list:
            if np.allclose(new_pos, barrier_pos):
                return None
        # Update agent position if new position is within bounds
        if np.all(new_pos<self.bounds+1) and np.all(new_pos >= [0,0]):
            self.agent_pos = new_pos
        return None

    def get_state(self)->np.ndarray:
        return self.agent_pos

    def get_reward(self)->int:
        if np.allclose(self.agent_pos, self.goal_pos):
            return 20
        else:
            return -1

    def get_valid_adj_coords(self, coord: np.ndarray)->List[np.ndarray]:
        # Get all adjacent coordinates
        adj_coords = []
        for adj_coord_move in self.adj_coord_moves:
            adj_coords.append(coord + adj_coord_move)
        # Filter out invalid coordinates
        valid_coords = []
        for adj_coord in adj_coords:
            within_bounds = np.all(adj_coord <= self.bounds) and np.all(adj_coord >= [0,0])
            is_barrier_list = [ np.all(np.isclose(adj_coord, barrier_coord)) for barrier_coord in self.barrier_pos_list]
            not_barrier = not np.any(is_barrier_list)
            if within_bounds and not_barrier:
                valid_coords.append(adj_coord)
        return valid_coords

    def choose_random_adj_coord(self, coord: np.ndarray)->np.ndarray:
        # Grab valid adjacent coordinates
        valid_coords = self.get_valid_adj_coords(coord)
        # Randomly chose a coordinate
        random_ind = np.random.choice(len(valid_coords))
        return valid_coords[random_ind]

    def get_adj_states_and_actions(self)->Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get all states adjacent to agent's current state with corresponding actions"""
        # Save current position
        current_state = np.copy(self.agent_pos)
        # Step simulation forward for each action
        adj_states = []
        for action in self.possible_actions:
            self.step(action)
            adj_states.append(np.copy(self.agent_pos))
            self.agent_pos = np.copy(current_state)
            # Agent position is always the same as how it started
        return adj_states, self.possible_actions

    def get_world_state(self):
        world_state = np.zeros(self.bounds+1, dtype=str)
        for barrier_pos in self.barrier_pos_list:
            world_state[barrier_pos[0], barrier_pos[1]] = 'b'
        world_state[self.goal_pos[0], self.goal_pos[1]] = 'g'
        world_state[self.agent_pos[0], self.agent_pos[1]] = 'a'
        return world_state