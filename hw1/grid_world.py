import enum
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from base_utils import Action

class GridWorld():
    def __init__(self, agents_start_pos: Optional[np.ndarray] = None, targets_pos = Optional[np.ndarray], random_restart: bool = False) -> None:
        # self.goal_pos = np.array([9,1])
        self.targets_pos = targets_pos
        self.bounds = np.array([9, 4])
        if agents_start_pos is not None:
            self.start_pos = agents_start_pos
            # self.agents_pos = agents_start_pos
        else:
            self.start_pos = self.generate_random_starting_pos()
            # self.agents_pos = self.generate_random_starting_pos()
        self.agents_pos = np.copy(self.start_pos)
        self.ActionToCoordChange = {
            Action.LEFT.value: [-1,0],
            Action.RIGHT.value: [1,0],
            Action.DOWN.value: [0,-1],
            Action.UP.value: [0,1]
        }
        self.possible_actions = [Action.LEFT, Action.RIGHT, Action.DOWN, Action.UP]
        self.random_restart = random_restart
        self.targets_pos = targets_pos
        self.adj_coord_moves = [
            np.array([0,1]),
            np.array([0,-1]),
            np.array([1,0]),
            np.array([-1,0])
        ]

    def reset(self)->None:
        if self.random_restart:
            self.agents_pos = self.generate_random_starting_pos()
        else:
            self.agents_pos = np.copy(self.start_pos)

    def generate_random_starting_pos(self)->np.ndarray:
        # Generate a random position within bounds
        start_pos = np.random.randint([0,0], self.bounds)
        return start_pos

    def step(self, action: Action, agent_id: int)->None:
        """Update simulation state based on action of an agent."""
        # Generate a new position assuming no constraints
        new_pos = self.agents_pos[agent_id] + self.ActionToCoordChange[action.value]
        # Update agent position if new position is within bounds
        if np.all(new_pos<self.bounds+1) and np.all(new_pos >= [0,0]):
            self.agents_pos[agent_id] = new_pos
        return None

    def get_state(self)->np.ndarray:
        return self.agents_pos

    def get_reward(self)->int:
        if any([np.allclose(self.agents_pos, target_pos) for target_pos in self.targets_pos]):
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
            if within_bounds:
                valid_coords.append(adj_coord)
        return valid_coords

    def choose_random_adj_coord(self, coord: np.ndarray)->np.ndarray:
        # Grab valid adjacent coordinates
        valid_coords = self.get_valid_adj_coords(coord)
        # Randomly chose a coordinate
        random_ind = np.random.choice(len(valid_coords))
        return valid_coords[random_ind]

    def get_adj_states_and_actions(self, agent_id)->Tuple[List[np.ndarray], List[np.ndarray]]:
        """Get all states adjacent to agent's current state with corresponding actions"""
        # Save current position
        current_state = np.copy(self.agents_pos[agent_id])
        # Step simulation forward for each action for this agent
        adj_states = []
        for action in self.possible_actions:
            self.step(action, agent_id)
            adj_states.append(np.copy(self.agents_pos[agent_id]))
            self.agents_pos[agent_id] = np.copy(current_state)
            # Agent position is always the same as how it started
        return adj_states, self.possible_actions

    def get_cli_state(self):
        world_state = np.zeros((self.bounds[1]+1, self.bounds[0]+1), dtype=object)
        world_state[:,:] = '  '
        for ind, target_pos in enumerate(self.targets_pos):
            # print(ind, ind+1, str(ind+1), 't'+str(ind+1))
            world_state[self.bounds[1]-target_pos[1], target_pos[0]] = 't'+str(ind+1)
        # Agents go on second so they override the targets on display
        for ind, agent_pos in enumerate(self.agents_pos):
            world_state[self.bounds[1]-agent_pos[1], agent_pos[0]] = 'a'+str(ind+1)
        return world_state

    def plot_state(self, ax, show: bool = True):
        colors = [(255,255,255), (200,0,0), (0,0,200)]

        cli_state = self.get_cli_state()
        display_map = np.zeros((cli_state.shape[0], cli_state.shape[1], 3), dtype=int)
        display_map[:,:] = colors[0]
        for row in range(display_map.shape[0]):
            for col in range(display_map.shape[1]):
                # Use 1 where there is a target
                if "t" in cli_state[row, col]:
                    display_map[display_map.shape[0]-row-1, col] = colors[1]
                # Use 2 where there is an agent
                elif "a" in cli_state[display_map.shape[0]-row-1, col]:
                    display_map[row, col] = colors[2]

        ax.imshow(display_map)
        x_range = [-0.5, self.bounds[0]+0.5]
        y_range = [-0.5, self.bounds[1]+0.5]
        ax.set_xticks(np.arange(self.bounds[0]+2)-0.5)
        ax.set_yticks(np.arange(self.bounds[1]+2)-0.5)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.grid(color=(0.3,0.3,0.3),linewidth=2)


