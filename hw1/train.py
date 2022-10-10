from typing import Tuple, List
import numpy as np

from grid_world import GridWorld
from learners import QLearner

def run_episode(learners: List[QLearner], grid_world: GridWorld, num_steps: int, update_q_tables: bool, use_episilon: bool)->np.ndarray:
    episode_rewards = []
    _, all_actions = grid_world.get_adj_states_and_actions()
    for _ in range(num_steps):
        # Save last state
        last_state = np.copy(grid_world.agent_pos)
        # Pick an action
        adj_states, actions = grid_world.get_adj_states_and_actions()
        if greedy == True:
            action = learner.greedy_pick_next_action(adj_states, actions)
        else:
            action = learner.epsilon_pick_next_action(adj_states, actions)
        # Take that action
        grid_world.step(action)
        # Get a reward
        reward = grid_world.get_reward()
        # Save the reward
        episode_rewards.append(reward)
        # Update value table
        adj_states, _ = grid_world.get_adj_states_and_actions()
        if learning:
            learner.update_q_table(last_state, action, reward, adj_states)
    return np.array(episode_rewards)

def train_learners(world: GridWorld, learners: List[QLearner], num_steps: int, num_episodes: int, testing: bool = True)->Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    # Setup grid world and appropriate learner

    all_rewards = []
    all_test_rewards = []
    test_episodes = []
    for num_episode in range(num_episodes):
        episode_rewards = run_episode()
        if learner_type.value == LearnerType.TD.value:
            episode_rewards = run_episode_td(learner, grid_world, num_steps)
        else:
            episode_rewards = run_episode_q(learner, grid_world, num_steps)

        all_rewards.append(episode_rewards)
        grid_world.reset()
        # Test the agent at intervals
        if (num_episode % 2 == 0 and testing) or num_episode == num_episodes-1:
        # if num_episode == num_episodes-1:
            test_coords = []
            # random test coordinates
            while len(test_coords) < 20:
                new_test_coord = grid_world.generate_random_starting_pos()
                already_chosen = False
                for test_coord in test_coords:
                    if np.allclose(new_test_coord, test_coord):
                        already_chosen = True
                if not already_chosen:
                    test_coords.append(new_test_coord)
            test_rewards = np.array(test_learner(test_coords, grid_world, learner, learner_type, num_steps))
            all_test_rewards.append(test_rewards)
            test_episodes.append(num_episode)

    if learner_type.value == LearnerType.TD.value:
        final_table = learner.v_table
    else:
        final_table = learner.q_table
    return all_rewards, all_test_rewards, test_episodes, final_table

def test_learners(start_coords: List[np.ndarray], grid_world: GridWorld, learner, learner_type: LearnerType, num_steps: int)->float:
    test_rewards = []
    for start_coord in start_coords:
        grid_world.agent_pos = start_coord
        if learner_type.value == LearnerType.TD.value:
            episode_reward = np.sum(run_episode_td(learner, grid_world, num_steps, learning=False, greedy=True))
        else:
            episode_reward = np.sum(run_episode_q(learner, grid_world, num_steps, learning=False, greedy=True))
        test_rewards.append(episode_reward)
    grid_world.reset()
    return test_rewards