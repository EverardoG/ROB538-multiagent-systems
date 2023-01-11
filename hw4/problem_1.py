from bar import *

np.random.seed(0)


# Problem 1a
runBarProblem(num_agents=42, K_nights=6, num_episodes=200, epsilon=0.1, epsilon_off=0, alpha=0.05, b=5, reward_type=RewardType.Local, filename="42 agents | 6 nights | Local Reward")

# Problem 1b
runBarProblem(num_agents=42, K_nights=6, num_episodes=200, epsilon=0.1, epsilon_off=0, alpha=0.05, b=5, reward_type=RewardType.Global, filename="42 agents | 6 nights | Global Reward")

# Problem 1c
runBarProblem(num_agents=42, K_nights=6, num_episodes=200, epsilon=0.1, epsilon_off=0, alpha=0.05, b=5, reward_type=RewardType.Difference, filename="42 agents | 6 nights | Difference Reward")

# def runExperiment(num_agents, k, b, reward_type, filename):
#     return runBarProblem(num_agents=num_agents, K_nights=k, num_episodes=200, epsilon_off=150, epsilon=0.1, alpha=0.1, b=b, reward_type=reward_type, filename=filename)

# for reward_type in [RewardType.Local, RewardType.Difference, RewardType.Global]:
#     runExperiment(num_agents=20, k=7, b=5, reward_type=reward_type, filename="20 agents | 7 nights | "+str(reward_type)+" Reward")
#     runExperiment(num_agents=50, k=6, b=4, reward_type=reward_type, filename="50 agents | 6 nights | "+str(reward_type)+" Reward")
