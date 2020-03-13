import gym
from Part1.temporal_difference_methods import TDSolver
from Part1.monte_carlo_methods import MCSolver
from Part1.train_loop import train

if __name__=="__main__":

    env = gym.make("FrozenLake-v0")

    learning_rate = 0.1
    gamma = 0.99
    initial_epsilon=1
    episodes = 5000
    n = 20 # for dyna Q
    method = "DynaQ"  # MC or ["Qlearning", "DoubleQlearning", "ExpectedSarsa", "Sarsa"]

    if method in ["Qlearning", "DoubleQlearning", "ExpectedSarsa", "Sarsa", "DynaQ"]:
        agent = TDSolver(method, env.observation_space.n, env.action_space.n, learning_rate, gamma, initial_epsilon, n)
    elif method in "MC":
        agent = MCSolver(method, env.observation_space.n, env.action_space.n, learning_rate,
                         gamma, initial_epsilon, 'First Visit')
    else:
        raise NotImplementedError

    train_loop = train(env, agent, method)

    for ep in range(episodes):
        train_loop.train()
        if ep % 1000 == 0:
            print(ep)

    train_loop.plot_metrics()
