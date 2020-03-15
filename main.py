import gym
from TabularMethods.temporal_difference_methods import TDSolver
from TabularMethods.monte_carlo_methods import MCSolver
from DeepRLMethods.DeepQLearningMethods import DeepQlearning
from train_loop import train

if __name__=="__main__":

    #env = gym.make("FrozenLake-v0")
    env = gym.make("LunarLander-v2")

    learning_rate = 0.1
    gamma = 0.99
    initial_epsilon=1
    episodes = 100
    n = 20 # for dyna Q
    method = "DeepQlearning"  # MC or ["Qlearning", "DoubleQlearning", "ExpectedSarsa", "Sarsa"]
    batch_size = 32

    if method in ["Qlearning", "DoubleQlearning", "ExpectedSarsa", "Sarsa", "DynaQ"]:
        agent = TDSolver(method, env.observation_space.n, env.action_space.n, learning_rate, gamma, initial_epsilon, n)
    elif method in "MC":
        agent = MCSolver(method, env.observation_space.n, env.action_space.n, learning_rate,
                         gamma, initial_epsilon, 'First Visit')
    elif method in "DeepQlearning":
        agent = DeepQlearning(env.observation_space.shape[0], env.action_space.n, learning_rate, gamma, initial_epsilon, batch_size)
    else:
        raise NotImplementedError

    train_loop = train(env, agent, method)

    for ep in range(episodes):
        train_loop.train()
        if ep % 1 == 0:
            print(ep)

    train_loop.plot_metrics()
