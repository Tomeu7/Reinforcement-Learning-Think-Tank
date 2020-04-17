import gym
from TabularMethods.TD.TD_methods import TDSolver
from TabularMethods.MC.MC_methods import MCSolver
from DeepRLMethods.DeepQLearningMethods import DeepQlearning
from DeepRLMethods.DDPG import DDPG
import matplotlib.pyplot as plt
import numpy as np

if __name__=="__main__":

    env = gym.make("FrozenLake-v0")
    env = gym.make("Taxi-v3")
    #env = gym.make("LunarLander-v2")
    #env = gym.make("LunarLanderContinuous-v2")



    episodes = 10001
    n = 20 # for dyna Q
    methods = ["Qlearning","ExpectedSarsa","Sarsa","DoubleQlearning"]  # MC or ["Qlearning", "DoubleQlearning", "ExpectedSarsa", "Sarsa"] or DeepQlearning or DDPG
    # methods = ["MC"]
    batch_size = 32

    for method in methods:
        if method in ["Qlearning", "Sarsa", "DoubleQlearning", "ExpectedSarsa", "DynaQ"]:
            from TabularMethods.TD.TD_train import TDtrain as train
            agent = TDSolver(method, env.observation_space.n, env.action_space.n, n)
            #agent = TDSolver(method, env.observation_space.n, env.action_space.n, n)
        elif method in "MC":
            from TabularMethods.MC.MC_train import MCtrain as train
            agent = MCSolver(method, env.observation_space.n, env.action_space.n)
        elif method in "DeepQlearning":
            agent = DeepQlearning(env.observation_space.shape[0], env.action_space.n, 5e-4, gamma, initial_epsilon, batch_size)
        elif method in "DDPG":
            agent = DDPG(env.observation_space.shape[0], env.action_space.shape[0], 5e-4, gamma, batch_size, env.action_space.low, env.action_space.high)
        else:
            raise NotImplementedError

        train_loop = train(env, agent, method)

        for ep in range(episodes):
            train_loop.train()
            if ep % 100 == 0:
                print(ep, method, agent.solver.epsilon, np.average(train_loop.episode_reward_list[-100:]))

            if ep % (episodes-1) == 0 and ep > 1:
                train_loop.plot_metrics()

    plt.legend()
    plt.show()