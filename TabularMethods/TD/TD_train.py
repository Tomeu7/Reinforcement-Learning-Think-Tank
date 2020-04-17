import matplotlib.pyplot as plt
import numpy as np

class TDtrain:
    def __init__(self, env, agent, method):
        self.env = env
        self.agent = agent

        self.episode_reward_list = []
        self.episode_epsilon_list = []
        self.episode_td_error_list = []
        self.episode = 0
        self.score = 0

        if method in ["Qlearning", "DoubleQlearning", "ExpectedSarsa", "Sarsa"]:
            self.train = self.td_training
        elif method in "DynaQ":
            self.train = self.model_based_td_training
        else:
            raise NotImplementedError

        self.method = method

    def action_selection(self, s, a_next):
        if self.method == "Sarsa" and a_next is not None:
            a = a_next
        else:
            a = self.agent.act(s)

        return a

    def td_training(self):
        episode_reward = 0

        s = self.env.reset()
        done = False
        steps = 0
        a_next = None
        while not done:
            a = self.action_selection(s, a_next)
            s_next, r, done, _ = self.env.step(a)
            if self.method == "Sarsa":
                a_next = self.agent.act(s_next)
            self.agent.update(s, a, r, s_next, a_next)
            s = s_next
            episode_reward += r
            steps += 1

        self.agent.decrease_epsilon()
        self.update_metrics(episode_reward, self.agent.solver.epsilon)

    def model_based_td_training(self):
        episode_reward = 0

        s = self.env.reset()
        done = False
        steps = 0
        while not done:
            a = self.agent.act(s)
            s_next, r, done, _ = self.env.step(a)
            self.agent.update(s, a, r, s_next, None)
            self.agent.learn_model(s, a, r, s_next)
            self.agent.dream()

            s = s_next
            episode_reward += r
            steps += 1

        self.update_metrics(episode_reward, self.agent.solver.epsilon)

    def update_metrics(self, episode_reward, episode_epsilon, episode_td_error=0):
        self.episode_reward_list.append(episode_reward)
        self.episode_epsilon_list.append(episode_epsilon)
        self.episode_td_error_list.append(episode_td_error)

    def plot_metrics(self):
        plt.plot((np.convolve(self.episode_reward_list, np.ones(20), 'valid')/20), label=self.method)
        plt.title("TD Methods")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward over last 20 episodes")
