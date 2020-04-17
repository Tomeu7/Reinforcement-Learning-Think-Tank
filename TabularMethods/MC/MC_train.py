import matplotlib.pyplot as plt
import numpy as np

class MCtrain:
    def __init__(self, env, agent, method):
        self.env = env
        self.agent = agent

        self.episode_reward_list = []
        self.episode_epsilon_list = []
        self.episode_td_error_list = []
        self.episode = 0
        self.score = 0
        if method in "MC":
            self.train = self.mc_training
        else:
            raise NotImplementedError

        self.method = method

    def action_selection(self, s, a_next):
        if self.method == "Sarsa" and a_next is not None:
            a = a_next
        else:
            a = self.agent.act(s)

        return a

    def mc_training(self):
        episode_reward = 0
        s = self.env.reset()
        done = False
        steps = 0
        trajectory = []
        while not done:
            a = self.agent.act(s)
            s_next, r, done, _ = self.env.step(a)
            trajectory.append((s, a, r))
            s = s_next
            episode_reward += r
            steps += 1
        self.agent.decrease_epsilon()
        self.agent.update(trajectory)
        self.update_metrics(episode_reward, self.agent.solver.epsilon)

    def update_metrics(self, episode_reward, episode_epsilon, episode_td_error=0):
        self.episode_reward_list.append(episode_reward)
        self.episode_epsilon_list.append(episode_epsilon)
        self.episode_td_error_list.append(episode_td_error)

    def plot_metrics(self):
        #plt.plot(self.episode_reward_list, label=self.method)
        plt.plot((np.convolve(self.episode_reward_list, np.ones(20), 'valid') / 20), label=self.method)
        plt.title("Monte Carlo Methods")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward over last 20 episodes")

