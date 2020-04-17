import numpy as np
import random
import json


class MCSolver:
    def __init__(self, method,observation_space, action_space, mc_config_path="TabularMethods/MC/MC_config.cfg"):

        with open(mc_config_path, 'r') as datafile:
            config = json.load(datafile)

        gamma = float(config["gamma"])
        m = str(config["visit"])
        epsilon_min = float(config['epsilon_min'])
        number_steps_decay_epsilon = float(config['number_steps_decay_epsilon'])

        self.epsilon_decay = (1 - epsilon_min)/number_steps_decay_epsilon

        if method == "MC":
            self.solver = OnPolicyMonteCarlo(observation_space, action_space, gamma, m)
        else:
            raise NotImplementedError

    def act(self, s):
        a = self.solver.act(s)
        return a

    def decrease_epsilon(self):
        self.solver.epsilon -= self.epsilon_decay
        self.solver.epsilon = max(0.001, self.solver.epsilon)

    def update(self, trajectory):
        self.solver.update(trajectory)


class OnPolicyMonteCarlo:
    def __init__(self,state_space_size, action_space_size, gamma, method="First visit"):
        self.action_space_size = action_space_size
        self.Q = np.zeros((state_space_size, action_space_size))
        self.N = np.zeros((state_space_size, action_space_size))
        self.gamma = gamma
        self.epsilon = 1
        self.method = method


    def update(self, trajectory):
        G = 0

        for i in range(len(trajectory)):

            s, a, r = list(reversed(trajectory))[i]
            G = self.gamma * G + r

            if self.method == "Every Visit":
                self.N[s, a] += 1
                self.Q[s, a] += 1 / self.N[s, a] * (G - self.Q[s, a])
            elif self.method == "First Visit":

                if not (s, a) in [(s_a_[0], s_a_[1]) for s_a_ in trajectory[0:i]]:
                    self.N[s, a] += 1
                    self.Q[s, a] += 1/self.N[s, a]*(G - self.Q[s, a])





    def act(self, s):
        rand = random.random()
        if rand < self.epsilon:
            a = np.random.randint(0, self.action_space_size)
        else:
            #a = np.argmax(self.Q[s, :])
            a = np.random.choice(np.flatnonzero(self.Q[s, :] == self.Q[s, :].max()))

        return a