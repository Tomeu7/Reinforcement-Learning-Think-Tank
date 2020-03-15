import numpy as np
import random


class MCSolver:
    def __init__(self, method,observation_space, action_space, learning_rate, gamma, epsilon, m):
        if method == "MC":
            self.solver = OnPolicyMonteCarlo(observation_space, action_space, learning_rate, gamma, epsilon, m)
        else:
            raise NotImplementedError

    def act(self, s):
        a = self.solver.act(s)
        self.decrease_epsilon()
        return a

    def decrease_epsilon(self):
        self.solver.epsilon -= 0.000001
        self.solver.epsilon = max(0.001, self.solver.epsilon)

    def update(self, trajectory, ret):
        self.solver.update(trajectory, ret)


class OnPolicyMonteCarlo:
    def __init__(self,state_space_size, action_space_size, learning_rate, gamma, epsilon=1, method="First visit"):
        self.action_space_size = action_space_size
        self.Q = np.zeros((state_space_size, action_space_size))
        self.N = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.method = method

    def update(self, trajectory, ret):
        G = ret
        for s,a,r in trajectory[-1:]:

            if self.method=="Every Visit":
                self.N[s, a] +=1
            elif self.method=="First Visit":
                self.N[s,a] = 1

            self.Q[s, a] += 1/self.N[s,a]*(G - self.Q[s,a])

            G = self.gamma*G + r

    def act(self, s):
        rand = random.random()
        if rand > self.epsilon:
            a = np.random.randint(0, self.action_space_size)
        else:
            a = np.argmax(self.Q[s, :])

        return a