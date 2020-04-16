import numpy as np
import random
from collections import deque


class TDSolver:
    def __init__(self, method,observation_space, action_space, learning_rate, gamma, epsilon, n=None):
        if method == "Qlearning":
            self.solver = Qlearning(observation_space, action_space, learning_rate, gamma, epsilon)
        elif method == "Sarsa":
            self.solver = Sarsa(observation_space, action_space, learning_rate, gamma, epsilon)
        elif method == "DoubleQlearning":
            self.solver = DoubleQlearning(observation_space, action_space, learning_rate, gamma, epsilon)
        elif method == "ExpectedSarsa":
            self.solver = ExpectedSarsa(observation_space, action_space, learning_rate, gamma, epsilon)
        elif method == "DynaQ":
            self.solver = DynaQ(observation_space, action_space, learning_rate, gamma, epsilon, n)
        else:
            raise NotImplementedError

    def act(self, s):
        a = self.solver.act(s)
        self.decrease_epsilon()
        return int(a)

    def decrease_epsilon(self):
        self.solver.epsilon -= 0.000001
        self.solver.epsilon = max(0.001, self.solver.epsilon)

    def update(self, s, a, r, s_next, a_next):
        self.solver.update(s, a, r, s_next, a_next)

    def learn_model(self,s, a, r, s_next):
        self.solver.model.add(s, a, r, s_next)

    def dream(self):
        self.solver.dream()


class Qlearning:
    def __init__(self,state_space_size, action_space_size, learning_rate, gamma, epsilon):
        self.action_space_size = action_space_size
        self.Q = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

    def update(self, s, a, r, s_next, _):
        self.Q[s, a] += self.learning_rate*(r+self.gamma*np.max(self.Q[s_next,:]) - self.Q[s,a])

    def act(self, s):
        rand = random.random()
        if rand > self.epsilon:
            a = np.random.randint(0, self.action_space_size)
        else:
            a = np.argmax(self.Q[s, :])

        return a


class Sarsa:
    def __init__(self, state_space_size, action_space_size, learning_rate, gamma, epsilon = 1):
        self.Q = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space_size = action_space_size

    def update(self, s, a, r, s_next, a_next):
        self.Q[s, a] += self.learning_rate * (r + self.gamma * self.Q[s_next, a_next] - self.Q[s, a])

    def act(self, s):
        rand = random.random()
        if rand > self.epsilon:
            a = np.random.randint(0, self.action_space_size)
        else:
            a = np.argmax(self.Q[s, :])

        return a


class DoubleQlearning:
    def __init__(self,state_space_size, action_space_size, learning_rate, gamma, epsilon):
        self.Q1 = np.zeros((state_space_size, action_space_size))
        self.Q2 = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space_size = action_space_size

    def update(self, s, a, r, s_next, _):

        coin = random.random()

        if coin > 0.5:
            self.Q1[s, a] += self.learning_rate*(r+self.gamma*np.max(self.Q2[s_next,:]) - self.Q1[s, a])
        else:
            self.Q2[s, a] += self.learning_rate * (r + self.gamma * np.max(self.Q1[s_next, :]) - self.Q2[s, a])

    def act(self, s):
        rand = random.random()
        if rand > self.epsilon:
            a = np.random.randint(0, self.action_space_size)
        else:
            a = np.argmax((self.Q1+self.Q2)[s, :])

        return a


class ExpectedSarsa:
    def __init__(self, state_space_size, action_space_size, learning_rate, gamma, epsilon = 1):
        self.Q = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.action_space_size = action_space_size

    def update(self, s, a, r, s_next, _):

        expectation_of_actions = (1-self.epsilon)*np.max(self.Q[s_next,:])
        expectation_of_actions += self.epsilon*np.sum(self.Q[s_next,:])
        self.Q[s, a] += self.learning_rate * (r + self.gamma * expectation_of_actions - self.Q[s, a])

    def act(self, s):
        rand = random.random()
        if rand > self.epsilon:
            a = np.random.randint(0, self.action_space_size)
        else:
            a = np.argmax(self.Q[s, :])

        return a


class DynaQ:
    def __init__(self, state_space_size, action_space_size, learning_rate, gamma, epsilon, n=20):
        self.action_space_size = action_space_size
        self.Q = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.model = self.Model(state_space_size, action_space_size)
        self.n = n
        self.sa_list = deque()

    def update(self, s, a, r, s_next, _):
        self.Q[s, a] += self.learning_rate*(r+self.gamma*np.max(self.Q[s_next, :]) - self.Q[s, a])

    def dream(self):
        for i in range(self.n):
            (s_imagined, a_imagined) = self.model.sample()
            r_imagined, s_next_imaged = self.model.step(s_imagined, a_imagined)
            self.update(s_imagined, a_imagined, r_imagined, s_next_imaged, None)

    def act(self, s):
        rand = random.random()
        if rand > self.epsilon:
            a = np.random.randint(0, self.action_space_size)
        else:
            a = np.argmax(self.Q[s, :])

        return a

    class Model:
        def __init__(self, n_states, n_actions):
            self.transitions = np.zeros((n_states, n_actions), np.uint32)
            self.rewards = np.zeros((n_states, n_actions))
            # self.visited = np.zeros((n_states, n_actions), dtype=bool)
            self.visited = list()


        def add(self, s, a, r, s_prime):
            self.visited.append((s, a))
            self.visited = list(set(self.visited))
            self.transitions[s, a] = s_prime
            self.rewards[s, a] = r
            #self.visited[s, a] = True


        def sample(self):
            """ Return random state, action"""
            # Random visited state
            s, a = self.visited[random.randint(0,len(self.visited)-1)]

            return s, a

        def step(self, s, a):
            """ Return state_prime and reward for state-action pair"""
            s_prime = self.transitions[s, a]
            r = self.rewards[s, a]
            return r, s_prime