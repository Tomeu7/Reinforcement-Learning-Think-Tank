import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class NeuralNetwork(nn.Module):
    def __init__(self, state_space_size, action_space_size, num_neurons=500, num_layers=2):
        super(NeuralNetwork, self).__init__()

        layers = [state_space_size] + [num_neurons] * num_layers
        self.hidden = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.hidden.append(nn.Linear(layers[i], layers[i + 1]))

        self.final_layer = nn.Linear(num_neurons, action_space_size)
        self.activation = nn.ReLU()
        self.apply(init_weights)

        self.activation = nn.ReLU()
        self.apply(init_weights)

    def forward(self, x):

        for item in self.hidden:
            x = self.activation(item(x))
        x = self.final_layer(x)
        return x


class DeepQlearning:
    def __init__(self,state_space_size, action_space_size, learning_rate, gamma, epsilon, batch_size):
        self.action_space_size = action_space_size
        self.Q = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = NeuralNetwork(state_space_size, action_space_size)
        self.Q_target = NeuralNetwork(state_space_size, action_space_size)
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(params=self.Q.parameters(), lr=learning_rate)

        self.memory = deque(maxlen=100000)
        self.batch_size = batch_size
        self.update_number = 0

    def push(self, s, a, r, s_next):
        self.memory.append((s, a, r, s_next))

    def update(self):
        if len(self.memory)>self.batch_size:
            mini_batches = random.sample(self.memory, self.batch_size)
            for (s, a, r, s_next) in mini_batches:
                self.optimizer.zero_grad()
                target = r + self.gamma*torch.max(self.Q_target.forward(s_next))
                loss = self.criterion(self.Q.forward(s)[a], target)
                loss.backward()
                self.optimizer.step()
            self.update_number += self.batch_size

            if self.update_number % 1000 == 0:
                self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, s):
        rand = random.random()
        if rand > self.epsilon:
            a = np.random.randint(0, self.action_space_size)
        else:
            a = np.argmax(self.Q.forward(s).detach().numpy())

        return a

