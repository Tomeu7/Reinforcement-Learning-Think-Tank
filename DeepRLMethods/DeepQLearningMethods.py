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
    def __init__(self, state_space_size, action_space_size, num_neurons=64, num_layers=2):
        super(NeuralNetwork, self).__init__()

        layers = [state_space_size] + [num_neurons] * num_layers
        self.hidden = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.hidden.append(nn.Linear(layers[i], layers[i + 1]))

        self.final_layer = nn.Linear(num_neurons, action_space_size)
        self.activation = nn.ReLU()
        self.final_activation = nn.Softmax()
        self.apply(init_weights)

        self.activation = nn.ReLU()
        self.apply(init_weights)

    def forward(self, x):
        for item in self.hidden:
            x = self.activation(item(x))
        x = self.final_layer(x)
        return x


class DeepQlearning:
    def __init__(self, state_space_size, action_space_size, learning_rate, gamma,
                 epsilon, batch_size, update_type, gradient_clamp_bool, criterion_type = "MSE"):
        self.action_space_size = action_space_size
        self.Q = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_type = update_type
        self.gradient_clamp_bool = gradient_clamp_bool

        self.Q = NeuralNetwork(state_space_size, action_space_size)
        self.Q_target = NeuralNetwork(state_space_size, action_space_size)
        self.Q_target.load_state_dict(self.Q.state_dict())
        if criterion_type == "MSE":
            self.criterion = nn.MSELoss()
        elif criterion_type == "HUBER":
            self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(params=self.Q.parameters(), lr=learning_rate)

        self.memory = deque(maxlen=100000)
        self.batch_size = batch_size
        self.update_number = 0
        self.tau = 0.05

    def decrease_epsilon(self):
        self.epsilon -= 0.001
        self.epsilon = max(0.05, self.epsilon)
        print(self.epsilon)

    def push(self, s, a, r, s_next, done):
        self.memory.append((np.expand_dims(s, 0), a, r, np.expand_dims(s_next, 0), not done))

    def update(self):
        # based on optimize_model() function in https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # as initially I was doing a for loop to iterate over the batches
        if len(self.memory) > self.batch_size:

            mini_batches = random.sample(self.memory, self.batch_size)
            s_, a_, r_, s_next_, done_ = zip(*mini_batches)
            s = torch.tensor(np.concatenate(s_, 0))
            a = torch.tensor(a_)
            r = torch.tensor(r_)
            s_next = torch.tensor(np.concatenate(s_next_, 0))

            with torch.no_grad():
                target_values = torch.zeros(self.batch_size)
                target_values[done_] = torch.max(self.Q_target.forward(s_next), axis=1).values.detach()
                target = (r + self.gamma * target_values).unsqueeze(1)
            self.optimizer.zero_grad()
            obtained = self.Q.forward(s).gather(1, a.unsqueeze(1))
            loss = self.criterion(obtained, target)
            loss.backward()
            if self.gradient_clamp_bool:
                for param in self.Q.parameters():
                    param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            # for (s, a, r, s_next) in mini_batches:
            #    self.optimizer.zero_grad()
            #    target = r + self.gamma*torch.max(self.Q_target.forward(s_next))
            #    loss = self.criterion(self.Q.forward(s)[a], target)
            #    loss.backward()
            #    self.optimizer.step()

            self.update_number += 1

    def update_target(self):
        """
        Updating target
        """
        if self.update == "Soft":
            new_state_dict = {}
            for key in self.Q.state_dict().keys():
                new_state_dict[key] = self.tau * self.Q.state_dict()[key] + (1.0 - self.tau) * self.Q_target.state_dict()[key]
            self.Q_target.load_state_dict(new_state_dict)
        elif self.update == "Hard":
            self.Q_target.load_state_dict(self.Q.state_dict())

    def act(self, s):
        s = torch.Tensor(s)
        rand = random.random()
        if rand > self.epsilon:
            a = np.random.randint(0, self.action_space_size)
        else:
            a = np.argmax(self.Q.forward(s).detach().numpy())
        return a

