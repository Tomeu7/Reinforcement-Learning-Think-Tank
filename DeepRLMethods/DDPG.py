import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


class Critic(nn.Module):
    def __init__(self, state_space_size, action_space_size, num_neurons=64, num_layers=2):
        super(Critic, self).__init__()

        layers = [state_space_size + action_space_size] + [num_neurons] * num_layers
        self.hidden = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.hidden.append(nn.Linear(layers[i], layers[i + 1]))

        self.final_layer = nn.Linear(num_neurons, 1)
        self.activation = nn.ReLU()
        self.apply(init_weights)

        self.activation = nn.ReLU()
        self.apply(init_weights)

    def forward(self, x):
        for item in self.hidden:
            x = self.activation(item(x))
        x = self.final_layer(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_space_size, action_space_size, num_neurons=64, num_layers=2):
        super(Actor, self).__init__()

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


class DDPG:
    def __init__(self, state_space_size, action_space_size, learning_rate, gamma, batch_size, action_min, action_max,
                 gradient_clamp_bool="True", criterion_type="MSE", sigma=1):
        self.action_space_size = action_space_size
        self.Q = np.zeros((state_space_size, action_space_size))
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gradient_clamp_bool = gradient_clamp_bool

        self.Actor = Actor(state_space_size, action_space_size)
        self.Actor_target = Actor(state_space_size, action_space_size)
        self.Actor_target.load_state_dict(self.Actor.state_dict())

        self.Critic = Critic(state_space_size, action_space_size)
        self.Critic_target = Critic(state_space_size, action_space_size)
        self.Critic_target.load_state_dict(self.Critic.state_dict())

        if criterion_type == "MSE":
            self.criterion = nn.MSELoss()
        elif criterion_type == "HUBER":
            self.criterion = nn.SmoothL1Loss()
        self.optimizer_actor = torch.optim.Adam(params=self.Actor.parameters(), lr=learning_rate)
        self.optimizer_critic = torch.optim.Adam(params=self.Critic.parameters(), lr=learning_rate)

        self.memory = deque(maxlen=100000)
        self.batch_size = batch_size
        self.update_number = 0
        self.tau = 0.05
        self.sigma = sigma
        self.action_max = action_max
        self.action_min = action_min

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
            s = torch.tensor(np.concatenate(s_, 0)).float()
            a = torch.tensor(a_).float()
            r = torch.tensor(r_).float()
            s_next = torch.tensor(np.concatenate(s_next_, 0)).float()

            # Optimizing Critic

            with torch.no_grad():
                target_values = torch.zeros(self.batch_size)
                target_values[done_] = torch.transpose(self.Critic_target(torch.cat((s_next, self.Actor_target(s_next)),1)),0,1)
                target = (r + self.gamma * target_values).unsqueeze(1)
            self.optimizer_critic.zero_grad()
            obtained = self.Critic(torch.cat((s,a),1))
            loss_critic = self.criterion(obtained, target)
            loss_critic.backward()
            if self.gradient_clamp_bool:
                for param in self.Critic.parameters():
                    param.grad.data.clamp_(-1, 1)
            self.optimizer_critic.step()

            # Optimizing Actor

            self.optimizer_actor.zero_grad()
            loss_actor = torch.sum(self.Critic(torch.cat((s, self.Actor(s)),1)))
            loss_actor.backward()
            if self.gradient_clamp_bool:
                for param in self.Actor.parameters():
                    param.grad.data.clamp_(-1, 1)
            self.optimizer_critic.step()

            self.update_number += 1

    def update_target(self):
        """
        Updating target
        """
        new_state_dict = {}
        for key in self.Critic.state_dict().keys():
            new_state_dict[key] = self.tau * self.Critic.state_dict()[key] + (1.0 - self.tau) * self.Critic_target.state_dict()[key]
        self.Critic.load_state_dict(new_state_dict)

        new_state_dict = {}
        for key in self.Actor.state_dict().keys():
            new_state_dict[key] = self.tau * self.Actor.state_dict()[key] + (1.0 - self.tau) * \
                                  self.Actor_target.state_dict()[key]
        self.Actor.load_state_dict(new_state_dict)

    def act(self, s, eval=False):
        s = torch.Tensor(s)
        with torch.no_grad():
            a = self.Actor(s)
        if not eval:
            noise = np.random.normal(loc=0, scale=self.sigma, size=self.action_space_size)
            a = np.clip(a + noise, self.action_min, self.action_max)

        return a.numpy()