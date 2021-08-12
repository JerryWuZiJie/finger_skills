'''
Network of policy
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim, out_dim, hidden_layer, activation):

    # first layer
    layers = [nn.Linear(in_dim, hidden_layer[0]), activation()]
    # hidden layers
    for i in range(len(hidden_layer)-1):
        layers.append(nn.Linear(hidden_layer[i], hidden_layer[i+1]))
        layers.append(activation())
    # output layer
    layers.append(nn.Linear(hidden_layer[-1], out_dim))

    net = nn.Sequential(*layers)

    return net


class Network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layer=[64, 64], activation=nn.ReLU):
        super(Network, self).__init__()

        self.net = mlp(in_dim, out_dim, hidden_layer, activation)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        return self.net(obs)


class ActorNetwork(Network):
    def __init__(self, in_dim, out_dim, hidden_layer=[64, 64], activation=nn.ReLU):
        super().__init__(in_dim, out_dim, hidden_layer=hidden_layer, activation=activation)

        log_std = torch.full(size=(out_dim,), fill_value=0.05)
        self.log_std = nn.Parameter(log_std)


class CriticNetwork(Network):
    pass
