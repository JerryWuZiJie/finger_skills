'''
Network of policy
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layer=[64, 64], activation=nn.ReLU):
        super(Network, self).__init__()

        # first layer
        layers = [nn.Linear(in_dim, hidden_layer[0]), activation()]
        # hidden layers
        for i in range(len(hidden_layer)-1):
            layers.append(nn.Linear(hidden_layer[i], hidden_layer[i+1]))
            layers.append(activation())
        # output layer
        layers.append(nn.Linear(hidden_layer[-1], out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        return self.net(obs)
