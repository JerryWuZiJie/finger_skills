'''
Network of policy
'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layer=[64, 64]):
        super(Network, self).__init__()
        self.layer0 = nn.Linear(in_dim, 64)
        self.layer1 = nn.Linear(64, 64)
        self.layer2 = nn.Linear(64, out_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        # a0 = nn.ReLU(self.layer0(obs))
        # a1 = nn.ReLU(self.layer1(a0))
        a0 = F.relu(self.layer0(obs))
        a1 = F.relu(self.layer1(a0))
        out = self.layer2(a1)

        return out
