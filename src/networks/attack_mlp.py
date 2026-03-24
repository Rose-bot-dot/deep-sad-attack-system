import torch
import torch.nn as nn
import torch.nn.functional as F


class AttackMLP(nn.Module):
    def __init__(self, x_dim=77, rep_dim=32):
        super().__init__()
        self.rep_dim = rep_dim

        self.fc1 = nn.Linear(x_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, rep_dim)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x