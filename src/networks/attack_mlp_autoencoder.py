import torch
import torch.nn as nn
import torch.nn.functional as F


class AttackMLPAutoencoder(nn.Module):
    def __init__(self, x_dim=77, rep_dim=32):
        super().__init__()
        self.rep_dim = rep_dim

        # encoder
        self.enc_fc1 = nn.Linear(x_dim, 128)
        self.enc_fc2 = nn.Linear(128, 64)
        self.enc_fc3 = nn.Linear(64, rep_dim)

        # decoder
        self.dec_fc1 = nn.Linear(rep_dim, 64)
        self.dec_fc2 = nn.Linear(64, 128)
        self.dec_fc3 = nn.Linear(128, x_dim)

    def forward(self, x):
        x = x.float()

        x = F.relu(self.enc_fc1(x))
        x = F.relu(self.enc_fc2(x))
        z = self.enc_fc3(x)

        x = F.relu(self.dec_fc1(z))
        x = F.relu(self.dec_fc2(x))
        x = self.dec_fc3(x)

        return x