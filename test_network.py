import os
import sys

# 把 src 目录加入 Python 搜索路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import torch
from networks.attack_mlp import AttackMLP

net = AttackMLP(x_dim=4, rep_dim=32)
x = torch.randn(5, 4)
y = net(x)

print("输出形状:", y.shape)