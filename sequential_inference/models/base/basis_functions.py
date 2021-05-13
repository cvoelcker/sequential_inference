import torch
from torch import nn

from torch.distributions import normal, uniform

import math


class RandomFourierFeatures(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = normal.Normal(0.0, 1.0).sample(in_dim, out_dim)
        self.b = uniform.Uniform(0.0, 1.0).sample(out_dim) * 1 * math.pi

        self.w = nn.parameter.Parameter(self.w, requires_grad=False)
        self.b = nn.parameter.Parameter(self.b, requires_grad=False)

        self.out_dim = out_dim

    def forward(self, x):
        return math.sqrt(2 / self.out_dim) * torch.cos(x @ self.w + self.b)
