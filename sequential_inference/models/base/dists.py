import math

import torch
from torch import nn
from torch.distributions import normal


class TanhNormal:
    def __init__(self, mean, std, multiplier=1.0):
        self.dist = normal.Normal(mean, std)
        self.multiplier = multiplier

    def sample(self):
        return torch.tanh(self.dist.sample())

    def rsample(self):
        norm_samples = self.dist.rsample()
        samples = torch.tanh(norm_samples)

        mu_log_probs = torch.sum(self.dist.log_prob(norm_samples), -1, keepdim=True)
        # tanh_correction = torch.sum(1 - torch.tanh(norm_samples) ** 2, -1, keepdim=True)
        # from https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        tanh_correction = torch.sum(
            2.0
            * (
                math.log(2.0)
                - norm_samples
                - nn.functional.softplus(-2.0 * norm_samples)
            ),
            -1,
            keepdim=True,
        )
        return samples * self.multiplier, mu_log_probs - tanh_correction

    @property
    def mean(self):
        return torch.tanh(self.dist.mean) * self.multiplier
