import torch
from torch import nn


class TanhNormal:
    def __init__(self, mean, std):
        self.dist = normal.Normal(mean, std)

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
        return samples, mu_log_probs - tanh_correction

    @property
    def mean(self):
        return torch.tanh(self.dist.mean)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, config):
        super().__init__()
        network_layers = []

        nodes = input_dim + action_dim
        for layer in config.layers:
            network_layers.append(nn.Linear(nodes, layer))
            network_layers.append(nn.ReLU(inplace=False))
            nodes = layer

        network_layers.append(nn.Linear(nodes, 1))

        self.net = nn.Sequential(*network_layers)

    def forward(self, x, a):
        inp = torch.cat((x, a), -1)
        return self.net(inp)


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, config):
        super().__init__()

        output_dim = output_dim

        network_layers = []
        nodes = input_dim
        for layer in config.layers:
            network_layers.append(nn.Linear(nodes, layer))
            network_layers.append(nn.ReLU())
            nodes = layer
        latent_nodes = nodes
        self.core_net = nn.Sequential(*network_layers)

        network_layers = []
        for layer in config.mean_layers:
            network_layers.append(nn.Linear(nodes, layer))
            network_layers.append(nn.ReLU())
            nodes = layer
        network_layers.append(nn.Linear(nodes, output_dim))
        self.mean_net = nn.Sequential(*network_layers)

        nodes = latent_nodes

        network_layers = []
        for layer in config.std_layers:
            network_layers.append(nn.Linear(nodes, layer))
            network_layers.append(nn.ReLU())
            nodes = layer

        network_layers.append(nn.Linear(nodes, output_dim))
        self.std_net = nn.Sequential(*network_layers)

    def forward(self, x):
        latent = self.core_net(x)
        mean = self.mean_net(latent)
        std = torch.nn.functional.softplus(self.std_net(latent)) + 1e-3
        assert torch.all(std > 0.0), std
        return mean, std

    def sample_action(self, x):
        mean, std = self(x)
        dist = TanhNormal(mean, std)
        act, logprob = dist.rsample()
        return act, logprob

    def get_deterministic_action(self, x):
        mean, _ = self(x)
        return mean

    def get_action_dist(self, x):
        mean, std = self(x)
        dist = TanhNormal(mean, std)
        return dist
