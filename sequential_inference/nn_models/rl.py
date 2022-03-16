import numpy as np
import torch
from torch import nn

from sequential_inference.nn_models.base.base_nets import (
    Flatten,
    TanhGaussian,
    create_mlp,
    create_conv_net,
)
from sequential_inference.nn_models.base.network_util import calc_output_shape


class MultiOutputHead(nn.Module):
    def __init__(self, n_tasks, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.n_tasks = n_tasks
        self.networks = [
            create_mlp(input_dim, hidden_dim, output_dim) for _ in range(n_tasks)
        ]

    def forward(self, x, task_id=None):
        if task_id is None:
            return torch.stack([c(x) for c in self.networks], dim=-1)
        return self.networks[task_id](x)


class SharedEmbeddingMultiOutputNet(nn.Module):
    def __init__(self, n_tasks, input_dim, shared_dim, hidden_dim, output_dim):
        super().__init__()
        self.n_tasks = n_tasks
        self.embedding = create_mlp(input_dim, hidden_dim[0], shared_dim)
        self.multi_output_head = MultiOutputHead(
            n_tasks, hidden_dim[0], hidden_dim[1:], output_dim
        )

    def forward(self, x, task_id=None):
        e = self.embedding(x)
        return self.multi_output_head(e, task_id)


class ConvMultiOutputNet(nn.Module):
    def __init__(
        self, n_tasks, input_dim, conv_layers, latent_rep_dim, hidden_dim, output_dim
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.embedding = create_conv_net(input_dim, latent_rep_dim, conv_layers)
        latent_size = np.prod(calc_output_shape(self.embedding, input_dim))
        self.embedding = nn.Sequential(self.embedding, Flatten())
        self.multi_output_head = MultiOutputHead(
            n_tasks, latent_size, hidden_dim, output_dim
        )

    def forward(self, x, task_id=None):
        e = self.embedding(x)
        return self.multi_output_head(e, task_id)
