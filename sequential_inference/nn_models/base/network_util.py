from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dists
from torch.distributions.kl import kl_divergence


def calc_output_shape(conv_layers, input_size):
    return tuple(conv_layers(torch.rand(1, *input_size)).shape)


def create_coord_buffer(patch_shape):
    ys = torch.linspace(-1, 1, patch_shape[0])
    xs = torch.linspace(-1, 1, patch_shape[1])
    xv, yv = torch.meshgrid(ys, xs)
    coord_map = torch.stack((xv, yv)).unsqueeze(0)
    return coord_map


def alternate_inverse(theta):
    inv_theta = torch.zeros_like(theta)
    inv_theta[:, 0, 0] = 1 / theta[:, 0, 0]
    inv_theta[:, 1, 1] = 1 / theta[:, 1, 1]
    inv_theta[:, 0, 2] = -theta[:, 0, 2] / theta[:, 0, 0]
    inv_theta[:, 1, 2] = -theta[:, 1, 2] / theta[:, 1, 1]
    return inv_theta


def invert(x, theta, image_shape, padding="zeros"):
    inverse_theta = alternate_inverse(theta)
    if x.size()[1] == 1:
        size = [x.size()[0], 1, *image_shape]
    elif x.size()[1] == 3:
        size = [x.size()[0], 3, *image_shape]
    else:
        raise ValueError("invalid tensor shape")
    grid = F.affine_grid(inverse_theta, size)
    x = F.grid_sample(x, grid, padding_mode=padding)

    return x


def differentiable_sampling(mean, sigma, prior_sigma, samples=1):
    dist = dists.Normal(mean, sigma)
    dist_0 = dists.Normal(0.0, prior_sigma)
    z = dist.rsample((samples,))  # type: ignore
    kl_z = dists.kl_divergence(dist, dist_0)
    return z, torch.sum(kl_z, -1)


def calc_kl_divergence(
    p_list: List[torch.distributions.Distribution],
    q_list: List[torch.distributions.Distribution],
) -> torch.Tensor:
    assert len(p_list) == len(q_list)
    kld = []
    for i in range(len(p_list)):
        # (N, L) shaped array of kl divergences.
        kld.append(kl_divergence(p_list[i], q_list[i]))
    kld = torch.stack(kld, 1)
    return kld


def reconstruction_likelihood(x, recon, sigma):
    dist = dists.Normal(x, sigma)
    p_x = dist.log_prob(recon)
    return p_x


def kl_mask(mask_pred, mask):
    tr_masks = mask.view(mask.size()[0], -1)
    tr_mask_preds = mask_pred.view(mask_pred.size()[0], -1)

    q_masks = dists.Bernoulli(probs=tr_masks)
    q_masks_recon = dists.Bernoulli(probs=tr_mask_preds)
    kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
    return kl_masks


def transform(x, grid, theta):
    x = F.grid_sample(x, grid)
    return x


def center_of_mass(mask, device="cuda"):
    grids = [
        torch.Tensor(grid).to(device)
        for grid in np.ogrid[[slice(0, i) for i in mask.shape[-2:]]]
    ]
    norm = torch.sum(mask, [-2, -1])
    return torch.stack(
        [torch.sum(mask * grids[d], [-2, -1]) / norm for d in range(2)], -1
    )


class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


# Modified from https://github.com/juliusfrost/dreamer-pytorch
class FreezeParameters:
    def __init__(self, parameters: List[nn.parameter.Parameter]):
        self.parameters = parameters
        self.param_states = [p.requires_grad for p in self.parameters]

    def __enter__(self):
        for param in self.parameters:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.parameters):
            param.requires_grad = self.param_states[i]
