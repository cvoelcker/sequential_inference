from typing import Tuple
import torch
from torch import distributions as td
from torch import nn
from torch.nn import Linear, Conv2d, ConvTranspose2d, GRUCell

from sequential_inference.nn_models.base.network_util import Reshape


# Encoder, part of PlaNET
class ConvEncoder(nn.Module):
    """Standard Convolutional Encoder for BADreamer. This encoder is used
    to encode images from an environment into a latent state for the
    BARSSM model
    """

    def __init__(
        self,
        depth: int = 32,
        latent_dim: int = 64,
        shape: Tuple[int, int, int] = (3, 64, 64),
        additional_channels: int = 0,
        return_scale: bool = True,
    ):
        """Initializes Conv Encoder
        Args:
            depth (int): Number of channels in the first conv layer
            act (Any): Activation for Encoder, default ReLU
            shape (List): Shape of observation input
        """
        super().__init__()
        self.act = nn.ReLU
        self.depth = depth
        self.shape = shape
        self.latent_dim = latent_dim
        self.return_scale = return_scale

        init_channels = self.shape[0] + additional_channels
        self.out_channels = 2 * self.latent_dim if return_scale else self.latent_dim

        self.layers = [
            Conv2d(init_channels, self.depth, 4, stride=2),
            self.act(),
            Conv2d(self.depth, 2 * self.depth, 4, stride=2),
            self.act(),
            Conv2d(2 * self.depth, 4 * self.depth, 4, stride=2),
            self.act(),
            Conv2d(4 * self.depth, 8 * self.depth, 4, stride=2),
            self.act(),
            Reshape((-1, 32 * self.depth)),
            Linear(32 * self.depth, self.out_channels),
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # Flatten to [batch*horizon, 3, 64, 64] in loss function
        # Returns tensor of shape [batch, horizon, 32 * depth]
        orig_shape = list(x.size())
        x = x.view(-1, *(orig_shape[-3:]))
        x = self.model(x)

        if self.return_scale:
            mean, scale = torch.chunk(x, 2, -1)
            mean = mean.view(orig_shape[:2] + [self.latent_dim])
            scale = scale.view(orig_shape[:2] + [self.latent_dim])
            return mean, scale
        return x.view(orig_shape[:2] + [self.latent_dim])


# Decoder, part of PlaNET
class ConvDecoder(nn.Module):
    """Convolutional Decoder for BADreamer.
    This decoder is used to decode images from the latent state and belief generated
    by BARSSM. This is used in calculating loss and logging gifs for imagined trajectories.
    """

    def __init__(
        self,
        latent_dim: int,
        depth: int = 32,
        shape: Tuple[int, int, int] = (3, 64, 64),
    ):
        """Initializes a ConvDecoder instance.
        Args:
            input_size (int): Input size
            depth (int): Number of channels in the first conv layer
            act (Any): Activation, default ReLU
            shape (List): Shape of observation input
        """
        super().__init__()
        self.act = nn.ReLU
        self.depth = depth
        self.shape = shape

        self.layers = [
            Linear(latent_dim, 32 * self.depth),
            Reshape([-1, 32 * self.depth, 1, 1]),
            ConvTranspose2d(32 * self.depth, 4 * self.depth, 5, stride=2),
            self.act(),
            ConvTranspose2d(4 * self.depth, 2 * self.depth, 5, stride=2),
            self.act(),
            ConvTranspose2d(2 * self.depth, self.depth, 6, stride=2),
            self.act(),
            ConvTranspose2d(self.depth, self.shape[0], 6, stride=2),
        ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        # x is [batch, horizon, input_size]
        orig_shape = list(x.size())
        x = self.model(x)

        # self.model reshapes x with Reshape layer, we need to return [batch, horizon ,...] structure
        reshape_size = orig_shape[:-1] + list(self.shape)
        mean = x.view(*reshape_size)

        # Equivalent to making a multivariate diag
        return mean
