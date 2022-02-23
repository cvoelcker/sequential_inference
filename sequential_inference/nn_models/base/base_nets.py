"""Contains very basic neural network components and some quick
factory functions for construction

Adapted from every repo ever ;)

@author Claas
"""

from typing import List, Sequence, Tuple, Union
from sequential_inference.nn_models.base.dists import TanhNormal
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dists

import sequential_inference.nn_models.base.network_util as net_util


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def create_conv_layers(
    input_size, output_size, layer_sizes, max_pool=True, padding=True, kernel_size=3
):
    if padding:
        padding = ((kernel_size - 1) // 2, (kernel_size - 1) // 2)
    else:
        padding = (0, 0)

    layers = []

    for layer in layer_sizes:
        layers.append(nn.Conv2d(input_size, layer, kernel_size, padding=padding))
        layers.append(nn.ReLU(inplace=True))
        if max_pool:
            layers.append(nn.MaxPool2d(2, stride=2))
        input_size = layer
    layers.append(nn.Conv2d(input_size, output_size, kernel_size, padding=padding))

    return nn.Sequential(*layers)


def create_mlp(
    input_size: Union[Sequence[int], int],
    output_size: Union[Sequence[int], int],
    layer_sizes: Sequence[int],
):
    layers = []

    if isinstance(input_size, Sequence):
        assert len(input_size) == 1, "Input shape must be a single dimension"
        input_size = input_size[0]
    else:
        input_size = input_size
    if isinstance(output_size, Sequence):
        assert len(output_size) == 1, "Output shape must be a single dimension"
        output_size = output_size[0]
    else:
        output_size = output_size

    for layer in layer_sizes:
        layers.append(nn.Linear(input_size, layer))
        layers.append(nn.SiLU(inplace=True))
        input_size = layer
    layers.append(nn.Linear(input_size, output_size))

    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=(1,)):
        super().__init__()
        if type(input_dim) == torch.Size:
            input_dim = input_dim[-1]
        if type(output_dim) == torch.Size:
            output_dim = output_dim[-1]
        self.net = create_mlp(input_dim, output_dim, hidden_units)

    def forward(self, x):
        return self.net(x)


class EncoderNet(nn.Module):
    """
    General parameterized encoding architecture for image components
    """

    def __init__(
        self,
        img_shape: Tuple[int, int, int],
        output_size: int,
        layer_sizes: List[int],
        **kwargs
    ):
        super().__init__()
        input_size = img_shape[0]

        self.latent_dim = output_size
        self.img_shape = img_shape[1:]

        self.network = create_conv_layers(
            input_size + 3,
            output_size,
            layer_sizes,
            max_pool=True,
            padding=True,
            kernel_size=3,
        )

        num_layers = len(layer_sizes)

        coord_map = net_util.create_coord_buffer(self.img_shape)
        self.register_buffer("coord_map_const", coord_map)

        # computes the number of flattened output neurons in the convolutions
        self.conv_size = int(
            self.latent_dim
            * self.img_shape[0]
            * self.img_shape[1]
            / (2 ** (2 * num_layers))
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.conv_size, 2 * self.latent_dim),
            nn.ReLU(inplace=False),
            nn.Linear(2 * self.latent_dim, 2 * self.latent_dim),
            nn.ReLU(inplace=False),
            nn.Linear(2 * self.latent_dim, self.latent_dim),
        )

    def forward(self, x):
        batch_size = x.shape[0]
        time_steps = x.shape[1]
        channels = x.shape[2]
        x = x.view(batch_size * time_steps, channels, *self.img_shape)
        coord_map = self.coord_map_const.repeat(batch_size * time_steps, 1, 1, 1)  # type: ignore
        inp = torch.cat((x, coord_map), 1)
        x = self.network(inp)
        x = x.view(-1, self.conv_size)
        x = self.mlp(x)
        x = x.view(batch_size, time_steps, -1)
        return x


class BroadcastDecoderNet(nn.Module):
    """
    General parameterized encoding architecture for VAE components
    """

    def __init__(
        self, latent_dim: int, output_shape: Tuple[int, int, int], layers=[32, 32, 32]
    ):
        super().__init__()

        self.output_channels = output_shape[0]
        img_shape = output_shape[1:]

        self.latent_dim = latent_dim
        self.img_shape = img_shape

        # construct the core conv structure
        self.network = create_conv_layers(
            self.latent_dim + 2,
            self.output_channels,
            layers,
            max_pool=False,
            padding=True,
            kernel_size=3,
        )

        # coordinate patching trick
        coord_map = net_util.create_coord_buffer(self.img_shape)
        self.register_buffer("coord_map_const", coord_map)

    def forward(self, x):
        # adds coordinate information to z and
        # produces a tiled representation of z
        batch_size = x.shape[0]
        time_steps = x.shape[1]
        x = x.view(batch_size * time_steps, self.latent_dim)
        z_scaled = x.unsqueeze(-1).unsqueeze(-1)
        z_tiled = z_scaled.repeat(1, 1, self.img_shape[0], self.img_shape[1])
        coord_map = self.coord_map_const.repeat(x.shape[0], 1, 1, 1)  # type: ignore
        inp = torch.cat((z_tiled, coord_map), 1)
        result = self.network(inp)
        result = result.view(
            batch_size, time_steps, self.output_channels, *self.img_shape
        )
        return result


class UNet(nn.Module):

    num_blocks: int
    in_channels: int
    out_channels: int
    channel_base: int

    def __init__(self, num_blocks, in_channels, out_channels, channel_base=64):
        super().__init__()
        self.num_blocks = num_blocks
        self.down_convs = nn.ModuleList()
        cur_in_channels = in_channels
        for i in range(num_blocks):
            self.down_convs.append(double_conv(cur_in_channels, channel_base * 2**i))
            cur_in_channels = channel_base * 2**i

        self.tconvs = nn.ModuleList()
        for i in range(num_blocks - 1, 0, -1):
            self.tconvs.append(
                nn.ConvTranspose2d(
                    channel_base * 2**i, channel_base * 2 ** (i - 1), 2, stride=2
                )
            )

        self.up_convs = nn.ModuleList()
        for i in range(num_blocks - 2, -1, -1):
            self.up_convs.append(
                double_conv(channel_base * 2 ** (i + 1), channel_base * 2**i)
            )

        self.final_conv = nn.Conv2d(channel_base, out_channels, 1)

    def forward(self, x):
        intermediates = []
        cur = x
        for down_conv in self.down_convs[:-1]:  # type: ignore
            cur = down_conv(cur)
            intermediates.append(cur)
            cur = nn.MaxPool2d(2)(cur)

        cur = self.down_convs[-1](cur)

        for i in range(self.num_blocks - 1):
            cur = self.tconvs[i](cur)
            cur = torch.cat((cur, intermediates[-i - 1]), 1)
            cur = self.up_convs[i](cur)

        return self.final_conv(cur)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class Gaussian(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_units=[256, 256], std=None, leaky_slope=0.2
    ):
        super(Gaussian, self).__init__()
        self.net = create_mlp(
            input_dim,
            2 * output_dim if std is None else output_dim,
            hidden_units,
        )

        self.std = std

    def forward(self, x):
        x = self.net(x)
        if self.std:
            mean = x
            std = torch.ones_like(mean) * self.std
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 1e-5

        return dists.Normal(loc=mean, scale=std)


class TanhGaussian(Gaussian):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_units=[256, 256],
        std=None,
        leaky_slope=0.2,
        multiplier=1.0,
    ):
        super(Gaussian, self).__init__()
        self.net = create_mlp(
            input_dim,
            2 * output_dim[-1] if std is None else output_dim,
            hidden_units,
        )

        self.std = std
        self.multiplier = torch.Tensor([multiplier])

    def forward(self, x):
        x = self.net(x)
        if self.std:
            mean = x
            std = torch.ones_like(mean) * self.std
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 1e-5

        return TanhNormal(mean=mean, std=std, multiplier=self.multiplier)


class OffsetGaussian(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_units=[256, 256], std=None, leaky_slope=0.2
    ):
        super().__init__()
        self.net = create_mlp(
            input_dim,
            2 * output_dim if std is None else output_dim,
            hidden_units,
        )

        self.std = std
        self.output_dim = output_dim

    def forward(self, inp):
        x = self.net(inp)
        if self.std:
            mean = x
            std = torch.ones_like(mean) * self.std
        else:
            mean, std = torch.chunk(x, 2, dim=-1)
            std = F.softplus(std) + 1e-5
        offset = inp[..., : self.output_dim]
        return dists.Normal(loc=mean + offset, scale=std)


class ConstantGaussian(nn.Module):
    def __init__(self, output_dim, std=1.0):
        super(ConstantGaussian, self).__init__()
        self.output_dim = output_dim
        self.std = std

    def forward(self, x):
        mean = torch.zeros((x.size(0), self.output_dim)).to(x)
        std = torch.ones((x.size(0), self.output_dim)).to(x) * self.std
        return dists.Normal(loc=mean, scale=std)


class SLACDecoder(nn.Module):
    def __init__(self, input_dim=288, output_dim=3, std=1.0, leaky_slope=0.2):
        super().__init__()
        self.std = std

        if type(input_dim) == torch.Size:
            input_dim = input_dim[0]
        if type(output_dim) == torch.Size:
            output_dim = output_dim[0]

        self.net = nn.Sequential(
            # (32+256, 1, 1) -> (256, 4, 4)
            nn.ConvTranspose2d(input_dim, 256, 4),
            nn.LeakyReLU(leaky_slope),
            # (256, 4, 4) -> (128, 8, 8)
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.LeakyReLU(leaky_slope),
            # (128, 8, 8) -> (64, 16, 16)
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.LeakyReLU(leaky_slope),
            # (64, 16, 16) -> (32, 32, 32)
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.LeakyReLU(leaky_slope),
            # (32, 32, 32) -> (3, 64, 64)
            nn.ConvTranspose2d(32, 3, 5, 2, 2, 1),
            nn.LeakyReLU(leaky_slope),
        )

    def forward(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = torch.cat(x, dim=-1)

        num_batches, num_sequences, latent_dim = x.size()
        x = x.view(num_batches * num_sequences, latent_dim, 1, 1)
        x = self.net(x)
        _, C, W, H = x.size()
        x = x.view(num_batches, num_sequences, C, W, H)
        return dists.Normal(loc=x, scale=torch.ones_like(x) * self.std)


class SLACEncoder(nn.Module):
    def __init__(self, input_dim=3, output_dim=256, leaky_slope=0.2):
        super().__init__()

        if type(input_dim) == torch.Size:
            input_dim = input_dim[0]
        if type(output_dim) == torch.Size:
            output_dim = output_dim[0]

        self.net = nn.Sequential(
            # (3, 64, 64) -> (32, 32, 32)
            nn.Conv2d(input_dim, 32, 5, 2, 2),
            nn.LeakyReLU(leaky_slope),
            # (32, 32, 32) -> (64, 16, 16)
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(leaky_slope),
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.LeakyReLU(leaky_slope),
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.LeakyReLU(leaky_slope),
            # (256, 4, 4) -> (256, 1, 1)
            nn.Conv2d(256, output_dim, 4),
            nn.LeakyReLU(leaky_slope),
        )

    def forward(self, x):
        num_batches, num_sequences, C, H, W = x.size()
        x = x.view(num_batches * num_sequences, C, H, W)
        x = self.net(x)
        x = x.view(num_batches, num_sequences, -1)

        return x


class TwinnedMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units):
        super().__init__()

        self.net1 = create_mlp(input_dim, output_dim, hidden_units)
        self.net2 = create_mlp(input_dim, output_dim, hidden_units)

    def forward(self, x):
        return self.net1(x), self.net2(x)
