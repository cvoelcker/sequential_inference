import torch
from torch import nn

from sequential_inference.models.base.base_nets import EncoderNet, BroadcastDecoderNet
from sequential_inference.models.vae.vae_core import VAECore
from sequential_inference.models.base.network_util import (
    differentiable_sampling,
    reconstruction_likelihood,
)


class BroadcastVAE(VAECore):
    def __init__(
        self,
        input_channels,
        output_channels,
        latent_dim,
        enc_layers,
        dec_layers,
        img_shape,
        sigma_p=0.1,
    ):
        """Initialization of a regular broadcast VAE

        Args:
            input_channels (int): number of input channels in the images
            latent_dim (int): 2 * mu for mean field embedding
            enc_layers (list[int]): list of encode layer sizes (channels)
            dec_layers (list[int]): list of decode layer sizes (channels)
            img_shape (tuple[int, int]): size (length and width) of image
        """

        self.latent_dim = latent_dim
        self.latent_mu = self.latent_dim
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.img_shape = img_shape

        encoder = EncoderNet(enc_layers, latent_dim, img_shape, input_channels)
        decoder = BroadcastDecoderNet(
            dec_layers, latent_dim, img_shape, output_channels
        )

        self.sigma_p = sigma_p

        self.img_shape = img_shape

        super().__init__(encoder, decoder)

    def decode(self, z):
        """Computes decoding path of VAE

        Args:
            z (torch.Tensor): latent encoding (sampled) for the image

        Returns:
            [torch.Tensor]: image prediction
        """
        x_pred = self.decoder(z)
        # bias to prevent exploding gradients in the beginning
        # this is data dependent, but it makes sense to bias the
        # output to output the mean of images
        x_pred = torch.sigmoid(x_pred)
        return x_pred


class OneStepBroadcastVAE(BroadcastVAE):
    def forward(self, x, samples=1):
        """Full encoding decoding pipeline

        Args:
            x (torch.Tensor): image input
            prior_sigma (float, optional): Prior sigma (don't change unless
            you really know what you are doing) Defaults to 1.

        Returns:
            torch.Tensor: all output of intermediate networks for complete control
        """
        # compute predictions
        mu, sigma, z, kl_z = self.encode(x, samples)
        z = z.flatten(0, 1)
        x_pred = self.decode(z)
        # x_pred = x_pred.reshape((samples, -1, *self.img_shape))
        # x_pred = torch.transpose(x_pred, 0, 1)
        return x_pred, {"mu": mu, "sigma": sigma, "z": z, "kl_z": kl_z}

    def encode(self, x, samples, prior_sigma=1.0):
        """Computes encoding path of VAE

        Args:
            x (torch.Tensor): image input
            prior_sigma (float, optional): Prior sigma (don't change unless
            you really know what you are doing) Defaults to 1.

        Returns:
            torch.Tensor: all output of intermediate networks for complete control
        """
        # compute predictions
        encoding = self.encoder(x)

        mu = encoding[..., : self.latent_dim]
        sigma = (
            torch.nn.functional.softplus(encoding[..., self.latent_dim :]) + 0.01
        )  # enforces positive sigma
        z, kl_z = differentiable_sampling(mu, sigma, prior_sigma, samples=1)
        return mu, sigma, z, kl_z

    def decode(self, z):
        """Computes decoding path of VAE

        Args:
            z (torch.Tensor): latent encoding (sampled) for the image

        Returns:
            [torch.Tensor]: image prediction
        """
        x_pred = self.decoder(z)
        return x_pred

    def loss(self, x, x_pred, kl_z, beta=1.0):
        """Computes -ELBO as differentiable loss function

        Args:
            x (torch.Tensor): original img
            x_pred (torch.Tensor): predicted img
            kl_z (torch.Tensor): kl between prior and posterior of vae
            beta (float, optional): weighing parameter between reconstruction and kl. Defaults to 1..

        Returns:
            [torch.float]: -elbo
        """
        elbo = torch.mean(
            torch.sum(reconstruction_likelihood(x, x_pred, self.sigma_p), (-1, -2, -3))
            - beta * kl_z
        )
        return -elbo

    def train(self, x, prior_sigma=1.0, beta=1.0):
        """Full network + loss computation

        Args:
            x (torch.Tensor): image input
            prior_sigma (float, optional): Prior sigma (don't change unless
            you really know what you are doing) Defaults to 1.
            beta (float, optional): weighing parameter between reconstruction and kl. Defaults to 1..

        Returns:
            [torch.float]: - elbo
            [dict]: intermediate tensors for bookkeeping and full control
        """
        x_pred, stats = self(x, prior_sigma)
        mu, sigma, z, kl_z = stats.values()
        loss = self.loss(x, x_pred, kl_z, beta)
        return loss, {"x_pred": x_pred, "mu": mu, "sigma": sigma, "z": z, "kl_zk": kl_z}
