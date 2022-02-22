import torch
from torch import nn

from sequential_inference.nn_models.base.network_util import (
    differentiable_sampling,
    reconstruction_likelihood,
)


class VAECore(nn.Module):
    def __init__(self, encoder, decoder):
        """Initialization of a regular broadcast VAE

        Args:
            input_channels (int): number of input channels in the images
            latent_dim (int): 2 * mu for mean field embedding
            enc_layers (list[int]): list of encode layer sizes (channels)
            dec_layers (list[int]): list of decode layer sizes (channels)
            img_shape (tuple[int, int]): size (length and width) of image
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """Full encoding decoding pipeline

        Args:
            x (torch.Tensor): image input
            prior_sigma (float, optional): Prior sigma (don't change unless
            you really know what you are doing) Defaults to 1.

        Returns:
            torch.Tensor: all output of intermediate networks for complete control
        """
        # compute predictions
        mu, sigma, z, kl_z = self.encode(x)
        x_pred = self.decode(z)

        return x_pred, mu, sigma, z, kl_z

    def encode(self, x, prior_sigma=1.0):
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

        mu = encoding[:, : self.latent_dim // 2]
        sigma = (
            torch.nn.functional.elu(encoding[:, self.latent_dim // 2 :]) + 0.01
        )  # enforces positive sigma
        z, kl_z = differentiable_sampling(mu, sigma, prior_sigma)
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
            torch.sum(reconstruction_likelihood(x, x_pred, self.sigma_p), (1, 2, 3))
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
        x_pred, mu, sigma, z, kl_z = self(x, prior_sigma)
        loss = self.loss(x, x_pred, kl_z, beta)
        return loss, {"x_pred": x_pred, "mu": mu, "sigma": sigma, "z": z, "kl_zk": kl_z}
