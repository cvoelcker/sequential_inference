import torch

from sequential_inference.models.vae.vae_core import VAECore
from sequential_inference.models.base.base_nets import create_mlp
from sequential_inference.models.base.network_util import reconstruction_likelihood


class InfoBottleneckEncoder(VAECore):
    def __init__(self, dim, latent_dim, output_dim, layers):

        self.dim = dim
        self.latent_dim = latent_dim
        self.layers = layers

        encoder = create_mlp(dim, latent_dim, layers)
        decoder = create_mlp(latent_dim // 2, output_dim, reversed(layers))

        super().__init__(encoder, decoder)

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
            torch.sum(reconstruction_likelihood(x, x_pred, 1.0), (1, 2, 3))
            - beta * kl_z
        )
        return -elbo

    def train(self, x, y, prior_sigma=1.0, beta=1.0):
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
        loss = self.loss(y, x_pred, kl_z, beta)
        return loss, {"x_pred": x_pred, "mu": mu, "sigma": sigma, "z": z, "kl_zk": kl_z}
