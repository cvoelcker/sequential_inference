from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.distributions import Distribution

from sequential_inference.abc.static_model import AbstractLatentStaticAlgorithm
from sequential_inference.nn_models.base.network_util import calc_kl_divergence


class VAEAlgorithm(AbstractLatentStaticAlgorithm):
    def __init__(self, encoder, decoder, lr=3e-4):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()), lr=lr
        )

        self.register_module("encoder", encoder)
        self.register_module("decoder", decoder)
        self.register_module("optimizer", self.optimizer)

    def infer_latent(
        self,
        obs: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.distributions.Distribution]]:
        mu, log_sigma = self.encoder(obs)
        latent_sample = mu + torch.exp(log_sigma) * torch.randn_like(log_sigma)
        return [(latent_sample, torch.distributions.Normal(mu, torch.exp(log_sigma)))]

    def reconstruct(self, latent_sample):
        return self.decoder(latent_sample)

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor], Dict]:

        latents = self.infer_latent(obs)

        posterior_dists = self.get_dists(latents)
        prior_dists: List[Distribution] = [
            torch.distributions.Normal(
                torch.zeros_like(d.mean), torch.ones_like(d.mean)
            )
            for d in posterior_dists
        ]
        kld_loss = calc_kl_divergence(posterior_dists, prior_dists)
        kld_loss = kld_loss.sum(-1).mean()

        posterior_latents = self.get_samples(latents)
        reconstruction = self.reconstruct(posterior_latents)
        reconstruction = reconstruction.view(reconstruction.shape[0], -1)
        obs = obs.view(obs.shape[0], -1)
        reconstruction_loss = torch.mean(torch.sum((reconstruction - obs) ** 2, -1))

        return (reconstruction_loss + kld_loss,), {
            "kld_loss": kld_loss.detach().cpu(),
            "reconstruction_loss": reconstruction_loss.detach().cpu(),
            "loss": (reconstruction_loss + kld_loss).detach().cpu(),
        }

    def get_dists(
        self, latents: List[Tuple[torch.Tensor, torch.distributions.Distribution]]
    ) -> List[torch.distributions.Distribution]:
        return [dist for _, dist in latents]

    def get_samples(self, latents):
        return torch.stack([latent_sample for latent_sample, _ in latents], 1)

    def get_step(self) -> Callable:
        def _step(losses, stats):
            self.optimizer.zero_grad()
            losses[0].backward()
            self.optimizer.step()
            return stats

        return _step
