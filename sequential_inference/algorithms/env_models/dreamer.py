import torch

from sequential_inference.algorithms.env_models.simple_vi import VIModelAlgorithm
from sequential_inference.nn_models.dynamics.dynamics import PlaNetLatentNetwork


class DreamerModelAlgorithm(VIModelAlgorithm):
    def __init__(
        self,
        encoder,
        decoder,
        reward_decoder_hidden_units,
        action_dim,
        kl_factor: float,
        state_factor: float,
        reward_factor: float,
        feature_dim=64,
        latent_dim=32,
        recurrent_hidden_dim=32,
        belief_dim=0,
        hidden_units=[64, 64],
        leaky_slope=0.2,
        free_nats=3.0,
        predict_from_prior=False,
        condition_on_posterior=False,
    ):
        latent = PlaNetLatentNetwork(
            action_dim,
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            recurrent_hidden_dim=recurrent_hidden_dim,
            belief_dim=belief_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope,
        )
        self.latent_dim = latent_dim

        super().__init__(
            encoder,
            decoder,
            latent,
            reward_decoder_hidden_units,
            action_dim,
            kl_factor,
            state_factor,
            reward_factor,
            free_nats=3,
            predict_from_prior=predict_from_prior,
            condition_on_posterior=condition_on_posterior,
        )

    def get_samples(self, latent, full=False):
        latent_ = []

        for l in latent:
            latent_.append(l[0])

        latent = torch.stack(latent_, 1)
        # if full:
        #     return latent
        # else:
        #     return latent[..., : self.latent_dim]
        return latent
