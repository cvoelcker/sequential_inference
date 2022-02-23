from sequential_inference.util.rl_util import join_state_with_array
import torch

from sequential_inference.nn_models.base.network_util import calc_kl_divergence
from sequential_inference.nn_models.dynamics.dynamics import (
    SLACLatentNetwork,
)
from sequential_inference.algorithms.env_models.simple_vi import VIModelAlgorithm


class SLACModelAlgorithm(VIModelAlgorithm):
    def __init__(
        self,
        encoder,
        decoder,
        reward_decoder,
        action_shape,
        kl_factor: float,
        state_factor: float,
        reward_factor: float,
        feature_dim=64,
        latent_dim=64,
        latent1_dim=32,
        latent2_dim=32,
        hidden_units=[64, 64],
        leaky_slope=0.2,
        predict_from_prior=False,
        condition_on_posterior=False,
    ):
        latent = SLACLatentNetwork(
            action_shape,
            feature_dim=feature_dim,
            latent1_dim=latent1_dim,
            latent2_dim=latent2_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope,
        )
        self.latent1_dim = latent1_dim

        super().__init__(
            encoder,
            decoder,
            reward_decoder,
            latent,
            kl_factor,
            state_factor,
            reward_factor,
            predict_from_prior=predict_from_prior,
            condition_on_posterior=condition_on_posterior,
        )

    def infer_full_sequence(self, obs, actions=None, rewards=None):
        features_seq = self.encoder(join_state_with_array(obs, rewards))
        horizon = obs.shape[1]
        priors = []
        posteriors = []

        prior, posterior = self.latent.obtain_initial(features_seq[:, 0])
        prior_latent = prior[1]
        posterior_latent = posterior[1]
        priors.append(prior)
        posteriors.append(posterior)

        for t in range(1, horizon):
            prior, posterior = self.latent(
                prior_latent,
                posterior_latent,
                features_seq[:, t],
                action=actions[:, t - 1],
            )
            prior_latent = prior[1]
            posterior_latent = posterior[1]
            if self.condition_on_posterior:
                prior_latent = posterior_latent
            priors.append(prior)
            posteriors.append(posterior)
        return priors, posteriors

    def infer_single_step(self, last_latent, obs, action=None, rewards=None):
        features = self.encoder(join_state_with_array(obs, rewards))
        last_latent = last_latent[:, self.latent1_dim :]
        _, posterior = self.latent(last_latent, last_latent, features, action=action)
        return self.get_samples(posterior)

    def predict_sequence(self, initial_latent, actions=None, reward=None):
        prior_latent = initial_latent[:, self.latent1_dim :]
        horizon = actions.shape[1]

        priors = []
        for t in range(1, horizon):
            prior = self.latent.infer_prior(prior_latent, actions[:, t])
            prior_latent = prior[1]
            priors.append(prior)
        return self.get_samples(priors)

    def get_samples(self, latent):
        latent1 = []
        latent2 = []

        for l in latent:
            latent1.append(l[0])
            latent2.append(l[1])

        latent1 = torch.stack(latent1, 1)
        latent2 = torch.stack(latent2, 1)

        return torch.cat((latent1, latent2), -1)

    def get_dists(self, latent):
        latent1 = []

        for l in latent:
            latent1.append(l[2])

        return latent1
