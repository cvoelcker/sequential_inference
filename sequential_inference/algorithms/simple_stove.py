from sequential_inference.models.dynamics.dynamics import (
    SimpleStoveLatentNetwork,
)
from sequential_inference.algorithms.simple_vi import VIModelAlgorithm


class SimplifiedStoveAlgorithm(VIModelAlgorithm):
    def __init__(
        self,
        encoder,
        decoder,
        reward_decoder,
        action_shape,
        latent_dim=32,
        dynamics_latent_dim=32,
        hidden_units=[64, 64],
        leaky_slope=0.2,
        predict_from_prior=False,
        condition_on_posterior=False,
    ):
        latent = SimpleStoveLatentNetwork(
            action_shape,
            latent_dim=latent_dim,
            dynamics_latent_dim=dynamics_latent_dim,
            hidden_units=hidden_units,
            leaky_slope=leaky_slope,
        )
        super().__init__(
            encoder,
            decoder,
            reward_decoder,
            latent,
            predict_from_prior=predict_from_prior,
            condition_on_posterior=condition_on_posterior,
        )

    def infer_full_sequence(self, obs, actions=None, rewards=None):
        features_seq = self.encoder(obs, rewards)
        horizon = obs.shape[1]
        priors = []
        posteriors = []

        prior, posterior = self.latent.obtain_initial(features_seq[:, 0])
        prior_latent = posterior[0]
        posterior_latent = posterior[1]
        priors.append(prior)
        posteriors.append(posteriors)

        for t in range(1, horizon):
            prior, posterior = self.latent(
                prior_latent,
                posterior_latent,
                features_seq[:, t],
                action=actions[:, t - 1],
            )
            prior_latent = posterior[0]
            posterior_latent = posterior[1]

            priors.append(prior)
            posteriors.append(posteriors)
        return priors, posteriors
