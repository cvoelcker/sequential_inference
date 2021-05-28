from sequential_inference.util.rl_util import join_state_with_array
from typing import Optional
import torch
from sequential_inference.models.base.network_util import calc_kl_divergence
from sequential_inference.models.dynamics.dynamics import SimpleLatentNetwork
from sequential_inference.abc.sequence_model import (
    AbstractLatentModel,
    AbstractSequenceAlgorithm,
)


class VIModelAlgorithm(AbstractSequenceAlgorithm):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        reward_decoder: torch.nn.Module,
        latent: AbstractLatentModel,
        kl_factor: float,
        state_factor: float,
        reward_factor: float,
        predict_from_prior: bool = False,
        condition_on_posterior: bool = False,
    ):
        super().__init__()
        self.latent = latent
        self.encoder = encoder
        self.decoder = decoder
        self.reward_decoder = reward_decoder

        self.predict_from_prior = predict_from_prior
        self.condition_on_posterior = condition_on_posterior

        self.kl_factor = kl_factor
        self.state_factor = state_factor
        self.reward_factor = reward_factor

        self.register_module("encoder", self.encoder)
        self.register_module("decoder", self.decoder)
        self.register_module("reward_decoder", self.reward_decoder)
        self.register_module("latent", self.latent)

    def infer_sequence(self, obs, actions=None, rewards=None):
        priors, posteriors = self.infer_full_sequence(
            obs, actions=actions, rewards=rewards
        )
        priors = self.get_samples(priors)
        posteriors = self.get_samples(posteriors)
        return priors, posteriors

    def infer_full_sequence(self, obs, actions=None, rewards=None):
        features_seq = self.encoder(join_state_with_array(obs, rewards))
        horizon = obs.shape[1]
        priors = []
        posteriors = []

        prior, posterior = self.latent.obtain_initial(features_seq[:, 0])
        prior_latent = prior[0]
        posterior_latent = posterior[0]
        priors.append(prior)
        posteriors.append(posterior)

        for t in range(1, horizon):
            prior, posterior = self.latent(
                prior_latent,
                posterior_latent,
                features_seq[:, t],
                action=actions[:, t - 1],
            )
            prior_latent = prior[0]
            posterior_latent = posterior[0]
            if self.condition_on_posterior:
                prior_latent = posterior_latent
            priors.append(prior)
            posteriors.append(posterior)
        return priors, posteriors

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):

        dones = 1. - dones
        stats = {}

        priors, posteriors = self.infer_full_sequence(
            obs, actions=actions, rewards=rewards
        )
        prior_dists = self.get_dists(priors)
        posterior_dists = self.get_dists(posteriors)
        prior_latents = self.get_samples(priors)
        posterior_latents = self.get_samples(priors)

        loss = 0.0

        # KL divergence loss.
        kld_loss = calc_kl_divergence(prior_dists, posterior_dists)
        kld_loss = (kld_loss * dones).sum(-1).mean()
        loss += self.kl_factor * kld_loss
        stats["kld_loss"] = kld_loss.detach().cpu()

        # Log likelihood loss of generated observations.
        images_seq = self.decoder(posterior_latents)
        log_likelihood_loss = (((images_seq - obs) ** 2) * dones).sum(-1).mean()
        loss += self.state_factor * log_likelihood_loss
        stats["log_lik_loss"] = log_likelihood_loss.detach().cpu()

        if self.predict_from_prior:
            # Log likelihood loss of generated observations from prior (if more supervision is needed).
            prior_images_seq = self.decoder(prior_latents)
            log_likelihood_prior_loss = (
                (((prior_images_seq - obs) ** 2) * dones).sum(-1).mean()
            )
            loss += self.state_factor * log_likelihood_prior_loss
            stats["log_lik_prior_loss"] = log_likelihood_prior_loss.detach().cpu()

        if rewards is not None:
            # Log likelihood loss of generated rewards.
            rewards_seq = self.reward_decoder(posterior_latents)
            reward_log_likelihood_loss = (
                (((rewards_seq - rewards) ** 2) * dones).sum(-1).mean()
            )
            loss += self.state_factor * reward_log_likelihood_loss
            stats["log_lik_rew"] = reward_log_likelihood_loss.detach().cpu()

        if rewards is not None and self.predict_from_prior:
            # Log likelihood loss of generated rewards.
            prior_rewards_seq = self.reward_decoder(prior_latents)
            prior_reward_log_likelihood_loss = (
                (((prior_rewards_seq - rewards) ** 2) * dones).sum(-1).mean()
            )
            loss += self.state_factor * prior_reward_log_likelihood_loss
            stats["log_lik_prior_rew"] = prior_reward_log_likelihood_loss.detach().cpu()

        stats["elbo"] = loss.detach().cpu()

        return loss, stats

    def infer_single_step(self, last_latent, obs, action=None, rewards=None):
        features = self.encoder(join_state_with_array(obs, rewards))
        last_latent = last_latent
        _, posterior = self.latent(last_latent, last_latent, features, action=action)
        return self.get_samples(posterior)

    def predict_sequence(self, initial_latent, actions=None, reward=None):
        prior_latent = initial_latent
        horizon = actions.shape[1]

        priors = []
        for t in range(1, horizon):
            prior = self.latent.infer_prior(prior_latent, actions[:, t])
            prior_latent = prior[0]
            priors.append(prior)
        return self.get_samples(priors)

    def get_samples(self, latent):
        latent1 = []

        for l in latent:
            latent1.append(l[0])

        latent1 = torch.stack(latent1, 1)

        return latent1

    def get_dists(self, latent):
        latent1 = []

        for l in latent:
            latent1.append(l[1])

        return latent1

    def reconstruct(self, latent):
        return self.decoder(latent)

    def reconstruct_reward(self, latent):
        return self.reward_decoder(latent)


class SimpleVIModelAlgorithm(VIModelAlgorithm):
    def __init__(
        self,
        encoder,
        decoder,
        reward_decoder,
        action_shape,
        feature_dim=64,
        latent_dim=32,
        hidden_units=[64, 64],
        leaky_slope=0.2,
        predict_from_prior=False,
        condition_on_posterior=False,
    ):
        latent = SimpleLatentNetwork(
            action_shape,
            feature_dim=feature_dim,
            latent_dim=latent_dim,
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
