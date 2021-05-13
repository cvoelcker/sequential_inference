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

        self.register_module("encoder", self.encoder)
        self.register_module("decoder", self.decoder)
        self.register_module("reward_decoder", self.reward_decoder)
        self.register_module("latent", self.latent)

    def infer_sequence(self, obs, actions=None, rewards=None, global_belief=None):
        priors, posteriors = self.infer_full_sequence(
            obs, actions=actions, rewards=rewards, global_belief=global_belief
        )
        priors = self.get_samples(priors)
        posteriors = self.get_samples(posteriors)
        return priors, posteriors

    def infer_full_sequence(self, obs, actions=None, rewards=None, global_belief=None):
        features_seq = self.encoder(obs, rewards)
        horizon = obs.shape[1]
        priors = []
        posteriors = []

        prior, posterior = self.latent.obtain_initial(features_seq[:, 0])
        prior_latent = prior[0]
        posterior_latent = posterior[0]
        priors.append(prior)
        posteriors.append(posteriors)

        for t in range(1, horizon):
            prior, posterior = self.latent(
                prior_latent,
                posterior_latent,
                features_seq[:, t],
                action=actions[:, t - 1],
                global_belief=global_belief[:, t],
            )
            prior_latent = prior[0]
            posterior_latent = posterior[0]
            if self.condition_on_posterior:
                prior_latent = posterior_latent
            priors.append(prior)
            posteriors.append(posteriors)
        return priors, posteriors

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        global_beliefs: Optional[torch.Tensor] = None,
    ):
        stats = {}

        priors, posteriors = self.infer_full_sequence(
            obs, actions=actions, rewards=rewards, global_beliefs=global_beliefs
        )
        prior_dists = self.get_dists(priors)
        posterior_dists = self.get_dists(posteriors)
        prior_latents = self.get_samples(priors)
        posterior_latents = self.get_samples(priors)

        loss = 0.0

        # KL divergence loss.
        kld_loss = calc_kl_divergence(prior_dists, posterior_dists)
        loss -= kld_loss
        stats["kld_loss"] = kld_loss.detach().cpu()

        # Log likelihood loss of generated observations.
        images_seq = self.decoder(posterior_latents)
        log_likelihood_loss = torch.mean((images_seq - obs) ** 2)
        loss += log_likelihood_loss
        stats["log_lik_loss"] = log_likelihood_loss.detach().cpu()

        if self.predict_from_prior:
            # Log likelihood loss of generated observations from prior (if more supervision is needed).
            prior_images_seq = self.decoder(prior_latents)
            log_likelihood_prior_loss = torch.mean((prior_images_seq - obs) ** 2)
            loss += log_likelihood_prior_loss
            stats["log_lik_prior_loss"] = log_likelihood_prior_loss.detach().cpu()

        if rewards is not None:
            # Log likelihood loss of generated rewards. do not reconstruct first reward, since it is a dummy reward
            rewards_seq = self.reward_decoder(posterior_latents)[:, 1:]
            reward_log_likelihood_loss = torch.mean(
                (rewards_seq - rewards.unsqueeze(-1)) ** 2
            ).mean()
            loss += reward_log_likelihood_loss
            stats["log_lik_rew"] = reward_log_likelihood_loss.detach().cpu()

        if rewards is not None and self.predict_from_prior:
            # Log likelihood loss of generated rewards. do not reconstruct first reward, since it is a dummy reward
            prior_rewards_seq = self.reward_decoder(prior_latents)[:, 1:]
            prior_reward_log_likelihood_loss = torch.mean(
                (prior_rewards_seq - rewards.unsqueeze(-1)) ** 2
            ).mean()
            loss += prior_reward_log_likelihood_loss
            stats["log_lik_prior_rew"] = prior_reward_log_likelihood_loss.detach().cpu()

        return -loss, stats

    def infer_single_step(
        self, last_latent, obs, action=None, rewards=None, global_belief=None
    ):
        features = self.encoder(obs, rewards)
        last_latent = last_latent
        _, posterior = self.latent(last_latent, last_latent, features, action=action)
        return self.get_samples(posterior), global_belief

    def predict_sequence(
        self, initial_latent, actions=None, reward=None, global_belief=None
    ):
        prior_latent = initial_latent
        horizon = actions.shape[1]

        priors = []
        for t in range(1, horizon):
            prior = self.latent.infer_prior(
                prior_latent, actions[:, t], global_belief=global_belief
            )
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
