from typing import List, Tuple
from sequential_inference.abc.rl import AbstractAgent
from sequential_inference.nn_models.base.base_nets import create_mlp
from sequential_inference.util.rl_util import join_state_with_array
import torch
from sequential_inference.nn_models.base.network_util import calc_kl_divergence
from sequential_inference.nn_models.dynamics.dynamics import SimpleLatentNetwork
from sequential_inference.abc.sequence_model import (
    AbstractLatentModel,
    AbstractLatentSequenceAlgorithm,
)


class VIModelAlgorithm(AbstractLatentSequenceAlgorithm):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent: AbstractLatentModel,
        reward_decoder_hidden_units: List[int],
        action_dim: List[int],
        kl_factor: float,
        state_factor: float,
        reward_factor: float,
        free_nats=0.0,
        predict_from_prior: bool = False,
        condition_on_posterior: bool = False,
        lr: float = 6e-4,
    ):
        super().__init__()
        self.latent = latent
        self.encoder = encoder
        self.decoder = decoder

        reward_decoder = create_mlp(
            self.latent.latent_dim + action_dim[0], 1, reward_decoder_hidden_units
        )
        self.reward_decoder = reward_decoder

        self.predict_from_prior = predict_from_prior
        self.condition_on_posterior = condition_on_posterior

        self.kl_factor = kl_factor
        self.state_factor = state_factor
        self.reward_factor = reward_factor
        self.free_nats = free_nats

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.register_module("encoder", self.encoder)
        self.register_module("decoder", self.decoder)
        self.register_module("reward_decoder", self.reward_decoder)
        self.register_module("latent", self.latent)
        self.register_module("optimizer", self.optimizer)

    def infer_sequence(self, obs, actions=None, rewards=None, full=False):
        priors, posteriors = self.infer_full_sequence(
            obs, actions=actions, rewards=rewards
        )
        priors = self.get_samples(priors, full=full)
        posteriors = self.get_samples(posteriors, full=full)
        return priors, posteriors

    def infer_full_sequence(self, obs, actions=None, rewards=None):
        # shift rewards by 1 timestep
        if rewards is not None:
            rewards = rewards[:, :-1]
            rewards = torch.cat([torch.zeros_like(rewards[:, :1]), rewards], dim=1)
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

        dones = 1.0 - dones
        stats = {}

        priors, posteriors = self.infer_full_sequence(
            obs, actions=actions, rewards=rewards
        )
        prior_dists = self.get_dists(priors)
        posterior_dists = self.get_dists(posteriors)
        prior_latents = self.get_samples(priors)
        posterior_latents = self.get_samples(posteriors)

        loss = 0.0

        # flatten for easier loss calculation
        obs = obs.view(obs.shape[0], obs.shape[1], -1)

        # KL divergence loss.
        kld_loss = calc_kl_divergence(prior_dists, posterior_dists)
        kld_loss = (kld_loss * dones).sum(-1).mean()
        loss += self.kl_factor * torch.clamp(kld_loss, self.free_nats)
        stats["kld_loss"] = kld_loss.detach().cpu()

        # Log likelihood loss of generated observations.
        images_seq = self.decoder(posterior_latents)
        images_seq = images_seq.view(images_seq.shape[0], images_seq.shape[1], -1)
        log_likelihood_loss = (((images_seq - obs) ** 2) * dones).sum(-1).mean()
        loss += self.state_factor * log_likelihood_loss
        stats["log_lik_loss"] = log_likelihood_loss.detach().cpu()

        if self.predict_from_prior:
            # Log likelihood loss of generated observations from prior (if more supervision is needed).
            prior_images_seq = self.decoder(prior_latents)
            prior_images_seq = images_seq.view(
                images_seq.shape[0], images_seq.shape[1], -1
            )
            log_likelihood_prior_loss = (
                (((prior_images_seq - obs) ** 2) * dones).sum(-1).mean()
            )
            loss += self.state_factor * log_likelihood_prior_loss
            stats["log_lik_prior_loss"] = log_likelihood_prior_loss.detach().cpu()

        if rewards is not None:
            # Log likelihood loss of generated rewards.
            rewards_seq = self.reward_decoder(
                torch.cat([posterior_latents, actions], dim=-1)
            )
            reward_log_likelihood_loss = (
                (((rewards_seq - rewards) ** 2) * dones).sum(-1).mean()
            )
            loss += self.state_factor * reward_log_likelihood_loss
            stats["log_lik_rew"] = reward_log_likelihood_loss.detach().cpu()

        if rewards is not None and self.predict_from_prior:
            # Log likelihood loss of generated rewards.
            prior_rewards_seq = self.reward_decoder(
                torch.cat([posterior_latents, actions], dim=-1)
            )
            prior_reward_log_likelihood_loss = (
                (((prior_rewards_seq - rewards) ** 2) * dones).sum(-1).mean()
            )
            loss += self.state_factor * prior_reward_log_likelihood_loss
            stats["log_lik_prior_rew"] = prior_reward_log_likelihood_loss.detach().cpu()

        stats["elbo"] = loss.detach().cpu()

        return loss, stats

    def infer_single_step(
        self, last_prior, last_latent, obs, action=None, rewards=None, full=False
    ):
        features = self.encoder(join_state_with_array(obs.unsqueeze(1), rewards))[:, 0]
        return self.latent(last_prior, last_latent, features, action=action)

    def predict_latent_sequence(
        self, initial_latent, actions=None, reward=None, full=False
    ):
        prior_latent = initial_latent
        horizon = actions.shape[1]

        priors = []
        for t in range(1, horizon):
            prior = self.latent.infer_prior(prior_latent, actions[:, t])
            prior_latent = prior[0]
            priors.append(prior)
        return self.get_samples(priors, full=full)

    def predict_latent_step(self, latent, action, full=False):
        prior = self.latent.infer_prior(latent, action)
        return self.get_samples([prior], full=full)

    def get_samples(self, latent, full=False):
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

    def get_step(self):
        def _step(losses, stats):
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            return stats

        return _step

    def parameters(self):
        return (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.latent.parameters())
            + list(self.reward_decoder.parameters())
        )

    def rollout_with_policy(
        self,
        latent: torch.Tensor,
        policy: AbstractAgent,
        horizon: int,
        reconstruct: bool = False,
        explore: bool = False,
    ) -> Tuple[torch.Tensor, ...]:

        predicted_latents = []
        predicted_actions = []
        reconstructions = []

        # save first to conform to regular RL convention
        predicted_latents.append((latent, None))

        # iterate over horizon
        for i in range(horizon):
            # decide whether to obtain reconstructions (needed for policies which directly
            # predict in observation space)
            if reconstruct:
                obs = self.reconstruct(latent)
            else:
                obs = None

            act = policy.act(obs, context=latent, explore=explore)
            latent = self.predict_latent_step(latent, act, full=True).squeeze(1)

            predicted_actions.append(act)
            predicted_latents.append((latent, None))

            if reconstruct:
                reconstructions.append(obs)

        # put together results
        predicted_latents = self.get_samples(predicted_latents)
        predicted_actions = torch.stack(predicted_actions, 1)
        rewards = self.reconstruct_reward(
            torch.cat([predicted_latents[:, :-1], predicted_actions], dim=-1)
        )
        if reconstruct:
            reconstructions = torch.stack(reconstructions, 1)
            return predicted_latents, predicted_actions, rewards, reconstructions
        else:
            return predicted_latents, predicted_actions, rewards


class SimpleVIModelAlgorithm(VIModelAlgorithm):
    def __init__(
        self,
        encoder,
        decoder,
        reward_decoder_hidden_units,
        action_shape,
        kl_factor: float = 1.0,
        state_factor: float = 1.0,
        reward_factor: float = 1.0,
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
            latent,
            reward_decoder_hidden_units,
            action_shape,
            kl_factor,
            state_factor,
            reward_factor,
            predict_from_prior=predict_from_prior,
            condition_on_posterior=condition_on_posterior,
        )
