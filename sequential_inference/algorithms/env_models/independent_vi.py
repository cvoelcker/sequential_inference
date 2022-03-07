from typing import List, Tuple

import torch
from torch.distributions.kl import kl_divergence

from sequential_inference.abc.rl import AbstractAgent
from sequential_inference.nn_models.base.base_nets import RecurrentMLP, create_mlp
from sequential_inference.util.rl_util import join_state_with_array
from sequential_inference.nn_models.base.network_util import calc_kl_divergence
from sequential_inference.nn_models.dynamics.dynamics import SimpleLatentNetwork
from sequential_inference.abc.sequence_model import (
    AbstractLatentModel,
    AbstractLatentSequenceAlgorithm,
)
from sequential_inference.util.torch import exponential_moving_matrix


class IndependentVIModelAlgorithm(AbstractLatentSequenceAlgorithm):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_hidden_units: List[int],
        recurrent_dim: int,
        reward_decoder_hidden_units: List[int],
        action_dim: List[int],
        lr: float = 6e-4,
        recurrent=False,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if recurrent:
            self.latent = RecurrentMLP(  # type: ignore
                self.encoder.feature_dim + action_dim[0],  # type: ignore
                recurrent_dim,
                self.encoder.feature_dim,
                latent_hidden_units,
            )
        else:
            self.latent = create_mlp(  # type: ignore
                self.encoder.feature_dim + action_dim[0], self.encoder.feature_dim, latent_hidden_units  # type: ignore
            )

        reward_decoder = create_mlp(
            self.encoder.feature_dim + action_dim[0], 1, reward_decoder_hidden_units  # type: ignore
        )
        self.reward_decoder = reward_decoder
        self.recurrent = recurrent

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
        return priors, posteriors

    def infer_full_sequence(self, obs, actions=None, rewards=None):
        # shift rewards by 1 timestep
        if rewards is not None:
            rewards = rewards[:, :-1]
            rewards = torch.cat([torch.zeros_like(rewards[:, :1]), rewards], dim=1)
        if rewards is None:
            inp = obs
        else:
            inp = join_state_with_array(obs, rewards)
        features_seq = self.encoder(inp)
        features_seq_predicted = self._predict_features(features_seq, actions)
        return features_seq, features_seq_predicted

    def _predict_features(self, features_seq, actions):
        h = actions.shape[1]
        features_seq = features_seq.unsqueeze(2).repeat(1, 1, h, 1)
        predictions = [features_seq]

        for i in range(h):
            next_step_cont = self.latent(predictions[-1], actions[:, i])
            next_step = self.latent(features_seq[:, :, i], actions[:, i])
            next_step = torch.cat(
                [next_step_cont[:, :, :i], next_step[:, :, i:]], dim=1
            )
            predictions.append(next_step)
        return torch.stack(predictions, 1)

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ):

        dones = 1.0 - dones
        stats = {}

        b = obs.shape[0]
        h = obs.shape[1]
        obs_shape = obs.shape[2:]

        # encoder_decoder loss

        obs = obs.view(b * h, *obs_shape)
        features_mean, features_scale = self.encoder(
            join_state_with_array(obs, rewards)
        )

        posterior = torch.distributions.Normal(features_mean, features_scale)
        prior = torch.distributions.Normal(torch.zeros_like(features_mean.loc), 1.0)
        kl = kl_divergence(posterior, prior)
        kl_loss = kl.sum(dim=1)

        recon = self.decoder(posterior.rsample())
        recon_loss = torch.sum((recon - obs) ** 2, dim=[1, 2, 3])

        loss = torch.mean(recon_loss + kl_loss)

        stats["vae_loss"] = loss.detach().cpu()
        stats["kl"] = kl.detach().cpu()
        stats["recon"] = recon_loss.detach().cpu()

        # latent dynamics loss

        features = features_mean.view(b, h, -1).detach()
        predictions = self._predict_features(features, actions)
        masking_matrix = torch.triu(torch.ones(h, h), 1).unsqueeze(0).unsqueeze(-1)
        discount_matrix = exponential_moving_matrix(h, 0.99).unsqueeze(0).unsqueeze(-1)

        features = features.unsqueeze(2).repeat(1, 1, h, 1)

        prediction_loss = torch.mean(
            torch.sum(
                masking_matrix * discount_matrix * (features - predictions) ** 2, dim=1
            )
            * (1 / discount_matrix.sum(1))
        )

        stats["prediction_loss"] = prediction_loss.detach().cpu()

        with torch.no_grad():
            recon_predictions = self.decoder(predictions).view(b, h, h, -1)
            obs = obs.view(b, h, -1).unsqueeze(1).repeat(1, h, 1, 1)
            recon_loss_prediction = torch.mean((recon_predictions - obs) ** 2)
            stats["recon_loss_prediction"] = recon_loss_prediction.detach().cpu()

        return prediction_loss + loss, stats

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
