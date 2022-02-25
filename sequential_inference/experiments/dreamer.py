from typing import Dict

import torch
from sequential_inference.abc.data import Env
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from sequential_inference.algorithms.rl.dreamer import DreamerRLAlgorithm

from sequential_inference.experiments.base import ModelBasedRLTrainingExperiment
from sequential_inference.nn_models.base.network_util import FreezeParameters
from sequential_inference.util.errors import NotInitializedException


class DreamerExperiment(ModelBasedRLTrainingExperiment):

    is_rl = True
    is_model = True

    def __init__(
        self,
        horizon: bool,
        epoch_steps: int,
        epochs: int,
        model_algorithm: AbstractSequenceAlgorithm,
        rl_algorithm: DreamerRLAlgorithm,
        model_batch_size: int = 32,
        rl_batch_size: int = 32,
        log_interval: int = 100,
    ):
        assert isinstance(
            rl_algorithm, DreamerRLAlgorithm
        ), "rl_algorithm must be a DreamerAlgorithm"
        super().__init__(
            epoch_steps,
            epochs,
            model_algorithm,
            rl_algorithm,
            model_batch_size=model_batch_size,
            rl_batch_size=rl_batch_size,
            log_interval=log_interval,
        )

        self.horizon = horizon

    def train_step(self):
        # update the planet model
        if self.data is None:
            raise NotInitializedException("data must be set before training")
        batch = self.data.get_batch(self.model_batch_size)
        unpacked_batch = self.unpack_batch(batch)
        stats = self.model_train_step(*unpacked_batch)
        stats["model_step_cuda"] = torch.Tensor([torch.cuda.memory_reserved()]).float()
        # update the rl model
        rl_stats = self.rl_train_step()
        stats["rl_step_cuda"] = torch.Tensor([torch.cuda.memory_reserved()]).float()
        return {**stats, **rl_stats}

    def rl_train_step(self) -> Dict[str, torch.Tensor]:
        # part one: calculate regression target for value function prediction
        if self.data is None:
            raise NotInitializedException("data must be set before training")
        batch = self.data.get_batch(self.rl_batch_size)
        obs, act, rew, done = self.unpack_batch(batch)
        batch_size = obs.shape[0]

        with FreezeParameters(self.model_algorithm.parameters()):
            with torch.no_grad():
                _, latents = self.model_algorithm.infer_sequence(
                    obs, act, rew, full=True
                )
            assert latents.shape[0] == self.rl_batch_size
            (
                predicted_latents,
                predicted_actions,
                predicted_rewards,
            ) = self.model_algorithm.rollout_with_policy(
                latents[:, -1],
                self.rl_algorithm.get_agent(),
                self.horizon,
                reconstruct=False,
                explore=True,
            )

            obs = self.model_algorithm.get_samples([[latents[:, -1:].detach()]])
            next_obs = predicted_latents[:, 1:]
            rewards = predicted_rewards

            losses, stats = self.rl_algorithm.compute_loss(
                obs, next_obs, None, rewards, None
            )

            stats = self._step_rl(losses, stats)

        return stats
