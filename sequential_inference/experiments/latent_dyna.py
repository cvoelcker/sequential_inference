from typing import Dict

import torch

from sequential_inference.abc.data import Env
from sequential_inference.experiments.base import ModelBasedRLTrainingExperiment
from sequential_inference.util.rl_util import rollout_with_policy


class DynaTrainingExperiment(ModelBasedRLTrainingExperiment):

    # TODO: Handle dual data sampling strategies

    is_rl = True
    is_model = True

    def train_step(self) -> Dict[str, torch.Tensor]:
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        stats = self.model_train_step(*unpacked_batch)

        rl_batch = self.data.get_model_batch(self.model_batch_size)
        unpacked_rl_batch = self.unpack_batch(rl_batch)
        rl_stats = self.rl_train_step(*unpacked_rl_batch)

        return {**stats, **rl_stats}


class LatentImaginationExperiment(ModelBasedRLTrainingExperiment):

    is_rl = True
    is_model = True

    def __init__(
        self,
        env: Env,
        horizon: bool,
        batch_size: int,
        epoch_steps: int,
        epochs: int,
        log_frequency: int,
    ):
        super().__init__(env, batch_size, epoch_steps, epochs, log_frequency)

        self.horizon = horizon

    def train_step(self) -> Dict[str, torch.Tensor]:
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        stats = self.model_train_step(*unpacked_batch)

        rl_batch = self.data.get_batch(self.rl_batch_size)
        obs, act, rew, done = self.unpack_batch(rl_batch)
        with torch.no_grad():
            _, latents = self.model_algorithm.infer_sequence(obs, act, rew, done)

        predicted_latents, predicted_actions, rewards = rollout_with_policy(
            latents[:, -1],
            self.model_algorithm,
            self.rl_algorithm.get_agent(),
            self.horizon,
            reconstruct=False,
            explore=True,
        )

        rl_stats = self.rl_train_step(
            predicted_latents, predicted_actions, rewards, None
        )

        return {**stats, **rl_stats}
