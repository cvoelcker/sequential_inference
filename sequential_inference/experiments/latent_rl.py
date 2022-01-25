from typing import Dict

import torch

from sequential_inference.abc.data import Env
from sequential_inference.experiments.base import ModelBasedRLTrainingExperiment
from sequential_inference.algorithms.rl.agents import InferencePolicyAgent


class LatentTrainingExperiment(ModelBasedRLTrainingExperiment):

    is_rl = True
    is_model = True

    def __init__(
        self,
        env: Env,
        pass_rl_gradients_to_model: bool,
        batch_size: int,
        rl_batch_size: int,
        epoch_steps: int,
        epochs: int,
        log_frequency: int,
    ):
        super().__init__(env, batch_size, epoch_steps, epochs, log_frequency)
        self.rl_batch_size = rl_batch_size
        self.pass_rl_gradients_to_model = pass_rl_gradients_to_model

    def train_step(self) -> Dict[str, torch.Tensor]:
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        model_loss = self.model_algorithm.compute_loss(*unpacked_batch)

        rl_batch = self.data.get_batch(self.rl_batch_size)
        rl_batch = self.unpack_batch(rl_batch)
        obs, act, rew, done = rl_batch
        if self.pass_rl_gradients_to_model:
            _, latents = self.model_algorithm.infer_sequence(obs, act, rew)
            rl_loss, rl_stats = self.rl_algorithm.compute_loss(latents, act, rew, done)
            self._step_rl(
                rl_loss,
            )
            self._step_model(model_loss + rl_loss)
        else:
            with torch.no_grad():
                _, latents = self.model_algorithm.infer_sequence(obs, act, rew)
            loss, rl_stats = self.rl_algorithm.compute_loss(latents, act, rew, done)
            self.step_rl_optimizer(loss)

        return {**model_stats, **rl_stats}

    def get_agent(self) -> InferencePolicyAgent:
        agent = self.rl_algorithm.get_agent()
        agent = InferencePolicyAgent(agent, self.model_algorithm)
        agent.latent = True
        agent.observation = False
        return agent
