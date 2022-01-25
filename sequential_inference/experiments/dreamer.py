from typing import Dict

import torch
from sequential_inference.abc.env import Env

from sequential_inference.experiments.base import ModelTrainingExperiment, RLTrainingExperiment
from sequential_inference.models.base.network_util import FreezeParameters
from sequential_inference.util.rl_util import rollout_with_policy


class DreamerExperiment(RLTrainingExperiment, ModelTrainingExperiment):

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
        self.discount = (torch.ones((1, self.horizon)) * self.rl_algorithm.discount) ** torch.arange(self.horizon)
        self.horizon_discount = (torch.ones((1, self.horizon)) * self.discount) ** torch.arange(self.horizon)
        self.discount = self.discount.to(self.device)
        self.horizon_discount = self.horizon_discount.to(self.device)

    def train_step(self):
        # update the planet model
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        stats = self.model_train_step(*unpacked_batch)

        # update the rl model
        rl_stats = self.rl_train_step()
        
        return {**stats, **rl_stats}


    def rl_train_step(self) -> Dict[str, torch.Tensor]:
        # part one: calculate regression target for value function prediction
        batch = self.data.get_batch(self.batch_size)
        obs, act, rew, done = self.unpack_batch(batch)

        with FreezeParameters(self.model_algorithm.get_parameters()):
            _, latents = self.model_algorithm.infer_sequence(obs, act, rew, done)
            assert latents.shape[0] == self.batch_size
            predicted_latents, predicted_actions, predicted_rewards = rollout_with_policy(
                latents[:, -1], 
                self.model_algorithm, 
                self.rl_algorithm.get_agent(), 
                self.horizon, 
                reconstruct=False, 
                explore=True)
            predicted_values = self.rl_algorithm.get_values(predicted_latents)

            # DEBUG assertions
            assert predicted_latents.shape[0] == self.batch_size, f"predicted_latents.shape[0]: {predicted_latents.shape[0]} != self.batch_size: {self.batch_size}"
            assert predicted_actions.shape[0] == self.batch_size, f"predicted_actions.shape[0]: {predicted_actions.shape[0]} != self.batch_size: {self.batch_size}"
            assert predicted_rewards.shape[0] == self.batch_size, f"predicted_rewards.shape[0]: {predicted_rewards.shape[0]} != self.batch_size: {self.batch_size}"
            assert predicted_latents.shape[1] == self.horizon, f"predicted_latents.shape[1]: {predicted_latents.shape[1]} != self.horizon: {self.horizon}"
            assert predicted_actions.shape[1] == self.horizon, f"predicted_actions.shape[1]: {predicted_actions.shape[1]} != self.horizon: {self.horizon}"
            assert predicted_rewards.shape[1] == self.horizon, f"predicted_rewards.shape[1]: {predicted_rewards.shape[1]} != self.horizon: {self.horizon}"
            assert predicted_values.shape == predicted_rewards.shape, f"predicted_values.shape: {predicted_values.shape} != predicted_rewards.shape: {predicted_rewards.shape}"

            obs = torch.cat([latents[:, -1:], predicted_latents], dim=1)
            rewards = torch.cat([rew, predicted_rewards], dim=1)

            value_loss, actor_loss = self.rl_algorithm.compute_loss(obs, None, rewards, None)

        self.value_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        value_loss.backward()
        actor_loss.backward()

        self.value_optimizer.step()
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.detach(), "value_loss": value_loss.detach()}
