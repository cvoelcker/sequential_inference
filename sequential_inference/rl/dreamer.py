from typing import Optional
from torch import nn
import torch
from sequential_inference.abc.rl import AbstractAgent, AbstractRLAlgorithm
from sequential_inference.rl.agents import PolicyNetworkAgent


class DreamerAlgorithm(AbstractRLAlgorithm):

    def __init__(
        self, 
        actor: nn.Module,
        value: nn.Module,
        lambda_discount: float,
        gamma_discount: float, 
        horizon: int):

        super().__init__()
        self.actor = actor
        self.value = value
        self.lambda_discount = lambda_discount
        self.gamma_discount = gamma_discount
        self.horizon = horizon

        self.register_module("actor", actor)
        self.register_module("value", value)


    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

            values = self.value(obs)
            predicted_values = values[:, 1:]
            values = values[:, :1]

            predicted_rewards = rewards

            # create the weighted prediction matrices
            predicted_values = predicted_values.view(self.batch_size, self.horizon) * self.discount
            predicted_values_mat = torch.diag_embed(predicted_values, offset=1)
            predicted_rewards = predicted_rewards.view(self.batch_size, self.horizon) * self.discount
            predicted_rewards_mat = predicted_rewards[:, None, :].repeat(1, self.horizon, 1)
            predicted_rewards_mat = torch.tril(predicted_rewards_mat)

            # calculate the weighted target
            predicted_results = predicted_rewards_mat + predicted_values_mat
            predicted_results = torch.sum(predicted_results, dim=1)
            predicted_results = predicted_results * self.horizon_discount
            target = (torch.sum(predicted_results, -1) * (1 - self.discount)).detach()

            # value loss
            value_loss = torch.mean((target - values) ** 2)

            # action loss
            action_loss = -torch.mean(torch.sum(predicted_rewards, -1)  + predicted_values[:, -1].detach())

            return value_loss, action_loss


    def get_agent(self) -> AbstractAgent:
        return PolicyNetworkAgent(self.policy, latent=True, observation=False)