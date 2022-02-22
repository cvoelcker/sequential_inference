from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.optim import Adam

from sequential_inference.abc.rl import AbstractAgent, AbstractRLAlgorithm
from sequential_inference.algorithms.rl.agents import PolicyNetworkAgent


class DreamerRLAlgorithm(AbstractRLAlgorithm):
    def __init__(
        self,
        actor: nn.Module,
        value: nn.Module,
        actor_lr: float,
        critic_lr: float,
        lambda_discount: float,
        gamma_discount: float,
        horizon: int,
    ):

        super().__init__()
        self.actor = actor
        self.value = value

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.value.parameters(), lr=critic_lr)

        self.lambda_discount = lambda_discount
        self.horizon_discount = self.lambda_discount ** torch.arange(horizon).to(
            self.device
        )
        self.gamma_discount = gamma_discount
        self.discount = self.gamma_discount ** torch.arange(horizon).to(self.device)
        self.horizon = horizon

        self.register_module("actor", actor)
        self.register_module("value", value)
        self.register_module("actor_optimizer", self.actor_optimizer)
        self.register_module("critic_optimizer", self.critic_optimizer)

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict]:
        assert rewards is not None, "Rewards must be provided"

        batch_size = obs.shape[0]
        values: torch.Tensor = self.value(obs)
        predicted_values = values[:, 1:]
        values = values[:, :1]

        predicted_rewards = rewards

        # create the weighted prediction matrices
        predicted_values = (
            predicted_values.view(batch_size, self.horizon) * self.discount
        )
        predicted_rewards = (
            predicted_rewards.view(batch_size, self.horizon) * self.discount
        )
        cumulative_predicted_rewards = torch.cumsum(predicted_rewards, dim=1)
        cumulative_predicted_rewards = cumulative_predicted_rewards + values
        target = (cumulative_predicted_rewards * self.horizon_discount).sum(-1).detach()

        # value loss
        value_loss = torch.mean((target - values) ** 2)

        # action loss
        action_loss = -torch.mean(
            torch.sum(predicted_rewards, -1) + predicted_values[:, -1].detach()
        )

        return (value_loss, action_loss), {
            "action_loss": action_loss.detach(),
            "value_loss": value_loss.detach(),
        }

    def get_step(self) -> Callable[[Tuple, Dict], Dict]:
        def _step(losses, stats):
            value_loss, actor_loss = losses

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            return stats

        return _step

    def get_agent(self) -> AbstractAgent:
        return PolicyNetworkAgent(self.actor, latent=True, observation=False)
