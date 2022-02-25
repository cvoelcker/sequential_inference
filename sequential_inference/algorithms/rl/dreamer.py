from re import A
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import nn
from torch.optim import Adam

from sequential_inference.abc.rl import AbstractAgent, AbstractRLAlgorithm
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from sequential_inference.algorithms.rl.agents import (
    InferencePolicyAgent,
    PolicyNetworkAgent,
)


class DreamerRLAlgorithm(AbstractRLAlgorithm):
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        latent_size: int,
        actor_lr: float,
        critic_lr: float,
        lambda_discount: float,
        gamma_discount: float,
        horizon: int,
    ):

        super().__init__()
        self.latent_size = latent_size

        self.actor = actor
        self.value = critic

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
        self.register_module("value", critic)
        self.register_module("actor_optimizer", self.actor_optimizer)
        self.register_module("critic_optimizer", self.critic_optimizer)

    def to(self, device):
        super().to(device)
        self.horizon_discount = self.horizon_discount.to(device)
        self.discount = self.discount.to(device)

    def compute_loss(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict]:
        if rewards is None:
            raise ValueError("Rewards are required")

        batch_size = obs.shape[0]

        # predict values
        predicted_values = self.value(next_obs)
        values = self.value(obs)

        # create the weighted prediction matrices
        predicted_values = (
            predicted_values.view(batch_size, self.horizon) * self.discount
        )
        rewards = rewards.view(batch_size, self.horizon) * self.discount
        cumulative_predicted_rewards = torch.cumsum(rewards, dim=1)  # type: ignore
        cumulative_predicted_rewards = cumulative_predicted_rewards + predicted_values
        target = (cumulative_predicted_rewards * self.horizon_discount).sum(-1).detach()

        # value loss
        value_loss = torch.mean((target - values) ** 2)

        # action loss
        action_loss = -torch.mean(
            torch.sum(rewards, -1) + predicted_values[:, -1]  # type: ignore
        )

        return (value_loss, action_loss), {
            "action_loss": action_loss.detach(),
            "value_loss": value_loss.detach(),
        }

    def get_step(self) -> Callable[[Tuple, Dict], Dict]:
        def _step(losses, stats):
            value_loss, actor_loss = losses

            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            value_loss.backward()
            self.critic_optimizer.step()

            return stats

        return _step

    def get_agent(self, model: AbstractSequenceAlgorithm) -> AbstractAgent:
        inner_agent = PolicyNetworkAgent(
            self.actor, latent=True, observation=False, max_latent_size=self.latent_size
        )
        return InferencePolicyAgent(
            inner_agent, model, max_latent_size=self.latent_size
        )
