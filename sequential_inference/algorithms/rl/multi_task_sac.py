import copy
from typing import Callable, Dict, Optional, Tuple

import math

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from sequential_inference.abc.rl import AbstractAgent, AbstractRLAlgorithm
from sequential_inference.algorithms.rl.agents import PolicyNetworkAgent
from sequential_inference.util.rl_util import join_state_with_array


class AbstractActorCriticAlgorithm(AbstractRLAlgorithm):
    actor: nn.Module
    critic: nn.Module


class AlphaModule(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.log_alpha = torch.log(torch.tensor([alpha], requires_grad=True))
        self.log_alpha = nn.Parameter(self.log_alpha)  # type: ignore

    def forward(self):
        return torch.exp(self.log_alpha)


class SACAlgorithm(AbstractRLAlgorithm):
    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        alpha: float,
        gamma: float,
        target_gamma: float,
        target_entropy: float,
        n_tasks: int,
        update_alpha: bool = False,
        latent: bool = False,
        observation: bool = True,
    ):
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.q_target = copy.deepcopy(self.critic)
        self.q_target.requires_grad = False
        self.alpha = AlphaModule(alpha)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr

        self.update_alpha = update_alpha
        self.gamma = gamma
        self.target_entropy = target_entropy

        self.target_gamma = target_gamma

        self.n_tasks = n_tasks

        assert (
            self.n_tasks == self.actor.n_tasks and self.n_tasks == self.critic.n_tasks
        ), "Actor and critic not compatible with n_tasks"

        self.latent = latent
        self.observation = observation

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        self.alpha_optimizer = Adam(self.alpha.parameters(), lr=self.alpha_lr)

        self.register_module("q", self.critic)
        self.register_module("qt", self.q_target)
        self.register_module("policy", self.actor)
        self.register_module("alpha", self.alpha)
        self.register_module("actor_optimizer", self.actor_optimizer)
        self.register_module("critic_optimizer", self.critic_optimizer)
        self.register_module("alpha_optimizer", self.alpha_optimizer)

    def to(self, device):
        super().to(device)
        self.actor.to(device)
        self.critic.to(device)
        self.alpha.to(device)
        self.q_target.to(device)

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict]:
        assert actions is not None
        assert rewards is not None
        assert done is not None

        assert rewards.shape[-1] == self.n_tasks

        next_obs = obs[:, -1]
        obs = obs[:, -2]
        actions = actions[:, -2]
        rewards = rewards[:, -2]
        done = done[:, -2]

        q_loss = self.critic_loss(obs, next_obs, actions, rewards, done)

        self.critic.requires_grad = False
        actor_loss, log_probs = self.actor_loss(obs)
        self.critic.requires_grad = True

        if self.update_alpha:
            alpha_loss = self.alpha_loss(log_probs.detach())
        else:
            alpha_loss = torch.zeros_like(q_loss)

        stats = {
            "value_loss": q_loss.detach(),
            "actor_loss": actor_loss.detach(),
            "alpha_loss": alpha_loss.detach(),
            "alpha": self.alpha().detach(),
        }

        return (q_loss, actor_loss, alpha_loss), stats

    def critic_loss(self, obs, next_obs, actions, rewards, done):
        with torch.no_grad():
            # calculate one step lookahead of policy
            action_dist = self.actor(next_obs)
            act, logprob = action_dist.rsample()
            next_q1, next_q2 = self.q_target(torch.cat((next_obs, act), -1))
            next_q = torch.min(next_q1, next_q2) - self.alpha() * logprob
            target_q = rewards.view(next_q.shape) + self.gamma * next_q

        # explicit shape handling to counter some serious bugs previously observed
        q1, q2 = self.critic(torch.cat((obs, actions), -1))
        q1 = q1.view(target_q.shape)
        q2 = q2.view(target_q.shape)

        # masking out loss instead of transition on done to prevent introducing false terminating states
        q1_loss = torch.mean(
            (1.0 - done.view(next_q.shape)) * (q1 - target_q.detach()).pow(2)
        )
        q2_loss = torch.mean(
            (1.0 - done.view(next_q.shape)) * (q2 - target_q.detach()).pow(2)
        )

        return q1_loss + q2_loss

    def actor_loss(self, obs):
        action_dist = self.actor(obs)
        actions, logprob = action_dist.rsample()
        q1, q2 = self.critic(torch.cat((obs, actions), -1))
        q = torch.min(q1, q2)
        loss = torch.mean((self.alpha().detach() * logprob - q))
        return loss, logprob

    def alpha_loss(self, log_probs) -> torch.Tensor:
        return -(self.alpha() * (log_probs.detach() + self.target_entropy)).mean()

    def update_target_networks(self):
        """
        Updates the target q function networks with a moving average of params
        """

        def softupdate(param1, param2):
            return (self.target_gamma * param1.data) + (
                (1 - self.target_gamma) * param2.data
            )

        with torch.no_grad():
            for param_target, param in zip(
                self.q_target.parameters(), self.critic.parameters()
            ):
                param_target.data.copy_(softupdate(param_target, param))

    def copy_q(self, net):
        copy_net = copy.deepcopy(net)
        for p in copy_net.parameters():
            p.requires_grad = False
        return copy_net

    def get_agent(self) -> AbstractAgent:
        return PolicyNetworkAgent(self.actor, self.latent, self.observation)

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_dist = self.actor(obs, 0)
        action, logprob = action_dist.rsample()
        return action, logprob

    def get_step(self) -> Callable[[Tuple, Dict], Dict]:
        def _step(losses: Tuple, stats: Dict) -> Dict:
            update_stats = {}

            (q_loss, actor_loss, alpha_loss) = losses

            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            q_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            if self.update_alpha:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

            self.update_target_networks()

            return {**stats, **update_stats}

        return _step
