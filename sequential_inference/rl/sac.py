import copy
from sequential_inference.abc.trainer import AbstractRLTrainer
from sequential_inference.models.rl import PolicyNetwork, ValueNetwork
from sequential_inference.abc.rl import AbstractRLAlgorithm

import numpy as np
import math

import torch
from torch import nn
from torch.distributions import normal


class SACTrainer(AbstractRLTrainer):

    EMPTY_STAT_DICT = {
        "value_loss": 0.0,
        "actor_loss": 0.0,
        "value_grad_norm": 0.0,
        "actor_grad_norm": 0.0,
        "alpha": 0.0,
    }

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        value_network_layers: int,
        policy_network_layers: int,
        alpha: float,
        gamma: float,
        grad_clip: float,
        target_gamma: float,
        q_learning_rate: float,
        policy_learning_rate: float,
        update_alpha: bool = False,
        pass_task: bool = False,
    ):
        super().__init__()

        if pass_task:
            input_dim = input_dim + 1

        self.q_1 = ValueNetwork(input_dim, action_dim, value_network_layers)
        self.q_2 = ValueNetwork(input_dim, action_dim, value_network_layers)
        self.q_1_target = ValueNetwork(input_dim, action_dim, value_network_layers)
        self.q_2_target = ValueNetwork(input_dim, action_dim, value_network_layers)

        self.policy = PolicyNetwork(input_dim, action_dim, policy_network_layers)

        self.log_alpha = torch.log(torch.tensor([alpha], requires_grad=True))
        self.log_alpha = nn.Parameter(self.log_alpha)
        self.update_alpha = update_alpha
        self.gamma = gamma
        self.target_entropy = -action_dim

        self.grad_clip_rl = grad_clip
        self.target_gamma = target_gamma

        self.optim_q_1 = torch.optim.Adam(self.q_1.parameters(), lr=q_learning_rate)
        self.optim_q_2 = torch.optim.Adam(self.q_2.parameters(), lr=q_learning_rate)
        self.optim_policy = torch.optim.Adam(
            self.policy.parameters(), policy_learning_rate
        )
        self.optim_alpha = torch.optim.Adam(
            [
                self.log_alpha,
            ],
            0.01,
        )

        self.register_module("q1", self.q_1)
        self.register_module("q2", self.q_2)
        self.register_module("q1t", self.q_1_target)
        self.register_module("q2t", self.q_2_target)
        self.register_module("policy", self.policy)
        self.register_module("alpha", self.log_alpha)
        self.register_module("optim1", self.optim_q_1)
        self.register_module("optim2", self.optim_q_2)
        self.register_module("optim_policy", self.optim_policy)
        self.register_module("optim_alpha", self.optim_alpha)

    @property
    def alpha(self):
        return torch.exp(self.log_alpha) + 1e-5

    def train_step(self, batch):
        s, a, r, s_n, done = self.unpack_batch(batch)

        return self.sac_update(s, s_n, a, r, done)

    def sac_update(self, obs, next_obs, actions, rewards, done):
        """
        Gathers the latent inference for RL, the q and policy losses and performsthe update step
        """
        q1_loss, q2_loss = self.critic_loss(obs, next_obs, actions, rewards, done)
        q1_grad_norm = self.optim_step(
            self.optim_q_1, q1_loss, self.q_1, self.grad_clip_rl, retain=True
        )
        q2_grad_norm = self.optim_step(
            self.optim_q_2, q2_loss, self.q_2, self.grad_clip_rl, retain=True
        )

        self.q_1.requires_grad = False
        self.q_2.requires_grad = False
        actor_loss, log_probs = self.actor_loss(obs)
        actor_grad_norm = self.optim_step(
            self.optim_policy, actor_loss, self.policy, self.grad_clip_rl, retain=True
        )
        self.q_1.requires_grad = True
        self.q_2.requires_grad = True

        if self.update_alpha:
            # print("alpha:   {}".format(self.alpha))
            alpha_loss = self.alpha_loss(log_probs)
            self.optim_step(self.optim_alpha, alpha_loss, self.alpha, 0.0, no_clip=True)

        # update target networks for next round
        self.update_target_networks()

        with torch.no_grad():
            actions = self.policy.get_deterministic_action(obs)

        return {
            "value_loss": ((q1_loss + q2_loss) / 2).detach(),
            "actor_loss": actor_loss.detach(),
            "value_grad_norm": (q1_grad_norm + q2_grad_norm) / 2,
            "actor_grad_norm": actor_grad_norm,
            "alpha": self.alpha.detach(),
            "mean_action": actions.mean().detach(),
        }

    def critic_loss(self, obs, next_obs, actions, rewards, done):
        with torch.no_grad():
            # calculate one step lookahead of policy
            act, logprob = self.policy.sample_action(next_obs)
            next_q1 = self.q_1_target(next_obs, act)
            next_q2 = self.q_2_target(next_obs, act)
            next_q = torch.min(next_q1, next_q2) - self.alpha * logprob
            target_q = rewards.view(next_q.shape) + self.gamma * next_q

        # explicit shape handling to counter some serious bugs previously observed
        q1 = self.q_1(obs, actions).view(target_q.shape)
        q2 = self.q_2(obs, actions).view(target_q.shape)

        # masking out loss instead of transition on done to prevent introducing false terminating states
        q1_loss = torch.mean(
            (1.0 - done.view(next_q.shape)) * (q1 - target_q.detach()).pow(2)
        )
        q2_loss = torch.mean(
            (1.0 - done.view(next_q.shape)) * (q2 - target_q.detach()).pow(2)
        )

        return q1_loss, q2_loss

    def actor_loss(self, obs):
        actions, logprob = self.policy.sample_action(obs)
        q = torch.min(self.q_1(obs, actions), self.q_2(obs, actions))
        loss = torch.mean((self.alpha.detach() * logprob - q))
        return loss, logprob

    def alpha_loss(self, log_probs):
        return -(
            torch.exp(self.log_alpha) * (log_probs.detach() + self.target_entropy)
        ).mean()

    def optim_step(self, optim, loss, net, clip, no_clip=False, retain=True):
        optim.zero_grad()
        loss.backward(retain_graph=retain)
        if not no_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
        else:
            grad_norm = 1.0
        optim.step()
        return grad_norm

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
                self.q_1_target.parameters(), self.q_1.parameters()
            ):
                param_target.data.copy_(softupdate(param_target, param))
            for param_target, param in zip(
                self.q_2_target.parameters(), self.q_2.parameters()
            ):
                param_target.data.copy_(softupdate(param_target, param))

    def copy_q(self, net):
        copy_net = copy.deepcopy(net)
        for p in copy_net.parameters():
            p.requires_grad = False
        return copy_net

    def get_policy(self):
        return PolicyNetworkAgent(self.policy)
