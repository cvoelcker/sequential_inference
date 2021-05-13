from typing import Optional

import torch

from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from sequential_inference.abc.rl import AbstractAgent


class InferencePolicyAgent(AbstractAgent):
    """Wraps a policy and an inference model to provide a stateful representation
    of the current latent state. Used with autoregressive and recurrent algorithms

    Stateful!
    """

    def __init__(self, policy: AbstractAgent, model: AbstractSequenceAlgorithm):
        self.policy = policy
        self.model = model

        self.state: Optional[torch.Tensor] = None
        self.last_action: Optional[torch.Tensor] = None

    def reset(self):
        """Resets the state to None so that the next inference call to the model will initialize
        the sequence
        """
        self.state: Optional[torch.Tensor] = None
        self.last_action: Optional[torch.Tensor] = None

    def act(
        self,
        observation: torch.Tensor,
        reward: Optional[torch.Tensor],
        context: Optional[torch.Tensor] = None,
        explore: bool = False,
    ):
        self.state = self.model.infer_single_step(
            self.state, observation, self.last_action, reward
        )
        action = self.policy.act(observation, reward, self.state, explore)
        self.last_action = action

        return action


class PolicyNetworkAgent(AbstractAgent):
    def __init__(self, policy: torch.nn.Module, latent: bool, observation: bool):
        self.policy = policy
        self.latent = latent
        self.observation = observation

    def act(
        self,
        observation: torch.Tensor,
        reward: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        explore: bool,
    ) -> torch.Tensor:
        if self.latent and not self.observation:
            inp = [context]
        elif not self.latent and self.observation:
            inp = [observation]
        elif self.latent and self.observation:
            inp = [observation, context]
        if explore:
            return self.policy.sample_action(*inp)[0]
        return self.policy.get_deterministic_action(*inp)
