from typing import Optional
from gym import Space

import torch

from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from sequential_inference.abc.rl import AbstractAgent


class RandomAgent(AbstractAgent):
    def __init__(self, action_space: Space):
        self.action_space = action_space

    def act(
        self,
        observation: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        explore: bool = False,
    ) -> torch.Tensor:
        action = self.action_space.sample()
        if len(action.shape) == 1:
            return torch.from_numpy(action).unsqueeze(0)
        return torch.from_numpy(action)


class DummyAgent(AbstractAgent):
    def __init__(self):
        pass

    def act(
        self,
        observations: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        contexts: Optional[torch.Tensor] = None,
        explore: bool = False,
    ) -> torch.Tensor:
        raise RuntimeError(
            "DummyAgent is not meant to act and should only be \
            used in model training experiments or offline RL experiments during \
            training"
        )


class PolicyNetworkAgent(AbstractAgent):
    """SAC type agent where the policy is a neural network representing a distribution"""

    def __init__(self, policy: torch.nn.Module, latent: bool, observation: bool):
        self.policy = policy
        self.latent = latent
        self.observation = observation

    def act(
        self,
        observation: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        explore: bool = False,
    ) -> torch.Tensor:
        if self.latent and not self.observation:
            inp = [context]
        elif not self.latent and self.observation:
            inp = [observation]
        elif self.latent and self.observation:
            inp = [observation, context]
        else:
            raise ValueError("Policy needs to depend on something ^^")
        action_dist = self.policy(*inp)
        if explore:
            action = action_dist.rsample()[0]
        action = action_dist.mean
        return action


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
        reward: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        explore: bool = False,
    ) -> torch.Tensor:
        # TODO: check if inference handles reward=None gracefully
        # TODO: check how initial_state = None is handled
        self.state = self.model.infer_single_step(
            self.state, observation, self.last_action, reward
        )
        action = self.policy.act(observation, reward, self.state, explore)
        self.last_action = action
        return action
