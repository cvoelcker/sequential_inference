from typing import Optional
from gym import Space

import torch

from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from sequential_inference.abc.rl import AbstractAgent


class RandomAgent(AbstractAgent):
    def __init__(self, action_space: Space):
        self.action_space = action_space

    def _sample_n(self, n: int):
        for _ in range(n):
            action = self.action_space.sample()
            yield action

    def act(
        self,
        observation: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        explore: bool = False,
    ) -> torch.Tensor:
        device = observation.device
        batch_size = observation.shape[0]

        return torch.stack(
            [torch.from_numpy(a).to(device) for a in self._sample_n(batch_size)], 0
        )


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

    def __init__(
        self,
        policy: torch.nn.Module,
        latent: bool,
        observation: bool,
        max_latent_size: Optional[int] = None,
    ):
        self.policy = policy
        self.latent = latent
        self.observation = observation
        self.max_latent_size = max_latent_size

    def act(
        self,
        observation: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        explore: bool = False,
    ) -> torch.Tensor:
        if self.latent:
            assert context is not None, "Context must be provided for latent policy"
            if self.max_latent_size is None:
                latent = context
            else:
                latent = context[:, : self.max_latent_size]
        else:
            latent = None

        if self.latent and not self.observation:
            inp = [latent]
        elif not self.latent and self.observation:
            inp = [observation]
        elif self.latent and self.observation:
            inp = [latent, context]
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

    def __init__(
        self,
        policy: AbstractAgent,
        model: AbstractSequenceAlgorithm,
        max_latent_size: Optional[int] = None,
    ):
        self.policy = policy
        self.model = model

        self.state: Optional[torch.Tensor] = None
        self.last_action: Optional[torch.Tensor] = None
        self.last_reward: Optional[torch.Tensor] = None

        self.max_latent_size = max_latent_size

    def reset(self):
        """Resets the state to None so that the next inference call to the model will initialize
        the sequence
        """
        self.state: Optional[torch.Tensor] = None
        self.last_action: Optional[torch.Tensor] = None
        self.last_reward: Optional[torch.Tensor] = None

    def act(
        self,
        observation: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        explore: bool = False,
    ) -> torch.Tensor:
        if context:
            assert context is not None, "Context must be provided for latent policy"
            if self.max_latent_size is None:
                latent = context
            else:
                latent = context[:, : self.max_latent_size]
        elif not context and observation:
            if self.state is None:
                self.state = self.model.latent.obtain_initial(observation)
            if self.last_reward is None:
                self.last_reward = torch.zeros_like(self.state[:, :, :1])
            latent = self.model.infer_single_step(
                self.state, observation, self.last_action, self.last_reward
            )
        else:
            raise ValueError("Policy needs to depend on something ^^")

        action = self.policy.act(observation, reward, latent, explore)
        self.last_action = action
        return action
