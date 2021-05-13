from _typeshed import NoneType
import abc
from sequential_inference.envs.mujoco.rand_param_envs.gym.core import ActionWrapper
from sequential_inference.abc.common import AbstractAlgorithm
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from typing import Optional

import torch


class AbstractAgent(abc.ABC):
    @abc.abstractmethod
    def act(
        self,
        observation: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        explore: bool = False,
    ) -> torch.Tensor:
        """Calculate a single action from the policy given observation (and optionally a last reward
        and a context variable that can hold e.g. a latent state)

        Args:
            observation (torch.Tensor): the current environment observation
            reward (Optional[torch.Tensor]): the last recieved reward. Defaults to None.
            context (Optional[torch.Tensor], optional): a context vector, e.g. with the latent state of an inference model. Defaults to None.

        Returns:
            torch.Tensor: the next action to take
        """
        pass


class AbstractRLAlgorithm(AbstractAlgorithm, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_agent(self) -> AbstractAgent:
        pass


class AbstractExplorationAgent(AbstractAgent, metaclass=abc.ABCMeta):
    def __init__(self, agent: AbstractAgent):
        self.agent = agent

    @abc.abstractmethod
    def exploration_policy(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def act(
        self,
        observation: torch.Tensor,
        reward: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        explore: bool,
    ) -> torch.Tensor:
        proposed_action = self.agent.act(observation, reward, context, explore)
        return self.exploration_policy(observation, proposed_action, reward, context)
