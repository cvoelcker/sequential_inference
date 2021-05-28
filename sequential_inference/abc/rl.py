import abc
from typing import Optional

import numpy as np
import torch

from sequential_inference.abc.common import AbstractAlgorithm


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
        action: np.ndarray,
        reward: torch.Tensor,
        context: torch.Tensor,
    ) -> np.ndarray:
        pass

    def act(
        self,
        observation: torch.Tensor,
        reward: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        explore: bool = False,
    ) -> np.ndarray:
        proposed_action = self.agent.act(observation, reward, context, explore)
        return self.exploration_policy(observation, proposed_action, reward, context)


class AbstractStatefulAgent(AbstractAgent, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reset(self):
        """Resets the internal agent on environment reset"""
        pass
