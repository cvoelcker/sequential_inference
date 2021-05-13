import abc
from typing import Dict, Optional, Tuple

import torch

from sequential_inference.abc.common import AbstractAlgorithm, Checkpointable


class AbstractTrainer(Checkpointable, metaclass=abc.ABCMeta):
    def __init__(
        self,
        algorithm: Optional[AbstractAlgorithm] = None,
        lr: float = 3e-4,
        gradient_norm_clipping: float = -1.0,
    ):

        self.algorithm = algorithm
        self.optimizer = torch.optim.Adam(self.algorithm.get_parameters(), lr=lr)
        self.gradient_norm_clipping = gradient_norm_clipping

        self.register_module("algo", self.algorithm)
        self.register_module("optim", self.optimizer)

        self.device = "cpu"

    def train_step(self, batch):
        batch = self.unpack_batch(batch)

        loss, stats = self.algorithm.compute_loss(*batch)
        stats["loss"] = loss.detach().cpu()

        self.optim.zero_grad()
        loss.backward()

        if self.gradient_norm_clipping > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.algorithm.get_parameters(), self.gradient_norm_clipping
            )

        self.optim.step()
        return stats

    @abc.abstractmethod
    def unpack_batch(self, batch):
        pass


class AbstractRLTrainer(AbstractTrainer):
    def unpack_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assumes a full RL batch
        Assumes RL convention r_t = f(s_t, a_t) and s_{t+1} = f(s_t, a_t)

        Args:
            batch (Dict[str, torch.Tensor]): dictionary of batch tensors

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: separate torch Tensors for observation, action, reward, next observation
        """
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = obs[:, 1:]
        obs = obs[:, :-1]
        return obs, act, rew, next_obs

    @abc.abstractmethod
    def get_policy(self):
        pass


class AbstractModelOnlineRLTrainer(AbstractTrainer):
    """TODO: the dreamer style training needs to be integrated

    Args:
        AbstractTrainer ([type]): [description]
    """

    def unpack_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @abc.abstractmethod
    def get_policy(self):
        pass
