import abc
from sequential_inference.abc.trainer import AbstractRLTrainer, AbstractTrainer
from sequential_inference.abc.common import AbstractAlgorithm
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from typing import Dict, Optional, Tuple

import torch


class ModelTrainer(AbstractTrainer):
    def __init__(
        self,
        algorithm: Optional[AbstractSequenceAlgorithm] = None,
        lr: float = 3e-4,
        gradient_norm_clipping: float = -1.0,
    ):
        super().__init__(algorithm, lr, gradient_norm_clipping)

    def unpack_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assumes a sequence batch
        Assumes RL convention r_t = f(s_t, a_t) and s_{t+1} = f(s_t, a_t)

        Args:
            batch (Dict[str, torch.Tensor]): dictionary of batch tensors

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: separate torch Tensors for observation, action, reward (next observation is implicit)
        """
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        return obs, act, rew
