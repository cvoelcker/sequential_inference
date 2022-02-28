import abc
from typing import Dict, Sized

import torch
from torch.utils.data import Dataset

from sequential_inference.abc.common import (
    Env,
    TorchContainer,
)
from sequential_inference.abc.rl import AbstractAgent


class AbstractDataBuffer(Dataset, TorchContainer, Sized):
    sample_length: int

    def insert(self, **kwargs: Dict[str, torch.Tensor]) -> None:
        pass


class AbstractDataHandler(abc.ABC):

    buffer: AbstractDataBuffer
    env: Env

    def __init__(
        self,
        env: Env,
        buffer: AbstractDataBuffer,
    ):
        self.env = env
        self.buffer = buffer

    @abc.abstractmethod
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        pass

    @abc.abstractmethod
    def initialize(self, cfg, preempted: bool):
        pass

    def load(self, path: str) -> None:
        pass

    def update(
        self, log: Dict[str, torch.Tensor], agent: AbstractAgent, **kwargs
    ) -> Dict[str, torch.Tensor]:
        return log


class AbstractDataSampler(abc.ABC):
    @abc.abstractmethod
    def get_next(self, batch_size: int):
        pass
