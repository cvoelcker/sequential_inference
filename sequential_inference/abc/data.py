import abc
from typing import Dict

import torch
from torch.utils.data import Dataset

from sequential_inference.abc.common import Checkpointable, Env, Saveable


class AbstractDataBuffer(Dataset, Checkpointable):
    pass


class AbstractDataHandler(Saveable):

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

    def update(self, log, agent, **kwargs):
        return log


class AbstractDataSampler(abc.ABC):
    @abc.abstractmethod
    def get_next(self, batch_size: int):
        pass
