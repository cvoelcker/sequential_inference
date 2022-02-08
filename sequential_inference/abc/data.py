import abc
from typing import Dict

import gym
import torch

from sequential_inference.abc.common import Checkpointable


class Env(abc.ABC):
    pass


class AbstractDataBuffer(Checkpointable):
    pass


class AbstractDataHandler(Checkpointable):

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
