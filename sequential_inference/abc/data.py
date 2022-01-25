import abc
from typing import Dict

import gym
import torch

from sequential_inference.abc.common import Checkpointable


class Env(abc.Abc):
    pass


Env.register(gym.Env)


class AbstractDataBuffer(Checkpointable):
    pass

class AbstractDataHandler(Checkpointable):

    buffer: AbstractDataBuffer

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
    def initialize(self, cfg, preempted: bool, run_dir: str):
        pass

    def update(self, log, agent, **kwargs):
        return log