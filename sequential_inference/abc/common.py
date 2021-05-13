import abc
from typing import Dict, Optional, OrderedDict

import torch


class Checkpointable:

    device: str

    def __init__(self):
        self.buffer: Dict[str, OrderedDict] = {}
        super().__init__()

    def state_dict(self):
        to_save = {}
        for k, v in self.buffer.items():
            to_save[k] = v.state_dict()
        return to_save

    def load_state_dict(self, chp: Dict[str, OrderedDict]):
        for k, v in self.buffer.items():
            v.load_state_dict(chp[k])

    def register_module(self, key: str, module: torch.nn.Module):
        if key in self.buffer.keys():
            raise KeyError("Key in module buffer is not unique")
        else:
            self.buffer[key] = module

    def to(self, device: str):
        self.device = device
        for _, v in self.buffer.items():
            v.to(device)


class AbstractAlgorithm(Checkpointable, metaclass=abc.ABCMeta):
    def __init__(self):
        self.device: str = "cpu"
        self.buffer: Dict[str, OrderedDict] = {}
        super().__init__()

    @abc.abstractmethod
    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate Abstract")

    def to(self, device: str):
        self.device = device
        for k, v in self.buffer.items():
            v.to(self.device)

    def get_parameters(self):
        params = []
        for k, v in self.buffer.items():
            # needs to concatenate the list to flatten them
            params += list(v.parameters())
        return params
