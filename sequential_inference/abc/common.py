import abc
from typing import Dict, Optional, OrderedDict

import torch


class Checkpointable(abc.ABC):

    device: str

    def __init__(self):
        self.device = "cpu"
        self.model_buffer: Dict[str, Checkpointable] = {}
        super().__init__()

    def state_dict(self):
        to_save = {}
        for k, v in self.model_buffer.items():
            to_save[k] = v.state_dict()
        return to_save

    def load_state_dict(self, chp: Dict[str, OrderedDict]):
        for k, v in self.model_buffer.items():
            v.load_state_dict(chp[k])

    def load(self, directory: str):
        chp = torch.load(directory, map_location=self.device)
        self.load_state_dict(chp)

    def register_module(self, key: str, module: torch.nn.Module):
        if key in self.model_buffer.keys():
            raise KeyError("Key in module buffer is not unique")
        else:
            self.model_buffer[key] = module

    def to(self, device: str):
        self.device = device
        for _, v in self.model_buffer.items():
            v.to(device)


Checkpointable.register(torch.nn.Module)


class AbstractAlgorithm(Checkpointable, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        done: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate Abstract")

    def get_parameters(self):
        params = []
        for k, v in self.model_buffer.items():
            # needs to concatenate the list to flatten them
            params += list(v.parameters())
        return params
