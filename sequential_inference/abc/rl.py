import abc
from typing import Optional

import torch


class AbstractPolicy(abc.ABC):
    @abc.abstractmethod
    def act(self, observation: torch.Tensor, context: Optional[torch.Tensor] = None):
        pass
