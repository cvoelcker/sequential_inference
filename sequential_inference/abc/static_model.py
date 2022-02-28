import abc
from typing import List, Tuple
import torch
from sequential_inference.abc.common import AbstractAlgorithm


class AbstractLatentStaticAlgorithm(AbstractAlgorithm):
    @abc.abstractmethod
    def infer_latent(
        self, obs: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.distributions.Distribution]]:
        pass

    @abc.abstractmethod
    def reconstruct(self, latent_sample: torch.Tensor) -> torch.Tensor:
        pass
