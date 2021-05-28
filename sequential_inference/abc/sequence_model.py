import abc
from sequential_inference.abc.common import AbstractAlgorithm
from typing import Dict, Optional, OrderedDict, Tuple
import torch


class AbstractSequenceAlgorithm(AbstractAlgorithm):
    @abc.abstractmethod
    def infer_sequence(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Cannot instantiate AbstractSequenceModel")

    @abc.abstractmethod
    def infer_single_step(
        self,
        last_latent: torch.Tensor,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate AbstractSequenceModel")

    @abc.abstractmethod
    def predict_sequence(
        self,
        initial_latent: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate AbstractSequenceModel")

    @abc.abstractmethod
    def reconstruct(self, latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate AbstractSequenceModel")

    @abc.abstractmethod
    def reconstruct_reward(self, latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate AbstractSequenceModel")


class AbstractLatentModel(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def infer_prior(
        self,
        last_latent: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        global_belief: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate Abstract")

    @abc.abstractmethod
    def infer_posterior(
        self,
        last_latent: torch.Tensor,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        global_belief: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate Abstract")

    @abc.abstractmethod
    def obtain_initial(
        self, state: torch.Tensor, global_belief: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate Abstract")

    def forward(
        self,
        last_latent_prior: Optional[torch.Tensor],
        last_latent_posterior: Optional[torch.Tensor],
        state: torch.Tensor,
        action: torch.Tensor,
        global_belief: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if last_latent_prior is None:
            return self.obtain_initial(state, global_belief=global_belief)
        prior = self.infer_prior(
            last_latent_posterior, action=action, global_belief=global_belief
        )
        posterior = self.infer_posterior(
            last_latent_posterior, state, action=action, global_belief=global_belief
        )
        return prior, posterior
