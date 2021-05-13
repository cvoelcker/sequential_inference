import abc
from typing import Dict, Optional, OrderedDict, Tuple
import torch


class AbstractSequenceAlgorithm(abc.ABC):
    def __init__(self):
        self.device: str = "cpu"
        self.buffer: Dict[str, OrderedDict] = {}
        super().__init__()

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
        global_belief: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate AbstractSequenceModel")

    @abc.abstractmethod
    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate AbstractSequenceModel")

    @abc.abstractmethod
    def predict_sequence(
        self,
        initial_latent: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
        global_belief: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate AbstractSequenceModel")

    @abc.abstractmethod
    def reconstruct(self, latent: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate AbstractSequenceModel")

    @abc.abstractmethod
    def reconstruct_reward(
        self, latent: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("Cannot instantiate AbstractSequenceModel")

    def register_module(self, key: str, module: torch.nn.Module):
        if key in self.buffer.keys():
            raise KeyError("Key in module buffer is not unique")
        else:
            self.buffer[key] = module

    def to(self, device: str):
        self.device = device
        for k, v in self.buffer.items():
            v.to(self.device)

    def get_checkpoint(self):
        to_save = {}
        for k, v in self.buffer.items():
            to_save[k] = v.state_dict()
        return to_save

    def load_checkpoint(self, chp: Dict[str, OrderedDict]):
        for k, v in self.buffer.items():
            v.load_state_dict(chp[k])

    def get_parameters(self):
        params = []
        for k, v in self.buffer.items():
            # needs to concatenate the list to flatten them
            params += list(v.parameters())
        return params


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
        action: Optional[torch.Tensor] = None,
        global_belief: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if last_latent_prior is None:
            return self.obtain_initial(state, global_belief=global_belief)
        prior = self.infer_prior(
            last_latent_prior, action=action, global_belief=global_belief
        )
        posterior = self.infer_posterior(
            last_latent_posterior, state, action=action, global_belief=global_belief
        )
        return prior, posterior
