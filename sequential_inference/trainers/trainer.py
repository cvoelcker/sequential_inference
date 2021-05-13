import abc
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from typing import Dict, Optional, Tuple

import torch


class AbstractTrainer(abc.ABC):
    def __init__(
        self,
        algorithm: Optional[AbstractSequenceAlgorithm] = None,
        lr: float = 3e-4,
        gradient_norm_clipping: float = -1.0,
    ):

        self.algorithm = algorithm
        self.optimizer = torch.optim.Adam(self.algorithm.get_parameters(), lr=lr)
        self.gradient_norm_clipping = gradient_norm_clipping

        self.device = "cpu"

    def train_step(self, batch):
        batch = self.unpack_batch(batch)

        loss, stats = self.algorithm.compute_loss(*batch)
        stats["loss"] = loss.detach().cpu()

        self.optim.zero_grad()
        loss.backward()

        if self.gradient_norm_clipping > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.algorithm.parameters(), self.gradient_norm_clipping
            )

        self.optim.step()
        return stats

    @abc.abstractmethod
    def unpack_batch(self, batch):
        pass

    def to(self, device):
        self.device = device
        self.algorithm.to(self.device)

    def get_checkpoint(self):
        to_save = {}
        to_save["algo"] = self.algorithm.get_checkpoint()
        to_save["optim"] = self.optimizer
        return to_save

    def load_checkpoint(self, chp):
        self.optimizer.load_state_dict(chp["optim"])
        self.algorithm.load_checkpoint(chp["algo"])


class ModelTrainer(AbstractTrainer):
    def unpack_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assumes a sequence batch
        Assumes RL convention r_t = f(s_t, a_t) and s_{t+1} = f(s_t, a_t)

        Args:
            batch (Dict[str, torch.Tensor]): dictionary of batch tensors

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: separate torch Tensors for observation, action, reward (next observation is implicit)
        """
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        return obs, act, rew


class RLTrainer(AbstractTrainer):
    def unpack_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assumes a full RL batch
        Assumes RL convention r_t = f(s_t, a_t) and s_{t+1} = f(s_t, a_t)

        Args:
            batch (Dict[str, torch.Tensor]): dictionary of batch tensors

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: separate torch Tensors for observation, action, reward, next observation
        """
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = obs[:, 1:]
        obs = obs[:, :-1]
        return obs, act, rew, next_obs


class RLLatentModelTrainer:
    def __init__(
        self,
        model_trainer: Optional[AbstractTrainer] = None,
        rl_trainer: Optional[AbstractTrainer] = None,
    ):

        self.model_trainer: AbstractTrainer = model_trainer
        self.model: AbstractSequenceAlgorithm = self.model_trainer.algorithm
        self.rl_trainer: AbstractTrainer = rl_trainer

        self.device = "cpu"

    def train_step(self, batch, train_model=True, train_rl=True):
        stats = {}
        if train_model:
            stats_model = self.model_trainer.train_step(batch)
            for k, v in stats_model:
                stats["model_" + k] = v
        if train_rl:
            # obtain embedding prediction from model
            with torch.no_grad():
                s, a, r = self.rl_trainer.unpack_batch(batch)
                latents = self.model.infer_sequence(s, a, r)
            batch = {"obs": latents, "act": a, "rew": r}
            stats_rl = self.rl_trainer.train_step(batch)
            for k, v in stats_rl:
                stats["rl_" + k] = v
        return stats

    def to(self, device):
        self.device = device
        self.rl_trainer.to(self.device)
        self.model_trainer.to(self.device)

    def get_checkpoint(self):
        to_save = {}
        to_save["model"] = self.model_trainer.get_checkpoint()
        to_save["rl"] = self.rl_trainer.get_checkpoint()
        return to_save

    def load_checkpoint(self, chp):
        self.rl_trainer.load_checkpoint(chp["rl"])
        self.model_trainer.load_checkpoint(chp["model"])
