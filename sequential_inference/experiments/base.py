from typing import Dict, Optional, Tuple
import abc
from sequential_inference.abc.data import Env
from sequential_inference.setup.factory import (
    setup_data,
    setup_rl_algorithm,
    setup_model_algorithm,
    setup_optimizer,
)

from tqdm import tqdm
import torch

from sequential_inference.abc.experiment import AbstractExperiment, AbstractRLExperiment
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from sequential_inference.abc.rl import AbstractRLAlgorithm


class TrainingExperiment(AbstractExperiment):

    is_rl: bool = False
    is_model: bool = False

    def __init__(
        self,
        epoch_steps: int,
        epochs: int,
    ):
        super().__init__()
        self.epoch_steps = epoch_steps
        self.epochs = epochs

    def run(self, status: Dict[str, str]):
        self.train(int(status["epoch_number"]))

    def train(self, start_epoch: int = 0):
        total_train_steps = start_epoch * self.epoch_steps
        for _ in range(start_epoch, self.epochs):
            for _ in tqdm(range(self.epoch_steps)):
                stats = self.train_step()
                self.notify_observers("step", stats, total_train_steps)
                total_train_steps += 1
            epoch_log = self.after_epoch({})
            self.notify_observers("epoch", epoch_log, total_train_steps)
            self.checkpoint(self)
        self.close_observers()

    def after_epoch(self, d):
        d = super().after_epoch(d)
        return d

    def unpack_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        done = batch["done"]
        return obs, act, rew, done

    @abc.abstractmethod
    def train_step(self) -> Dict[str, torch.Tensor]:
        pass


class RLTrainingExperiment(AbstractRLExperiment, TrainingExperiment):

    rl_algorithm: AbstractRLAlgorithm

    is_rl: bool = True

    def __init__(
        self,
        epoch_steps: int,
        epochs: int,
        rl_algorithm: AbstractRLAlgorithm,
        batch_size: int = 32,
    ):
        super().__init__(epoch_steps, epochs)

        self.rl_algorithm = rl_algorithm
        self._step_rl = self.rl_algorithm.get_step()
        self.batch_size = batch_size

        self.register_module("rl_algorithm", self.rl_algorithm)

    def train_step(self) -> Dict[str, torch.Tensor]:
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        return self.rl_train_step(*unpacked_batch)

    def rl_train_step(self, obs, act, rew, done) -> Dict[str, torch.Tensor]:
        loss, stats = self.rl_algorithm.compute_loss(obs, act, rew, done)
        self._step_rl(loss)
        return stats


class ModelTrainingExperiment(TrainingExperiment):

    model_algorithm: AbstractSequenceAlgorithm

    is_model = True

    def __init__(
        self,
        epoch_steps: int,
        epochs: int,
        model_algorithm: AbstractSequenceAlgorithm,
        batch_size: int = 32,
    ):
        super().__init__(epoch_steps, epochs)
        self.model_algorithm = model_algorithm
        self._model_step = self.model_algorithm.get_step()
        self.batch_size = batch_size

        self.register_module("model_algorithm", self.model_algorithm)

    def train_step(self) -> Dict[str, torch.Tensor]:
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        return self.model_train_step(*unpacked_batch)

    def model_train_step(self, obs, act, rew, done) -> Dict[str, torch.Tensor]:
        loss, stats = self.model_algorithm.compute_loss(obs, act, rew, done)
        self._step_model(loss)
        return stats


class ModelBasedRLTrainingExperiment(TrainingExperiment, abc.ABC):

    model_algorithm: AbstractSequenceAlgorithm
    rl_algorithm: AbstractRLAlgorithm

    is_model = True
    is_rl = True

    def __init__(
        self,
        epoch_steps: int,
        epochs: int,
        model_algorithm: AbstractSequenceAlgorithm,
        rl_algorithm: AbstractRLAlgorithm,
        model_batch_size: int = 32,
        rl_batch_size: int = 32,
    ):
        super().__init__(epoch_steps, epochs)
        self.model_algorithm = model_algorithm
        self._step_model = self.model_algorithm.get_step()
        self.model_batch_size = model_batch_size

        self.register_module("model_algorithm", self.model_algorithm)

        self.rl_algorithm = rl_algorithm
        self._step_rl = self.rl_algorithm.get_step()
        self.rl_batch_size = rl_batch_size

        self.register_module("rl_algorithm", self.rl_algorithm)

    def model_train_step(self, obs, act, rew, done) -> Dict[str, torch.Tensor]:
        loss, stats = self.model_algorithm.compute_loss(obs, act, rew, done)
        self._step_model(loss)
        return stats

    def rl_train_step(self, obs, act, rew, done) -> Dict[str, torch.Tensor]:
        loss, stats = self.rl_algorithm.compute_loss(obs, act, rew, done)
        self._step_rl(loss)
        return stats
