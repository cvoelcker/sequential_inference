import os
import pickle
from typing import Dict, Optional, Tuple
import abc

from tqdm import tqdm
import torch

from sequential_inference.abc.experiment import AbstractExperiment, AbstractRLExperiment
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from sequential_inference.abc.rl import AbstractRLAlgorithm
from sequential_inference.log.logger import Checkpointing
from sequential_inference.util.errors import NotInitializedException


class AbstractTrainingExperiment(AbstractExperiment):

    is_rl: bool = False
    is_model: bool = False

    def __init__(self, epoch_steps: int, epochs: int, log_interval: int):
        super().__init__()
        self.checkpointing: Optional[Checkpointing] = None
        self.epoch_steps = epoch_steps
        self.epochs = epochs
        self.log_interval = log_interval

    def run(self, status: Dict[str, str]):
        self.train(int(status["epoch_number"]))

    def train(self, start_epoch: int = 0):
        if self.data is None:
            raise NotInitializedException("Data is not initialized")
        total_train_steps = start_epoch * self.epoch_steps
        for e in range(start_epoch, self.epochs):
            print(f"Epoch {e}")
            self.epoch = e
            for i in tqdm(range(self.epoch_steps)):
                stats = self.train_step()
                self.notify_observers("step", stats, total_train_steps)
                total_train_steps += 1
                if i % self.log_interval == 0:
                    self.notify_observers("log", stats, total_train_steps)
            epoch_log = self.after_epoch({})
            self.notify_observers("epoch", epoch_log, total_train_steps)
            self.checkpoint()
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

    def set_checkpoint(self, chp_dir):
        self.chp_dir = chp_dir
        self.checkpointing = Checkpointing(chp_dir, "checkpoints")
        self.data_checkpointing = Checkpointing(chp_dir, "data", overwrite=True)

    def checkpoint(self):
        if self.checkpointing is not None:
            with open(os.path.join(self.chp_dir, "status"), "wb") as f:
                pickle.dump(
                    {
                        "status": "training",
                        "epoch_number": self.epoch + 1,
                    },
                    f,
                )
            print("Checkpointing experiment")
            self.checkpointing(self)
        if self.data is not None and self.data_checkpointing is not None:
            self.data_checkpointing(self.data.buffer)


class RLTrainingExperiment(AbstractRLExperiment, AbstractTrainingExperiment):

    rl_algorithm: AbstractRLAlgorithm

    is_rl: bool = True

    def __init__(
        self,
        epoch_steps: int,
        epochs: int,
        rl_algorithm: AbstractRLAlgorithm,
        batch_size: int = 32,
        log_interval: int = 10,
    ):
        super().__init__(epoch_steps, epochs, log_interval)

        self.rl_algorithm = rl_algorithm
        self._step_rl = self.rl_algorithm.get_step()
        self.batch_size = batch_size

        self.register_module("rl_algorithm", self.rl_algorithm)

    def train_step(self) -> Dict[str, torch.Tensor]:
        if self.data is None:
            raise NotInitializedException("Data not initialized")
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        return self.rl_train_step(*unpacked_batch)

    def rl_train_step(self, obs, act, rew, done) -> Dict[str, torch.Tensor]:
        loss, stats = self.rl_algorithm.compute_loss(obs, act, rew, done)
        stats = self._step_rl(loss, stats)
        stats["rl_step_cuda"] = torch.Tensor([torch.cuda.memory_reserved()]).float()
        return stats

    def after_epoch(self, epoch_log: Dict[str, torch.Tensor]):
        if self.data is None:
            raise NotInitializedException("Data not initialized")
        epoch_log = super().after_epoch(epoch_log)
        epoch_log = self.data.update(epoch_log, self.rl_algorithm.get_agent())
        return epoch_log


class ModelTrainingExperiment(AbstractTrainingExperiment):

    model_algorithm: AbstractSequenceAlgorithm

    is_model = True

    def __init__(
        self,
        epoch_steps: int,
        epochs: int,
        model_algorithm: AbstractSequenceAlgorithm,
        batch_size: int = 32,
        log_interval: int = 10,
    ):
        super().__init__(epoch_steps, epochs, log_interval)
        self.model_algorithm = model_algorithm
        self._model_step = self.model_algorithm.get_step()
        self.batch_size = batch_size

        self.register_module("model_algorithm", self.model_algorithm)

    def train_step(self) -> Dict[str, torch.Tensor]:
        if self.data is None:
            raise NotInitializedException("Data not initialized")
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        return self.model_train_step(*unpacked_batch)

    def model_train_step(self, obs, act, rew, done) -> Dict[str, torch.Tensor]:
        loss, stats = self.model_algorithm.compute_loss(obs, act, rew, done)
        self._model_step(loss, stats)
        return stats


class ModelBasedRLTrainingExperiment(AbstractTrainingExperiment, abc.ABC):

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
        log_interval: int = 10,
    ):
        super().__init__(epoch_steps, epochs, log_interval)
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
        stats = self._step_model(loss, stats)
        return stats

    def rl_train_step(self, obs, act, rew, done) -> Dict[str, torch.Tensor]:
        loss, stats = self.rl_algorithm.compute_loss(obs, act, rew, done)
        self._step_rl(loss)
        return stats
