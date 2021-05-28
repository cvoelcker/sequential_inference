from sequential_inference.experiments.mixins.data import AbstractDataMixin
import torch

from sequential_inference.abc.trainer import AbstractRLTrainer
from sequential_inference.trainers.trainer import ModelTrainer
from sequential_inference.abc.experiment import AbstractRLExperiment


class IntegratedRLModelTrainingExperiment(AbstractDataMixin, AbstractRLExperiment):
    def __init__(
        self,
        joint_trainer: ModelRLTrainer,
        batch_size: int,
        epoch_steps: int,
        epochs: int,
        log_frequency: int,
    ):
        self.model = model_trainer
        self.rl_trainer = rl_trainer
        self.epoch_steps = epoch_steps
        self.epochs = epochs
        self.log_frequency = log_frequency

    def train(self, start_epoch: int = 0):
        total_train_steps = start_epoch * self.epoch_steps
        for e in range(start_epoch, self.epochs):
            print(f"Training epoch {e + 1}/{self.epochs} for {self.epoch_steps} steps")
            for i in range(self.epoch_steps):
                sample = self.get_batch()
                stats = self.train_step(sample)
                self.notify_observers("step", stats, total_train_steps)
                if total_train_steps % self.log_frequency == 0:
                    self.notify_observers("log", stats, total_train_steps)
                total_train_steps += 1
            epoch_log = self.after_epoch({})
            self.notify_observers("epoch", epoch_log, total_train_steps)
        self.close_observers()

    def train_step(self, batch, train_model=True, train_rl=True):
        stats = {}
        if train_model:
            stats_model = self.model.train_step(batch)
            for k, v in stats_model:
                stats["model_" + k] = v
        if train_rl:
            # obtain embedding prediction from model
            with torch.no_grad():
                s, a, r = self.rl_trainer.unpack_batch(batch)
                latents = self.model.algorithm.infer_sequence(s, a, r)
            batch = {"obs": latents, "act": a, "rew": r}
            stats_rl = self.rl_trainer.train_step(batch)
            for k, v in stats_rl:
                stats["rl_" + k] = v
        return stats
