import torch

from sequential_inference.abc.trainer import AbstractRLTrainer
from sequential_inference.trainers.trainer import ModelTrainer
from sequential_inference.abc.experiment import AbstractExperiment, AbstractRLExperiment
from sequential_inference.data.data import (
    BatchTrajectorySampler,
    TrajectoryReplayBuffer,
)


class IntegratedRLModelTrainingExperiment(AbstractRLExperiment):
    def __init__(
        self,
        model_trainer: ModelTrainer,
        rl_trainer: AbstractRLTrainer,
        data: TrajectoryReplayBuffer,
        batch_size: int,
        epochs: int,
    ):
        self.model = model_trainer
        self.rl_trainer = rl_trainer
        self.data = data
        self.dataset = BatchTrajectorySampler(self.data, self.batch_size)
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, start_epoch: int = 0):
        total_train_steps = start_epoch * len(self.dataset) // self.batch_size
        for e in range(start_epoch, self.epochs):
            print(f"Training epoch {e + 1}/{self.epochs} for {len(self.data)} steps")
            for sample in self.data:
                stats = self.train_step(sample)
                self.notify_observers("step", stats, total_train_steps)
                if total_train_steps % self.log_frequency == 0:
                    self.notify_observers("log", stats, total_train_steps)
                total_train_steps += 1
            epoch_log = self.after_epoch()
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
