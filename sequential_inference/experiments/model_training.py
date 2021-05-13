from sequential_inference.data.data import (
    BatchTrajectorySampler,
    TrajectoryReplayBuffer,
)
from sequential_inference.trainers.trainer import ModelTrainer
from sequential_inference.abc.experiment import AbstractExperiment


class ModelTrainingExperiment(AbstractExperiment):
    def __init__(
        self,
        model_trainer: ModelTrainer,
        data: TrajectoryReplayBuffer,
        batch_size: int,
        epochs: int,
    ):
        self.trainer = model_trainer
        self.data = data
        self.dataset = BatchTrajectorySampler(self.data, self.batch_size)
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, start_epoch: int = 0):
        total_train_steps = start_epoch * len(self.dataset) // self.batch_size
        for e in range(start_epoch, self.epochs):
            print(f"Training epoch {e + 1}/{self.epochs} for {len(self.data)} steps")
            for sample in self.data:
                stats = self.trainer.train_step(sample)
                self.notify_observers("step", stats, total_train_steps)
                if total_train_steps % self.log_frequency == 0:
                    self.notify_observers("log", stats, total_train_steps)
                total_train_steps += 1
            epoch_log = self.after_epoch()
            self.notify_observers("epoch", epoch_log, total_train_steps)
        self.close_observers()
