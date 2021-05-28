from sequential_inference.abc.trainer import AbstractRLTrainer
from sequential_inference.abc.experiment import AbstractRLExperiment
from sequential_inference.experiments.mixins.data import AbstractDataMixin


class PureRLTrainingExperiment(AbstractDataMixin, AbstractRLExperiment):
    def __init__(
        self,
        rl_trainer: AbstractRLTrainer,
        batch_size: int,
        epoch_steps: int,
        epochs: int,
        log_frequency: int,
    ):
        self.trainer = rl_trainer
        self.batch_size = batch_size
        self.epoch_steps = epoch_steps
        self.epochs = epochs
        self.log_frequency = log_frequency

    def train(self, start_epoch: int = 0):
        total_train_steps = start_epoch * self.epoch_steps
        for e in range(start_epoch, self.epochs):
            print(f"Training epoch {e + 1}/{self.epochs} for {self.epoch_steps} steps")
            for i in range(self.epoch_steps):
                sample = self.get_batch()
                stats = self.trainer.train_step(sample)
                self.notify_observers("step", stats, total_train_steps)
                if total_train_steps % self.log_frequency == 0:
                    self.notify_observers("log", stats, total_train_steps)
                total_train_steps += 1
            epoch_log = self.after_epoch({})
            self.notify_observers("epoch", epoch_log, total_train_steps)
        self.close_observers()
