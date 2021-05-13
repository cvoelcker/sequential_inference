import abc
from typing import Dict
from sequential_inference.trainers.trainer import AbstractTrainer, RLTrainer
from sequential_inference.abc.common import Checkpointable


class AbstractExperiment(Checkpointable, metaclass=abc.ABCMeta):
    def __init__(self):
        self.observers = []
        self.epoch_hooks = []
        self.trainers: Dict[str, AbstractTrainer] = {}

    @abc.abstractmethod
    def train(self, start_epoch=0):
        pass

    def after_epoch(self):
        epoch_log = {}
        for hook in self.epoch_hooks:
            log = hook(self)
            epoch_log.update(log)
        return epoch_log

    def register_epoch_hook(self, epoch_hook):
        self.epoch_hooks.append(epoch_hook)

    def register_observer(self, observer):
        self.observers.append(observer)

    def notify_observers(self, keyword, data, step):
        for observer in self.observers:
            observer.notify(keyword, data, step)

    def close_observers(self):
        for observer in self.observers:
            observer.close()


class AbstractRLExperiment(AbstractExperiment, metaclass=abc.ABCMeta):

    rl_trainer: RLTrainer

    def get_policy(self):
        return self.rl_trainer.get_policy()
