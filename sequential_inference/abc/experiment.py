import abc
from sequential_inference.abc.rl import AbstractRLAlgorithm
from typing import Dict

import torch

from sequential_inference.abc.common import Checkpointable


class AbstractExperiment(Checkpointable, metaclass=abc.ABCMeta):
    """Interface for an experiment class. An experiment provides the
    train method and various methods to handle hooks and observers
    """

    def __init__(self):
        super().__init__()
        self.observers = []
        self.epoch_hooks = []

    @abc.abstractmethod
    def build(self, cfg, run_dir, preempted):
        pass

    @abc.abstractmethod
    def train(self, start_epoch=0):
        pass

    def before_experiment(self):
        pass

    def after_experiment(self):
        self.close_observers()

    def after_epoch(self, epoch_log: Dict[str, torch.Tensor]):
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
    """Defines the minimum scaffolding for an RL experiment"""

    rl_algorithm: AbstractRLAlgorithm

    def get_agent(self):
        return self.rl_algorithm.get_policy()


class ExperimentMixin(abc.ABC):
    """This ABC mostly exists for the type hierarchy, the mixins
    can be fully flexible
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    pass
