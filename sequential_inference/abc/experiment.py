import os

import abc
import pickle

from omegaconf import omegaconf
from sequential_inference.abc.data import AbstractDataHandler
from sequential_inference.abc.rl import AbstractAgent, AbstractRLAlgorithm
from typing import Any, Dict, Optional

import torch

from sequential_inference.abc.common import Checkpointable
from sequential_inference.util.errors import NotInitializedException


class AbstractExperiment(Checkpointable, metaclass=abc.ABCMeta):
    """Interface for an experiment class. An experiment provides the
    train method and various methods to handle hooks and observers
    """

    def __init__(self):
        super().__init__()
        self.observers = []
        self.epoch_hooks = []
        self.data: Optional[AbstractDataHandler] = None

    @abc.abstractmethod
    def run(self, status: Dict[str, str]) -> None:
        """Run the experiment

        Args:
            status (Dict[str, str]): the status of the experiment
        """
        pass

    def set_data_handler(self, data: AbstractDataHandler) -> None:
        self.data = data

    def initialize(self, cfg: omegaconf.DictConfig, preempted: bool) -> Dict[str, Any]:
        """Initialize the experiment, either collecting new data and using the default models, or loading a checkpointed model

        Args:
            cfg (omegaconf.DictConfig):the global config object
            preempted (bool): a flag checking for preempted training
            run_dir (str): the directory to store the experiment
        """
        if self.data is None:
            raise NotInitializedException("Data handler not set")
        experiment_status = {}
        if preempted:
            print("Reloading preempted experiment")
            with open(os.path.join(cfg.chp_dir, "status"), "rb") as f:
                experiment_status = pickle.load(f)
            checkpoints = os.path.join(cfg.chp_dir, "checkpoints")
            list_dir = os.listdir(checkpoints)
            checkpoint_location = sorted(list_dir)[-1]
            self.load(os.path.join(checkpoints, checkpoint_location))
        else:
            experiment_status["status"] = "new"
            experiment_status["epoch_number"] = 0

        self.data.initialize(cfg, preempted)

        return experiment_status

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

    def register_observers(self, observers):
        for observer in observers:
            self.register_observer(observer)

    def notify_observers(self, keyword, data, step):
        for observer in self.observers:
            observer.notify(keyword, data, step)

    def close_observers(self):
        for observer in self.observers:
            observer.close()

    def set_checkpoint(self, chp_dir):
        pass

    def checkpoint(self):
        pass


class AbstractRLExperiment(AbstractExperiment, metaclass=abc.ABCMeta):
    """Defines the minimum scaffolding for an RL experiment"""

    rl_algorithm: AbstractRLAlgorithm

    def get_agent(self) -> AbstractAgent:
        return self.rl_algorithm.get_agent()
