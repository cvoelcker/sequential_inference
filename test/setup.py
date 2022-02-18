import unittest

from typing import List

import hydra
from hydra import initialize, compose

import torch

from sequential_inference.abc.common import Env
from sequential_inference.abc.util import AbstractLogger
from sequential_inference.data.handler import AbstractDataHandler, setup_data
from sequential_inference.experiments.base import RLTrainingExperiment
from sequential_inference.log.logger import Checkpointing
from sequential_inference.util.setup import fix_env_config
from sequential_inference.util.vector import vector_preface


class TestSetup(unittest.TestCase):
    def test_initialize(self):
        with initialize(config_path="../config"):
            # config is relative to a module
            cfg = compose(config_name="test_initialize", overrides=[])

            print(cfg)

            preempted = vector_preface(cfg)

            env: Env = hydra.utils.instantiate(cfg.env)
            cfg = fix_env_config(cfg, env)
            data: AbstractDataHandler = setup_data(cfg, env)
            experiment: RLTrainingExperiment = hydra.utils.instantiate(cfg.experiment)
            logging: List[AbstractLogger] = hydra.utils.instantiate(cfg.logging)

            # connect data handler, logging and handle preemption
            experiment.register_observers(logging)
            experiment.set_data_handler(data)
            experiment.initialize(cfg, preempted)

            data_1 = data.buffer.s[0, 0]
            parameters_1 = list(experiment.rl_algorithm.actor.parameters())[0]  # type: ignore

            checkpointer = Checkpointing(".", "checkpoint", 0)
            checkpointer(experiment)
            data_checkpointer = Checkpointing(".", "data", 0)
            data_checkpointer(data.buffer)

            env: Env = hydra.utils.instantiate(cfg.env)
            cfg = fix_env_config(cfg, env)
            data: AbstractDataHandler = setup_data(cfg, env)
            experiment: RLTrainingExperiment = hydra.utils.instantiate(cfg.experiment)
            logging: List[AbstractLogger] = hydra.utils.instantiate(cfg.logging)

            # connect data handler, logging and handle preemption
            experiment.register_observers(logging)
            experiment.set_data_handler(data)

            experiment.load(checkpointer.get_latest())
            data.buffer.load(data_checkpointer.get_latest())

            data_2 = data.buffer.s[0, 0]
            parameters_2 = list(experiment.rl_algorithm.actor.parameters())[0]  # type: ignore

            self.assertTrue(torch.all(data_1 == data_2))
            self.assertTrue(torch.all(parameters_1 == parameters_2))

            env: Env = hydra.utils.instantiate(cfg.env)
            cfg = fix_env_config(cfg, env)
            data: AbstractDataHandler = setup_data(cfg, env)
            experiment: RLTrainingExperiment = hydra.utils.instantiate(cfg.experiment)
            logging: List[AbstractLogger] = hydra.utils.instantiate(cfg.logging)

            # connect data handler, logging and handle preemption
            experiment.register_observers(logging)
            experiment.set_data_handler(data)
            experiment.initialize(cfg, preempted)

            data_2 = data.buffer.s[0, 0]
            parameters_2 = list(experiment.rl_algorithm.actor.parameters())[0]  # type: ignore

            self.assertFalse(torch.all(data_1 == data_2))
            self.assertFalse(torch.all(parameters_1 == parameters_2))

    def test_train(self):

        with initialize(config_path="../config"):
            # config is relative to a module
            cfg = compose(config_name="test_initialize", overrides=[])

            print(cfg)

            preempted = vector_preface(cfg)

            env: Env = hydra.utils.instantiate(cfg.env)
            cfg = fix_env_config(cfg, env)
            data: AbstractDataHandler = setup_data(cfg, env)
            experiment: RLTrainingExperiment = hydra.utils.instantiate(cfg.experiment)
            logging: List[AbstractLogger] = hydra.utils.instantiate(cfg.logging)

            # connect data handler, logging and handle preemption
            experiment.register_observers(logging)
            experiment.set_data_handler(data)
            experiment.initialize(cfg, preempted)
            experiment.set_checkpoint(cfg.chp_dir)

            experiment.train()


if __name__ == "__main__":
    TestSetup().test_initialize()
    TestSetup().test_train()
