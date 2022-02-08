from typing import List

import hydra
from omegaconf import OmegaConf

from sequential_inference.abc.data import Env
from sequential_inference.abc.experiment import AbstractExperiment
from sequential_inference.abc.util import AbstractLogger
from sequential_inference.data.handler import AbstractDataHandler, setup_data

from sequential_inference.setup.vector import vector_preface
from sequential_inference.envs.env import setup_environment


@hydra.main(config_path="../config", config_name="test_init")
def test_init(cfg):
    preempted = vector_preface(cfg)
    print(OmegaConf.to_yaml(cfg))
    print(cfg.chp_dir)
    print(cfg.logging.tensorboard.log_dir)

    env: Env = setup_environment(cfg)
    data: AbstractDataHandler = setup_data(cfg, env)
    experiment: AbstractExperiment = hydra.utils.instantiate(cfg.experiment)
    logging: List[AbstractLogger] = hydra.utils.instantiate(cfg.logging)


@hydra.main(config_path="../config", config_name="test_initialize")
def test_initialize(cfg):
    preempted = vector_preface(cfg)

    env: Env = setup_environment(cfg)
    data: AbstractDataHandler = setup_data(cfg, env)
    experiment: AbstractExperiment = hydra.utils.instantiate(cfg.experiment)
    logging: List[AbstractLogger] = hydra.utils.instantiate(cfg.logging)

    # connect data handler, logging and handle preemption
    experiment.register_observers(logging)
    experiment.set_data_handler(data)
    experiment_status = experiment.initialize(cfg, preempted)

    # run the experiment (most of the time just starts the training loop)
    experiment.run(experiment_status)


if __name__ == "__main__":
    test_init()
