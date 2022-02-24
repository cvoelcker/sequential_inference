from typing import List

import hydra
from omegaconf import OmegaConf

from sequential_inference.abc.common import Env
from sequential_inference.abc.util import AbstractLogger
from sequential_inference.data.handler import AbstractDataHandler, setup_data
from sequential_inference.experiments.base import RLTrainingExperiment
from sequential_inference.util.setup import fix_env_config
from sequential_inference.util.vector import vector_preface


@hydra.main(config_path="../config", config_name="test_dreamer")
def test_train(cfg):
    print(OmegaConf.to_yaml(cfg))

    preempted = vector_preface(cfg)

    env: Env = hydra.utils.instantiate(cfg.env)
    cfg = fix_env_config(cfg, env)
    data: AbstractDataHandler = setup_data(cfg, env)
    experiment: RLTrainingExperiment = hydra.utils.instantiate(cfg.experiment)
    experiment.to(cfg.device)
    logging: List[AbstractLogger] = hydra.utils.instantiate(cfg.logging)

    # connect data handler, logging and handle preemption
    experiment.register_observers(logging)
    experiment.set_data_handler(data)
    status = experiment.initialize(cfg, preempted)
    experiment.set_checkpoint(cfg.chp_dir)

    print(status)

    experiment.train(status["epoch_number"])


if __name__ == "__main__":
    # TestSetup().test_initialize()
    # TestSetup().test_train()
    test_train()
