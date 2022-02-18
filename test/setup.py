from typing import List, Tuple

import hydra
from omegaconf import OmegaConf

from sequential_inference.abc.common import Env
from sequential_inference.abc.experiment import AbstractExperiment
from sequential_inference.abc.util import AbstractLogger
from sequential_inference.data.handler import AbstractDataHandler, setup_data
from sequential_inference.log.logger import Checkpointing
from sequential_inference.util.setup import fix_env_config
from sequential_inference.util.vector import vector_preface


@hydra.main(config_path="../config", config_name="test_initialize")
def test_initialize(cfg):
    preempted = vector_preface(cfg)

    print(cfg)

    env: Env = hydra.utils.instantiate(cfg.env)
    cfg = fix_env_config(cfg, env)
    data: AbstractDataHandler = setup_data(cfg, env)
    experiment: AbstractExperiment = hydra.utils.instantiate(cfg.experiment)
    logging: List[AbstractLogger] = hydra.utils.instantiate(cfg.logging)

    checkpointer = Checkpointing("./checkpoints", "checkpoint", 0)
    checkpointer(experiment)

    # connect data handler, logging and handle preemption
    experiment.register_observers(logging)
    experiment.set_data_handler(data)
    experiment_status = experiment.initialize(cfg, preempted)

    # run the experiment (most of the time just starts the training loop)
    experiment.run(experiment_status)


if __name__ == "__main__":
    test_initialize()
