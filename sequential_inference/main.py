from typing import List

import hydra
from omegaconf import OmegaConf

from sequential_inference.abc.data import Env
from sequential_inference.abc.experiment import AbstractExperiment
from sequential_inference.abc.util import AbstractLogger
from sequential_inference.data.handler import AbstractDataHandler, setup_data


@hydra.main(config_path="../config", config_name="main")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))

    preempted = vector_preface(cfg)

    env: Env = hydra.utils.instantiate(cfg.env)
    cfg = fix_env_config(cfg, env)
    data: AbstractDataHandler = setup_data(cfg, env)
    experiment: RLTrainingExperiment = hydra.utils.instantiate(cfg.experiment)
    experiment.to(cfg.device)
    experiment.summarize()
    logging: List[AbstractLogger] = hydra.utils.instantiate(cfg.logging)

    # connect data handler, logging and handle preemption
    experiment.register_observers(logging)
    experiment.set_data_handler(data)
    status = experiment.initialize(cfg, preempted)
    experiment.set_checkpoint(cfg.chp_dir)

    experiment.train(status["epoch_number"])


if __name__ == "__main__":
    main()
