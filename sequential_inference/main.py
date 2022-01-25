from typing import List
import hydra

from sequential_inference.abc.data import Env
from sequential_inference.abc.experiment import AbstractExperiment
from sequential_inference.abc.util import AbstractDataSampler, AbstractLogger
from sequential_inference.data.handler import AbstractDataHandler, setup_data

from sequential_inference.setup.vector import vector_preface
from sequential_inference.envs.env import setup_environment


@hydra.main(config_path="../config", config_name="main")
def main(cfg):

    run_dir, preempted = vector_preface(cfg)

    env: Env = setup_environment(cfg)
    data: AbstractDataHandler = setup_data(cfg, env)
    experiment: AbstractExperiment = hydra.utils.instantiate(cfg.experiment)
    logging: List[AbstractLogger] = hydra.utils.instantiate(cfg.logging)

    # connect data handler, logging and handle preemption
    experiment.register_observers(logging)
    experiment.set_data_handler(data)
    experiment_status = experiment.initialize(cfg, preempted, run_dir)

    # run the experiment (most of the time just starts the training loop)
    experiment.run(experiment_status)
