import os
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from sequential_inference.abc.data import Env
from sequential_inference.abc.evaluation import AbstractEvaluator
from sequential_inference.algorithms.rl.agents import RandomAgent
from sequential_inference.data.handler import AbstractDataHandler, setup_data
from sequential_inference.experiments.base import AbstractTrainingExperiment
from sequential_inference.log.logger import Checkpointing
from sequential_inference.util.setup import fix_env_config


@hydra.main(config_path="../config", config_name="main_evaluate")
def evaluate(cfg):
    # load run config
    run_cfg: DictConfig = OmegaConf.load(
        os.path.join(cfg.load_dir, ".hydra/config.yaml")
    )

    print(OmegaConf.to_yaml(cfg))
    print(OmegaConf.to_yaml(run_cfg))

    env: Env = hydra.utils.instantiate(run_cfg.env)
    run_cfg = fix_env_config(run_cfg, env)
    data: AbstractDataHandler = setup_data(run_cfg, env)
    data.initialize(run_cfg, False)
    experiment: AbstractTrainingExperiment = hydra.utils.instantiate(run_cfg.experiment)
    experiment.to(run_cfg.device)
    experiment.summarize()

    # connect data handler, logging and handle preemption
    experiment.set_data_handler(data)

    # data.load(cfg.load_dir)
    checkpoint: Checkpointing = Checkpointing(cfg.load_dir)
    load_path = checkpoint.get_latest()
    experiment.load(load_path)

    data.update({}, RandomAgent(env.action_space))

    # run evaluation
    evaluators: List[AbstractEvaluator] = hydra.utils.instantiate(cfg.evaluators)

    for evaluator in evaluators:
        evaluator.evaluate(experiment, 0)


if __name__ == "__main__":
    evaluate()
