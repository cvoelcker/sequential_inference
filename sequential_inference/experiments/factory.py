from sequential_inference.data.data import TrajectoryReplayBuffer
from sequential_inference.rl.factory import setup_rl_algorithm
from sequential_inference.algorithms.factory import setup_model_algorithm
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from sequential_inference.abc.rl import AbstractRLAlgorithm
from typing import Dict, Type

import sequential_inference.envs.types import Env

from sequential_inference.experiments.data import *
from sequential_inference.experiments.training_experiments import *
from sequential_inference.abc.experiment import AbstractExperiment, ExperimentMixin


EXPERIMENT_REGISTRY: Dict[str, Type[AbstractExperiment]] = {
    "latent_model_rl": LatentTrainingExperiment,
    "latent_imagination_rl": LatentImaginationExperiment,
    "dyna_rl": DynaTrainingExperiment,
    "model_training": ModelTrainingExperiment,
    "rl_training": RLTrainingExperiment,
}

def setup_experiment(env, cfg) -> AbstractExperiment:

    # model free RL + model-based inference
    if cfg.experiment.type == "latent_model_rl":
        experiment: TrainingExperiment = LatentTrainingExperiment(
            cfg.experiment.pass_rl_gradients_to_model,
            cfg.train.epoch_steps,
            cfg.train_steps.epochs,
            cfg.log.log_frequency,
        )
    # model-based inference + model-based rollouts for RL
    elif cfg.experiment.type == "latent_imagination_rl":
        experiment: TrainingExperiment = LatentImaginationExperiment(
            cfg.train.epoch_steps,
            cfg.train_steps.epochs,
            cfg.log.log_frequency,
        )
    # model-based data generation + data-augmented RL
    elif cfg.experiment.type == "dyna_rl":
        experiment: TrainingExperiment = DynaTrainingExperiment(
            cfg.train.epoch_steps,
            cfg.train_steps.epochs,
            cfg.log.log_frequency,
        )
    # model-based inference + direct differentiation for policy optimization (after Dreamer)
    elif cfg.experiment.type == "dreamer":
        experiment: TrainingExperiment = DreamerExperiment(
            cfg.train.epoch_steps,
            cfg.train_steps.epochs,
            cfg.log.log_frequency,)
    
    # separate training functions for simplified debugging and potential pretraining
    elif cfg.experiment.type == "model_training":
        experiment: TrainingExperiment = ModelTrainingExperiment(
            cfg.train.epoch_steps,
            cfg.train_steps.epochs,
            cfg.log.log_frequency,
        )
    elif cfg.experiment.type == "rl_training":
        experiment: TrainingExperiment = RLTrainingExperiment(
            cfg.train.epoch_steps,
            cfg.train_steps.epochs,
            cfg.log.log_frequency,
        )
    else:
        raise KeyError(f"Experiment type {cfg.experiment.type} no known")

    if experiment.is_rl:
        rl_algorithm: AbstractRLAlgorithm = setup_rl_algorithm(env, cfg)
        experiment.set_rl_algorithm(rl_algorithm)
    if experiment.is_model:
        model: AbstractSequenceAlgorithm = setup_model_algorithm(env, cfg)
        experiment.set_model_algorithm(model)

    return experiment
