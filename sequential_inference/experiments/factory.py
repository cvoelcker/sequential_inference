from sequential_inference.data.data import TrajectoryReplayBuffer
from sequential_inference.rl.factory import setup_rl_algorithm
from sequential_inference.algorithms.factory import setup_model_algorithm
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from sequential_inference.abc.rl import AbstractRLAlgorithm
from typing import Dict, Type

import gym

from sequential_inference.experiments.mixins.data import (
    AbstractDataMixin,
    DataSamplingMixin,
    DynaSamplingMixin,
    FixedDataSamplingMixin,
    OnlineDataSamplingMixin,
)
from sequential_inference.experiments.training_experiments import (
    ModelTrainingExperiment,
    LatentImaginationExperiment,
    LatentTrainingExperiment,
    DynaTrainingExperiment,
    RLTrainingExperiment,
    TrainingExperiment,
)
from sequential_inference.abc.experiment import AbstractExperiment, ExperimentMixin


EXPERIMENT_REGISTRY: Dict[str, Type[AbstractExperiment]] = {
    "latent_model_rl": LatentTrainingExperiment,
    "latent_imagination_rl": LatentImaginationExperiment,
    "dyna_rl": DynaTrainingExperiment,
    "model_training": ModelTrainingExperiment,
    "rl_training": RLTrainingExperiment,
}

DATA_MIXIN_REGISTRY: Dict[str, Type[AbstractDataMixin]] = {
    "online": OnlineDataSamplingMixin,
    "fixed": FixedDataSamplingMixin,
    "epoch": DataSamplingMixin,
    "dyna": DynaSamplingMixin,
}


def setup_data_mixin(env: gym.Env, cfg) -> AbstractDataMixin:
    buffer = TrajectoryReplayBuffer(
        cfg.data.num_trajectories,
        cfg.data.trajectory_length,
        env,
        cfg.train.sample_length,
        chp_dir=cfg.chp_dir,
    )

    if cfg.data.sampler == "online":
        sampler = OnlineDataSamplingMixin(env, buffer, cfg.data.batch_size)
        return sampler
    elif cfg.data.sampler == "fixed":
        sampler = FixedDataSamplingMixin(env, buffer, cfg.data.batch_size)
        sampler.set_num_sampling_steps(cfg.data.initial_samples)
        sampler.set_env(env)
        return sampler
    elif cfg.data.sampler == "epoch":
        sampler = DataSamplingMixin(env, buffer, cfg.data.batch_size)
        sampler.set_num_sampling_steps(cfg.data.epoch_samples, cfg.data.initial_samples)
        sampler.set_env(env)
        return sampler
    elif cfg.data.sampler == "dyna":
        sampler = DynaTrainingExperiment(env, buffer, cfg.data.batch_size)
        sampler.set_env(env)
        return sampler
    else:
        raise KeyError(f"Data sampling strategy {cfg.data.sampler} no known")


def setup_experiment(env, cfg) -> AbstractExperiment:

    data: AbstractDataMixin = setup_data_mixin(env, cfg)

    if cfg.experiment.type == "latent_model_rl":
        experiment: TrainingExperiment = LatentTrainingExperiment(
            cfg.experiment.pass_rl_gradients_to_model,
            cfg.train.epoch_steps,
            cfg.train_steps.epochs,
            cfg.log.log_frequency,
        )
    elif cfg.experiment.type == "latent_imagination_rl":
        experiment: TrainingExperiment = LatentImaginationExperiment(
            cfg.train.epoch_steps,
            cfg.train_steps.epochs,
            cfg.log.log_frequency,
        )
    elif cfg.experiment.type == "dyna_rl":
        experiment: TrainingExperiment = DynaTrainingExperiment(
            cfg.train.epoch_steps,
            cfg.train_steps.epochs,
            cfg.log.log_frequency,
        )
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

    experiment.set_data_sampler(data)
    if experiment.is_rl:
        rl_algorithm: AbstractRLAlgorithm = setup_rl_algorithm(env, cfg)
        experiment.set_rl_algorithm(rl_algorithm)
    if experiment.is_model:
        model: AbstractSequenceAlgorithm = setup_model_algorithm(env, cfg)
        experiment.set_model_algorithm(model)

    return experiment
