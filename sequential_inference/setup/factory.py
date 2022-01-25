from torch import nn
from typing import Dict, Type

from sequential_inference.abc.env import Env
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from sequential_inference.abc.experiment import AbstractExperiment

from sequential_inference.data.strategy import *

from sequential_inference.experiments.base import *
from sequential_inference.experiments.latent_dyna import *
from sequential_inference.experiments.latent_rl import *
from sequential_inference.experiments.dreamer import *

from sequential_inference.models.base.base_nets import *

from sequential_inference.rl.sac import SACAlgorithm

from sequential_inference.algorithms.simple_stove import SimplifiedStoveAlgorithm
from sequential_inference.algorithms.slac import SLACModelAlgorithm
from sequential_inference.algorithms.simple_vi import VIModelAlgorithm
from sequential_inference.algorithms.dreamer import DreamerAlgorithm


EXPERIMENT_REGISTRY: Dict[str, Type[AbstractExperiment]] = {
    "latent_model_rl": LatentTrainingExperiment,
    "latent_imagination_rl": LatentImaginationExperiment,
    "dyna_rl": DynaTrainingExperiment,
    "model_training": ModelTrainingExperiment,
    "rl_training": RLTrainingExperiment,
}

MODEL_ALGORITHM_REGISTRY = {
    "Dreamer": DreamerAlgorithm,
    "SimpleVI": VIModelAlgorithm,
    "SLAC": SLACModelAlgorithm,
    "SimpleStove": SimplifiedStoveAlgorithm,
}

ENCODER_DECODER_REGISTRY = {
    "Visual": (EncoderNet, BroadcastDecoderNet),
    "MLP": (MLP, MLP),
    "Slac": (SLACEncoder, SLACDecoder),
}

OPTIMIZER_REGISTRY = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "ada_grad": torch.optim.Adagrad,
    "rms_prop": torch.optim.RMSprop,
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

    return experiment

def setup_rl_algorithm(env, cfg):

    if cfg.rl.algorithm == "sac":
        if cfg.rl.latent_input:
            actor = TanhGaussian(
                cfg.algorithm.parameters.latent_dim,
                env.action_space.shape[0],
                cfg.rl.hidden_units,
                multiplier=env.action_space.high,
            )
            critic = TwinnedMLP(cfg.algorithm.parameters.latent_dim + env.action_space.shape[0], 1, cfg.rl.hidden_units)
            latent = True
            observation = False
        elif cfg.data.visual_obs:
            encoder = EncoderNet(
                env.observation_space[-1],
                cfg.rl.latent_dim,
                cfg.rl.encoder_hidden_units,
                env.observation_space[:-1],
            )
            actor_head = TanhGaussian(
                cfg.rl.encoder_hidden_units,
                env.action_space.shape,
                cfg.rl.hidden_units,
                multiplier=cfg.action_space.high,
            )
            critic_head = TwinnedMLP(
                cfg.rl.encoder_hidden_units + env.action_space.shape[0], 1, cfg.rl.hidden_units
            )
            actor = nn.Sequential(encoder, actor_head)
            critic = nn.Sequential(encoder, critic_head)
            latent = False
            observation = True
        else:
            actor = Gaussian(
                env.observation_space.shape, env.action_space.shape, cfg.rl.hidden_units
            )
            critic = TwinnedMLP(env.observation_space.shape + env.action_space.shape[0], 1, cfg.rl.hidden_units)
            latent = False
            observation = True
        return SACAlgorithm(
            actor,
            critic,
            len(env.action_space.shape),
            cfg.rl.init_alpha,
            cfg.rl.gamma,
            cfg.rl.tau,
            cfg.rl.update_alpha,
            latent=latent,
            observation=observation,
        )


def setup_model_algorithm(env: Env, cfg) -> AbstractSequenceAlgorithm:
    input_dim = env.observation_space.shape[-1]
    output_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    AlgorithmClass = MODEL_ALGORITHM_REGISTRY[cfg.algorithm.name]
    Encoder, Decoder = ENCODER_DECODER_REGISTRY[cfg.encoder_decoder.name]

    encoder = Encoder(
        input_dim + 1,
        cfg.algorithm.parameters.feature_dim,
        **cfg.encoder_decoder.encoder.parameters
    )
    decoder = Decoder(
        cfg.algorithm.parameters.latent_dim,
        output_dim,
        **cfg.encoder_decoder.decoder.parameters
    )
    reward_decoder = MLP(
        cfg.algorithm.parameters.latent_dim, 1, **cfg.reward_decoder.parameters
    )

    algorithm = AlgorithmClass(
        encoder, decoder, reward_decoder, action_dim, **cfg.algorithm.parameters
    )

    if cfg.add_global_belief:
        raise NotImplementedError("Belief is not ready")
    return algorithm


def setup_optimizer(cfg):
    return OPTIMIZER_REGISTRY[cfg]


def setup_data(env, cfg):
    buffer = TrajectoryReplayBuffer(cfg.data.buffer_num_trajectories, cfg.data.buffer_trajectory_length, env, sample_length=cfg.data.sample_length, device=cfg.device)

    if cfg.data.name == "fixed":
        return FixedDataStrategy(env, )