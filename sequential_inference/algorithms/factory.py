import gym

from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from sequential_inference.models.base.base_nets import (
    BroadcastDecoderNet,
    EncoderNet,
    MLP,
    SLACDecoder,
    SLACEncoder,
)
from sequential_inference.algorithms.simple_stove import SimplifiedStoveAlgorithm
from sequential_inference.algorithms.slac import SLACModelAlgorithm
from sequential_inference.algorithms.simple_vi import VIModelAlgorithm
from sequential_inference.algorithms.dreamer import DreamerAlgorithm


REGISTERED_ALGORITHMS = {
    "Dreamer": DreamerAlgorithm,
    "SimpleVI": VIModelAlgorithm,
    "SLAC": SLACModelAlgorithm,
    "SimpleStove": SimplifiedStoveAlgorithm,
}

REGISTERED_ENCODER_DECODER = {
    "Visual": (EncoderNet, BroadcastDecoderNet),
    "MLP": (MLP, MLP),
    "Slac": (SLACEncoder, SLACDecoder),
}


def setup_model_algorithm(env: gym.Env, cfg) -> AbstractSequenceAlgorithm:
    input_dim = env.observation_space.shape[-1]
    output_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    AlgorithmClass = REGISTERED_ALGORITHMS[cfg.algorithm.name]
    Encoder, Decoder = REGISTERED_ENCODER_DECODER[cfg.encoder_decoder.name]

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
