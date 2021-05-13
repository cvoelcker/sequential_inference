from sequential_inference.algorithms.belief import BeliefModelAlgorithm
from torch import nn

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
    "Slac": SLACModelAlgorithm,
    "SimpleStove": SimplifiedStoveAlgorithm,
}

REGISTERED_ENCODER_DECODER = {
    "Visual": (EncoderNet, BroadcastDecoderNet),
    "MLP": (MLP, MLP),
    "Slac": (SLACEncoder, SLACDecoder),
}

REGISTERED_BELIEF_ALGORITHMS = {}


def make_sequence_algorithm(input_dim, output_dim, action_dim, cfg):
    AlgorithmClass = REGISTERED_ALGORITHMS[cfg.algorithm.name]
    Encoder, Decoder = REGISTERED_ENCODER_DECODER[cfg.encoder_decoder.name]

    encoder = Encoder(
        input_dim,
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
        Encoder, Decoder = REGISTERED_ENCODER_DECODER[
            cfg.belief_algorithm.encoder_decoder_architecture
        ]

        encoder = Encoder(
            input_dim,
            cfg.belief_algorithm.feature_dim,
            **cfg.belief_algorithm.encoder_parameters
        )
        decoder = Decoder(
            cfg.belief_algorithm.latent_dim,
            output_dim,
            **cfg.belief_algorithm.decoder_parameters
        )

        BeliefAlgorithmClass = REGISTERED_BELIEF_ALGORITHMS[cfg.belief_algorithm.name]
        belief_algorithm = BeliefAlgorithmClass(
            encoder,
            action_dim,
            cfg.belief_feature_dim,
            cfg.belief_latent_dim,
            **cfg.algorithm_parameters
        )

        algorithm = BeliefModelAlgorithm(
            algorithm, belief_algorithm, cfg.belief_algorithm.expand_belief
        )

    return algorithm
