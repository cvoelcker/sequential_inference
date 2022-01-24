from torch import nn

from sequential_inference.models.base.base_nets import (
    EncoderNet,
    Gaussian,
    TanhGaussian,
    TwinnedMLP,
)
from sequential_inference.rl.sac import SACAlgorithm


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
