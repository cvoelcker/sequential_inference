import omegaconf

from sequential_inference.abc.common import Env


def hydra_make_list(**kwargs):
    return list(kwargs.values())


def fix_env_config(cfg: omegaconf.DictConfig, env: Env) -> omegaconf.DictConfig:
    setattr(cfg.env, "obs_dim", env.observation_space.shape)  # type: ignore
    setattr(cfg.env, "action_dim", env.action_space.shape)  # type: ignore
    return cfg
