import os
from typing import Tuple

import omegaconf


def get_slurm_id() -> str:
    return os.getenv("SLURM_JOB_ID") or "0"


def vector_preface(cfg: omegaconf.DictConfig) -> bool:
    """Modifies the config to include the correct checkpointing path and checks whether a previous checkpoint exists

    Args:
        cfg (omegaconf.DictConfig): the current config

    Returns:
        bool: checks for existing chp
    """
    chp_dir = os.getcwd()
    if cfg.cluster == "vector" and get_slurm_id() != "0":
        chp_dir = "/checkpoint/{}/{}/chp".format(os.getenv("USER"), get_slurm_id())
    chp_dir = os.path.join(chp_dir, "chp")
    setattr(cfg, "chp_dir", chp_dir)
    if not os.path.exists(chp_dir):
        os.mkdir(chp_dir)
        return False
    return True
