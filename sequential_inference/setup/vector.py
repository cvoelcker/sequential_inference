import os
from typing import Tuple


def get_slurm_id() -> str:
    return os.getenv("SLURM_JOB_ID") or "0"


def vector_preface(cfg) -> Tuple[str, bool]:
    log_dir = os.getcwd()

    if cfg.cluster == "vector":
        chp_dir = "/checkpoint/{}/{}/chp".format(os.getenv("USER"), get_slurm_id())
    elif cfg.cluster == "local":
        chp_dir = os.path.join(log_dir, "chp")
    else:
        raise NotImplementedError(f"Computing on {cfg.cluster} is not implemented")

    if not os.path.exists(chp_dir):
        os.mkdir(chp_dir)
        return chp_dir, False
    return chp_dir, True
