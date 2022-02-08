import os


def get_slurm_id():
    return os.getenv("SLURM_JOB_ID") or "0"


def vector_preface(cfg):
    log_dir = "."
    if cfg.cluster == "vector":
        print("Running on Vector cluster")
        log_dir = "/checkpoint/{}/{}".format(os.getenv("USER"), get_slurm_id())
        setattr(cfg, "chp_dir", log_dir)
    chp_dir = os.path.join(log_dir, "chp")

    if not os.path.exists(chp_dir):
        os.mkdir(chp_dir)
        return False
    return True
