import os


def get_slurm_id():
    return os.getenv("SLURM_JOB_ID") or "0"


def vector_preface(cfg):
    log_dir = "."
    if cfg.cluster == "vector":
        print("Running on Vector cluster")
        log_dir = "/checkpoints/{}/{}".format(os.getenv("USER"), get_slurm_id())
        setattr(cfg, "chp_dir", log_dir)

    if os.path.exists(os.path.join(log_dir, "status")):
        return True

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return False
