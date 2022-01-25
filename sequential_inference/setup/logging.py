from sequential_inference.logging import *
import os
from sequential_inference.logging.logger import CsvLogger, TorchTensorboardHandler


def setup_logging(cfg, run_dir: str, preempted: bool = False):
    loggers = []
    for item in cfg.logging:
        if item.type == "file_logger":
            logger = CsvLogger(run_dir, item.file_name, keys=(item.key), overwrite=False)
        if item.type == "tensorboard":
            logger = TorchTensorboardHandler(
                logdir = os.path.join(run_dir, "tb"),
                namedir = item.name,
                keys = (item.key),
                reset_logdir=False)
        loggers.append(logger)
    return loggers

