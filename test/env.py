from typing import List

import hydra
from omegaconf import OmegaConf

from sequential_inference.abc.common import Env
from sequential_inference.abc.data import AbstractDataHandler
from sequential_inference.data.handler import setup_data
from sequential_inference.util.setup import fix_env_config
from sequential_inference.util.vector import vector_preface


@hydra.main(config_path="../config", config_name="test_dreamer")
def test_train(cfg):
    print(OmegaConf.to_yaml(cfg))

    env: Env = hydra.utils.instantiate(cfg.env)
    cfg = fix_env_config(cfg, env)

    data: AbstractDataHandler = setup_data(cfg, env)

    data.initialize(cfg, False)

    print(env)


if __name__ == "__main__":
    test_train()
