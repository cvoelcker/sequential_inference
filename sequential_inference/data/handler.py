import os
from typing import Dict

import omegaconf
import torch

from sequential_inference.abc.common import Env
from sequential_inference.abc.data import AbstractDataHandler
from sequential_inference.abc.rl import AbstractAgent
from sequential_inference.data.storage import BatchDataSampler, TrajectoryReplayBuffer
from sequential_inference.data.offline import load_offline_data
from sequential_inference.algorithms.rl.agents import RandomAgent
from sequential_inference.data.collection import gather_data
from sequential_inference.util.rl_util import load_agent


def setup_data(cfg: omegaconf.DictConfig, env: Env) -> AbstractDataHandler:
    buffer = TrajectoryReplayBuffer(
        cfg.data.buffer_num_trajectories,
        cfg.data.buffer_trajectory_length,
        env,
        sample_length=cfg.data.sample_length,
        device=cfg.device,
    )

    if cfg.data.name == "fixed":
        handler = FixedDataStrategy(env, buffer)
    elif cfg.data.name == "online":
        handler = OnlineDataSamplingStrategy(env, buffer)
        handler.set_num_sampling_steps(cfg.data.n, cfg.data.n_init)
    else:
        raise ValueError(f"Unknown data strategy {cfg.data.name}")
    return handler


class DataStrategy(AbstractDataHandler):
    def initialize(self, cfg: omegaconf.DictConfig, preempted: bool):
        if preempted:
            self.reload_preempted(cfg)
        if cfg.data.init_data_source == "random":
            agent = RandomAgent(self.env.action_space)
            gather_data(self.env, agent, self.buffer, cfg.data.n_init)
            self.dataset = BatchDataSampler(self.buffer)
        elif cfg.data.init_data_source == "agent":
            agent = load_agent(cfg.data.agent_path)
            gather_data(self.env, agent, self.buffer, cfg.data.n_init)
            self.dataset = BatchDataSampler(self.buffer)
        elif cfg.data.init_data_source == "dataset":
            self.dataset = load_offline_data(cfg.data.dataset_path)
        elif cfg.data.init_data_source == "empty":
            self.dataset = BatchDataSampler(self.buffer)
        return super().initialize(cfg, preempted)

    def reload_preempted(self, cfg):
        chp_dir = os.path.join(cfg.chp_dir, "data/save.torch")
        self.buffer.load(chp_dir)

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.dataset.get_next(batch_size)


class FixedDataStrategy(DataStrategy):
    def update(
        self, epoch_log: Dict[str, torch.Tensor], agent: AbstractAgent, **kwargs
    ):
        return epoch_log


class OnlineDataSamplingStrategy(DataStrategy):
    def set_num_sampling_steps(self, n: int, n_init: int) -> None:
        self.n = n
        self.n_init = n_init

    def update(
        self, epoch_log: Dict[str, torch.Tensor], agent: AbstractAgent, **kwargs
    ):
        print("Called")
        with torch.no_grad():
            log = gather_data(self.env, agent, self.buffer, self.n, explore=True)

        self.dataset = BatchDataSampler(self.buffer)
        if log is None:
            return epoch_log
        else:
            return {**log, **epoch_log}
