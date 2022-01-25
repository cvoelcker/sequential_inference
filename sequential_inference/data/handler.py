from abc import abstractmethod
from sequential_inference.abc.data import AbstractDataHandler
from sequential_inference.abc.rl import AbstractAgent
from typing import Dict

import torch

from sequential_inference.util.data import gather_trajectory_data, load_offline_data
from sequential_inference.util.rl_util import load_agent
from sequential_inference.data.storage import (
    BatchTrajectorySampler,
    TrajectoryReplayBuffer,
)
from sequential_inference.algorithms.rl.agents import RandomAgent


def setup_data(env, cfg):
    buffer = TrajectoryReplayBuffer(
        cfg.data.buffer_num_trajectories,
        cfg.data.buffer_trajectory_length,
        env,
        sample_length=cfg.data.sample_length,
        device=cfg.device,
    )

    if cfg.data.name == "fixed":
        return FixedDataStrategy(
            env,
        )
    # TODO finish


class FixedDataStrategy(AbstractDataHandler):
    buffer: TrajectoryReplayBuffer
    n: int
    dataset: BatchTrajectorySampler

    def initialize(self, cfg, preeempted: bool, run_dir: str):
        if preeempted:
            self.reload_preempted(run_dir)
        if cfg.data.data_source == "random":
            agent = RandomAgent(self.env.action_space)
            gather_trajectory_data(self.env, agent, self.buffer, cfg.data.n_init)
            self.dataset = BatchTrajectorySampler(self.buffer)
        elif cfg.data.data_source == "agent":
            agent = load_agent(cfg.data.agent_path)
            gather_trajectory_data(self.env, agent, self.buffer, cfg.data.n_init)
            self.dataset = BatchTrajectorySampler(self.buffer)
        elif cfg.data.data_source == "dataset":
            self.dataset = load_offline_data(cfg.data.dataset_path)
        elif cfg.data.data_source == "empty":
            self.dataset = BatchTrajectorySampler(self.buffer)
        super().initialize(cfg)

    def reload_preempted(self, run_dir):
        # TODO: add data reloading shenanigans here
        pass

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.dataset.get_next(batch_size)


class OnlineDataSamplingStrategy(AbstractDataHandler):
    buffer: TrajectoryReplayBuffer
    n: int
    n_init: int

    def initialize(self, cfg, preeempted: bool, run_dir: str):

        self.set_num_sampling_steps(cfg.data.n, cfg.data.n_init)

        if preeempted:
            self.reload_preempted(run_dir)
        if cfg.data.data_source == "random":
            agent = RandomAgent(self.env.action_space)
            gather_trajectory_data(self.env, agent, self.buffer, cfg.data.n_init)
            self.dataset = BatchTrajectorySampler(self.buffer)
        elif cfg.data.data_source == "agent":
            agent = load_agent(cfg.data.agent_path)
            gather_trajectory_data(self.env, agent, self.buffer, cfg.data.n_init)
            self.dataset = BatchTrajectorySampler(self.buffer)
        elif cfg.data.data_source == "dataset":
            self.dataset = load_offline_data(cfg.data.dataset_path)
        super().initialize(cfg)

    def set_num_sampling_steps(self, n: int, n_init: int) -> None:
        self.n = n
        self.n_init = n_init

    def update(
        self, epoch_log: Dict[str, torch.Tensor], agent: AbstractAgent, **kwargs
    ):
        gather_trajectory_data(self.env, agent, self.buffer, self.n)
        self.dataset = BatchTrajectorySampler(self.buffer)
        super().after_epoch(epoch_log)

    def reload_preempted(self, run_dir):
        # TODO: add data reloading shenanigans here
        pass

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.dataset.get_next(batch_size)


class ModelBasedOnlineDataSamplingStrategy(AbstractDataHandler):
    buffer: TrajectoryReplayBuffer
    n: int
    n_init: int

    def initialize(self, cfg, preeempted: bool, run_dir: str):

        self.set_num_sampling_steps(cfg.data.n, cfg.data.n_init)

        if preeempted:
            self.reload_preempted(run_dir)
        if cfg.data.data_source == "random":
            agent = RandomAgent(self.env.action_space)
            gather_trajectory_data(self.env, agent, self.buffer, cfg.data.n_init)
            self.dataset = BatchTrajectorySampler(self.buffer)
        elif cfg.data.data_source == "agent":
            agent = load_agent(cfg.data.agent_path)
            gather_trajectory_data(self.env, agent, self.buffer, cfg.data.n_init)
            self.dataset = BatchTrajectorySampler(self.buffer)
        elif cfg.data.data_source == "dataset":
            self.dataset = load_offline_data(cfg.data.dataset_path)
        super().initialize(cfg)

    def set_num_sampling_steps(self, n: int, n_init: int) -> None:
        self.n = n
        self.n_init = n_init

    def update(
        self, epoch_log: Dict[str, torch.Tensor], agent: AbstractAgent, **kwargs
    ):
        gather_trajectory_data(self.env, agent, self.buffer, self.n)
        self.dataset = BatchTrajectorySampler(self.buffer)
        super().after_epoch(epoch_log)

    def reload_preempted(self, run_dir):
        # TODO: add data reloading shenanigans here
        pass

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.dataset.get_next(batch_size)
