from abc import abstractmethod
from sequential_inference.abc.common import Checkpointable
from sequential_inference.abc.rl import AbstractAgent
from sequential_inference.envs.vec_env.vec_env import VecEnv
from typing import Dict

import gym
import torch

from sequential_inference.util.data import gather_trajectory_data, load_offline_data
from sequential_inference.util.rl_util import load_agent
from sequential_inference.data.storage import (
    BatchTrajectorySampler,
    TrajectoryReplayBuffer,
)
from sequential_inference.abc.experiment import (
    AbstractExperiment,
    AbstractRLExperiment,
    ExperimentMixin,
)
from sequential_inference.rl.agents import RandomAgent


class AbstractDataStrategy(ExperimentMixin, Checkpointable):

    experiment: AbstractExperiment
    env: VecEnv
    buffer: TrajectoryReplayBuffer

    def __init__(
        self,
        env: VecEnv,
        buffer: TrajectoryReplayBuffer,
    ):
        self.env = env
        self.buffer = buffer

    @abstractmethod
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        pass

    def initialize(self, cfg, preeempted: bool, run_dir: str):
        pass

    def update(self, log, agent, **kwargs):
        return log


class FixedDataStrategy(AbstractDataStrategy):
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


class OnlineDataSamplingStrategy(AbstractDataStrategy):
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

    def update(self, epoch_log: Dict[str, torch.Tensor], agent: AbstractAgent, **kwargs):
        gather_trajectory_data(self.env, agent, self.buffer, self.n)
        self.dataset = BatchTrajectorySampler(self.buffer)
        super().after_epoch(epoch_log)

    def reload_preempted(self, run_dir):
        # TODO: add data reloading shenanigans here
        pass

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.dataset.get_next(batch_size)


class ModelBasedOnlineDataSamplingStrategy(AbstractDataStrategy):
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

    def update(self, epoch_log: Dict[str, torch.Tensor], agent: AbstractAgent, **kwargs):
        gather_trajectory_data(self.env, agent, self.buffer, self.n)
        self.dataset = BatchTrajectorySampler(self.buffer)
        super().after_epoch(epoch_log)

    def reload_preempted(self, run_dir):
        # TODO: add data reloading shenanigans here
        pass

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.dataset.get_next(batch_size)