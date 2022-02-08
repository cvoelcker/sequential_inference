from typing import Dict

from gym import Env
import omegaconf
import torch

from sequential_inference.abc.data import AbstractDataHandler, AbstractDataSampler
from sequential_inference.abc.rl import AbstractAgent
from sequential_inference.data.storage import TrajectoryReplayBuffer
from sequential_inference.data.offline import load_offline_data
from sequential_inference.algorithms.rl.agents import RandomAgent
from sequential_inference.util.data import gather_trajectory_data
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
        return FixedDataStrategy(env, buffer)
    elif cfg.data.name == "online":
        return OnlineDataSamplingStrategy(env, buffer)


class DataStrategy(AbstractDataHandler):
    def initialize(self, cfg, preempted: bool):
        run_dir = "chp"
        if preempted:
            self.reload_preempted(run_dir)
        if cfg.data.init_data_source == "random":
            agent = RandomAgent(self.env.action_space)
            gather_trajectory_data(self.env, agent, self.buffer, cfg.data.n_init)
            self.dataset = AbstractDataSampler(self.buffer)
        elif cfg.data.init_data_source == "agent":
            agent = load_agent(cfg.data.agent_path)
            gather_trajectory_data(self.env, agent, self.buffer, cfg.data.n_init)
            self.dataset = AbstractDataSampler(self.buffer)
        elif cfg.data.init_data_source == "dataset":
            self.dataset = load_offline_data(cfg.data.dataset_path)
        elif cfg.data.init_data_source == "empty":
            self.dataset = AbstractDataSampler(self.buffer)
        return super().initialize(cfg)

    def reload_preempted(self, run_dir):
        # TODO: add data reloading shenanigans here
        pass

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.dataset.get_next(batch_size)


class FixedDataStrategy(DataStrategy):
    def update(
        self, epoch_log: Dict[str, torch.Tensor], agent: AbstractAgent, **kwargs
    ):
        super().after_epoch(epoch_log)


class OnlineDataSamplingStrategy(DataStrategy):
    def set_num_sampling_steps(self, n: int, n_init: int) -> None:
        self.n = n
        self.n_init = n_init

    def update(
        self, epoch_log: Dict[str, torch.Tensor], agent: AbstractAgent, **kwargs
    ):
        gather_trajectory_data(self.env, agent, self.buffer, self.n)
        self.dataset = AbstractDataSampler(self.buffer)
        super().after_epoch(epoch_log)
