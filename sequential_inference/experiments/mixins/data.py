from abc import abstractmethod
from sequential_inference.envs.vec_env.vec_env import VecEnv
from typing import Dict

import gym
import torch

from sequential_inference.util.data import gather_trajectory_data
from sequential_inference.data.data import (
    BatchTrajectorySampler,
    TrajectoryReplayBuffer,
)
from sequential_inference.abc.experiment import (
    AbstractExperiment,
    AbstractRLExperiment,
    ExperimentMixin,
)
from sequential_inference.rl.agents import RandomAgent


class AbstractDataMixin(ExperimentMixin):

    experiment: AbstractExperiment
    env: VecEnv
    buffer: TrajectoryReplayBuffer

    def __init__(
        self,
        env: VecEnv,
        buffer: TrajectoryReplayBuffer,
        experiment: AbstractExperiment = None,
    ):
        self.env = env
        self.buffer = buffer
        self.experiment = experiment

    @abstractmethod
    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        pass

    def set_experiment(self, experiment: AbstractExperiment):
        self.experiment = experiment

    def before_experiment(self):
        pass

    def after_epoch(self, log):
        return log


class FixedDataSamplingMixin(AbstractDataMixin):
    buffer: TrajectoryReplayBuffer
    n: int
    dataset: BatchTrajectorySampler

    def set_num_sampling_steps(self, n_init: int) -> None:
        self.n_init = n_init

    def before_experiment(self) -> None:
        agent = RandomAgent(self.env.action_space)
        gather_trajectory_data(self.env, agent, self.buffer, self.n_init)
        self.dataset = BatchTrajectorySampler(self.buffer)

    def reload_preempted(self, run_dir):
        # TODO: add data reloading shenanigans here
        pass

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.dataset.get_next(batch_size)


class DataSamplingMixin(AbstractDataMixin):
    buffer: TrajectoryReplayBuffer
    n: int
    n_init: int
    experiment: AbstractRLExperiment

    def set_num_sampling_steps(self, n: int, n_init: int) -> None:
        self.n = n
        self.n_init = n_init

    def before_experiment(self) -> None:
        agent = RandomAgent(self.env.action_space)
        gather_trajectory_data(self.env, agent, self.buffer, self.n_init)
        self.dataset = BatchTrajectorySampler(self.buffer)

    def after_epoch(self, epoch_log: Dict[str, torch.Tensor]):
        agent = self.experiment.get_agent()
        gather_trajectory_data(self.env, agent, self.buffer, self.n)
        self.dataset = BatchTrajectorySampler(self.buffer)
        super().after_epoch(epoch_log)

    def reload_preempted(self, run_dir):
        # TODO: add data reloading shenanigans here
        pass

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        return self.dataset.get_next(batch_size)


class OnlineDataSamplingMixin(AbstractDataMixin):
    n: int
    n_init: int

    def set_num_sampling_steps(self, n: int, n_init: int) -> None:
        self.n = n
        self.n_init = n_init

    def reload_preempted(self, run_dir):
        # TODO: add data reloading shenanigans here
        pass

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        # TODO: fix this shit
        return {"TODO": torch.Tensor([0.0])}


class DynaSamplingMixin(AbstractDataMixin):
    n: int
    n_init: int

    def set_num_sampling_steps(self, n: int, n_init: int) -> None:
        self.n = n
        self.n_init = n_init

    def reload_preempted(self, run_dir):
        super.reload_preempted(run_dir)
        # TODO: add data reloading shenanigans here

    def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        # TODO: fix this shit
        return {"TODO": torch.Tensor([0.0])}

    def get_model_batch(self) -> Dict[str, torch.Tensor]:
        # TODO: fix this shit
        return {"TODO": torch.Tensor([0.0])}
