import itertools
from typing import Dict, Iterator

import torch
from sequential_inference.abc.data import AbstractDataBuffer, AbstractDataSampler


class DataBuffer(AbstractDataBuffer):
    def __init__(
        self,
        capacity: int,
        env,
        sample_length=1,
        device="cpu",
    ):
        self.device = device

        obs_space = env.observation_space
        obs_type = env.reset().dtype
        act_space = env.action_space

        self.obs_shape = obs_space.shape
        self.act_shape = act_space.shape

        self.s = torch.zeros((capacity, *obs_space.shape), dtype=obs_type)  # .to(
        #   self.device
        # )
        self.a = torch.zeros((capacity, *act_space.shape), dtype=torch.float)
        # .to(
        #     self.device
        # )
        self.r = torch.zeros((capacity, 1))  # .to(self.device)
        self.d = torch.zeros((capacity, 1))  # .to(self.device)
        self.t = torch.zeros((capacity, 1))  # .to(self.device)

        self.sample_length = sample_length
        self.capacity = capacity
        self.length = 0
        self.fill_counter = 0
        self.full = False

        self.register_module("s", self.s)  # type: ignore
        self.register_module("a", self.a)  # type: ignore
        self.register_module("r", self.r)  # type: ignore
        self.register_module("d", self.d)  # type: ignore
        self.register_module("t", self.t)  # type: ignore

    def insert(self, trajectory):
        traj_len = trajectory["obs"].shape[0]

        if self.fill_counter + traj_len > self.capacity:
            self.full = True
            self.length = self.fill_counter
            self.fill_counter = 0
        else:
            self.length = self.length + traj_len

        self.s[self.fill_counter : self.fill_counter + traj_len] = (
            trajectory["obs"].detach().to("cpu")
        )  # .to(self.device).detach()
        self.a[self.fill_counter : self.fill_counter + traj_len] = (
            trajectory["act"].detach().to("cpu")
        )  # .to(self.device).detach()
        self.r[self.fill_counter : self.fill_counter + traj_len] = (
            trajectory["rew"].detach().to("cpu")
        )  # .to(self.device).detach()
        self.d[self.fill_counter : self.fill_counter + traj_len] = (
            trajectory["done"].detach().to("cpu")
        )  # .to(self.device).detach()

    def __getitem__(self, i):
        s = self.s[i : i + self.sample_length].float().squeeze(0).to(self.device)
        a = self.a[i : i + self.sample_length].float().squeeze(0).to(self.device)
        r = self.r[i : i + self.sample_length].float().squeeze(0).to(self.device)
        d = self.d[i : i + self.sample_length].float().squeeze(0).to(self.device)
        s_n = (
            self.s[i + 1 : i + 1 + self.sample_length]
            .float()
            .squeeze(0)
            .to(self.device)
        )
        return dict(obs=s, act=a, rew=r, next_obs=s_n, done=d)

    def __len__(self):
        return min(self.length - self.sample_length, 0)


class TrajectoryReplayBuffer(AbstractDataBuffer):
    def __init__(
        self,
        num_trajectories,
        trajectory_length,
        env,
        sample_length=1,
        device="cpu",
    ):
        super().__init__()
        self.device = device

        obs_space = env.observation_space
        if len(obs_space.shape) == 3:
            obs_type = torch.uint8
            self.visual = True
        else:
            obs_type = env.reset().dtype
            self.visual = False
        act_space = env.action_space

        self.obs_shape = obs_space.shape
        self.act_shape = act_space.shape

        self.s = torch.zeros(
            (num_trajectories, trajectory_length, *obs_space.shape), dtype=obs_type
        )  # .to(self.device)
        self.a = torch.zeros(
            (num_trajectories, trajectory_length, *act_space.shape), dtype=torch.float
        )  # .to(self.device)
        self.r = torch.zeros(
            (num_trajectories, trajectory_length, 1)
        )  # .to(self.device)
        self.d = torch.ones(
            (num_trajectories, trajectory_length, 1)
        )  # .to(self.device)
        self.t = torch.zeros(
            (num_trajectories, trajectory_length, 1)
        )  # .to(self.device)

        self.trajectory_length = trajectory_length
        self.sample_length = sample_length
        self.capacity = num_trajectories
        self.fill_counter = 0
        self.full = False

        print(self.state_dict())

        self.register_module("s", self.s)  # type: ignore
        self.register_module("a", self.a)  # type: ignore
        self.register_module("r", self.r)  # type: ignore
        self.register_module("d", self.d)  # type: ignore
        self.register_module("t", self.t)  # type: ignore

    def insert(self, trajectory):
        self.s[self.fill_counter] = trajectory["obs"].to("cpu").detach()
        self.a[self.fill_counter] = trajectory["act"].to("cpu").detach()
        self.r[self.fill_counter] = trajectory["rew"].to("cpu").detach()
        self.d[self.fill_counter] = trajectory["done"].to("cpu").detach()
        self.t[self.fill_counter] = trajectory["task"].to("cpu").detach()

        self.fill_counter += 1
        if self.fill_counter == self.capacity:
            self.full = True
            self.fill_counter = 0

        self.length = (
            self.capacity * (self.trajectory_length - self.sample_length - 1)
            if self.full
            else self.fill_counter * (self.trajectory_length - self.sample_length - 1)
        )

    def __getitem__(self, k):
        t = k // (self.trajectory_length - self.sample_length - 1)
        i = k % (self.trajectory_length - self.sample_length - 1)
        s = self.s[t, i : i + self.sample_length].float().squeeze(0).to(self.device)
        a = self.a[t, i : i + self.sample_length].float().squeeze(0).to(self.device)
        r = self.r[t, i : i + self.sample_length].float().squeeze(0).to(self.device)
        d = self.d[t, i : i + self.sample_length].float().squeeze(0).to(self.device)
        s_n = (
            self.s[t, i + 1 : i + 1 + self.sample_length]
            .float()
            .squeeze(0)
            .to(self.device)
        )
        if self.visual:
            s = s / 255.0
            s_n = s_n / 255.0
        return dict(obs=s, act=a, rew=r, next_obs=s_n, done=d)

    def __len__(self):
        return self.length


# class BatchDataSampler(AbstractDataSampler):
#     """The BatchDataSampler iterates infinitely over a given dataset. It models the behavior of many RL algorithms that do not sample full batches.
#     To provide flexible batch sizes with minimal overhead, it holds a set of iterators for different batch sizes as "views" on the data.
#     """
#
#     def __init__(self, buffer: AbstractDataBuffer):
#         self.buffer = buffer
#
#         self.iterators: Dict[int, Iterator[Dict[str, torch.Tensor]]] = {}
#
#     def _make_iterator(self, batch_size: int):
#         _dataloader = torch.utils.data.DataLoader(  # type: ignore
#             self.buffer, batch_size=batch_size, drop_last=True, shuffle=True
#         )
#         self.iterators[batch_size] = itertools.cycle(iter(_dataloader))
#
#     def get_next(self, batch_size: int):
#         if batch_size not in self.iterators.keys():
#             self._make_iterator(batch_size)
#         return self.iterators[batch_size].__next__()


class BatchDataSampler(AbstractDataSampler):
    def __init__(self, buffer: AbstractDataBuffer):
        self.buffer = buffer
        self.iterators: Dict[int, Iterator[Dict[str, torch.Tensor]]] = {}

    def _make_iterator(self, batch_size: int):
        _dataloader = torch.utils.data.DataLoader(  # type: ignore
            self.buffer, batch_size=batch_size, drop_last=True, shuffle=True
        )
        self.iterators[batch_size] = iter(_dataloader)

    def get_next(self, batch_size):
        if batch_size not in self.iterators.keys():
            self._make_iterator(batch_size)
        try:
            return self.iterators[batch_size].__next__()
        except StopIteration as e:
            self._make_iterator(batch_size)
            return self.iterators[batch_size].__next__()
