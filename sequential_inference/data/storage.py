import os
import pickle
import itertools
from typing import Dict, Iterator
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import numpy as np

from sequential_inference.abc.util import AbstractDataSampler

USE_GPU = torch.cuda.is_available()
DEVICE = "cuda" if USE_GPU else "cpu"


class TrajectoryReplayBuffer(Dataset):
    def __init__(
        self,
        num_trajectories,
        trajectory_length,
        env,
        sample_length=1,
        device="cpu",
    ):
        obs_space = env.observation_space
        obs_type = env.reset().dtype
        act_space = env.action_space

        self.obs_shape = obs_space.shape
        self.act_shape = act_space.shape

        self.s = torch.zeros(
            (num_trajectories, trajectory_length, *obs_space.shape), dtype=obs_type
        ).to(DEVICE)
        self.a = torch.zeros(
            (num_trajectories, trajectory_length, *act_space.shape), dtype=torch.float
        ).to(DEVICE)
        self.r = torch.zeros((num_trajectories, trajectory_length, 1)).to(DEVICE)
        self.d = torch.zeros((num_trajectories, trajectory_length, 1)).to(DEVICE)

        self.trajectory_length = trajectory_length
        self.sample_length = sample_length
        self.capacity = num_trajectories
        self.fill_counter = 0
        self.full = False

    def insert(self, trajectory):
        self.s[self.fill_counter] = trajectory["obs"].to(DEVICE).detach()
        self.a[self.fill_counter] = trajectory["act"].to(DEVICE).detach()
        self.r[self.fill_counter] = trajectory["rew"].to(DEVICE).detach()
        self.d[self.fill_counter] = trajectory["done"].to(DEVICE).detach().unsqueeze(-1)

        self.length = (
            self.capacity * (self.trajectory_length - self.sample_length - 1)
            if self.full
            else self.fill_counter * (self.trajectory_length - self.sample_length - 1)
        )

        self.fill_counter += 1
        if self.fill_counter == self.capacity:
            self.full = True
            self.fill_counter = 0

    def __getitem__(self, k):
        t = k // (self.trajectory_length - self.sample_length - 1)
        i = k % (self.trajectory_length - self.sample_length - 1)
        s = self.s[t, i : i + self.sample_length].float().squeeze(0)
        a = self.a[t, i : i + self.sample_length].float().squeeze(0)
        r = self.r[t, i : i + self.sample_length].float().squeeze(0)
        d = self.d[t, i : i + self.sample_length].float().squeeze(0)
        s_n = self.s[t, i + 1 : i + 1 + self.sample_length].float().squeeze(0)
        return dict(obs=s, act=a, rew=r, next_obs=s_n, done=d)

    def __len__(self):
        return self.length


class BatchDataSampler(AbstractDataSampler):
    """The BatchDataSampler iterates infinitely over a given dataset. It models the behavior of many RL algorithms that do not sample full batches.
    To provide flexible batch sizes with minimal overhead, it holds a set of iterators for different batch sizes as "views" on the data.
    """

    def __init__(self, buffer: Dataset):
        self.buffer = buffer

        self.iterators: Dict[int, Iterator[Dict[str, torch.Tensor]]] = {}

    def _make_iterator(self, batch_size: int):
        _dataloader = torch.utils.data.DataLoader(
            self.buffer, batch_size=batch_size, drop_last=True, shuffle=True
        )
        self.iterators[batch_size] = itertools.cycle(iter(_dataloader))

    def get_next(self, batch_size: int):
        if batch_size not in self.iterators.keys():
            self._make_iterator(batch_size)
        return self.iterators[batch_size].__next__()
