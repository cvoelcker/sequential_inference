import os
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

import numpy as np

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
        chp_dir=None,
    ):
        obs_space = env.observation_space
        obs_type = torch.from_numpy(env.reset()).dtype
        act_space = env.action_space

        self.obs_shape = obs_space.shape
        self.act_shape = act_space.shape

        self.s = torch.zeros(
            (num_trajectories, trajectory_length, *obs_space.shape), dtype=obs_type
        ).to(DEVICE)
        self.a = torch.zeros(
            (num_trajectories, trajectory_length, *act_space.shape), dtype=torch.float
        ).to(DEVICE)
        self.r = torch.zeros((num_trajectories, trajectory_length)).to(DEVICE)

        self.trajectory_length = trajectory_length
        self.sample_length = sample_length
        self.capacity = num_trajectories
        self.fill_counter = 0
        self.full = False

        self.chp_dir = chp_dir

    def insert(self, trajectory, save=True):
        assert not (save and self.chp_dir is None)
        if save:
            with torch.no_grad():
                torch.save(
                    {k: v.detach().cpu().numpy() for k, v in trajectory.items()},
                    os.path.join(
                        self.chp_dir, f"checkpoint_data_{self.fill_counter:09d}"
                    ),
                )

        self.s[self.fill_counter] = trajectory["states"].to(DEVICE).detach()
        self.a[self.fill_counter] = trajectory["actions"].to(DEVICE).detach()
        self.r[self.fill_counter] = trajectory["rewards"].to(DEVICE).detach()

        self.length = (
            self.capacity * (self.trajectory_length - 1)
            if self.full
            else self.fill_counter * (self.trajectory_length - 1)
        )

        self.fill_counter += 1
        if self.fill_counter == self.capacity:
            self.full = True
            self.fill_counter = 0

    def __getitem__(self, k):
        t = k // (self.trajectory_length - self.sample_length)
        i = k % (self.trajectory_length - self.sample_length)
        s = self.s[t, i : i + self.sample_length].float().squeeze(0)
        a = self.a[t, i : i + self.sample_length].float().squeeze(0)
        r = self.r[t, i : i + self.sample_length].float().squeeze(0)
        s_n = self.s[t, i + 1 : i + 1 + self.sample_length].float().squeeze(0)
        return s, a, r, s_n

    def __len__(self):
        return self.length


class BatchTrajectorySampler:
    def __init__(self, buffer, batch_size):
        self.buffer = buffer
        self.batch_size = batch_size
        self._dataloader = torch.utils.data.DataLoader(
            self.buffer, batch_size=self.batch_size, drop_last=True, shuffle=True
        )

    def __iter__(self):
        self.iterator = iter(self._dataloader)
        return self

    def __next__(self):
        return next(self.iterator)

    def __len__(self):
        return int(len(self.buffer) // self.batch_size)
