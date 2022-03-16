import random
import torch
from tqdm import tqdm
from sequential_inference.abc.data import AbstractDataBuffer


class AuxilliaryTaskReplayBuffer(AbstractDataBuffer):
    def __init__(
        self,
        num_trajectories,
        trajectory_length,
        env,
        num_tasks,
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
            (num_trajectories, trajectory_length, num_tasks)
        )  # .to(self.device)
        self.d = torch.ones(
            (num_trajectories, trajectory_length, 1)
        )  # .to(self.device)
        self.t = torch.zeros(
            (num_trajectories, trajectory_length, 1)
        )  # .to(self.device)

        self.targets = []
        self.num_tasks = num_tasks

        self.trajectory_length = trajectory_length
        self.sample_length = sample_length
        self.capacity = num_trajectories
        self.fill_counter = torch.Tensor([0]).long()
        self.full = torch.Tensor([False]).bool()

    def insert(self, trajectory):
        self.s[self.fill_counter] = (
            trajectory["obs"].to("cpu").detach().to(self.s.dtype)
        )
        self.a[self.fill_counter] = (
            trajectory["act"].to(self.a.dtype).to("cpu").detach()
        )
        self.r[self.fill_counter, :, :1] = (
            trajectory["rew"].to(self.r.dtype).to("cpu").detach()
        )
        self.d[self.fill_counter] = (
            trajectory["done"].to(self.d.dtype).to("cpu").detach()
        )
        self.t[self.fill_counter] = (
            trajectory["task"].to(self.t.dtype).to("cpu").detach()
        )

        self.fill_counter += 1
        if self.fill_counter == self.capacity:
            self.full = torch.Tensor([True]).bool()
            self.fill_counter = torch.Tensor([0]).long()

    def generate_goal_similarity_tasks(self):
        for i in tqdm(range(1, self.num_tasks - 1)):
            item = random.randint(0, len(self) - 1)
            item = self[item]["obs"][0]
            self.targets.append(item)
            for k, sample in enumerate(self):
                m = k // (self.trajectory_length - self.sample_length - 1)
                n = k % (self.trajectory_length - self.sample_length - 1)
                for j, obs in enumerate(sample["obs"]):
                    self.r[m, n + j, i] = torch.exp(-torch.sum((obs - item) ** 2))

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

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
        return int(
            self.capacity * (self.trajectory_length - self.sample_length - 1)
            if self.full
            else self.fill_counter * (self.trajectory_length - self.sample_length - 1)
        )
