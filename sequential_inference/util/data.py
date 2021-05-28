import torch

from sequential_inference.abc.rl import AbstractAgent
from sequential_inference.util.rl_util import run_agent_in_environment
from sequential_inference.data.data import TrajectoryReplayBuffer
from sequential_inference.envs.vec_env.vec_env import VecEnv


def gather_trajectory_data(
    env: VecEnv,
    agent: AbstractAgent,
    buffer: TrajectoryReplayBuffer,
    steps: int,
    explore: bool = True,
    randomize_tasks: bool = True,
) -> None:

    trajectory_length = buffer.trajectory_length

    # TODO think about saving task in replay buffer
    observations, actions, rewards, _, dones = run_agent_in_environment(
        env, agent, steps, explore, randomize_tasks
    )
    for obs, act, rew, done in zip(observations, actions, rewards, dones):
        obs_len = obs.shape[0]
        act_len = act.shape[0]
        rew_len = rew.shape[0]
        done_len = done.shape[0]
        if obs_len > trajectory_length:
            raise ValueError("Data sampling: sampled trajectory longer then max buffer")
        obs = torch.cat(
            [obs, torch.zeros(trajectory_length - obs_len, *obs.shape[1:])], 0
        )
        act = torch.cat(
            [act, torch.zeros(trajectory_length - act_len, *act.shape[1:])], 0
        )
        rew = torch.cat(
            [rew, torch.zeros(trajectory_length - rew_len, *rew.shape[1:])], 0
        )
        done = torch.cat(
            [done, torch.ones(trajectory_length - done_len, *done.shape[1:])], 0
        )
        buffer.insert({"obs": obs, "act": act, "rew": rew, "done": done})
