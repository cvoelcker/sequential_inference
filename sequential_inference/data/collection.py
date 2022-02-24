from typing import Dict, Optional
import torch

from sequential_inference.abc.common import Env
from sequential_inference.abc.data import AbstractDataBuffer

from sequential_inference.abc.rl import AbstractAgent
from sequential_inference.util.rl_util import run_agent_in_vec_environment
from sequential_inference.data.storage import TrajectoryReplayBuffer


def gather_data(
    env: Env,
    agent: AbstractAgent,
    buffer: AbstractDataBuffer,
    steps: int,
    explore: bool = True,
) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(buffer, TrajectoryReplayBuffer):
        log = gather_trajectory_data(env, agent, buffer, steps, explore)
    elif isinstance(buffer, AbstractDataBuffer):
        log = gather_sequential_data(env, agent, buffer, steps, explore)
    else:
        raise NotImplementedError(
            f"Data gathering not implemented for this buffer {type(buffer)}"
        )
    return log


def gather_sequential_data(
    env: Env,
    agent: AbstractAgent,
    buffer: AbstractDataBuffer,
    steps: int,
    explore: bool = True,
) -> None:
    pass


def gather_trajectory_data(
    env: Env,
    agent: AbstractAgent,
    buffer: TrajectoryReplayBuffer,
    steps: int,
    explore: bool = True,
) -> Dict[str, torch.Tensor]:

    trajectory_length = buffer.trajectory_length

    observations, actions, rewards, tasks, dones = run_agent_in_vec_environment(
        env, agent, steps, explore
    )
    for obs, act, rew, task, done in zip(observations, actions, rewards, tasks, dones):
        obs_len = obs.shape[0]
        act_len = act.shape[0]
        rew_len = rew.shape[0]
        done_len = done.shape[0]
        task_len = task.shape[0]
        if obs_len > trajectory_length:
            raise ValueError(
                f"Data sampling: sampled trajectory longer then max buffer: {obs_len} in {trajectory_length}"
            )
        obs = torch.cat(
            [
                obs,
                torch.zeros(trajectory_length - obs_len, *obs.shape[1:]).to(obs.device),
            ],
            0,
        )
        act = torch.cat(
            [
                act,
                torch.zeros(trajectory_length - act_len, *act.shape[1:]).to(obs.device),
            ],
            0,
        )
        rew = torch.cat(
            [
                rew,
                torch.zeros(trajectory_length - rew_len, *rew.shape[1:]).to(obs.device),
            ],
            0,
        )
        done = torch.cat(
            [
                done,
                torch.ones(trajectory_length - done_len, *done.shape[1:]).to(
                    obs.device
                ),
            ],
            0,
        )
        task = torch.cat(
            [
                task,
                torch.zeros(trajectory_length - task_len, *task.shape[1:]).to(
                    obs.device
                ),
            ],
            0,
        )
        buffer.insert({"obs": obs, "act": act, "rew": rew, "task": task, "done": done})
    return {"average_reward": torch.stack(rewards).sum(1).mean().cpu().detach()}


def load_offline_data():
    pass
