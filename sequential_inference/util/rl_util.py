from re import A
from typing import Dict, Iterator, List, Tuple

from tqdm import tqdm
import torch
import numpy as np

from sequential_inference.envs.vec_env.vec_env import VecEnv
from sequential_inference.abc.rl import AbstractAgent
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm


def rollout_with_policy(
    initial_context: Dict[str, torch.Tensor],
    model: AbstractSequenceAlgorithm,
    policy: AbstractAgent,
    horizon: int,
    reconstruct: bool = False,
    explore: bool = False,
) -> Tuple[torch.Tensor, ...]:
    _, posteriors = model.infer_sequence(
        initial_context["obs"], initial_context["act"], initial_context["rew"]
    )

    latent = posteriors[:, -1]

    predicted_latents = []
    predicted_actions = []
    reconstructions = []

    # save first to conform to regular RL convention
    predicted_latents.append(latent)

    # iterate over horizon
    for i in range(horizon):
        # decide whether to obtain reconstructions (needed for policies which directly
        # predict in observation space)
        obs = model.reconstruct(latent)

        act = policy.act(obs, context=latent, explore=explore)
        latent = model.predict_sequence(latent, act)

        predicted_actions.append(act)
        predicted_latents.append(latent)
        if reconstruct:
            reconstructions.append(obs)

    # put together results
    predicted_latents = torch.stack(predicted_latents, 1)
    predicted_actions = torch.stack(predicted_actions, 1)
    rewards = model.reconstruct_reward(predicted_latents)
    if reconstruct:
        reconstructions = torch.stack(reconstructions, 1)
        return predicted_latents, predicted_actions, rewards, reconstructions
    else:
        return predicted_latents, predicted_actions, rewards


class Buffer:
    def __init__(self, num_envs: int, return_latent: bool = False):
        self.num_envs = num_envs
        self.return_latent = return_latent
        self.obs: List[torch.Tensor] = []
        self.act: List[torch.Tensor] = []
        self.rew: List[torch.Tensor] = []
        self.done: List[torch.Tensor] = []
        self.task: List[torch.Tensor] = []
        self.ltn: List[torch.Tensor] = []

    def add(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        done: torch.Tensor,
        task: torch.Tensor,
        ltn: torch.Tensor = None,
    ):
        self.obs.append(obs)
        self.act.append(act)
        self.rew.append(rew)
        self.done.append(done)
        self.task.append(task)
        self.ltn.append(ltn)

    def empty(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        o = torch.stack(self.obs, 1)
        a = torch.stack(self.act, 1)
        r = torch.stack(self.rew, 1)
        d = torch.stack(self.done, 1)
        t = torch.stack(self.task, 1)
        self.obs = []
        self.act = []
        self.rew = []
        self.done = []
        self.task = []
        self.ltn = []
        if self.return_latent:
            l = torch.Tensor(self.ltn)
            for i in range(self.num_envs):
                yield (o[i], a[i], r[i], d[i], t[i], l[i])
        for i in range(self.num_envs):
            yield (o[i], a[i], r[i], d[i], t[i])


def run_agent_in_environment(
    environment: VecEnv,
    policy: AbstractAgent,
    steps: int,
    explore: bool = False,
    randomize_tasks: bool = True,
    return_contexts: bool = False,
) -> Tuple[List[torch.Tensor], ...]:
    actions = []
    observations = []
    rewards = []
    tasks = []
    dones = []
    latents = []

    steps = steps // environment.num_envs

    buffer = Buffer(environment.num_envs)

    if randomize_tasks:
        last_obs = environment.reset()
    else:
        last_obs = environment.reset_mdp()
    reward = None

    policy.reset()
    for _ in tqdm(range(steps)):
        action = policy.act(last_obs, reward, explore=explore)
        next_obs, reward, done, info = environment.step(action)
        if return_contexts:
            latent = policy.state
        else:
            latent = None
        buffer.add(last_obs, action, reward, done, info["task"], latent)

        last_obs = next_obs

        if torch.all(done):
            # move buffer entries to return lists
            policy.reset()
            if return_contexts:
                for obs, act, rew, done, task, ltn in buffer.empty():
                    observations.append(obs)
                    actions.append(act)
                    rewards.append(rew)
                    dones.append(done)
                    tasks.append(task)
                    latents.append(ltn)
            else:
                for obs, act, rew, done, task in buffer.empty():
                    observations.append(obs)
                    actions.append(act)
                    rewards.append(rew)
                    dones.append(done)
                    tasks.append(task)

            if randomize_tasks:
                last_obs = environment.reset()
            else:
                last_obs = environment.reset_mdp()

    if return_contexts:
        return observations, actions, rewards, tasks, dones, latents
    else:
        return observations, actions, rewards, tasks, dones


def join_state_with_array(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    if len(action.shape) == 2:
        action = action.unsqueeze(-1)
    shape = state.shape
    action_shape = action.shape[-1]
    if len(state.shape) > 3:
        action = action.reshape(shape[0], shape[1], action_shape, 1, 1)
        action = action.expand(shape[0], shape[1], action_shape, shape[-2], shape[-1])
        inp = torch.cat((state, action), -3)
        return inp
    else:
        return torch.cat((state, action), -1)


def load_agent():
    raise NotImplementedError("Agent loading not yet available")
