from typing import Dict, Tuple, Union

import gym
import torch

from sequential_inference.envs.meta_wrappers.common import MetaTask
from sequential_inference.abc.rl import AbstractAgent, InferencePolicyAgent
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm


def rollout_with_policy(
    initial_context: Dict[str, torch.Tensor],
    model: AbstractSequenceAlgorithm,
    policy: AbstractAgent,
    horizon: int,
    reconstruct: bool = False,
    explore: bool = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
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
    rewards = model.reconstruct_rewards(predicted_latents)
    if reconstruct:
        reconstructions = torch.stack(reconstructions, 1)
        return predicted_latents, predicted_actions, rewards, reconstructions
    else:
        return predicted_latents, predicted_actions, rewards


def run_agent_in_environment(
    environment: MetaTask,
    policy: AbstractAgent,
    steps: int,
    explore: bool = False,
    randomize_tasks: bool = True,
    return_contexts: bool = False,
) -> Union[
    Tuple(
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ),
    Tuple(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor),
]:
    assert (
        not return_contexts or type(policy) is InferencePolicyAgent
    ), "Cannot return contexts if the inference agent does not hold any"
    actions = []
    observations = []
    rewards = []
    contexts = []
    dones = []
    tasks = []
    if randomize_tasks:
        obs = environment.reset()
    else:
        obs = environment.reset_mdp()
    reward = None
    observations.append(obs)
    if type(policy) is InferencePolicyAgent:
        policy.reset()
    for _ in steps:
        action = policy.act(obs, reward, explore=explore)
        actions.append(action)
        obs, reward, done, info = environment.step(action)
        observations.append(obs)
        rewards.append(reward)
        dones.append(done)
        tasks.append(info["task"])
        if return_contexts:
            contexts.append(policy.state)
        if done:
            if randomize_tasks:
                obs = environment.reset()
            else:
                obs = environment.reset_mdp()
            reward = None
            observations.append(obs)
            if type(policy) is InferencePolicyAgent:
                policy.reset()

    observations = torch.stack(observations, 1)
    actions = torch.stack(actions, 1)
    rewards = torch.stack(rewards, 1)
    dones = torch.stack(dones, 1, dtype=int)
    tasks = torch.stack(tasks, 1)
    if return_contexts:
        observations = torch.stack(observations, 1)
        return observations, actions, rewards, dones, tasks, contexts
    else:
        return observations, actions, rewards, dones, tasks


def join_state_action(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    if len(action.shape) == 1:
        action = action.unsqueeze(-1)
    shape = state.shape
    action_shape = action.shape[-1]
    if len(state.shape) > 3:
        action = action.reshape(shape[0], action_shape, 1, 1)
        action = action.expand(shape[0], action_shape, shape[-2], shape[-1])
        inp = torch.cat((state, action), -3)
        return inp
    else:
        return torch.cat((state, action), -1)
