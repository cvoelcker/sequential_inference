from typing import Dict, Tuple

import torch

from sequential_inference.abc.rl import AbstractPolicy
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm


def rollout_with_policy(
    initial_context: Dict[str, torch.Tensor],
    model: AbstractSequenceAlgorithm,
    policy: AbstractPolicy,
    horizon: int,
    reconstruct: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, posteriors = model.infer_sequence(
        initial_context["obs"], initial_context["act"], initial_context["rew"]
    )

    latent = posteriors[:, -1]

    predicted_latents = []
    predicted_actions = []

    # save first to conform to regular RL convention
    predicted_latents.append(latent)

    # iterate over horizon
    for i in range(horizon):
        # decide whether to obtain reconstructions (needed for policies which directly
        # predict in observation space)
        if reconstruct:
            obs = model.reconstruct(latent)
        else:
            obs = latent
        act = policy.act(obs)
        latent = model.predict_sequence(latent, act)

        predicted_actions.append(act)
        predicted_latents.append(latent)

    # pu together results
    predicted_latents = torch.stack(predicted_latents, 1)
    predicted_actions = torch.stack(predicted_actions, 1)
    rewards = model.reconstruct_rewards(predicted_latents)

    return predicted_latents, predicted_actions, rewards
