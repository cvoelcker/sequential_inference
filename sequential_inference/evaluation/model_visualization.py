import os
from typing import Dict, Union

import torch
from torchvision.utils import save_image

from sequential_inference.abc.data import AbstractDataBuffer
from sequential_inference.abc.rl import AbstractAgent
from sequential_inference.abc.sequence_model import AbstractLatentSequenceAlgorithm
from sequential_inference.abc.evaluation import (
    AbstractEvaluator,
    AbstractModelVisualizer,
)
from sequential_inference.data.storage import BatchDataSampler, TrajectoryReplayBuffer
from sequential_inference.experiments.base import (
    ModelBasedRLTrainingExperiment,
    ModelTrainingExperiment,
)


class LatentModelReconstructionEvaluator(AbstractEvaluator):
    def __init__(
        self,
        save_path: str = ".",
        save_name: str = "model_rollouts",
        batch_size: int = 16,
        inference_steps: int = 20,
        prediction_steps: int = 80,
    ):
        self.save_path = os.path.join(save_path, save_name)
        self.visualizer = LatentSequenceModelRollout(
            batch_size, inference_steps, prediction_steps
        )
        os.makedirs(self.save_path)

    def evaluate(
        self,
        experiment: Union[ModelTrainingExperiment, ModelBasedRLTrainingExperiment],
        epoch: int,
    ):
        model = experiment.model_algorithm
        data = experiment.data
        assert data is not None, "Data not initialized"
        image_dict = self.visualizer.visualize_model_prediction(model, data.buffer)
        predictions = image_dict["predicted_obs"]
        inferred = image_dict["inferred_obs"]
        ground_truth_inferred = image_dict["ground_inferred_obs"]
        ground_truth_predicted = image_dict["ground_predicted_obs"]

        for i, (inf, pred, inf_truth, pred_truth) in enumerate(
            zip(inferred, predictions, ground_truth_inferred, ground_truth_predicted)
        ):
            save_image(
                torch.cat([inf, pred], dim=0),
                os.path.join(self.save_path, f"{epoch}_{i}_model.png"),
            )
            save_image(
                torch.cat([inf_truth, pred_truth], dim=0),
                os.path.join(self.save_path, f"{epoch}_{i}_env.png"),
            )


class LatentSequenceModelRollout(AbstractModelVisualizer):
    def __init__(
        self,
        batch_size: int = 16,
        inference_steps: int = 20,
        prediction_steps: int = 80,
    ):
        self.batch_size = batch_size
        self.inference_steps = inference_steps
        self.prediction_steps = prediction_steps

    def visualize_model_prediction(
        self,
        model: AbstractLatentSequenceAlgorithm,
        data_buffer: AbstractDataBuffer,
        return_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        with DataBufferSample(
            data_buffer, self.batch_size, self.inference_steps, self.prediction_steps
        ) as batch:

            obs = batch["obs"]
            act = batch["act"]
            rew = batch["rew"]

            _, posteriors = model.infer_sequence(
                obs[:, : self.inference_steps],
                act[:, : self.inference_steps],
                rew[:, : self.inference_steps],
                full=True,
            )
            inferred_observations = model.reconstruct(posteriors)
            prediction_latents = model.predict_latent_sequence(
                posteriors[:, -1], act[:, self.inference_steps - 1 :], full=False
            )
            predicted_observations = model.reconstruct(prediction_latents)
            predicted_rewards = model.reconstruct_reward(
                torch.cat([prediction_latents, act[:, self.inference_steps :]], dim=-1)
            )

            if return_losses:
                image_reconstruction_loss = torch.mean(
                    (obs[:, self.inference_steps :] - predicted_observations) ** 2,
                    [0, 1],
                ).sum()
                reward_reconstruction_loss = torch.mean(
                    (rew[:, self.inference_steps :] - predicted_rewards) ** 2, [0, 1]
                ).sum()
                return {
                    "ground_obs": obs,
                    "infered_obs": inferred_observations,
                    "predicted_obs": predicted_observations,
                    "predicted_rew": predicted_rewards,
                    "mse_obs": image_reconstruction_loss,
                    "mse_rew": reward_reconstruction_loss,
                }
            else:
                return {
                    "ground_inferred_obs": obs[:, : self.inference_steps],
                    "ground_predicted_obs": obs[:, self.inference_steps :],
                    "inferred_obs": inferred_observations,
                    "predicted_obs": predicted_observations,
                    "predicted_rew": predicted_rewards,
                }

    def visualize_with_agent(
        self,
        model: AbstractLatentSequenceAlgorithm,
        agent: AbstractAgent,
        data_buffer: AbstractDataBuffer,
    ):
        with DataBufferSample(
            data_buffer, self.batch_size, self.inference_steps, 0
        ) as batch:

            obs = batch["obs"]
            act = batch["act"]
            rew = batch["rew"]

            _, posteriors = model.infer_sequence(
                obs[:, : self.inference_steps],
                act[:, : self.inference_steps],
                rew[:, : self.inference_steps],
                full=True,
            )
            (
                predicted_latents,
                predicted_actions,
                rewards,
                reconstructions,
            ) = model.rollout_with_policy(
                posteriors[-1], agent, self.prediction_steps, reconstruct=True
            )

            return {
                "pred_obs": reconstructions,
                "pred_act": predicted_actions,
                "pred_rew": rewards,
                "pred_latent": predicted_latents,
            }


class DataBufferSample:
    def __init__(
        self,
        data_buffer: AbstractDataBuffer,
        batch_size: int,
        inference_steps: int,
        prediction_steps: int,
    ):
        self.data_buffer = data_buffer
        self.batch_size = batch_size
        self.original_sample_length = data_buffer.sample_length

        # set sample length to include inference and prediction path
        if isinstance(data_buffer, TrajectoryReplayBuffer):
            assert (
                inference_steps + prediction_steps < data_buffer.trajectory_length
            ), "Cannot sample longer trajectories than in buffer"
        else:
            assert inference_steps + prediction_steps < len(
                data_buffer
            ), "Cannot sample more steps than in buffer"

        self.data_buffer.sample_length = inference_steps + prediction_steps
        self.sampler = BatchDataSampler(self.data_buffer)

    def __enter__(self) -> Dict[str, torch.Tensor]:
        batch = self.sampler.get_next(self.batch_size)
        return batch

    def __exit__(self, exc_type, exc_value, traceback):
        self.data_buffer.sample_length = self.original_sample_length
