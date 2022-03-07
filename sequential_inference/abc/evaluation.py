import abc
from typing import Dict
from enum import Enum

import torch
from sequential_inference.abc.common import Env

from sequential_inference.abc.data import AbstractDataBuffer
from sequential_inference.abc.experiment import AbstractExperiment
from sequential_inference.abc.rl import AbstractAgent
from sequential_inference.abc.sequence_model import AbstractLatentSequenceAlgorithm
from sequential_inference.abc.util import AbstractLogger


class AbstractEvaluator(abc.ABC):
    @abc.abstractmethod
    def evaluate(
        self,
        experiment: AbstractExperiment,
        epoch: int,
    ) -> None:
        pass


class AbstractModelVisualizer(abc.ABC):
    @abc.abstractmethod
    def visualize_model_prediction(
        self,
        model: AbstractLatentSequenceAlgorithm,
        data: AbstractDataBuffer,
        batch_size: int,
        inference_steps: int,
        prediction_steps: int,
        return_losses: bool = False,
    ) -> Dict[str, torch.Tensor]:
        pass

    @abc.abstractmethod
    def visualize_with_agent(
        self,
        model: AbstractLatentSequenceAlgorithm,
        agent: AbstractAgent,
        data_buffer: AbstractDataBuffer,
        batch_size: int,
        inference_steps: int,
        prediction_steps: int,
    ) -> Dict[str, torch.Tensor]:
        pass


class AbstractRLVisualizer(abc.ABC):
    @abc.abstractmethod
    def visualize_rl_agent(
        self,
        environment: Env,
        agent: AbstractAgent,
        evaluation_steps: int,
    ) -> torch.Tensor:
        pass
