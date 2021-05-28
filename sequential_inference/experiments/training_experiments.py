from typing import Dict, Tuple
import abc

from tqdm import tqdm
import torch

from sequential_inference.experiments.mixins.data import (
    AbstractDataMixin,
    DynaSamplingMixin,
)
from sequential_inference.abc.experiment import AbstractExperiment, AbstractRLExperiment
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from sequential_inference.abc.rl import AbstractRLAlgorithm
from sequential_inference.rl.sac import AbstractActorCriticAlgorithm, SACAlgorithm
from sequential_inference.rl.agents import InferencePolicyAgent
from sequential_inference.util.rl_util import rollout_with_policy


class TrainingExperiment(AbstractExperiment):

    data: AbstractDataMixin
    is_rl: bool = False
    is_model: bool = False

    def __init__(
        self,
        batch_size: int,
        epoch_steps: int,
        epochs: int,
        log_frequency: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.epoch_steps = epoch_steps
        self.epochs = epochs
        self.log_frequency = log_frequency

    def set_data_sampler(self, data_sampler: AbstractDataMixin):
        self.data = data_sampler
        self.data.set_experiment(self)
        self.buffer = data_sampler.buffer

    def train(self, start_epoch: int = 0):
        total_train_steps = start_epoch * self.epoch_steps
        self.before_experiment()
        for e in range(start_epoch, self.epochs):
            print(f"Training epoch {e + 1}/{self.epochs} for {self.epoch_steps} steps")
            for _ in tqdm(range(self.epoch_steps)):
                stats = self.train_step()
                self.notify_observers("step", stats, total_train_steps)
                if total_train_steps % self.log_frequency == 0:
                    self.notify_observers("log", stats, total_train_steps)
                total_train_steps += 1
            epoch_log = self.after_epoch({})
            print(stats)
            self.notify_observers("epoch", epoch_log, total_train_steps)
        self.close_observers()

    def after_epoch(self, d):
        d = super().after_epoch(d)
        d = self.data.after_epoch(d)
        return d

    def before_experiment(self):
        super().before_experiment()
        self.data.before_experiment()

    def unpack_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        done = batch["done"]
        return obs, act, rew, done

    @abc.abstractmethod
    def train_step(self) -> Dict[str, torch.Tensor]:
        pass


class RLTrainingExperiment(AbstractRLExperiment, TrainingExperiment):

    rl_algorithm: AbstractRLAlgorithm

    is_rl: bool = True

    def set_rl_algorithm(
        self,
        algorithm: AbstractRLAlgorithm,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        grad_clipping: float,
    ):
        self.rl_algorithm = algorithm
        self.register_module("rl_algo", algorithm)
        if isinstance(algorithm, SACAlgorithm):
            self.actor_optimizer = optimizer(
                algorithm.actor.parameters(), lr=learning_rate
            )
            self.register_module("actor_optim", self.actor_optimizer)
            self.value_optimizer = optimizer(
                algorithm.critic.parameters(), lr=learning_rate
            )
            self.register_module("value_optimizer", self.value_optimizer)
            self.alpha_optimizer = optimizer(algorithm.alpha, lr=learning_rate)
            self.register_module("alpha_optimizer", self.alpha_optimizer)

            def step(loss):
                q_loss, actor_loss, alpha_loss = loss
                self.value_optimizer.zero_grad()
                q_loss.backward()
                if self.rl_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.rl_algorithm.get_parameters(), self.model_grad_norm
                    )
                self.value_optimizer.step(retain_graph=True)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.rl_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.rl_algorithm.get_parameters(), self.model_grad_norm
                    )
                self.actor_optimizer.step(retain_graph=True)
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                if self.rl_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.rl_algorithm.get_parameters(), self.model_grad_norm
                    )
                self.alpha_optimizer.step(retain_graph=True)

            self._step_rl = step

        elif isinstance(algorithm, AbstractActorCriticAlgorithm):
            self.actor_optimizer = optimizer(
                algorithm.actor.parameters(), lr=learning_rate
            )
            self.register_module("actor_optim", self.actor_optimizer)
            self.value_optimizer = optimizer(
                algorithm.critic.parameters(), lr=learning_rate
            )
            self.register_module("value_optimizer", self.value_optimizer)

            def step(loss):
                critic_loss, actor_loss = loss
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                if self.rl_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.rl_algorithm.get_parameters(), self.model_grad_norm
                    )
                self.actor_optimizer.step(retain_graph=True)
                self.value_optimizer.zero_grad()
                critic_loss.backward()
                if self.rl_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.rl_algorithm.get_parameters(), self.model_grad_norm
                    )
                self.value_optimizer.step(retain_graph=True)

            self._step_rl = step

        else:
            self.rl_optimizer = optimizer(algorithm.get_parameters(), lr=learning_rate)
            self.register_module("rl", self.rl_optimizer)
            self.rl_grad_norm = grad_clipping

            def step(loss):
                self.rl_optimizer.zero_grad()
                loss.backward()
                if self.rl_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.rl_algorithm.get_parameters(), self.model_grad_norm
                    )
                self.rl_optimizer.step()

            self._step_rl = step

    def train_step(self) -> Dict[str, torch.Tensor]:
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        return self.rl_train_step(*unpacked_batch)

    def rl_train_step(self, obs, act, rew, done) -> Dict[str, torch.Tensor]:
        loss, stats = self.rl_algorithm.compute_loss(obs, act, rew, done)
        self.step_rl_optimizer(loss)
        stats["rl_loss"] = loss.detach().cpu().mean().detach().cpu()
        return stats

    def step_rl_optimizer(self, loss):
        self._step_rl(loss)


class ModelTrainingExperiment(TrainingExperiment):

    model_algorithm: AbstractSequenceAlgorithm

    is_model = True

    def set_model_algorithm(
        self,
        algorithm: AbstractSequenceAlgorithm,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        grad_clipping: float = -1.0,
    ):
        self.model_algorithm = algorithm
        self.register_module("model_algo", self.model_algorithm)
        self.model_optimizer = optimizer(algorithm.get_parameters(), lr=learning_rate)
        self.register_module("model_optim", self.model_optimizer)
        self.model_grad_norm = grad_clipping

    def train_step(self) -> Dict[str, torch.Tensor]:
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        return self.model_train_step(*unpacked_batch)

    def model_train_step(self, obs, act, rew, done) -> Dict[str, torch.Tensor]:
        loss, stats = self.model_algorithm.compute_loss(obs, act, rew, done)
        self.step_model_optimizer(loss)
        return stats

    def step_model_optimizer(self, loss):
        self.model_optimizer.zero_grad()
        loss.backward()
        if self.model_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.model_algorithm.get_parameters(), self.model_grad_norm
            )
        self.model_optimizer.step()


class DynaTrainingExperiment(RLTrainingExperiment, ModelTrainingExperiment):

    data: DynaSamplingMixin

    is_rl = True
    is_model = True

    def train_step(self) -> Dict[str, torch.Tensor]:
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        stats = self.model_train_step(*unpacked_batch)

        rl_batch = self.data.get_model_batch(self.model_batch_size)
        unpacked_rl_batch = self.unpack_batch(rl_batch)
        rl_stats = self.rl_train_step(*unpacked_rl_batch)

        return {**stats, **rl_stats}


class LatentTrainingExperiment(RLTrainingExperiment, ModelTrainingExperiment):

    is_rl = True
    is_model = True

    def __init__(
        self,
        pass_rl_gradients_to_model: bool,
        batch_size: int,
        epoch_steps: int,
        epochs: int,
        log_frequency: int,
    ):
        super().__init__(batch_size, epoch_steps, epochs, log_frequency)

        self.pass_rl_gradients_to_model = pass_rl_gradients_to_model

    def train_step(self) -> Dict[str, torch.Tensor]:
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        stats = self.model_train_step(*unpacked_batch)

        rl_batch = self.data.get_batch(self.rl_batch_size)
        rl_batch = self.unpack_batch(rl_batch)
        obs, act, rew, done = self.unpack_batch(rl_batch)
        if self.pass_rl_gradients_to_model:
            _, latents = self.model_algorithm.infer_sequence(obs, act, rew)
            loss, rl_stats = self.rl_algorithm.compute_loss(latents, act, rew, done)
            self.step_rl_optimizer(loss)
            self.step_model_optimizer(loss)
        else:
            with torch.no_grad():
                _, latents = self.model_algorithm.infer_sequence(obs, act, rew)
            loss, rl_stats = self.rl_algorithm.compute_loss(latents, act, rew, done)
            self.step_rl_optimizer(loss)

        return {**stats, **rl_stats}

    def get_agent(self) -> InferencePolicyAgent:
        agent: InferencePolicyAgent = self.rl_algorithm.get_agent()
        agent.latent = True
        agent.observation = False
        return InferencePolicyAgent(agent, self.model_algorithm)


class LatentImaginationExperiment(RLTrainingExperiment, ModelTrainingExperiment):

    is_rl = True
    is_model = True

    def __init__(
        self,
        horizon: bool,
        batch_size: int,
        epoch_steps: int,
        epochs: int,
        log_frequency: int,
    ):
        super().__init__(batch_size, epoch_steps, epochs, log_frequency)

        self.horizon = horizon

    def train_step(self) -> Dict[str, torch.Tensor]:
        batch = self.data.get_batch(self.batch_size)
        unpacked_batch = self.unpack_batch(batch)
        stats = self.model_train_step(*unpacked_batch)

        rl_batch = self.data.get_batch(self.rl_batch_size)
        obs, act, rew, done = self.unpack_batch(rl_batch)
        with torch.no_grad():
            _, latents = self.model_algorithm.infer_sequence(obs, act, rew, done)

        predicted_latents, predicted_actions, rewards = rollout_with_policy(
            latents[:, -1],
            self.model_algorithm,
            self.rl_algorithm.get_agent(),
            self.horizon,
            reconstruct=False,
            explore=True,
        )

        rl_stats = self.rl_train_step(
            predicted_latents, predicted_actions, rewards, None
        )

        return {**stats, **rl_stats}
