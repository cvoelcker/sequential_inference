from typing import Dict, Tuple
import abc
from sequential_inference.abc.env import Env
from sequential_inference.setup.factory import setup_data, setup_rl_algorithm, setup_model_algorithm, setup_optimizer
from sequential_inference.setup.factory import setup_optimizer

from tqdm import tqdm
import torch

from sequential_inference.abc.experiment import AbstractExperiment, AbstractRLExperiment
from sequential_inference.abc.sequence_model import AbstractSequenceAlgorithm
from sequential_inference.abc.rl import AbstractRLAlgorithm
from sequential_inference.rl.sac import AbstractActorCriticAlgorithm, SACAlgorithm


class TrainingExperiment(AbstractExperiment):

    is_rl: bool = False
    is_model: bool = False

    def __init__(
        self,
        env: Env,
        batch_size: int,
        epoch_steps: int,
        epochs: int,
    ):
        super().__init__()
        self.env = env
        self.batch_size = batch_size
        self.epoch_steps = epoch_steps
        self.epochs = epochs

    def build(self, cfg, preempted: bool, run_dir: str):
        #TODO: figure out the hydra instantiation toolkit, this should really help here
        self.data = setup_data(self.env, cfg)
        self.data.initialize(cfg, preempted, run_dir)

        if preempted:
            # TODO: make sure latest checkpoint is loaded
            self.load(run_dir)

    def train(self, start_epoch: int = 0):
        total_train_steps = start_epoch * self.epoch_steps
        self.before_experiment()
        for _ in range(start_epoch, self.epochs):
            for _ in tqdm(range(self.epoch_steps)):
                stats = self.train_step()
                self.notify_observers("step", stats, total_train_steps)
                total_train_steps += 1
            epoch_log = self.after_epoch({})
            self.notify_observers("epoch", epoch_log, total_train_steps)
            self.checkpoint(self)
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

    def build(self, cfg, run_dir: str, preempted: bool):
        rl_algorithm = setup_rl_algorithm(self.env, cfg)
        optimizer = setup_optimizer(cfg.rl.optimizer)
        self.set_rl_algorithm(rl_algorithm, optimizer, cfg.rl.learning_rate, cfg.rl.grad_clipping)
        super().build(cfg, preempted, run_dir)

    def set_rl_algorithm(
        self,
        algorithm: AbstractRLAlgorithm,
        optimizer: torch.optim.Optimizer,
        learning_rate: float,
        grad_clipping: float,
    ):
        #TODO: probably separate this out into different classes to reduce depth and compleity here
        self.rl_algorithm = algorithm
        self.rl_grad_norm = grad_clipping
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
            self.alpha_optimizer = optimizer(
                algorithm.alpha.parameters(), lr=learning_rate)
            self.register_module("alpha_optimizer", self.alpha_optimizer)

            def step(loss):
                q_loss, actor_loss, alpha_loss = loss
                self.value_optimizer.zero_grad()
                q_loss.backward(retain_graph=True)
                if self.rl_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.rl_algorithm.get_parameters(), self.model_grad_norm
                    )
                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                if self.rl_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.rl_algorithm.get_parameters(), self.model_grad_norm
                    )
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                if self.rl_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.rl_algorithm.get_parameters(), self.model_grad_norm
                    )
                self.actor_optimizer.step()
                self.value_optimizer.step()
                self.alpha_optimizer.step()

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

    def build(self, cfg, run_dir: str, preempted: bool):
        model_algorithm = setup_model_algorithm(self.env, cfg)
        optimizer = setup_optimizer(cfg.model.optimizer)
        self.set_model_algorithm(model_algorithm, optimizer, cfg.model.learning_rate, cfg.model.grad_clipping)
        super().build(cfg, preempted, run_dir)

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
