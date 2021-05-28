# from sequential_inference.abc.trainer import AbstractRLTrainer
# import torch
# import numpy as np
#
# from sequential_inference.data.data import (
#     BatchTrajectorySampler,
#     TrajectoryReplayBuffer,
# )
# from sequential_inference.envs.meta_wrappers.common import MetaTask
# from sequential_inference.trainers.trainer import ModelTrainer
# from sequential_inference.abc.experiment import AbstractRLExperiment
# from sequential_inference.util.rl_util import rollout_with_policy
#
#
# class DynaTrainingExperiment(AbstractRLExperiment):
#     def __init__(
#         self,
#         model_trainer: ModelTrainer,
#         rl_trainer: AbstractRLTrainer,
#         env: MetaTask,
#         batch_size: int,
#         epochs: int,
#         num_model_trajectories: int,
#         model_trajectory_length: int,
#         device: str,
#         model_data_ratio: float,
#         model_batch_size: int,
#         policy_update_steps: int,
#         update_model: int,
#     ):
#         self.trainer = model_trainer
#         self.batch_size = batch_size
#         self.epochs = epochs
#         self.rl_trainer = rl_trainer
#         self.model_trainer = model_trainer
#         self.batch_size = self.model_trainer.batch_size
#
#         self.model_dataset = TrajectoryReplayBuffer(
#             num_model_trajectories,
#             model_trajectory_length,
#             env,
#             device=device,
#             chp_dir="",  # model dataset is not backed up, since it is resampled every epoch
#         )
#         self.model_data = None
#         self.real_sampler = None
#         self.env = env
#         self.num_model_trajectories = num_model_trajectories
#         self.model_trajectory_length = model_trajectory_length
#         self.model_data_ratio = model_data_ratio
#         self.model_batch_size = model_batch_size
#         self.policy_update_steps = policy_update_steps
#         self.update_model = update_model
#
#     def gather_model_data(self):
#         assert self.data is not None
#
#         sample_idx = np.random.randint(
#             len(self.dataset), size=(self.num_model_trajectories,)
#         )
#         start_states = []
#         for idx in sample_idx:
#             start_states.append(self.dataset[idx][0])
#         start_states = torch.stack(start_states, 0)
#
#         data = rollout_with_policy(
#             start_states,
#             self.trainer.algorithm,
#             self.rl_trainer.get_agent(),
#             self.model_trajectory_length,
#             reconstruct=True,
#             explore=True,
#         )
#         for d in data:
#             self.model_dataset.insert(d, save=False)
#         self._model_data_loader = BatchTrajectorySampler(
#             self.model_dataset, self.model_batch_size
#         )
#         self.model_data = iter(self._model_data_loader)
#
#     def set_data(self, data):
#         super().set_data(data)
#         self.real_sampler = BatchTrajectorySampler(
#             self.dataset, int(self.model_batch_size * (1 - self.model_data_ratio))
#         )
#         self.real_sampler = iter(self.real_sampler)
#         self.gather_model_data()
#
#     def concatenate_samples(self, sample1, sample2):
#         s1, a1, r1, s_n1 = sample1
#         s2, a2, r2, s_n2 = sample2
#         return (
#             torch.cat((s1, s2), 0),
#             torch.cat((a1, a2), 0),
#             torch.cat((r1, r2), 0),
#             torch.cat((s_n1, s_n2), 0),
#         )
#
#     def get_next_model_batch(self):
#         try:
#             batch = next(self.model_data)
#         except StopIteration as e:
#             self.model_data = iter(self._model_data_loader)
#             batch = next(self.model_data)
#         return batch
#
#     def pretrain(self, iters):
#         for e in range(0, iters):
#             print(f"Pretraining epoch {e + 1}/{iters} for {len(self.data)} steps")
#             for sample in self.data:
#                 self.model_trainer.train_step(sample)
#
#     def train_step(self, real_sample):
#         if self.update_model:
#             model_train_stats = self.model_trainer.train_step(real_sample)
#
#         for _ in range(self.policy_update_steps):
#             model_sample = self.get_next_model_batch()
#             real_sample = next(self.real_sampler)
#             sample = self.concatenate_samples(real_sample, model_sample)
#             policy_train_stats = self.rl_trainer.train_step(sample)
#         return {**model_train_stats, **policy_train_stats}
#
#     def train(self, start_epoch: int = 0):
#         total_train_steps = start_epoch * len(self.dataset) // self.batch_size
#         for e in range(start_epoch, self.epochs):
#             print(f"Training epoch {e + 1}/{self.epochs} for {len(self.data)} steps")
#             for sample in self.data:
#                 stats = self.trainer.train_step(sample)
#                 self.notify_observers("step", stats, total_train_steps)
#                 if total_train_steps % self.log_frequency == 0:
#                     self.notify_observers("log", stats, total_train_steps)
#                 total_train_steps += 1
#             epoch_log = self.after_epoch()
#             self.notify_observers("epoch", epoch_log, total_train_steps)
#         self.close_observers()
