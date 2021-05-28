import hydra
import torch

from sequential_inference.algorithms.factory import setup_model_algorithm
from sequential_inference.experiments.mixins.data import FixedDataSamplingMixin
from sequential_inference.util.data import gather_trajectory_data
from sequential_inference.rl.agents import RandomAgent
from sequential_inference.data.data import TrajectoryReplayBuffer
from sequential_inference.experiments.training_experiments import (
    ModelTrainingExperiment,
)
from sequential_inference.envs.factory import make_vec_env


@hydra.main(config_path="../config", config_name="tests/slac_test")
def main(cfg):
    print(cfg)
    num_envs = 2
    env = make_vec_env(
        num_envs, "HalfCheetahDir-v0", time_limit=20, is_multi_env=True, suite="mujoco"
    )
    buffer = TrajectoryReplayBuffer(200, 200, env, sample_length=10, chp_dir="/tmp")

    data_sampler = FixedDataSamplingMixin(env, buffer, 16)
    data_sampler.set_num_sampling_steps(2 * num_envs * 20)

    algorithm = setup_model_algorithm(env, cfg)
    algorithm.to("cpu")
    state_dict = algorithm.state_dict()
    algorithm.load_state_dict(state_dict)

    experiment = ModelTrainingExperiment(64, 100, 100, 1)
    experiment.set_model_algorithm(algorithm, torch.optim.Adam, 0.0001)
    experiment.set_data_sampler(data_sampler)

    experiment.train()


if __name__ == "__main__":
    main()
