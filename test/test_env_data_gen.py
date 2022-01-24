import hydra

from sequential_inference.envs.factory import make_vec_env
from sequential_inference.data.data import TrajectoryReplayBuffer
from sequential_inference.rl.agents import RandomAgent
from sequential_inference.util.data import gather_trajectory_data
from sequential_inference.experiments.data import FixedDataSamplingStrategy


@hydra.main(config_path="../config", config_name="tests/dreamer_test")
def main_test(cfg):
    num_envs = 2
    env = make_vec_env(
        num_envs, "HalfCheetahDir-v0", time_limit=20, is_multi_env=True, suite="mujoco"
    )
    buffer = TrajectoryReplayBuffer(200, 200, env, sample_length=10, chp_dir="/tmp")
    agent = RandomAgent(env.action_space)
    gather_trajectory_data(env, agent, buffer, num_envs * 20 * 10)

    data_sampler = FixedDataSamplingStrategy(env, buffer, 16)
    data_sampler.set_num_sampling_steps(2 * num_envs * 20)
    data_sampler.before_experiment()
    print(data_sampler.get_batch())


if __name__ == "__main__":
    main_test()
