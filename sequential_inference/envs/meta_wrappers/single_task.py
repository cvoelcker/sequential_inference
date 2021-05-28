import gym
import numpy as np

from sequential_inference.envs.meta_wrappers.common import MetaTask


class SingleTaskEnv(MetaTask):
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        print("Made env")
        self.single_task = True
        self.task_size = 1
        self.task_classify = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset_task(self, task=None):
        pass

    def reset_mdp(self):
        obs = self.env.reset()
        return obs

    def render(self, resolution=(64, 64)):
        data = (
            self.env.render(mode="rgb_array", width=resolution[0], height=resolution[1])
            .astype(np.uint8)
            .transpose(2, 0, 1)
        )
        return data

    def close(self):
        return self.env.close()

    def get_task(self):
        return 0

    @property
    def _goal(self):
        return self.get_task()

    def get_all_task_idx(self):
        return [0]

    def get_dataset(self):
        return self.env.get_dataset()
