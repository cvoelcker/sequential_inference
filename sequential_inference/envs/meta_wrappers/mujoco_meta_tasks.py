import gym
import d4rl
import numpy as np

from gym.envs.registration import load
from PIL import Image

from .common import MetaTask


def mujoco_wrapper(entry_point, **kwargs):
    # Load the environment from its entry point
    env_cls = load(entry_point)
    env = env_cls(**kwargs)
    return env


class MujocoMetaTask(MetaTask):
    def __init__(self, task_name, single_task=False, **kwargs):
        super().__init__()
        if task_name == "HalfCheetahDir-v0" or task_name == "HalfCheetahVel-v0":
            kwargs = dict()
        self.env = gym.make(task_name, **kwargs)
        self.task_name = task_name
        if task_name == "HalfCheetahDir-v0":
            self.task_size = 2
            self.task_classify = True
        elif task_name == "HalfCheetahVel-v0":
            self.task_size = 1
            self.task_classify = False
        elif task_name == "SawyerReachGoal-v0":
            self.task_size = 3
            self.task_classify = False
        elif task_name == "SawyerHammerGoal-v0":
            self.task_size = 3
            self.task_classify = False
        elif task_name == "SawyerDrawerGoal-v0":
            self.task_size = 3
            self.task_classify = False
        elif task_name == "SawyerReachGoalOOD-v0":
            self.task_size = len(self.env.task_list)
            self.task_classify = False
        elif task_name == "SawyerReachGoalTwoModes-v0":
            self.task_size = len(self.env.task_list)
            self.task_classify = False
        else:
            self.task_size = None
            self.task_classify = None
        self.single_task = single_task

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset_task(self, task=None):
        if self.single_task:
            task = self.get_task()
        if hasattr(self.env, "reset_task"):
            self.env.reset_task(task)

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
        task = self.env.get_task()
        if self.task_name == "HalfCheetahDir-v0":
            # task[0] = 0 if task[0] == -1 else 1
            task = 0 if task[0] == -1 else 1
        return task

    @property
    def _goal(self):
        return self.get_task()

    def get_all_task_idx(self):
        if self.task_name == "HalfCheetahDir-v0":
            return [0, 1]
        return list(range(len(self.env.task_list)))

    def get_dataset(self):
        return self.env.get_dataset()
