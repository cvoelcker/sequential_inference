import gym
import metaworld
import random

from mujoco_py import MjRenderContextOffscreen

from .common import MetaTask


class ML1Env(MetaTask):
    cam_config_1 = {"elevation": -130, "azimuth": 70}

    def __init__(self, task_name, is_eval_env=False):
        super().__init__()
        self.task_name = task_name
        ml1 = metaworld.ML1(task_name)
        if not is_eval_env:
            self.classes = ml1.train_classes
            self.tasks = ml1.train_tasks
        else:
            self.classes = ml1.test_classes
            self.tasks = ml1.test_tasks
        self.env = self.classes[task_name]()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.task_size = None
        self.task_classify = None
        # for rendering
        self.viewer1 = MjRenderContextOffscreen(self.env.sim, device_id=-1)
        self.init_cam_fn(self.viewer1, self.cam_config_1)
        # reset task
        self.reset_task()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset_task(self):
        task = random.choice(self.tasks)
        self.task = task
        self.set_task(task)

    def reset_mdp(self):
        return self.env.reset()

    def init_cam_fn(self, viewer, config):
        viewer.cam.elevation = config[
            "elevation"
        ]  # camera rotation around the axis in the plane going through the frame origin (if 0 you just see a line)
        viewer.cam.azimuth = config[
            "azimuth"
        ]  # camera rotation around the camera's vertical axis
        viewer.cam.distance = 2.0
        viewer.cam.lookat[1] = 0.3

    def render(self, resolution=(64, 64)):
        return self.sim.render(*resolution)[:, :, ::-1].transpose(2, 0, 1)

    def close(self):
        return self.env.close()

    def get_task(self):
        return self.env.task

    def get_all_task_idx(self):
        return len(self.tasks)
