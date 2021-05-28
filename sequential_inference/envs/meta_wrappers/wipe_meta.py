import numpy as np
from gym.spaces.box import Box

from sequential_inference.envs.robosuite_envs.wipe import WipeLocated


class WipeMeta:
    def __init__(
        self, task_code="ecec", num_paths=30, num_markers=35, is_eval_task=False
    ):
        location_decode = {
            "e": "everywhere",
            "u": "up",
            "d": "down",
            "l": "left",
            "r": "right",
        }
        shape_decode = {"c": True, "d": False}
        if is_eval_task:
            location = location_decode[task_code[2]]
            continuous = shape_decode[task_code[3]]
            seed = 5
        else:
            location = location_decode[task_code[0]]
            continuous = shape_decode[task_code[1]]
            seed = 10

        self.env = WipeLocated(
            num_paths=num_paths,
            num_markers=num_markers,
            continuous_paths=continuous,
            location_paths=location,
            seed=seed,
        )
        self._define_spaces()
        # for task loss in BADream
        self.task_size = None
        self.task_classify = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _prepare_observation(self, raw_obs):
        return raw_obs["robot0_robot-state"]

    def _define_spaces(self):
        action_limits = self.env.robots[0].action_limits
        self.action_space = Box(
            low=action_limits[0],
            high=action_limits[1],
            shape=(self.env.robots[0].action_dim,),
        )
        obs = self.reset()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs.shape)

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self._prepare_observation(raw_obs)
        return obs, reward, done, info

    def reset(self):
        raw_obs = self.env.reset()
        return self._prepare_observation(raw_obs)

    def reset_mpd(self):
        self.env.deterministic_reset = True
        obs = self.reset()
        self.env.deterministic_reset = False
        return obs

    def render(self, resolution):
        return np.flip(
            self.env.sim.render(resolution[0], resolution[1], camera_name="sideview"),
            axis=0,
        ).transpose(2, 0, 1)
