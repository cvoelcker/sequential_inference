import numpy as np
from gym.spaces import Box

from envs.sawyer.asset_path_utils import full_v2_path_for
from envssawyer.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)

from pyquaternion import Quaternion
from envs.sawyer.mujoco.utils.rotation import euler2quat


class SawyerHammerEnvV2(SawyerXYZEnv):
    def __init__(self):

        liftThresh = 0.12
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        # obj_low = (-0.1, 0.4, 0.0)
        # obj_high = (0.1, 0.5, 0.0)
        goal_low = (-0.3, 0.85, 0.0)
        goal_high = (0.3, 0.85, 0.0)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            "hammer_init_pos": np.array([0.07, 0.4, 0.2]),
            "hand_init_pos": np.array([0, 0.4, 0.2]),
        }

        self.goal = np.array([-0.3, 0.85, 0.0])
        self.hammer_init_pos = self.init_config["hammer_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self.liftThresh = liftThresh
        self.max_path_length = 200

        self._random_reset_space = Box(np.array(goal_low), np.array(goal_high))
        self.goal_space = self._random_reset_space

        self.max_nail_dist = None
        self.max_hammer_dist = None
        self.maxHammerDist = 0.2

        # TODO only fixed. rotz works under MPPI motion planning
        # rotMode fixed, rotz, quat, euler
        self.rotMode = "fixed"

        if self.rotMode == "fixed":
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif self.rotMode == "rotz":
            self.action_rot_scale = 1.0 / 50
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        elif self.rotMode == "quat":
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2 * np.pi, 1, 1, 1, 1]),
            )
        elif self.rotMode == "euler":
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi / 2, -np.pi / 2, 0, -1]),
                np.array([1, 1, 1, np.pi / 2, np.pi / 2, np.pi * 2, 1]),
            )
        else:
            raise NotImplementedError

    def get_obs_dim(self):
        return len(self._get_obs())

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_hammer.xml")

    @_assert_task_is_set
    def step(self, action):
        if self.rotMode == "euler":
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == "fixed":
            self.set_xyz_action(action[:3])
        elif self.rotMode == "rotz":
            self.set_xyz_action_rotz(action[:4])
        else:
            self.set_xyz_action_rot(action[:7])
        self.do_simulation([action[-1], -action[-1]])

        ob = self._get_obs()
        reward, _, reachDist, pickRew, _, _, screwDist = self.compute_reward(action, ob)
        self.curr_path_length += 1

        info = {
            "reachDist": reachDist,
            "pickRew": pickRew,
            "epRew": reward,
            "goalDist": screwDist,
            "success": float(screwDist <= 0.05),
        }
        return ob, reward, False, info

    def _get_pos_objects(self):
        return np.hstack(
            (self.get_body_com("hammer").copy(), self.get_body_com("nail_link").copy())
        )

    def _set_hammer_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        # Set position of box & nail (these are not randomized)
        self.sim.model.body_pos[self.model.body_name2id("box")] = self.goal

        # Update _target_pos
        self._target_pos = self._get_site_pos("goal")

        # Randomize hammer position
        self.hammer_init_pos = (
            self._get_state_rand_vec()
            if self.random_init
            else self.init_config["hammer_init_pos"]
        )
        self._set_hammer_xyz(self.hammer_init_pos)

        # Update heights (for use in reward function)
        self.hammerHeight = self.get_body_com("hammer").copy()[2]
        self.heightTarget = self.hammerHeight + self.liftThresh

        # Update distances (for use in reward function)
        nail_init_pos = self._get_site_pos("nailHead")
        self.max_nail_dist = (self._target_pos - nail_init_pos)[1]
        self.max_hammer_dist = np.linalg.norm(
            np.array(
                [self.hammer_init_pos[0], self.hammer_init_pos[1], self.heightTarget]
            )
            - nail_init_pos
            + self.heightTarget
            + np.abs(self.max_nail_dist)
        )

        # close gripper at initial state
        # self.do_simulation([1, -1], self.frame_skip)

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        self.pickCompleted = False

    def compute_reward(self, actions, obs):
        hammerPos = obs[3:6]
        hammerHeadPos = self.data.get_geom_xpos("hammer_head").copy()
        hammerHandlePos = self.data.get_geom_xpos("hammer_handle").copy()
        objPos = self.data.site_xpos[self.model.site_name2id("nailHead")]

        rightFinger, leftFinger = (
            self._get_site_pos("rightEndEffector"),
            self._get_site_pos("leftEndEffector"),
        )
        fingerCOM = (rightFinger + leftFinger) / 2

        heightTarget = self.heightTarget

        hammerDist = np.linalg.norm(objPos - hammerHeadPos)
        screwDist = np.abs(objPos[1] - self._target_pos[1])
        reachDist = np.linalg.norm(hammerHandlePos - fingerCOM)

        rewards = 0

        # penalty for dropping the hammer

        drop_thresh = 0.03
        if hammerPos[2] < objPos[2] - drop_thresh:
            rewards -= 10

        hammer_nail_dist_reward = 1 - np.tanh(hammerDist)
        rewards += hammer_nail_dist_reward

        nail_strike_reward = 1 - np.tanh(screwDist)
        nail_striike_weight = 100
        rewards += nail_striike_weight * nail_strike_reward

        return [rewards, 0, reachDist, 0, 0, hammerDist, screwDist]

    def viewer_setup(self):
        # side view
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 180
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat[1] = 0
        self.viewer.cam.lookat[0] = 0.6
        self.viewer.cam.lookat[2] = 0.02

    def render(self, mode="human", width=500, height=500):
        if mode == "rgb_array":
            # self._get_viewer(mode='rgb_array').render()
            # window size used for old mujoco-py:
            # width, height = 500, 500
            self._get_viewer(mode="rgb_array").render(width, height)
            data = self._get_viewer(mode="rgb_array").read_pixels(
                width, height, depth=False
            )
            return np.flipud(data)
        elif mode == "human":
            self._get_viewer().render()


class SawyerHammerEnvGoal(SawyerHammerEnvV2):
    """
    This env is the multi-task version of sawyer with different position of the goal
    """

    def __init__(self, is_eval_env=False):
        self.goal_range = Box(
            low=np.array([-0.3, 0.85, 0.0]), high=np.array([0.3, 0.85, 0.0])
        )

        self.task_list = self.generate_list_of_task(
            num_tasks=30, is_eval_env=is_eval_env
        )

        super().__init__()

    def reset_task(self, task=None):
        if task is None:
            task_idx = np.random.randint(len(self.task_list))
        else:
            task_idx = task
        self.set_task(self.task_list[task_idx])

    def reset(self):
        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def generate_list_of_tasks(self, num_tasks, is_eval_env):
        """To be called externally to obtain samples from the task distribution"""
        if is_eval_env:
            np.random.seed(100)  # pick eval tasks as random from diff seed
        else:
            np.random.seed(101)

        possible_goals = [self.goal_range.sample() for _ in range(num_tasks)]
        np.random.seed()
        return possible_goals

    def set_task(self, goal):
        """To be called externally to set the task for this environment"""
        self.sim.model.body_pos[self.model.body_name2id("box")] = goal

    def get_task(self):
        return self.model.body_pos[self.model.body_name2id("box")]
