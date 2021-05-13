import numpy as np
from gym.spaces import Box

from pyquaternion import Quaternion

from sequential_inference.envs.sawyer.asset_path_utils import full_v2_path_for
from sequential_inference.envs.sawyer.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)


class SawyerDrawerEnv(SawyerXYZEnv):
    def __init__(
        self,
        xml_path=None,
        goal_site_name=None,
        action_mode="joint_position",
        reward_scaling=1.0,
    ):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.9, 0.0)
        obj_high = (0.1, 0.9, 0.0)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            "obj_init_angle": np.array([-1.0, 0], dtype=np.float32),
            "obj_init_pos": np.array([0.0, 0.9, 0.0], dtype=np.float32),
            "hand_init_pos": np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config["obj_init_pos"]
        self.obj_init_angle_state = self.init_config["obj_init_angle"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        goal_low = self.hand_low
        goal_high = self.hand_high

        self.max_path_length = 150

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxDist = 0.15
        self.target_reward = 1000 * self.maxDist + 1000 * 2

        angle = self.obj_init_angle_state[0]

        quat = Quaternion(axis=[1, 0, 0], angle=angle)
        self.rotate_matrix = quat.rotation_matrix

        # state >= 0, opening the drawer, state < 0, close the drawer
        self.open_close = self.obj_init_angle_state[1] >= 0

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

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_drawer.xml")

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
        reward, reachDist, pullDist = self.compute_reward(action, ob)
        self.curr_path_length += 1
        info = {
            "reachDist": reachDist,
            "goalDist": pullDist,
            "epRew": reward,
            "pickRew": None,
            "success": float(pullDist <= 0.03),
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        object_vector = np.array([0.0, -0.16, 0.05])
        object_rotate_vector = np.matmul(self.rotate_matrix, object_vector)

        return self.get_body_com("drawer_link") + object_rotate_vector

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        # Compute nightstand position
        self.obj_init_pos = self.init_config["obj_init_pos"]

        # Set mujoco body to computed position
        self.sim.model.body_pos[self.model.body_name2id("drawer")] = self.obj_init_pos

        angle = self.obj_init_angle_state[0]

        quat = Quaternion(axis=[1, 0, 0], angle=angle)

        self.sim.model.body_quat[self.model.body_name2id("drawer")] = quat.elements
        self.rotate_matrix = quat.rotation_matrix

        if self.open_close:
            target_vector_origin = np.array([0.0, -0.16 - self.maxDist, 0.09])
        else:
            target_vector_origin = np.array([0.0, -0.16, 0.09])
            self._set_obj_xyz(-self.maxDist)

        target_vector = np.matmul(self.rotate_matrix, target_vector_origin)

        # Set _target_pos to current drawer position (closed) minus an offset
        self._target_pos = self.obj_init_pos + target_vector

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        del actions

        objPos = obs[3:6]

        rightFinger, leftFinger = (
            self._get_site_pos("rightEndEffector"),
            self._get_site_pos("leftEndEffector"),
        )
        fingerCOM = (rightFinger + leftFinger) / 2
        pullGoal = self._target_pos
        pullDist = np.abs(objPos[1] - pullGoal[1])
        reachDist = np.linalg.norm(objPos - fingerCOM)
        reachRew = -reachDist

        self.reachCompleted = reachDist < 0.05

        def pullReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if self.reachCompleted:
                pullRew = 1000 * (self.maxDist - pullDist) + c1 * (
                    np.exp(-(pullDist ** 2) / c2) + np.exp(-(pullDist ** 2) / c3)
                )
                pullRew = max(pullRew, 0)
                return pullRew
            else:
                return 0

            pullRew = max(pullRew, 0)

            return pullRew

        pullRew = pullReward()
        reward = reachRew + pullRew

        return [reward, reachDist, pullDist]

    def viewer_setup(self):
        # side view
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.elevation = 0
        self.viewer.cam.azimuth = 180
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat[1] = 0.5
        self.viewer.cam.lookat[0] = 0.0
        self.viewer.cam.lookat[2] = 0.2

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


class SawyerDrawerEnvGoal(SawyerDrawerEnv):
    """
    This env is the multi-task version of sawyer with different position of the goal
    """

    def __init__(
        self,
        is_eval_env=False,
        reward_scaling=1.0,
        xml_path=None,
        goal_site_name=None,
        action_mode="joint_position",
        mode="ee",
        level="easy",
    ):
        self.goal_range = Box(low=np.array([-0.1, 0.0]), high=np.array([0.0, 1.0]))

        self.task_list = self.generate_list_of_tasks(
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
        self.obj_init_angle_state = goal.copy()

    def get_task(self):
        return self.obj_init_angle_state

    def step(self, action):
        obs, rew, done, info = super().step(action)
        info["task"] = self.get_task()
        return obs, rew, done, info
