import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import (
    SawyerXYZEnv,
    _assert_task_is_set,
)

from pyquaternion import Quaternion
from metaworld.envs.mujoco.utils.rotation import euler2quat


class SawyerHammerEnvV2(SawyerXYZEnv):
    def __init__(self):

        liftThresh = 0.12
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.4, 0.0)
        obj_high = (0.1, 0.5, 0.0)
        goal_low = (0.2399, 0.7399, 0.109)
        goal_high = (0.2401, 0.7401, 0.111)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            "hammer_init_pos": np.array([0.07, 0.4, 0.2]),
            "hand_init_pos": np.array([0, 0.4, 0.2]),
        }
        self.goal = self.init_config["hammer_init_pos"]
        self.hammer_init_pos = self.init_config["hammer_init_pos"]
        self.hand_init_pos = self.init_config["hand_init_pos"]

        self.liftThresh = liftThresh
        self.max_path_length = 200

        self._random_reset_space = Box(np.array(obj_low), np.array(obj_high))
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.max_nail_dist = None
        self.max_hammer_dist = None
        self.maxHammerDist = 0.2

    @property
    def model_name(self):
        return full_v2_path_for("sawyer_xyz/sawyer_hammer.xml")

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
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
        self.sim.model.body_pos[self.model.body_name2id("box")] = np.array(
            [-0.24, 0.85, 0.0]
        )
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
        # import pdb; pdb.set_trace()
        if hammerPos[2] < objPos[2] - drop_thresh:
            rewards -= 10

        hammer_nail_dist_reward = 1 - np.tanh(hammerDist)
        rewards += hammer_nail_dist_reward

        nail_strike_reward = 1 - np.tanh(screwDist)
        nail_striike_weight = 100
        rewards += nail_striike_weight * nail_strike_reward

        return [rewards, 0, reachDist, 0, 0, hammerDist, screwDist]
