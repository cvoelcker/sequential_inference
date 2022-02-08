import numpy as np
import multiprocessing
from robosuite.environments.manipulation.wipe import Wipe, DEFAULT_WIPE_CONFIG
from robosuite.models.tasks import ManipulationTask

from .wipe_located_arena import WipeLocatedArena


class WipeLocated(Wipe):
    def __init__(
        self,
        num_paths=20,
        continuous_paths=True,
        location_paths="everywhere",  # everywhere, left, right, up, down
        seed=None,
        num_markers=20,
    ):
        self.num_paths = num_paths
        self.continuous_paths = continuous_paths
        self.location_paths = location_paths
        self.seed = seed
        config = DEFAULT_WIPE_CONFIG
        config.update({"num_markers": num_markers, "arm_limit_collision_penalty": -200})

        controller_configs = {
            "type": "JOINT_POSITION",
            "input_max": 1,
            "input_min": -1,
            "output_max": 0.05,
            "output_min": -0.05,
            "kp": 50,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "qpos_limits": None,
            "interpolation": None,
            "ramp_ratio": 0.2,
        }

        super().__init__(
            task_config=config,
            robots="Panda",
            # robots="Sawyer",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            hard_reset=False,
            controller_configs=controller_configs,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Get robot's contact geoms
        self.robot_contact_geoms = self.robots[0].robot_model.contact_geoms

        mujoco_arena = WipeLocatedArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            table_friction_std=self.table_friction_std,
            coverage_factor=self.coverage_factor,
            num_markers=self.num_markers,
            line_width=self.line_width,
            num_paths=self.num_paths,
            continuous_paths=self.continuous_paths,
            location_paths=self.location_paths,
            seed=self.seed,
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of self.unit_wiped_reward is provided per single dirt (peg) wiped during this step
            - a discrete reward of self.task_complete_reward is provided if all dirt is wiped

        Note that if the arm is either colliding or near its joint limit, a reward of 0 will be automatically given

        Un-normalized summed components if using reward shaping (individual components can be set to 0:

            - Reaching: in [0, self.distance_multiplier], proportional to distance between wiper and centroid of dirt
              and zero if the table has been fully wiped clean of all the dirt
            - Table Contact: in {0, self.wipe_contact_reward}, non-zero if wiper is in contact with table
            - Wiping: in {0, self.unit_wiped_reward}, non-zero for each dirt (peg) wiped during this step
            - Cleaned: in {0, self.task_complete_reward}, non-zero if no dirt remains on the table
            - Collision / Joint Limit Penalty: in {self.arm_limit_collision_penalty, 0}, nonzero if robot arm
              is colliding with an object
              - Note that if this value is nonzero, no other reward components can be added
            - Large Force Penalty: in [-inf, 0], scaled by wiper force and directly proportional to
              self.excess_force_penalty_mul if the current force exceeds self.pressure_threshold_max
            - Large Acceleration Penalty: in [-inf, 0], scaled by estimated wiper acceleration and directly
              proportional to self.ee_accel_penalty

        Note that the final per-step reward is normalized given the theoretical best episode return and then scaled:
        reward_scale * (horizon /
        (num_markers * unit_wiped_reward + horizon * (wipe_contact_reward + task_complete_reward)))

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0

        total_force_ee = np.linalg.norm(
            np.array(self.robots[0].recent_ee_forcetorques.current[:3])
        )

        # Skip checking whether limits are achieved (maybe unsafe for real robots)
        active_markers = []

        # Current 3D location of the corners of the wiping tool in world frame
        c_geoms = self.robots[0].gripper.important_geoms["corners"]
        corner1_id = self.sim.model.geom_name2id(c_geoms[0])
        corner1_pos = np.array(self.sim.data.geom_xpos[corner1_id])
        corner2_id = self.sim.model.geom_name2id(c_geoms[1])
        corner2_pos = np.array(self.sim.data.geom_xpos[corner2_id])
        corner3_id = self.sim.model.geom_name2id(c_geoms[2])
        corner3_pos = np.array(self.sim.data.geom_xpos[corner3_id])
        corner4_id = self.sim.model.geom_name2id(c_geoms[3])
        corner4_pos = np.array(self.sim.data.geom_xpos[corner4_id])

        # Unit vectors on my plane
        v1 = corner1_pos - corner2_pos
        v1 /= np.linalg.norm(v1)
        v2 = corner4_pos - corner2_pos
        v2 /= np.linalg.norm(v2)

        # Corners of the tool in the coordinate frame of the plane
        t1 = np.array(
            [
                np.dot(corner1_pos - corner2_pos, v1),
                np.dot(corner1_pos - corner2_pos, v2),
            ]
        )
        t2 = np.array(
            [
                np.dot(corner2_pos - corner2_pos, v1),
                np.dot(corner2_pos - corner2_pos, v2),
            ]
        )
        t3 = np.array(
            [
                np.dot(corner3_pos - corner2_pos, v1),
                np.dot(corner3_pos - corner2_pos, v2),
            ]
        )
        t4 = np.array(
            [
                np.dot(corner4_pos - corner2_pos, v1),
                np.dot(corner4_pos - corner2_pos, v2),
            ]
        )

        pp = [t1, t2, t4, t3]

        # Normal of the plane defined by v1 and v2
        n = np.cross(v1, v2)
        n /= np.linalg.norm(n)

        def isLeft(P0, P1, P2):
            return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])

        def PointInRectangle(X, Y, Z, W, P):
            return (
                isLeft(X, Y, P) < 0
                and isLeft(Y, Z, P) < 0
                and isLeft(Z, W, P) < 0
                and isLeft(W, X, P) < 0
            )

        # Only go into this computation if there are contact points
        if self.sim.data.ncon != 0:

            # Check each marker that is still active
            for marker in self.model.mujoco_arena.markers:

                # Current marker 3D location in world frame
                marker_pos = np.array(
                    self.sim.data.body_xpos[
                        self.sim.model.body_name2id(marker.root_body)
                    ]
                )

                # We use the second tool corner as point on the plane and define the vector connecting
                # the marker position to that point
                v = marker_pos - corner2_pos

                # Shortest distance between the center of the marker and the plane
                dist = np.dot(v, n)

                # Projection of the center of the marker onto the plane
                projected_point = np.array(marker_pos) - dist * n

                # Positive distances means the center of the marker is over the plane
                # The plane is aligned with the bottom of the wiper and pointing up, so the marker would be over it
                if dist > 0.0:
                    # Distance smaller than this threshold means we are close to the plane on the upper part
                    if dist < 0.02:
                        # Write touching points and projected point in coordinates of the plane
                        pp_2 = np.array(
                            [
                                np.dot(projected_point - corner2_pos, v1),
                                np.dot(projected_point - corner2_pos, v2),
                            ]
                        )
                        # Check if marker is within the tool center:
                        if PointInRectangle(pp[0], pp[1], pp[2], pp[3], pp_2):
                            active_markers.append(marker)

        # Obtain the list of currently active (wiped) markers that where not wiped before
        # These are the markers we are wiping at this step
        lall = np.where(np.isin(active_markers, self.wiped_markers, invert=True))
        new_active_markers = np.array(active_markers)[lall]

        # Loop through all new markers we are wiping at this step
        for new_active_marker in new_active_markers:
            # Grab relevant marker id info
            new_active_marker_geom_id = self.sim.model.geom_name2id(
                new_active_marker.visual_geoms[0]
            )
            # Make this marker transparent since we wiped it (alpha = 0)
            self.sim.model.geom_rgba[new_active_marker_geom_id][3] = 0
            # Add this marker the wiped list
            self.wiped_markers.append(new_active_marker)
            # Add reward if we're using the dense reward
            if self.reward_shaping:
                reward += self.unit_wiped_reward

        # Additional reward components if using dense rewards
        if self.reward_shaping:
            # If we haven't wiped all the markers yet, add a smooth reward for getting closer
            # to the centroid of the dirt to wipe
            if len(self.wiped_markers) < self.num_markers:
                _, _, mean_pos_to_things_to_wipe = self._get_wipe_information
                mean_distance_to_things_to_wipe = np.linalg.norm(
                    mean_pos_to_things_to_wipe
                )
                dist = 5 * mean_distance_to_things_to_wipe
                reward += -(dist**2 + np.log(dist**2 + 1e-6))
                # reward += self.distance_multiplier * (
                #         1 - np.tanh(self.distance_th_multiplier * mean_distance_to_things_to_wipe))

            # Reward for keeping contact
            if self.sim.data.ncon != 0 and self._has_gripper_contact:
                reward += self.wipe_contact_reward

            # Penalty for excessive force with the end-effector
            if total_force_ee > self.pressure_threshold_max:
                reward -= self.excess_force_penalty_mul * total_force_ee
                self.f_excess += 1

            # Reward for pressing into table
            # TODO: Need to include this computation somehow in the scaled reward computation
            elif total_force_ee > self.pressure_threshold and self.sim.data.ncon > 1:
                reward += self.wipe_contact_reward + 0.01 * total_force_ee
                if self.sim.data.ncon > 50:
                    reward += 10.0 * self.wipe_contact_reward

            # Penalize large accelerations
            reward -= self.ee_accel_penalty * np.mean(
                abs(self.robots[0].recent_ee_acc.current)
            )

        # Final reward if all wiped
        if len(self.wiped_markers) == self.num_markers:
            reward += self.task_complete_reward

        # Printing results
        if self.print_results:
            string_to_print = (
                "Process {pid}, timestep {ts:>4}: reward: {rw:8.4f}"
                "wiped markers: {ws:>3} collisions: {sc:>3} f-excess: {fe:>3}".format(
                    pid=id(multiprocessing.current_process()),
                    ts=self.timestep,
                    rw=reward,
                    ws=len(self.wiped_markers),
                    sc=self.collisions,
                    fe=self.f_excess,
                )
            )
            print(string_to_print)

        # If we're scaling our reward, we normalize the per-step rewards given the theoretical best episode return
        # This is equivalent to scaling the reward by:
        #   reward_scale * (horizon /
        #       (num_markers * unit_wiped_reward + horizon * (wipe_contact_reward + task_complete_reward)))
        if self.reward_scale:
            reward *= self.reward_scale * self.reward_normalization_factor
        return reward
