from typing import List
import collections
import time

from aloha.constants import (
    DT,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    FOLLOWER_GRIPPER_JOINT_OPEN,
    FOLLOWER_GRIPPER_JOINT_UNNORMALIZE_FN,
    FOLLOWER_GRIPPER_POSITION_NORMALIZE_FN,
    FOLLOWER_GRIPPER_VELOCITY_NORMALIZE_FN,
    IS_MOBILE,
    LEADER_GRIPPER_JOINT_NORMALIZE_FN,
    START_ARM_POSE,
)
from aloha.robot_utils import (
    ImageRecorder,
    move_arms,
    move_grippers,
    Recorder,
    setup_follower_bot,
    setup_leader_bot,
    transMatrix_to_euler_vecter,
    euler_vector_to_transMatrix,
    get_delta_transMatrix,
)
import dm_env
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    InterbotixRobotNode,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_modules.xs_robot.slate import InterbotixSlate
from interbotix_xs_msgs.msg import JointSingleCommand
import matplotlib.pyplot as plt
import numpy as np
from rclpy.logging import LoggingSeverity


class RealEnv:
    """
    Environment for real robot bi-manual manipulation.

    Action space: [
        left_arm_qpos (6),             # absolute joint position
        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
        right_arm_qpos (6),            # absolute joint position
        right_gripper_positions (1),   # normalized gripper position (0: close, 1: open)
    ]

    Observation space: {
        "qpos": Concat[
            left_arm_qpos (6),          # absolute joint position
            left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
            right_arm_qpos (6),         # absolute joint position
            right_gripper_qpos (1)      # normalized gripper position (0: close, 1: open)
        ]
        "qvel": Concat[
            left_arm_qvel (6),          # absolute joint velocity (rad)
            left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
            right_arm_qvel (6),         # absolute joint velocity (rad)
            right_gripper_qvel (1)      # normalized gripper velocity (pos: opening, neg: closing)
        ]
        "ee_pos": Concat[
            left_arm_ee_pos (3+3),      # end effector position (x, y, z) + orientation (qx, qy, qz)
            left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
            right_arm_ee_pos (3+3),     # end effector position (x, y, z) + orientation (qx, qy, qz)
            right_gripper_position (1)  # normalized gripper position (0: close, 1: open)
        ]
        "images": {
            "cam_high": (480x640x3),        # h, w, c, dtype='uint8'
            "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
            "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
            "cam_right_wrist": (480x640x3)  # h, w, c, dtype='uint8'
        }
    """

    def __init__(
        self,
        node: InterbotixRobotNode,
        setup_robots: bool = True,
        setup_base: bool = False,
        camera_names: List[str] = None,
        is_mobile: bool = IS_MOBILE,
        arm_mask: List[bool] = [True, True],
        logging_level=LoggingSeverity.INFO,
        target_fps: int = 30,
    ):
        self.follower_bot_left = InterbotixManipulatorXS(
            robot_model="vx300s",
            group_name="arm",
            gripper_name="gripper",
            robot_name="follower_left",
            node=node,
            iterative_update_fk=True,
            logging_level=logging_level,
        )
        self.follower_bot_right = InterbotixManipulatorXS(
            robot_model="vx300s",
            group_name="arm",
            gripper_name="gripper",
            robot_name="follower_right",
            node=node,
            iterative_update_fk=True,
            logging_level=logging_level,
        )

        self.recorder_left = Recorder("left", node=node)
        self.recorder_right = Recorder("right", node=node)
        self.image_recorder = ImageRecorder(
            node=node,
            camera_names=camera_names,
            is_mobile=IS_MOBILE,
            is_monitor=False,
            target_fps=target_fps,
        )
        self.gripper_command = JointSingleCommand(name="gripper")
        self.previous_ee_pose = None
        if setup_robots:
            self.setup_robots()

        if setup_base:
            if is_mobile:
                self.setup_base(node)
            else:
                raise ValueError(
                    (
                        "Requested to set up base but robot is not mobile. "
                        "Hint: check the 'IS_MOBILE' constant."
                    )
                )

        self.arm_mask = arm_mask

    def setup_base(self, node):
        self.base = InterbotixSlate(
            "aloha",
            node=node,
        )
        self.base.base.set_motor_torque(False)

    def setup_robots(self):
        setup_follower_bot(self.follower_bot_left)
        setup_follower_bot(self.follower_bot_right)

    def get_ee_pose(self):
        left_qpos_raw = self.recorder_left.qpos
        right_qpos_raw = self.recorder_right.qpos

        left_eepos_matrix = self.follower_bot_left.arm.get_ee_pose()
        right_eepos_matrix = self.follower_bot_right.arm.get_ee_pose()
        left_arm_eepos = transMatrix_to_euler_vecter(left_eepos_matrix)
        right_arm_eepos = transMatrix_to_euler_vecter(right_eepos_matrix)

        left_gripper_qpos = [FOLLOWER_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[7])]
        right_gripper_qpos = [FOLLOWER_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])]
        return np.concatenate(
            [left_arm_eepos, left_gripper_qpos, right_arm_eepos, right_gripper_qpos]
        )

    def get_delta_ee_pose(self, prev_ee_pose, curr_ee_pose):
        prev_left_matrix = euler_vector_to_transMatrix(prev_ee_pose[:6])
        prev_right_matrix = euler_vector_to_transMatrix(prev_ee_pose[7:13])
        curr_left_matrix = euler_vector_to_transMatrix(curr_ee_pose[:6])
        curr_right_matrix = euler_vector_to_transMatrix(curr_ee_pose[7:13])
        delta_left_matrix = get_delta_transMatrix(prev_left_matrix, curr_left_matrix)
        delta_right_matrix = get_delta_transMatrix(prev_right_matrix, curr_right_matrix)

        delta_left_eepos = transMatrix_to_euler_vecter(delta_left_matrix)
        delta_right_eepos = transMatrix_to_euler_vecter(delta_right_matrix)

        return np.concatenate(
            [
                delta_left_eepos,
                [curr_ee_pose[6]],
                delta_right_eepos,
                [curr_ee_pose[13]],
            ]
        )

    def ee_pose_step(self, delta_ee_pose, prev_ee_pose=None):
        """
        Step the environment using delta end effector pose.
        """
        if prev_ee_pose is None:
            prev_ee_pose = self.get_ee_pose()

        prev_left_eepose = prev_ee_pose[:6]
        prev_right_eepose = prev_ee_pose[7:13]
        prev_left_matrix = euler_vector_to_transMatrix(prev_left_eepose)
        prev_right_matrix = euler_vector_to_transMatrix(prev_right_eepose)

        # set_ee_pose_matrix
        delta_left_eepose = delta_ee_pose[:6]
        delta_right_eepose = delta_ee_pose[7:13]
        delta_left_matrix = euler_vector_to_transMatrix(delta_left_eepose)
        delta_right_matrix = euler_vector_to_transMatrix(delta_right_eepose)

        # get new ee pose
        new_left_matrix = prev_left_matrix @ delta_left_matrix
        new_right_matrix = prev_right_matrix @ delta_right_matrix

        left_gripper_qpos = delta_ee_pose[6]
        right_gripper_qpos = delta_ee_pose[13]
        return new_left_matrix, left_gripper_qpos, new_right_matrix, right_gripper_qpos

    def get_qpos(self):
        left_qpos_raw = self.recorder_left.qpos
        right_qpos_raw = self.recorder_right.qpos
        left_arm_qpos = left_qpos_raw[:6]
        right_arm_qpos = right_qpos_raw[:6]
        left_gripper_qpos = [FOLLOWER_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[7])]
        right_gripper_qpos = [FOLLOWER_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])]
        return np.concatenate(
            [left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos]
        )

    def get_qvel(self):
        left_qvel_raw = self.recorder_left.qvel
        right_qvel_raw = self.recorder_right.qvel
        left_arm_qvel = left_qvel_raw[:6]
        right_arm_qvel = right_qvel_raw[:6]
        left_gripper_qvel = [FOLLOWER_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[7])]
        right_gripper_qvel = [FOLLOWER_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[7])]
        return np.concatenate(
            [left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel]
        )

    def get_effort(self):
        left_effort_raw = self.recorder_left.effort
        right_effort_raw = self.recorder_right.effort
        left_robot_effort = left_effort_raw[:7]
        right_robot_effort = right_effort_raw[:7]
        return np.concatenate([left_robot_effort, right_robot_effort])

    def get_images(self):
        return self.image_recorder.get_images()

    def get_base_vel(self):
        linear_vel = self.base.base.get_linear_velocity().x
        angular_vel = self.base.base.get_angular_velocity().z
        return np.array([linear_vel, angular_vel])

    def set_gripper_pose(
        self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized
    ):
        if self.arm_mask[0]:
            left_gripper_desired_joint = FOLLOWER_GRIPPER_JOINT_UNNORMALIZE_FN(
                left_gripper_desired_pos_normalized
            )
            self.gripper_command.cmd = left_gripper_desired_joint
            self.follower_bot_left.gripper.core.pub_single.publish(self.gripper_command)

        if self.arm_mask[1]:
            right_gripper_desired_joint = FOLLOWER_GRIPPER_JOINT_UNNORMALIZE_FN(
                right_gripper_desired_pos_normalized
            )
            self.gripper_command.cmd = right_gripper_desired_joint
            self.follower_bot_right.gripper.core.pub_single.publish(
                self.gripper_command
            )

    def _reset_joints(self):
        reset_position = START_ARM_POSE[:6]
        move_arms(
            [self.follower_bot_left, self.follower_bot_right],
            [reset_position, reset_position],
            moving_time=1.0,
        )

    def _reset_gripper(self):
        """
        Set to position mode and do position resets.

        First open then close, then change back to PWM mode
        """
        move_grippers(
            [self.follower_bot_left, self.follower_bot_right],
            [FOLLOWER_GRIPPER_JOINT_OPEN] * 2,
            moving_time=0.5,
        )
        move_grippers(
            [self.follower_bot_left, self.follower_bot_right],
            [FOLLOWER_GRIPPER_JOINT_CLOSE] * 2,
            moving_time=1.0,
        )

    def get_observation(self, get_base_vel=False):
        obs = collections.OrderedDict()
        obs["qpos"] = self.get_qpos()
        obs["qvel"] = self.get_qvel()
        obs["effort"] = self.get_effort()
        obs["images"] = self.get_images()
        obs["eepose"] = self.get_ee_pose()
        if getattr(self, "base", None) is not None:
            obs["base_vel"] = self.get_base_vel()
            if get_base_vel:
                obs["base_vel"] = self.get_base_vel()
        else:
            obs["base_vel"] = np.zeros(2)
        return obs

    def get_reward(self):
        return 0

    def reset(self, fake=False):
        if not fake:
            # Reboot follower robot gripper motors
            self.follower_bot_left.core.robot_reboot_motors("single", "gripper", True)
            self.follower_bot_right.core.robot_reboot_motors("single", "gripper", True)
            self._reset_joints()
            self._reset_gripper()
        obs = self.get_observation()
        self.previous_ee_pose = obs["eepose"].copy()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )

    def step(
        self,
        action,
        base_action=None,
        get_base_vel=False,
        get_obs=True,
        use_delta_ee=False,
        moving_time=DT,
    ):
        state_len = int(len(action) / 2)
        left_action = action[:state_len]
        right_action = action[state_len:]

        left_gripper_qpos = left_action[-1]
        right_gripper_qpos = right_action[-1]

        if use_delta_ee:
            left_action, left_gripper_qpos, right_action, right_gripper_qpos = (
                self.ee_pose_step(action, self.previous_ee_pose)
            )
        if use_delta_ee:
            if self.arm_mask[0]:
                curr_left_joint = self.follower_bot_left.arm.get_joint_positions()
                # print("curr_left_joint before ee pose matrix", curr_left_joint)
                theta_list, success = self.follower_bot_left.arm.set_ee_pose_matrix(
                    left_action,
                    custom_guess=curr_left_joint,
                    moving_time=moving_time,
                    blocking=False,
                    # execute=False,
                )
                # curr_left_joint = self.follower_bot_left.arm.get_joint_positions()
                # print("curr_left_joint after ee pose matrix", curr_left_joint)
                if not success:
                    print("Failed to set left arm ee pose")
                # else:
                #     num_steps = int(np.ceil(moving_time / DT))
                #     traj_list = np.linspace(curr_left_joint, theta_list, num_steps)
                #     print(f"{curr_left_joint} -> {theta_list}")
                #     for t in range(num_steps):
                #         success = self.follower_bot_left.arm.set_joint_positions(
                #             traj_list[t], blocking=False
                #         )
                #         time.sleep(DT)
            if self.arm_mask[1]:
                curr_right_joint = self.follower_bot_right.arm.get_joint_positions()
                self.follower_bot_right.arm.set_ee_pose_matrix(
                    right_action,
                    blocking=False,
                    custom_guess=curr_right_joint,
                    execute=False,
                )
        else:
            if self.arm_mask[0]:
                self.follower_bot_left.arm.set_joint_positions(
                    left_action[:6], blocking=False
                )
            if self.arm_mask[1]:
                self.follower_bot_right.arm.set_joint_positions(
                    right_action[:6], blocking=False
                )

        self.set_gripper_pose(left_gripper_qpos, right_gripper_qpos)
        if base_action is not None:
            base_action_linear, base_action_angular = base_action
            self.base.base.command_velocity_xyaw(
                x=base_action_linear, yaw=base_action_angular
            )
        if get_obs:
            obs = self.get_observation(get_base_vel)
            delta_ee_pose = self.get_delta_ee_pose(self.previous_ee_pose, obs["eepose"])
            obs["delta_eepose"] = delta_ee_pose
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs,
        )


def get_action(
    leader_bot_left: InterbotixManipulatorXS, leader_bot_right: InterbotixManipulatorXS
):
    action = np.zeros(14)  # 6 joint + 1 gripper, for two arms
    # Arm actions
    action[:6] = leader_bot_left.core.joint_states.position[:6]
    action[7 : 7 + 6] = leader_bot_right.core.joint_states.position[:6]
    # Gripper actions
    action[6] = LEADER_GRIPPER_JOINT_NORMALIZE_FN(
        leader_bot_left.core.joint_states.position[6]
    )
    action[7 + 6] = LEADER_GRIPPER_JOINT_NORMALIZE_FN(
        leader_bot_right.core.joint_states.position[6]
    )

    return action


def make_real_env(
    node: InterbotixRobotNode = None,
    setup_robots: bool = True,
    setup_base: bool = False,
    arm_mask: List[bool] = [True, True],
    logging_level=LoggingSeverity.INFO,
    camera_names: List[str] = None,
    target_fps: int = 30,
):
    if node is None:
        node = get_interbotix_global_node()
        if node is None:
            node = create_interbotix_global_node("aloha")
    env = RealEnv(
        node,
        setup_robots,
        setup_base,
        camera_names=camera_names,
        arm_mask=arm_mask,
        logging_level=logging_level,
        target_fps=target_fps,
    )
    return env


def test_real_teleop():
    """
    Test bimanual teleoperation and show image observations onscreen.

    It first reads joint poses from both leader arms.
    Then use it as actions to step the environment.
    The environment returns full observations including images.

    An alternative approach is to have separate scripts for teleop and observation recording.
    This script will result in higher fidelity (obs, action) pairs
    """
    onscreen_render = True
    render_cam = "cam_left_wrist"

    node = get_interbotix_global_node()

    # source of data
    leader_bot_left = InterbotixManipulatorXS(
        robot_model="wx250s",
        robot_name="leader_left",
        node=node,
    )
    leader_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s",
        robot_name="leader_right",
        node=node,
    )
    setup_leader_bot(leader_bot_left)
    setup_leader_bot(leader_bot_right)

    # environment setup
    env = make_real_env(node=node)
    ts = env.reset(fake=True)
    episode = [ts]
    # visualization setup
    if onscreen_render:
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation["images"][render_cam])
        plt.ion()

    for _ in range(1000):
        action = get_action(leader_bot_left, leader_bot_right)
        ts = env.step(action)
        episode.append(ts)

        if onscreen_render:
            plt_img.set_data(ts.observation["images"][render_cam])
            plt.pause(DT)
        else:
            time.sleep(DT)


if __name__ == "__main__":
    test_real_teleop()
