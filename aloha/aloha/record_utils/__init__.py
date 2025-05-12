from typing import List

import os
import time
import h5py
import numpy as np

import rclpy
from rclpy.logging import LoggingSeverity
from rclpy.signals import SignalHandlerOptions
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_startup,
)
from aloha.real_env import make_real_env
from aloha.constants import (
    DT,
    ARM_MASKS,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    FOLLOWER_GRIPPER_JOINT_OPEN,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEADER_GRIPPER_JOINT_MID,
    LEADER_GRIPPER_JOINT_OPEN,
    START_ARM_POSE,
    INACTIVE_START_ARM_POSE,
    CAMERA_RESOLUTIONS,
)
from aloha.robot_utils import (
    enable_gravity_compensation,
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_on,
    torque_off,
    sleep_arms,
)
from .cli import InputThread


LOGGING = dict(
    debug=LoggingSeverity.DEBUG,
    info=LoggingSeverity.INFO,
    warn=LoggingSeverity.WARN,
    error=LoggingSeverity.ERROR,
    fatal=LoggingSeverity.FATAL,
    unset=LoggingSeverity.UNSET,
    default=LoggingSeverity.INFO,
)


class RecordState:
    def __init__(self):
        self.debug = False
        self.shutdown_requested = False


def shutdown_requested(state: RecordState = None):
    if state is None:
        return False
    return state.shutdown_requested


def is_gripper_closed(leader_bots, threshold=LEADER_GRIPPER_CLOSE_THRESH):
    gripper_poses = [get_arm_gripper_positions(bot) for bot in leader_bots]
    return all([gripper_pose < threshold for gripper_pose in gripper_poses])


def wait_for_start(
    leader_bots,
    state: RecordState = None,
    verbose=True,
    use_gravity_compensation=False,
    torque_off=False,
):
    # press gripper to start data collection
    # disable torque for only gripper joint of leader robot to allow user movement
    for leader_bot in leader_bots:
        leader_bot.core.robot_torque_enable("single", "gripper", False)
    if verbose:
        print(f"Close the gripper to start")

    close_thresh = LEADER_GRIPPER_CLOSE_THRESH
    while not is_gripper_closed(
        leader_bots, threshold=close_thresh
    ) and not shutdown_requested(state):
        time.sleep(DT)
    if shutdown_requested(state):
        return
    if torque_off:
        for leader_bot in leader_bots:
            if use_gravity_compensation:
                enable_gravity_compensation(leader_bot)
            else:
                torque_off(leader_bot)
    if verbose:
        print(f"Started!")


def discard_or_save(leader_bots, state: RecordState = None):
    def closed():
        return is_gripper_closed(leader_bots, threshold=LEADER_GRIPPER_CLOSE_THRESH)

    for leader_bot in leader_bots:
        leader_bot.core.robot_torque_enable("single", "gripper", False)

    print("To continue and save data, close the gripper.")
    discard = False
    to_exit = False

    if state.debug:
        print("Debug mode, skipping discard_or_save")
        while not closed() and not shutdown_requested(state):
            time.sleep(DT)
        return False, False

    input_thread = InputThread("Discard/Stop/Exit? (h): ")
    input_thread.start()

    try:
        while not closed() and not shutdown_requested(state):
            input_text = input_thread.get_result()

            if input_text is None:
                time.sleep(DT)
                continue

            if input_text in ["y", "yes", "discard", "d"]:
                input_thread.stop()
                discard = True
                print("Close the gripper to continue...        ")
                while not closed():
                    time.sleep(DT)
                print("gripper close")
                break
            elif input_text in ["q", "quit", "e", "exit"]:
                discard = True
                to_exit = True
                break
            elif input_text in ["s", "stop"]:
                to_exit = True
                break
            elif input_text in ["h", "help", ""]:
                print("Commands:")
                print("  y / yes / discard / d: Discard the data")
                print("  q / quit / e / exit: Exit the program (Discard)")
                print("  s / stop: Save the data before Exit")
                print("  h / help: Show this help message")
            else:
                print("Invalid input.")
            input_thread.clear_result()

        if shutdown_requested(state):
            discard = True

    finally:
        input_thread.clear_result()
        input_thread.stop()

    return discard, to_exit


def init_ros(
    dt: float,
    active_bot: str,
    camera_names: List[str],
    logging_level=LoggingSeverity.WARN,
):
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)

    print("Creating node...")
    node = create_interbotix_global_node("aloha")
    # source of data
    print("Creating leader arms...")
    leader_bot_left = InterbotixManipulatorXS(
        robot_model="wx250s",
        robot_name="leader_left",
        node=node,
        iterative_update_fk=False,
        logging_level=logging_level,
    )
    leader_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s",
        robot_name="leader_right",
        node=node,
        iterative_update_fk=False,
        logging_level=logging_level,
    )

    print("Creating Env...")
    print("Using camera names: ", camera_names)
    env = make_real_env(
        node,
        setup_robots=False,
        setup_base=False,
        arm_mask=ARM_MASKS[active_bot],
        logging_level=logging_level,
        camera_names=camera_names,
        target_fps=(1 / dt) * 0.8,
    )
    print("Robot Startup...")
    robot_startup(node)

    active_leader_bots = []
    active_follower_bots = []
    inactive_leader_bots = []
    inactive_follower_bots = []
    if active_bot == "left":
        active_leader_bots.append(leader_bot_left)
        active_follower_bots.append(env.follower_bot_left)
        inactive_leader_bots.append(leader_bot_right)
        inactive_follower_bots.append(env.follower_bot_right)
    elif active_bot == "right":
        active_leader_bots.append(leader_bot_right)
        active_follower_bots.append(env.follower_bot_right)
        inactive_leader_bots.append(leader_bot_left)
        inactive_follower_bots.append(env.follower_bot_left)
    elif active_bot == "both":
        active_leader_bots.extend([leader_bot_left, leader_bot_right])
        active_follower_bots.extend([env.follower_bot_left, env.follower_bot_right])

    return dict(
        env=env,
        leader_bot_left=leader_bot_left,
        leader_bot_right=leader_bot_right,
        active_leader_bots=active_leader_bots,
        active_follower_bots=active_follower_bots,
        inactive_leader_bots=inactive_leader_bots,
        inactive_follower_bots=inactive_follower_bots,
    )


def start_position(
    active_leader_bots,
    active_follower_bots,
    inactive_leader_bots,
    inactive_follower_bots,
):
    # move arms to starting position
    active_start_arm_qpos = START_ARM_POSE[:6]
    inactive_start_arm_qpos = INACTIVE_START_ARM_POSE[:6]
    num_active_bots = len(active_leader_bots) + len(active_follower_bots)
    num_inactive_bots = len(inactive_leader_bots) + len(inactive_follower_bots)

    move_arms(
        [
            *active_leader_bots,
            *active_follower_bots,
            *inactive_leader_bots,
            *inactive_follower_bots,
        ],
        [active_start_arm_qpos] * num_active_bots
        + [inactive_start_arm_qpos] * num_inactive_bots,
        moving_time=2.5,
    )
    # move grippers to starting position
    move_grippers(
        [
            *active_leader_bots,
            *active_follower_bots,
            *inactive_leader_bots,
            *inactive_follower_bots,
        ],
        # [LEADER_GRIPPER_JOINT_MID, FOLLOWER_GRIPPER_JOINT_CLOSE],
        [LEADER_GRIPPER_JOINT_MID] * len(active_leader_bots)
        + [FOLLOWER_GRIPPER_JOINT_CLOSE] * len(active_follower_bots)
        + [LEADER_GRIPPER_JOINT_OPEN] * len(inactive_leader_bots)
        + [FOLLOWER_GRIPPER_JOINT_OPEN] * len(inactive_follower_bots),
        moving_time=1.5,
    )


def opening_ceremony(
    active_leader_bots,
    active_follower_bots,
    inactive_leader_bots,
    inactive_follower_bots,
):
    """Move all 4 robots to a pose where it is easy to start demonstration"""

    # reboot gripper motors, and set operating modes for all motors
    def set_bots(leader_bot, follower_bot):
        follower_bot.core.robot_reboot_motors("single", "gripper", True)
        follower_bot.core.robot_set_operating_modes("group", "arm", "position")
        follower_bot.core.robot_set_operating_modes(
            "single", "gripper", "current_based_position"
        )
        leader_bot.core.robot_set_operating_modes("group", "arm", "position")
        leader_bot.core.robot_set_operating_modes("single", "gripper", "position")
        # follower_bot.core.robot_set_motor_registers("single", "gripper", 'current_limit', 1000) # TODO(tonyzhaozh) figure out how to set this limit

        torque_on(leader_bot)
        torque_on(follower_bot)

    for leader_bot, follower_bot in zip(active_leader_bots, active_follower_bots):
        set_bots(leader_bot, follower_bot)
    for leader_bot, follower_bot in zip(inactive_leader_bots, inactive_follower_bots):
        set_bots(leader_bot, follower_bot)

    start_position(
        active_leader_bots,
        active_follower_bots,
        inactive_leader_bots,
        inactive_follower_bots,
    )


def end_ceremony(
    active_leader_bots: List[InterbotixManipulatorXS],
    active_follower_bots: List[InterbotixManipulatorXS],
    # inactive_leader_bots,
    # inactive_follower_bots,
):
    for leader_bot, follower_bot in zip(active_leader_bots, active_follower_bots):
        # Torque on leader bot
        torque_on(leader_bot)
        # Open follower gripper
        follower_bot.core.robot_set_operating_modes("single", "gripper", "position")

    move_grippers(
        active_follower_bots,
        [FOLLOWER_GRIPPER_JOINT_OPEN] * len(active_follower_bots),
        moving_time=1.25,
    )


def num_of_anomaly_frames(frames: np.ndarray, method="mse", threshold=1):
    """
    Calculate the number of frames that are different from the previous frame
    frames: np.ndarray, shape (N, H, W, C)
    method: "mse" or "abs"
    threshold: threshold for the difference
    """
    diffs = []
    diffs = frames[1:] - frames[:-1]
    if method == "mse":
        diffs = np.mean(diffs**2, axis=(1, 2, 3))
    elif method == "abs":
        diffs = np.mean(np.abs(diffs), axis=(1, 2, 3))
    else:
        raise ValueError("Unsupported method")

    return len(np.where(diffs < threshold)[0])


def save_dataset(
    camera_names,
    actions: List,
    timesteps: List,
    dataset_path,
    max_timesteps,
    compress=False,
    verbose=False,
):
    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 848, 3) 'uint8'
        - cam_low           (480, 848, 3) 'uint8'
        - cam_left_wrist    (480, 848, 3) 'uint8'
        - cam_right_wrist   (480, 848, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'
    - eepose                (14,)         'float64'

    actions
    - joint_action          (14,)         'float64'
    - delta_eepose          (14,)         'float64'
    - base_action             (2,)          'float64'
    """

    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/effort": [],
        "/actions/joint_action": [],
        "/actions/delta_eepose": [],
        "/actions/base_action": [],
        # '/base_action_t265': [],
    }
    for cam_name in camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []
        # data_dict[f"/observations/images/{cam_name}/depth"] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    if len(actions) + 1 != len(timesteps):
        print(
            f"Warning: len(actions) + 1 != len(timesteps), {len(actions)} + 1 != {len(timesteps)}"
        )
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict["/observations/qpos"].append(ts.observation["qpos"])
        data_dict["/observations/qvel"].append(ts.observation["qvel"])
        data_dict["/observations/effort"].append(ts.observation["effort"])
        data_dict["/actions/joint_action"].append(action)
        data_dict["/actions/delta_eepose"].append(
            timesteps[0].observation[
                "delta_eepose"
            ]  # shift one timestep to make it align with action
        )
        data_dict["/actions/base_action"].append(ts.observation["base_vel"])
        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        for cam_name in camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(
                ts.observation["images"][cam_name]["color"]
            )

    for key in data_dict.keys():
        if key.startswith("/actions/"):
            data_dict[key] = data_dict[key][:-1]  # remove last element
    # HDF5
    t0 = time.time()
    max_chunks = max(
        map(
            lambda resolution: (resolution[0] * resolution[1] * 3 + 1024) // 1024,
            CAMERA_RESOLUTIONS.values(),
        )
    )
    with h5py.File(
        dataset_path + ".temp.hdf5", "w", rdcc_nbytes=max_chunks * 2
    ) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = compress
        root.attrs["compress_method"] = "gzip:1:shuffle"
        obs = root.create_group("observations")
        acts = root.create_group("actions")
        image = obs.create_group("images")
        for cam_name in camera_names:
            resolution = CAMERA_RESOLUTIONS[cam_name]
            if compress:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, *resolution, 3),
                    dtype="uint8",
                    compression=1,
                    shuffle=True,
                )
            else:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, *resolution, 3),
                    dtype="uint8",
                    chunks=(1, *resolution, 3),
                )
        _ = obs.create_dataset("qpos", (max_timesteps, 14))
        _ = obs.create_dataset("qvel", (max_timesteps, 14))
        _ = obs.create_dataset("effort", (max_timesteps, 14))
        _ = acts.create_dataset("joint_action", (max_timesteps - 1, 14))
        _ = acts.create_dataset("delta_eepose", (max_timesteps - 1, 14))
        _ = acts.create_dataset("base_action", (max_timesteps - 1, 2))

        # breakpoint()
        for name, array in data_dict.items():
            if any([val is None for val in array]):
                print(f"Warning: {name} has None values, skipping...")
            root[name][...] = array

    os.rename(dataset_path + ".temp.hdf5", dataset_path + ".hdf5")
    if verbose:
        print(f"Saving: {time.time() - t0:.1f} secs")
    return True


def sleep(bots, moving_time=1):
    print(f"Sleeping...")
    for bot in bots:
        torque_on(bot)
    sleep_arms(bots, moving_time=moving_time, home_first=True)


def get_auto_index(dataset_dir, dataset_name_prefix="", data_suffix="hdf5"):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx + 1):
        if not os.path.isfile(
            os.path.join(dataset_dir, f"{dataset_name_prefix}episode_{i}.{data_suffix}")
        ):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")


def get_episode_idx(episode_idx, save_dir):
    if episode_idx is not None:
        return episode_idx
    else:
        return get_auto_index(save_dir)


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(
        f"Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3e} Step env: {np.mean(step_env_time):.3f}"
    )
    return freq_mean
