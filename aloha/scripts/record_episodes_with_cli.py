from pdb import set_trace
from pprint import pprint
import argparse
import os
import time
from typing import Dict
import cv2
import h5py

# import h5py_cache
import IPython
import numpy as np
from aloha.constants import (
    DT,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    FOLLOWER_GRIPPER_JOINT_OPEN,
    FPS,
    IS_MOBILE,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEADER_GRIPPER_JOINT_MID,
    LEADER_GRIPPER_JOINT_OPEN,
    START_ARM_POSE,
    SLEEP_ARM_POSE,
    DATA_DIR,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from aloha.real_env import RealEnv, get_action, make_real_env
from aloha.robot_utils import (
    ImageRecorder,
    Recorder,
    enable_gravity_compensation,
    get_arm_gripper_positions,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
    sleep_arms,
)

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tqdm import tqdm
from utils.AsyncQueueProcessor import AsyncQueueProcessor
from utils.cli import InputThread
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)

e = IPython.embed

TASK_CONFIGS = {
    "aloha_test_move_water": {
        "dataset_dir": DATA_DIR + "/aloha_test_move_water",
        "episode_len": 600,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "active_bot": "both",
    },
    "aloha_test": {
        "dataset_dir": DATA_DIR + "/aloha_test",
        "episode_len": 600,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "active_bot": "both",
    },
    "aloha_ise_bar": {
        "dataset_dir": DATA_DIR + "/aloha_ise_bar",
        "episode_len": 800,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "active_bot": "left",
    },
    "aloha_ise_door": {
        "dataset_dir": DATA_DIR + "/aloha_ise_door",
        "episode_len": 800,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "active_bot": "left",
    },
    "fd_button_press_topdown_narrow": {
        "dataset_dir": DATA_DIR + "/flow_decomp/button_press_topdown_narrow",
        "episode_len": 500,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "active_bot": "left",
    },
    "fd_button_press_topdown": {
        "dataset_dir": DATA_DIR + "/flow_decomp/button_press_topdown",
        "episode_len": 500,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "active_bot": "left",
    },
    "fd_assembly": {
        "dataset_dir": DATA_DIR + "/flow_decomp/assembly",
        "episode_len": 600,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "active_bot": "left",
    },
    "fd_pick_and_place": {
        "dataset_dir": DATA_DIR + "/flow_decomp/pick_and_place",
        "episode_len": 600,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "active_bot": "left",
    },
}


INACTIVE_START_ARM_POSE = SLEEP_ARM_POSE
ARM_MASKS = {
    "left": [True, False],
    "right": [False, True],
    "both": [True, True],
}


def is_gripper_closed(leader_bots, threshold=LEADER_GRIPPER_CLOSE_THRESH):
    gripper_poses = [get_arm_gripper_positions(bot) for bot in leader_bots]
    return all([gripper_pose < threshold for gripper_pose in gripper_poses])


def wait_for_start(leader_bots, use_gravity_compensation=False, verbose=True):
    # press gripper to start data collection
    # disable torque for only gripper joint of leader robot to allow user movement
    for leader_bot in leader_bots:
        leader_bot.core.robot_torque_enable("single", "gripper", False)
    if verbose:
        print(f"Close the gripper to start")

    close_thresh = LEADER_GRIPPER_CLOSE_THRESH
    while not is_gripper_closed(leader_bots, threshold=close_thresh):
        time.sleep(DT / 10)
    for leader_bot in leader_bots:
        if use_gravity_compensation:
            enable_gravity_compensation(leader_bot)
        else:
            torque_off(leader_bot)
    if verbose:
        print(f"Started!")


def discard_or_save(leader_bots):
    for leader_bot in leader_bots:
        leader_bot.core.robot_torque_enable("single", "gripper", False)

    print("To continue and save data, close the gripper.")
    discard = False
    to_exit = False

    input_thread = InputThread("Discard/Stop/Exit? (h): ")
    input_thread.start()

    def closed():
        return is_gripper_closed(leader_bots, threshold=LEADER_GRIPPER_CLOSE_THRESH)

    try:
        while not closed():
            input_text = input_thread.get_result()

            if input_text is None:
                time.sleep(DT / 10)
                continue

            if input_text in ["y", "yes", "discard", "d"]:
                input_thread.stop()
                discard = True
                print("Close the gripper to continue...        ")
                while not closed():
                    time.sleep(DT / 10)
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
    finally:
        input_thread.clear_result()
        input_thread.stop()

    return discard, to_exit


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


def capture_episodes(
    dt,
    max_timesteps,
    active_bot,
    camera_names,
    dataset_dir,
    dataset_name_template: str,
    base_count=0,
    overwrite=False,
    num_episodes=3,
    use_gravity_compensation=False,
):
    print(f'Saving Dataset to "{dataset_dir}"')

    node = create_interbotix_global_node("aloha")

    # source of data
    leader_bot_left = InterbotixManipulatorXS(
        robot_model="wx250s",
        robot_name="leader_left",
        node=node,
        iterative_update_fk=False,
    )
    leader_bot_right = InterbotixManipulatorXS(
        robot_model="wx250s",
        robot_name="leader_right",
        node=node,
        iterative_update_fk=False,
    )
    env = make_real_env(
        node, setup_robots=False, setup_base=False, arm_mask=ARM_MASKS[active_bot]
    )
    robot_startup(node)

    # saving dataset
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

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

    # move all 4 robots to a starting pose where it is easy to start teleoperation, then wait till both gripper closed
    opening_ceremony(
        active_leader_bots,
        active_follower_bots,
        inactive_leader_bots,
        inactive_follower_bots,
    )
    # env = make_real_env(init_node=False, setup_robots=False)
    counter = 0

    def save_dataset_wrapper(args):
        try:
            save_dataset(*args)
        except Exception as e:
            print(f"Error saving dataset with args: ")
            pprint(args)
            print(f"{e}")

    start_position(
        active_leader_bots,
        active_follower_bots,
        inactive_leader_bots,
        inactive_follower_bots,
    )
    print("leader_bot_left ee pose: \n", leader_bot_left.arm.get_ee_pose())

    wait_for_start(
        active_leader_bots, use_gravity_compensation=use_gravity_compensation
    )
    saving_worker = AsyncQueueProcessor(2, save_dataset_wrapper)
    while counter < num_episodes:
        dataset_name = dataset_name_template.format(episode_idx=base_count + counter)
        dataset_path = os.path.join(dataset_dir, dataset_name)
        if os.path.isfile(dataset_path) and not overwrite:
            print(
                f"Dataset already exist at \n{dataset_path}\nHint: set overwrite to True."
            )
            exit()

        is_healthy, timesteps, actions, freq_mean = capture_one_episode(
            dt,
            max_timesteps,
            camera_names,
            env,
            leader_bot_left,
            leader_bot_right,
            active_leader_bots,
            active_follower_bots,
            desc=f"[ {counter}/{num_episodes} ]",
            use_gravity_compensation=use_gravity_compensation,
        )
        time.sleep(0.5)
        start_position(
            active_leader_bots,
            active_follower_bots,
            inactive_leader_bots,
            inactive_follower_bots,
        )

        if not is_healthy:
            print(
                f"\n\nFreq_mean = {freq_mean}, lower than 30, re-collecting... \n\n\n\n"
            )
            continue
        discard, to_exit = discard_or_save(active_leader_bots)

        # breakpoint()
        if discard:
            print(f"Discard dataset.")
        else:
            try:
                saving_worker.add_data(
                    (
                        camera_names,
                        actions,
                        timesteps,
                        dataset_path,
                        max_timesteps,
                        True,  # compress
                    )
                )
                counter += 1
            except Exception as e:
                print(f"Error saving dataset: {e}\n\nre-collecting... \n\n\n\n")
        if to_exit:
            print(f"Exiting...")
            break
    sleep(
        # leader_bot_left,
        # leader_bot_right,
        # env.follower_bot_left,
        # env.follower_bot_right,
        active_leader_bots + active_follower_bots,
        moving_time=2,
    )
    robot_shutdown()
    saving_worker.join()


def end_ceremony(
    active_leader_bots,
    active_follower_bots,
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


def capture_one_episode(
    dt,
    max_timesteps: int,
    camera_names: list,
    env: RealEnv,
    leader_bot_left,
    leader_bot_right,
    active_leader_bots,
    active_follower_bots,
    desc: str = None,
    use_gravity_compensation=False,
):
    # Data collection
    ts = env.reset(fake=True)
    timesteps = [ts]
    actions = []
    actual_dt_history = []
    time0 = time.time()
    DT = 1 / FPS

    for leader_bot in active_leader_bots:
        if use_gravity_compensation:
            enable_gravity_compensation(leader_bot)
        else:
            torque_off(leader_bot)

    for t in tqdm(range(max_timesteps), desc=desc):
        t0 = time.time()  #
        action = get_action(leader_bot_left, leader_bot_right)
        t1 = time.time()  #
        ts = env.step(action)
        t2 = time.time()  #
        timesteps.append(ts)
        actions.append(action)
        actual_dt_history.append([t0, t1, t2])
        time.sleep(max(0, DT - (time.time() - t0)))
    print(f"Avg fps: {max_timesteps / (time.time() - time0)}")

    end_ceremony(
        # leader_bot_left, leader_bot_right, env.follower_bot_left, env.follower_bot_right
        active_leader_bots,
        active_follower_bots,
    )

    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 30:
        # not healthy
        return False, None, None, freq_mean
    return True, timesteps, actions, freq_mean


def save_dataset(
    camera_names,
    actions,
    timesteps,
    dataset_path,
    max_timesteps,
    compress=True,
    verbose=False,
):
    """
    For each timestep:
    observations
    - images
        - cam_high          (480, 640, 3) 'uint8'
        - cam_low           (480, 640, 3) 'uint8'
        - cam_left_wrist    (480, 640, 3) 'uint8'
        - cam_right_wrist   (480, 640, 3) 'uint8'
    - qpos                  (14,)         'float64'
    - qvel                  (14,)         'float64'

    action                  (14,)         'float64'
    base_action             (2,)          'float64'
    """

    # breakpoint()
    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/effort": [],
        "/action": [],
        "/base_action": [],
        # '/base_action_t265': [],
    }
    for cam_name in camera_names:
        data_dict[f"/observations/images/{cam_name}"] = []
        # data_dict[f"/observations/images/{cam_name}/depth"] = []

    # len(action): max_timesteps, len(time_steps): max_timesteps + 1
    while actions:
        action = actions.pop(0)
        ts = timesteps.pop(0)
        data_dict["/observations/qpos"].append(ts.observation["qpos"])
        data_dict["/observations/qvel"].append(ts.observation["qvel"])
        data_dict["/observations/effort"].append(ts.observation["effort"])
        data_dict["/action"].append(action)
        data_dict["/base_action"].append(ts.observation["base_vel"])
        # data_dict['/base_action_t265'].append(ts.observation['base_vel_t265'])
        for cam_name in camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(
                ts.observation["images"][cam_name]["color"]
            )
    if compress:
        # JPEG compression
        t0 = time.time()
        encode_param = [
            int(cv2.IMWRITE_JPEG_QUALITY),
            50,
        ]  # tried as low as 20, seems fine
        compressed_len = []
        for cam_name in camera_names:
            image_list = data_dict[f"/observations/images/{cam_name}"]
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                result, encoded_image = cv2.imencode(
                    ".jpg", image, encode_param
                )  # 0.02 sec # cv2.imdecode(encoded_image, 1)
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))
            data_dict[f"/observations/images/{cam_name}"] = compressed_list
        if verbose:
            print(f"compression: {time.time() - t0:.2f}s")

        # pad so it has same length
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f"/observations/images/{cam_name}"]
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype="uint8")
                image_len = len(compressed_image)
                padded_compressed_image[:image_len] = compressed_image
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f"/observations/images/{cam_name}"] = padded_compressed_image_list
        if verbose:
            print(f"padding: {time.time() - t0:.2f}s")

    # HDF5
    t0 = time.time()
    with h5py.File(dataset_path + ".temp.hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = compress
        obs = root.create_group("observations")
        image = obs.create_group("images")
        for cam_name in camera_names:
            if compress:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, padded_size),
                    dtype="uint8",
                    chunks=(1, padded_size),
                )
            else:
                _ = image.create_dataset(
                    cam_name,
                    (max_timesteps, 480, 640, 3),
                    dtype="uint8",
                    chunks=(1, 480, 640, 3),
                )
        _ = obs.create_dataset("qpos", (max_timesteps, 14))
        _ = obs.create_dataset("qvel", (max_timesteps, 14))
        _ = obs.create_dataset("effort", (max_timesteps, 14))
        _ = root.create_dataset("action", (max_timesteps, 14))
        _ = root.create_dataset("base_action", (max_timesteps, 2))

        # breakpoint()
        for name, array in data_dict.items():
            root[name][...] = array

        if compress:
            _ = root.create_dataset("compress_len", (len(camera_names), max_timesteps))
            root["/compress_len"][...] = compressed_len

    os.rename(dataset_path + ".temp.hdf5", dataset_path + ".hdf5")
    if verbose:
        print(f"Saving: {time.time() - t0:.1f} secs")
    return True


def main(args: Dict):
    task_config = TASK_CONFIGS[args["task_name"]]
    use_gravity_compensation = args["use_gravity_compensation"]
    dataset_dir = task_config["dataset_dir"]
    max_timesteps = task_config["episode_len"]
    camera_names = task_config["camera_names"]
    active_bot = task_config.get("active_bot", "both")

    inactive_bot_pos = task_config.get("inactive_bot_pos", None)
    if inactive_bot_pos is not None and len(inactive_bot_pos) == len(START_ARM_POSE):
        global INACTIVE_START_ARM_POSE
        INACTIVE_START_ARM_POSE = inactive_bot_pos

    num_episodes = args.get("num_episodes", 3)
    if args["episode_idx"] is not None:
        episode_idx = args["episode_idx"]
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    capture_episodes(
        DT,
        max_timesteps,
        active_bot,
        camera_names,
        dataset_dir,
        "episode_{episode_idx}",
        base_count=episode_idx,
        overwrite=overwrite,
        num_episodes=num_episodes,
        use_gravity_compensation=use_gravity_compensation,
    )


def sleep(
    bots,
    # leader_bot_left,
    # leader_bot_right,
    # follower_bot_left,
    # follower_bot_right,
    moving_time=1,
):
    for bot in bots:
        torque_on(bot)
    sleep_arms(bots, home_first=True)

    time.sleep(moving_time)
    # follower_sleep_position = (0, -1.7, 1.55, 0, 0.65, 0)
    # leader_sleep_left_position = (-0.61, 0.0, 0.43, 0.0, 1.04, -0.65)
    # leader_sleep_right_position = (0.61, 0.0, 0.43, 0.0, 1.04, 0.65)
    # all_positions = [follower_sleep_position] * 2 + [
    #     leader_sleep_left_position,
    #     leader_sleep_right_position,
    # ]
    # move_arms(all_bots, all_positions, moving_time=moving_time)

    # leader_sleep_left_position_2 = (0.0, 0.66, -0.27, -0.0, 1.1, 0)
    # leader_sleep_right_position_2 = (0.0, 0.66, -0.27, -0.0, 1.1, 0)
    # move_arms(
    #     leader_bots,
    #     [leader_sleep_left_position_2, leader_sleep_right_position_2],
    #     moving_time=moving_time,
    # )


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


def print_dt_diagnosis(actual_dt_history):
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]

    dt_mean = np.mean(total_time)
    dt_std = np.std(total_time)
    freq_mean = 1 / dt_mean
    print(
        f"Avg freq: {freq_mean:.2f} Get action: {np.mean(get_action_time):.3f} Step env: {np.mean(step_env_time):.3f}"
    )
    return freq_mean


def debug():
    print(f"====== Debug mode ======")
    recorder = Recorder("right", is_debug=True)
    image_recorder = ImageRecorder(init_node=False, is_debug=True)
    while True:
        time.sleep(1)
        recorder.print_diagnostics()
        image_recorder.print_diagnostics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name", action="store", type=str, help="Task name.", required=True
    )
    parser.add_argument(
        "--episode_idx",
        action="store",
        type=int,
        help="Episode index.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--num_episodes",
        "-n",
        action="store",
        type=int,
        help="Number of episodes to collect.",
        default=3,
        required=False,
    )
    parser.add_argument(
        "--use_gravity_compensation",
        action="store_true",
        help="Activate gravity compensation",
        default=False,
        required=False,
    )
    main(vars(parser.parse_args()))
