from typing import List, Tuple, Optional

import os
import time
import sys
import signal

import traceback
from collections import deque
from einops import rearrange
from pprint import pprint
from tqdm import tqdm

import dm_env
import numpy as np
import cv2
import torch

import rclpy
from rclpy.logging import LoggingSeverity

from interbotix_common_modules.common_robot.robot import robot_shutdown

from aloha.real_env import RealEnv
from aloha.record_utils.AsyncQueueProcessor import AsyncQueueProcessor
from aloha.record_utils import (
    RecordState,
    shutdown_requested,
    init_ros,
    wait_for_start,
    discard_or_save,
    start_position,
    opening_ceremony,
    end_ceremony,
    num_of_anomaly_frames,
    save_dataset,
    sleep,
    get_episode_idx,
    print_dt_diagnosis,
    LOGGING,
)

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from quest.algos.base import Policy
import quest.utils.utils as utils


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


record_state = RecordState()


def shutdown_handler(signum, frame):
    global record_state
    print(f"Received signal {signum}.")
    if not shutdown_requested(record_state):
        record_state.shutdown_requested = True
        print("Shutdown signal received. Exiting...")


signal.signal(signal.SIGINT, shutdown_handler)


def rollout_episodes(
    dt: float,
    max_timesteps: int,
    active_bot: str,
    camera_names: List[str],
    dataset_dir: str,
    dataset_name_template: str,
    model: Policy,
    crop: Optional[Tuple[int]] = None,
    resize: Optional[int] = None,
    base_count: int = 0,
    overwrite: bool = False,
    num_episodes: int = 3,
    compress: bool = False,
    save_verbose: bool = False,
    logging_level=LoggingSeverity.WARN,
    eps=None,  # for validation
):
    ros_init_dict = init_ros(dt, active_bot, camera_names, logging_level=logging_level)
    env = ros_init_dict["env"]
    leader_bot_left = ros_init_dict["leader_bot_left"]
    leader_bot_right = ros_init_dict["leader_bot_right"]
    active_leader_bots = ros_init_dict["active_leader_bots"]
    active_follower_bots = ros_init_dict["active_follower_bots"]
    inactive_leader_bots = ros_init_dict["inactive_leader_bots"]
    inactive_follower_bots = ros_init_dict["inactive_follower_bots"]

    # saving dataset
    print(f'Saving Dataset to "{dataset_dir}"')
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    # move all 4 robots to a starting pose where it is easy to start teleoperation, then wait till both gripper closed
    print(f"Hi! The data will be saved to {dataset_dir}")
    opening_ceremony(
        active_leader_bots,
        active_follower_bots,
        inactive_leader_bots,
        inactive_follower_bots,
    )
    # env = make_real_env(init_node=False, setup_robots=False)
    counter = 0

    def save_dataset_wrapper(*args, **kwargs):
        try:
            save_dataset(*args, **kwargs)
        except Exception as e:
            print(f"Error saving dataset with args: ")
            pprint(args)
            pprint(kwargs)
            print(f"Error: {e}")
            traceback.print_exc()

    saving_worker = AsyncQueueProcessor(2, save_dataset_wrapper)
    try:
        wait_for_start(active_leader_bots, state=record_state)
        while counter < num_episodes and not shutdown_requested(record_state):
            dataset_name = dataset_name_template.format(
                episode_idx=base_count + counter
            )
            dataset_path = os.path.join(dataset_dir, dataset_name)
            if os.path.isfile(dataset_path) and not overwrite:
                print(
                    f"Dataset already exist at \n{dataset_path}\nHint: set overwrite to True."
                )
                exit()

            is_healthy, timesteps, actions, stats = eval_one_episode(
                dt,
                max_timesteps,
                env,
                model,
                active_leader_bots,
                active_follower_bots,
                desc=f"[ {counter}/{num_episodes} ]",
                camera_names=camera_names,
                crop=crop,
                resize=resize,
                eps=eps,
            )
            freq_mean, anomaly_frames_count = stats
            time.sleep(0.5)
            start_position(
                active_leader_bots,
                active_follower_bots,
                inactive_leader_bots,
                inactive_follower_bots,
            )

            if not is_healthy or anomaly_frames_count > 0:
                print(
                    f"\n\n\tFreq_mean = {freq_mean}\n\tAnomaly frames count = {anomaly_frames_count}"
                )
                if not is_healthy:
                    print(f"\tDataset is unhealthy, re-collecting...")
                print("\n\n")
            discard, to_exit = discard_or_save(active_leader_bots, state=record_state)

            if not is_healthy:
                discard = True
            # breakpoint()
            if discard:
                print(f"\033[31mDiscard dataset.\033[0m")
            else:
                try:
                    saving_worker.add_data(
                        camera_names,
                        actions,
                        timesteps,
                        dataset_path,
                        max_timesteps,
                        compress=compress,
                        verbose=save_verbose,
                    )
                    counter += 1
                except Exception as e:
                    print(f"Error saving dataset: {e}\n\nre-collecting... \n\n\n\n")

            if to_exit:
                print(f"Exiting...")
                break
    finally:
        print("\033[0K\rShutting down... Please wait.")
        if not rclpy.ok():
            print("rclpy is not ok, shutting down...")
            robot_shutdown()
            return
        sleep(
            active_leader_bots + active_follower_bots,
            moving_time=2,
        )
        robot_shutdown()
        saving_worker.join()


def eval_one_episode(
    dt,
    max_timesteps: int,
    env: RealEnv,
    model: Policy,
    active_leader_bots,
    active_follower_bots,
    camera_names: List[str] = ["cam_high"],
    crop: Optional[Tuple[int]] = None,
    resize: Optional[int] = None,
    desc: str = None,
    eps=None,  # for validation
):
    # Data collection
    ts = env.reset(fake=True)  # TODO: make env take active and inactive arms
    timesteps = [ts]
    excuted_actions = []
    actual_dt_history = []

    actions = deque(maxlen=500)
    time0 = timer = time.time()
    for t in tqdm(range(max_timesteps), desc=desc):
        if record_state.shutdown_requested:
            print("Shutdown requested. Exiting...")
            break
        time_delta = time.time() - timer
        while time_delta < dt:
            if time_delta < 0.7 * dt:
                time.sleep(dt * 0.7)
            else:
                time.sleep(dt * 0.01)
            time_delta = time.time() - timer

        if len(actions) < 1 and eps is None:
            t0 = time.time()  #
            action = get_action(
                timesteps,
                model,
                horizon=10,
                cam_names=camera_names,
                crop=crop,
                resize=resize,
            )
            action = np.concatenate([action[:, :6] / 10, action[:, -1:]], axis=1)
            actions.extend(action)
            t1 = time.time()  #
        elif eps is not None and len(actions) < 1:
            t0 = time.time()
            t1 = time.time()
            actions.extend(eps["delta_ee_pose"])

        action = actions.popleft()
        action = np.concatenate(
            [action[:6], action[6][None], np.zeros((7,), dtype=np.float32)], axis=0
        )
        ts = env.step(action, use_delta_ee=True, moving_time=dt)
        t2 = time.time()  #
        timesteps.append(ts)
        excuted_actions.append(action)
        actual_dt_history.append([t0, t1, t2])
        timer = timer + dt

    print(f"Avg fps: {max_timesteps / (time.time() - time0)}")

    end_ceremony(
        active_leader_bots,
        active_follower_bots,
    )

    freq_mean = print_dt_diagnosis(actual_dt_history)
    try:
        anomaly_frames_count = num_of_anomaly_frames(
            np.array(
                [ts.observation["images"]["cam_high"]["color"] for ts in timesteps]
            ),
        )
    except Exception as e:
        print(f"Error calculating anomaly frames: {e}")
        anomaly_frames_count = len(timesteps)
    unhealthy = freq_mean < (1 / dt) or anomaly_frames_count > 10
    return not unhealthy, timesteps, actions, (freq_mean, anomaly_frames_count)


def get_action(
    ts_history: List[dm_env.TimeStep],
    model: Policy,
    horizon: int = 1,
    task_id: int = 0,
    cam_names: List[str] = ["cam_high"],
    crop: Optional[Tuple[int]] = None,
    resize: Optional[int] = None,
):
    raw_observation = {}

    min_horizon = min(horizon, len(ts_history))
    for cam_name in cam_names:
        raw_observation[cam_name] = np.array(
            [
                ts.observation["images"][cam_name]["color"]
                for ts in ts_history[-min_horizon:]
            ]
        )
        if crop is not None:
            raw_observation[cam_name] = raw_observation[cam_name][
                :, crop[0] : crop[2], crop[1] : crop[3]
            ]
        if resize is not None:
            raw_observation[cam_name] = cv2.resize(
                raw_observation[cam_name][0],
                (resize, resize),
                interpolation=cv2.INTER_LINEAR,
            )[None]
        # print(f"raw_observation[cam_name].shape: {raw_observation[cam_name].shape}")
    observation = {}
    observation["images"] = np.concatenate(
        [raw_observation[cam_name] for cam_name in cam_names], axis=2
    )
    # print(f"observation['images'].shape: {observation['images'].shape}")
    # observation["images"] = rearrange(observation["images"], "c t ... -> t c ...")
    if observation["images"].shape[0] < horizon:
        observation["images"] = np.concatenate(
            [
                observation["images"],
                observation["images"][-1:, :].repeat(
                    horizon - observation["images"].shape[0], axis=0
                ),
            ]
        )
    # print(f"observation['images'].shape: {observation['images'].shape}")
    observation["images"] = rearrange(observation["images"], "t h w c -> 1 t h w c")
    # save_img(observation["images"][0, 0], "test.png")
    actions = model.get_action(observation, task_id)
    actions = np.concatenate(
        [np.clip(actions[:, :6], -0.3, 0.3), actions[:, -1:]], axis=1
    )
    return actions


def save_img(img, path):
    if img.shape[2] != 3:
        img = rearrange(img, "c h w -> h w c")
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


OmegaConf.register_new_resolver("eval", eval, replace=True)


def get_arg_tuple(arg):
    if arg is not None:
        arg = tuple(map(int, arg.split(",")))
        if len(arg) == 1:
            arg = (arg[0], arg[0])
    return arg


def load_hdf5(dataset_dir, dataset_name):
    import h5py

    dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        # is_sim = root.attrs['sim']
        qpos = root["/observations/qpos"][()]
        qvel = root["/observations/qvel"][()]
        if "effort" in root.keys():
            effort = root["/observations/effort"][()]
        else:
            effort = None
        joint_actions = root["/actions/joint_action"][()]
        base_actions = root["/actions/base_action"][()]
        delta_ee_pose = root["/actions/delta_eepose"][()]
        image_dict = {}
        for cam_name in root[f"/observations/images/"].keys():
            image_dict[cam_name] = root[f"/observations/images/{cam_name}"][()]

    return dict(
        qpos=np.array(qpos),
        qvel=np.array(qvel),
        effort=np.array(effort),
        joint_actions=np.array(joint_actions),
        base_actions=np.array(base_actions),
        delta_ee_pose=np.array(delta_ee_pose),
        cam_images=image_dict,
    )


@hydra.main(config_path="config", config_name="evaluate", version_base=None)
def main(cfg):
    global record_state
    record_state.debug = cfg.debug

    ## Model configurations
    # model_path = args.get("model_path", None)
    # if model_path is not None:
    #     if not os.path.isfile(model_path):
    #         print(f"Model path does not exist: {model_path}")
    #         exit()
    #     model = BCTransformerPolicy.load(model_path)
    #     print(f"Loaded model from {model_path}")

    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    OmegaConf.resolve(cfg)

    # input_type = (
    #     cfg.task["dataset"]["keys"]
    #     if "keys" in cfg.task["dataset"]
    #     else cfg.task["dataset"]["dataset_keys"]
    # )

    # create model
    save_dir, _ = utils.get_experiment_dir(cfg, evaluate=True)
    os.makedirs(save_dir)

    if cfg.checkpoint_path is None:
        # Basically if you don't provide a checkpoint path it will automatically find one corresponding
        # to the experiment/variant name you provide
        checkpoint_path, _ = utils.get_experiment_dir(
            cfg, evaluate=False, allow_overlap=True
        )
        checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = utils.get_latest_checkpoint(cfg.checkpoint_path)
    state_dict = utils.load_state(checkpoint_path)
    if "config" in state_dict:
        print("autoloading based on saved parameters")
        model = instantiate(
            state_dict["config"]["algo"]["policy"], shape_meta=cfg.task.shape_meta
        )
    else:
        print(cfg.algo.policy)
        model = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.eval()

    model.load_state_dict(state_dict["model"])

    ## ALOHA configurations
    fps = cfg.fps
    crop = get_arg_tuple(cfg.crop)
    resize = cfg.resize

    max_timesteps = cfg.episode_len
    camera_names = cfg.camera_names
    active_bot = cfg.active_bot
    logging_level = LOGGING["default"]

    num_episodes = cfg.num_episodes
    episode_idx = get_episode_idx(cfg.episode_idx, save_dir)
    overwrite = cfg.overwrite

    compress = cfg.compress
    save_verbose = cfg.save_verbose

    # inactive_bot_pos = task_config.get("inactive_bot_pos", None)
    # if inactive_bot_pos is not None and len(inactive_bot_pos) == len(START_ARM_POSE):
    #     global INACTIVE_START_ARM_POSE
    #     INACTIVE_START_ARM_POSE = inactive_bot_pos
    eps = load_hdf5(
        "/home/aloha/aloha_data/flow_decomp/pick_and_place_black/", "episode_10"
    )

    rollout_episodes(
        1 / fps,
        max_timesteps,
        active_bot,
        camera_names,
        save_dir,
        "episode_{episode_idx}",
        model,
        crop=crop,
        resize=resize,
        base_count=episode_idx,
        overwrite=overwrite,
        num_episodes=num_episodes,
        compress=compress,
        save_verbose=save_verbose,
        logging_level=logging_level,
        eps=eps,
    )


if __name__ == "__main__":
    main()
