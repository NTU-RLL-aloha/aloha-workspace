from typing import List

import argparse
import os

from aloha.constants import DT
import cv2
import h5py
import IPython
import matplotlib.pyplot as plt
import numpy as np

e = IPython.embed

from aloha.constants import JOINT_NAMES

import interbotix_common_modules.angle_manipulation as ang
from interbotix_xs_modules.xs_robot import mr_descriptions as mrd
import modern_robotics as mr

STATE_NAMES = JOINT_NAMES + ["gripper"]
BASE_STATE_NAMES = ["linear_vel", "angular_vel"]


def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        # is_sim = root.attrs['sim']
        compressed = root.attrs.get("compress", False)
        qpos = root["/observations/qpos"][()]
        qvel = root["/observations/qvel"][()]
        if "effort" in root.keys():
            effort = root["/observations/effort"][()]
        else:
            effort = None
        # eepose = root["/observations/eepose"][()]
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
        # eepose=np.array(eepose),
        joint_actions=np.array(joint_actions),
        base_actions=np.array(base_actions),
        delta_ee_pose=np.array(delta_ee_pose),
        cam_images=image_dict,
    )


def visualize_episode(dataset_dir, episode_idx, overwrite=False, ismirror=False):
    if ismirror:
        datafile_name = f"mirror_episode_{episode_idx}"
    else:
        datafile_name = f"episode_{episode_idx}"

    dump_dir = os.path.join(dataset_dir, datafile_name)
    if os.path.exists(dump_dir) and not overwrite:
        print(
            f"Episode {episode_idx} already exists at {dump_dir}. Use --overwrite to overwrite."
        )
        print("Exiting...")
        return

    print(f"Visualizing episode {episode_idx}...")
    os.makedirs(dump_dir, exist_ok=True)
    data = load_hdf5(dataset_dir, datafile_name)
    print("data loaded!")

    save_videos(data["cam_images"], DT, video_path=os.path.join(dump_dir, "video.mp4"))
    visualize_joints(
        data["qpos"],
        data["joint_actions"],
        plot_path=os.path.join(dump_dir, "qpos.png"),
    )
    visualize_delta_ee(
        data["delta_ee_pose"],
        plot_path=os.path.join(dump_dir, "delta_ee.png"),
    )
    # visualize_solved_joints(
    #     data["delta_ee_pose"],
    #     data["qpos"],
    #     plot_path=os.path.join(dump_dir, "solved_joints.png"),
    # )
    # visualize_single(effort, "effort", plot_path=os.path.join(dump_dir, "effort.png"))
    # visualize_single(
    #     action - qpos, "tracking_error", plot_path=os.path.join(dump_dir, "error.png")
    # )
    # visualize_base(base_action, plot_path=os.path.join(dump_dir, "base_action.png"))
    # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back

    print("done!")


def main(args):
    dataset_dir = args["dataset_dir"]
    episode_idx = args["episode_idx"]
    overwrite = args["overwrite"]
    ismirror = args["ismirror"]

    if episode_idx is not None:
        visualize_episode(dataset_dir, episode_idx, overwrite, ismirror)
    else:
        for datafile_name in os.listdir(dataset_dir):
            if datafile_name.endswith(".hdf5"):
                episode_idx = int(datafile_name.split("_")[1].split(".")[0])
                visualize_episode(dataset_dir, episode_idx, overwrite, ismirror)


def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]]  # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f"Saved video to: {video_path}")
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f"Saved video to: {video_path}")


def visualize_joints(
    qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None
):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = "State", "Command"

    qpos = np.array(qpos_list)  # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    # h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = [f"{name}_left" for name in STATE_NAMES] + [
        f"{name}_right" for name in STATE_NAMES
    ]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f"Joint {dim_idx}: {all_names[dim_idx]}")
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved qpos plot to: {plot_path}")
    plt.close()


def solve_joints(T_sd, joints):
    REV: float = 2 * np.pi

    def _wrap_theta_list(theta_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Wrap an array of joint commands to [-pi, pi) and between the joint limits.

        :param theta_list: array of floats to wrap
        :return: array of floats wrapped between [-pi, pi)
        """
        theta_list = (theta_list + np.pi) % REV - np.pi
        # for x in range(len(theta_list)):
        #     if round(theta_list[x], 3) < round(self.group_info.joint_lower_limits[x], 3:
        #         theta_list[x] += REV
        #     elif round(theta_list[x], 3) > round(self.group_info.joint_upper_limits[x], 3:
        #         theta_list[x] -= REV
        return theta_list

    robot_des = getattr(mrd, "vx300s")

    initial_guesses = [joints]
    for guess in initial_guesses:
        theta_list, success = mr.IKinSpace(
            Slist=robot_des.Slist,
            M=robot_des.M,
            T=T_sd,
            thetalist0=guess,
            eomg=0.001,
            ev=0.001,
        )
        solution_found = True

        # Check to make sure a solution was found and that no joint limits were violated
        if success:
            theta_list = _wrap_theta_list(theta_list)
            # solution_found = _check_joint_limits(theta_list)
        else:
            solution_found = False

        if solution_found:
            return theta_list, True

    return theta_list, False


def ee2matrix(ee):
    # Convert end effector pose to transformation matrix
    x, y, z, roll, pitch, yaw = ee
    T_sd = np.identity(4)
    T_sd[:3, :3] = ang.euler_angles_to_rotation_matrix([roll, pitch, yaw])
    T_sd[:3, 3] = [x, y, z]
    return T_sd


def visualize_solved_joints(
    delta_ee_list, joints_list, plot_path=None, ylim=None, label_overwrite=None
):
    INITIAL_EE = np.array(
        [
            0.29542043956822583,
            -0.0006201612977695931,
            0.12199920482801078,
            1.1306219458685354,
            0.9953363753329612,
            -0.7508703869808562,
        ]
    )

    num_ee_dim = 6
    num_joint_dim = 6

    delta_ee_list = np.array(delta_ee_list)  # ts, dim
    joints_list = np.array(joints_list)

    delta_ee_list = np.concatenate([np.zeros((1, num_ee_dim)), delta_ee_list[:, :num_ee_dim]])
    joints_list = joints_list[:, :num_joint_dim]

    T_sd = ee2matrix(INITIAL_EE)
    solve_joints_list = []
    # WARNING: Solved joints are currently incorrect
    for delta_ee, joints in zip(delta_ee_list, joints_list):
        breakpoint()
        T_rel = ee2matrix(delta_ee)
        T_sd = T_rel @ T_sd
        solved_joints, solved = solve_joints(T_sd, joints)
        print(f"joints: {joints}")
        print(f"solved: {solved_joints}")
        print(f"error: {solved_joints - joints}")
        print()
        if not solved:
            raise ValueError("No solution found")
        solve_joints_list.append(solved_joints[0])

    label = "State"

    solve_joints_list = np.array(solve_joints_list)  # ts, dim
    num_ts, num_dim = solve_joints_list.shape
    # h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, num_dim))

    # plot joint state
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(solve_joints_list[:, dim_idx], label=label)
        ax.set_title(f"Joint {dim_idx}: {STATE_NAMES[dim_idx]}")
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved solved joints plot to: {plot_path}")
    plt.close()


def visualize_delta_ee(delta_ee_list, plot_path=None, ylim=None, label_overwrite=None):

    delta_ee = np.array(delta_ee_list)  # ts, dim
    num_dim = 7
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(delta_ee[:, dim_idx])
        ax.set_title(f"Delta EE {dim_idx}: {all_names[dim_idx]}")
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved delta ee plot to: {plot_path}")
    plt.close()


def visualize_single(
    efforts_list, label, plot_path=None, ylim=None, label_overwrite=None
):
    efforts = np.array(efforts_list)  # ts, dim
    num_ts, num_dim = efforts.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + "_left" for name in STATE_NAMES] + [
        name + "_right" for name in STATE_NAMES
    ]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(efforts[:, dim_idx], label=label)
        ax.set_title(f"Joint {dim_idx}: {all_names[dim_idx]}")
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved effort plot to: {plot_path}")
    plt.close()


def visualize_base(readings, plot_path=None):
    readings = np.array(readings)  # ts, dim
    num_ts, num_dim = readings.shape
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = BASE_STATE_NAMES
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(readings[:, dim_idx], label="raw")
        ax.plot(
            np.convolve(readings[:, dim_idx], np.ones(20) / 20, mode="same"),
            label="smoothed_20",
        )
        ax.plot(
            np.convolve(readings[:, dim_idx], np.ones(10) / 10, mode="same"),
            label="smoothed_10",
        )
        ax.plot(
            np.convolve(readings[:, dim_idx], np.ones(5) / 5, mode="same"),
            label="smoothed_5",
        )
        ax.set_title(f"Joint {dim_idx}: {all_names[dim_idx]}")
        ax.legend()

    # if ylim:
    #     for dim_idx in range(num_dim):
    #         ax = axs[dim_idx]
    #         ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved effort plot to: {plot_path}")
    plt.close()


def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace(".pkl", "_timestamp.png")
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h * 2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10e-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f"Camera frame timestamps")
    ax.set_xlabel("timestep")
    ax.set_ylabel("time (sec)")

    ax = axs[1]
    ax.plot(np.arange(len(t_float) - 1), t_float[:-1] - t_float[1:])
    ax.set_title(f"dt")
    ax.set_xlabel("timestep")
    ax.set_ylabel("time (sec)")

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved timestamp plot to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        action="store",
        type=str,
        help="Dataset dir.",
        required=True,
    )
    parser.add_argument(
        "--episode_idx",
        action="store",
        type=int,
        help="Episode index.",
        required=False,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing episode.",
    )
    parser.add_argument("--ismirror", action="store_true")
    main(vars(parser.parse_args()))
