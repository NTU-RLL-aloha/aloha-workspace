import numpy as np
from typing import Dict
import h5py
from pathlib import Path
import cv2
import contextlib


@contextlib.contextmanager
def timer_print(msg: str):
    import time

    start = time.time()
    yield
    end = time.time()
    print(f"{msg}: {end - start:.2f} seconds")


non_compress_datapath = Path(__file__).parent / "test_data" / "non_compress.hdf5"

non_compress_datapath.parent.mkdir(parents=True, exist_ok=True)

is_sim = False
compress = True
camera_names = [
    "camera_0",
    "camera_1",
    "camera_2",
]

T = 500
H = 480
W = 640

A = 14
B = 2
fake_qpos = np.random.rand(T, A).astype(np.float32)
fake_qvel = np.random.rand(T, A).astype(np.float32)
fake_joint_action = np.random.rand(T - 1, A).astype(np.float32)
fake_delta_eepose = np.random.rand(T - 1, A).astype(np.float32)
fake_base_action = np.random.rand(T - 1, B).astype(np.float32)

data_dict: Dict[str, np.ndarray] = {
    "/observations/qpos": fake_qpos,
    "/observations/qvel": fake_qvel,
    "/actions/joint_action": fake_joint_action,
    "/actions/delta_eepose": fake_delta_eepose,
    "/actions/base_action": fake_base_action,
}

image = cv2.imread(str(non_compress_datapath.parent / "cam_high_rgb.png"))
cropped_image = image[0:H, 0:W, :]
for cam_name in camera_names:
    fake_video = np.repeat(cropped_image[np.newaxis, ...], T, axis=0)
    data_dict[f"/observations/videos/{cam_name}"] = fake_video

for shuffle in [True, False]:
    for c in [1, 2, 5, 8, 9]:
        filename = f"gzip{c}"
        if not shuffle:
            filename = f"no_shuffle_gzip{c}"
        print(f"========== {filename} ===========")
        datapath = Path(__file__).parent / "test_data" / f"{filename}.hdf5"
        with h5py.File(str(datapath), "w") as root:
            root.attrs["sim"] = False
            root.attrs["compress"] = compress
            obs = root.create_group("observations")
            acts = root.create_group("actions")
            videos = obs.create_group("videos")
            for cam_name in camera_names:
                _ = videos.create_dataset(
                    cam_name,
                    (T, H, W, 3),
                    dtype="uint8",
                    compression=c,
                    shuffle=shuffle,
                    # # this is will take a big time/memory overhead, but should be faster when
                    # # reading the data in this pattern
                    # chunks=(8, H, W, 3),
                )
            _ = obs.create_dataset("qpos", (T, A))
            _ = obs.create_dataset("qvel", (T, A))
            _ = acts.create_dataset("joint_action", (T - 1, A))
            _ = acts.create_dataset("delta_eepose", (T - 1, A))
            _ = acts.create_dataset("base_action", (T - 1, B))

            # breakpoint()
            for name, array in data_dict.items():
                with timer_print(f"writing {name}"):
                    root[name][...] = array

        # verify the saved videos
        with h5py.File(str(datapath), "r") as root:
            for cam_name in camera_names:
                with timer_print(f"reading {cam_name}"):
                    video = root["observations/videos"][cam_name][...]
                    assert video.shape == (T, H, W, 3)
                    assert video.dtype == np.uint8
                    assert np.all(video[0] == cropped_image)
                    assert np.all(video[-1] == cropped_image)
                # save the video to a file
                video_path = (
                    Path(__file__).parent
                    / "test_data"
                    / f"{cam_name}_verify"
                    / f"{filename}.mp4"
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)
                vw = cv2.VideoWriter()
                vw.open(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
                for i in range(T):
                    vw.write(video[i])
                vw.release()

print("======== non-compressed ========")
with h5py.File(str(non_compress_datapath), "w") as root:
    root.attrs["sim"] = False
    root.attrs["compress"] = compress
    obs = root.create_group("observations")
    acts = root.create_group("actions")
    videos = obs.create_group("videos")
    for cam_name in camera_names:
        _ = videos.create_dataset(
            cam_name,
            (T, H, W, 3),
            dtype="uint8",
            compression=None,
        )
    _ = obs.create_dataset("qpos", (T, A))
    _ = obs.create_dataset("qvel", (T, A))
    _ = acts.create_dataset("joint_action", (T - 1, A))
    _ = acts.create_dataset("delta_eepose", (T - 1, A))
    _ = acts.create_dataset("base_action", (T - 1, B))

    # breakpoint()
    for name, array in data_dict.items():
        with timer_print(f"writing {name}"):
            root[name][...] = array
