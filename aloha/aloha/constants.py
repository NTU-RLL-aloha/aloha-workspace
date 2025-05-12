# flake8: noqa

import os

### Task parameters

# Set to 'true' for Mobile ALOHA, 'false' for Stationary ALOHA
IS_MOBILE = os.environ.get("INTERBOTIX_ALOHA_IS_MOBILE", "true").lower() == "true"
# IS_MOBILE = False

ARM_MASKS = {
    "left": [True, False],
    "right": [False, True],
    "both": [True, True],
}

COLOR_IMAGE_TOPIC_NAME = "{}/camera/color/image_rect_raw"  # for RealSense cameras
DEPTH_TOPIC_NAME = "{}/camera/aligned_depth_to_color/image_raw"  # for RealSense cameras

CONFIG_DIR = os.path.expanduser("../config")
DATA_DIR = os.path.expanduser("~/aloha_data")

### ALOHA Fixed Constants
FPS = 50
DT = 1 / FPS

try:
    from rclpy.duration import Duration
    from rclpy.constants import S_TO_NS

    DT_DURATION = Duration(seconds=0, nanoseconds=DT * S_TO_NS)
except ImportError:
    pass
JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]

CAMERA_RESOLUTIONS = {
    "cam_high": (480, 848),
    "cam_low": (480, 848),
    "cam_left_wrist": (480, 848),
    "cam_right_wrist": (480, 848),
}

# fmt: off
# START_ARM_POSE = [0.0,-0.96,1.16,0.0,-0.3,0.0,0.02239,-0.02239,0.0,-0.96,1.16,0.0,-0.3,0.0,0.02239,-0.02239] # original
START_ARM_POSE = [0.0,-0.32,0.80,0.0,0.66,1.3,0.02239,-0.02239,0.0,-0.32,0.80,0.0,0.66,1.3,0.02239,-0.02239] # flow decomposor
SLEEP_ARM_POSE = [0.0,-2.049999952316284,1.7000000476837158,0.0,-2.0,0.0,0.02239,-0.02239,0.0,-2.049999952316284,1.7000000476837158,0.0,-2.0,0.0,0.02239,-0.02239]
# START_ARM_POSE = SLEEP_ARM_POSE
# fmt: on
INACTIVE_START_ARM_POSE = SLEEP_ARM_POSE

LEADER_GRIPPER_CLOSE_THRESH = 0.0

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
LEADER_GRIPPER_POSITION_OPEN = 0.0323
LEADER_GRIPPER_POSITION_CLOSE = 0.0185

FOLLOWER_GRIPPER_POSITION_OPEN = 0.0579
FOLLOWER_GRIPPER_POSITION_CLOSE = 0.0440

# Gripper joint limits (qpos[6])
LEADER_GRIPPER_JOINT_OPEN = 0.8298
LEADER_GRIPPER_JOINT_CLOSE = -0.0552
LEADER_GRIPPER_JOINT_MID = (LEADER_GRIPPER_JOINT_OPEN + LEADER_GRIPPER_JOINT_CLOSE) / 2

FOLLOWER_GRIPPER_JOINT_OPEN = 1.6214
FOLLOWER_GRIPPER_JOINT_CLOSE = 0.6197

### Helper functions

LEADER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - LEADER_GRIPPER_POSITION_CLOSE) / (
    LEADER_GRIPPER_POSITION_OPEN - LEADER_GRIPPER_POSITION_CLOSE
)
FOLLOWER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (
    x - FOLLOWER_GRIPPER_POSITION_CLOSE
) / (FOLLOWER_GRIPPER_POSITION_OPEN - FOLLOWER_GRIPPER_POSITION_CLOSE)
LEADER_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (LEADER_GRIPPER_POSITION_OPEN - LEADER_GRIPPER_POSITION_CLOSE)
    + LEADER_GRIPPER_POSITION_CLOSE
)
FOLLOWER_GRIPPER_POSITION_UNNORMALIZE_FN = (
    lambda x: x * (FOLLOWER_GRIPPER_POSITION_OPEN - FOLLOWER_GRIPPER_POSITION_CLOSE)
    + FOLLOWER_GRIPPER_POSITION_CLOSE
)
LEADER2FOLLOWER_POSITION_FN = lambda x: FOLLOWER_GRIPPER_POSITION_UNNORMALIZE_FN(
    LEADER_GRIPPER_POSITION_NORMALIZE_FN(x)
)

LEADER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - LEADER_GRIPPER_JOINT_CLOSE) / (
    LEADER_GRIPPER_JOINT_OPEN - LEADER_GRIPPER_JOINT_CLOSE
)
FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - FOLLOWER_GRIPPER_JOINT_CLOSE) / (
    FOLLOWER_GRIPPER_JOINT_OPEN - FOLLOWER_GRIPPER_JOINT_CLOSE
)
LEADER_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (LEADER_GRIPPER_JOINT_OPEN - LEADER_GRIPPER_JOINT_CLOSE)
    + LEADER_GRIPPER_JOINT_CLOSE
)
FOLLOWER_GRIPPER_JOINT_UNNORMALIZE_FN = (
    lambda x: x * (FOLLOWER_GRIPPER_JOINT_OPEN - FOLLOWER_GRIPPER_JOINT_CLOSE)
    + FOLLOWER_GRIPPER_JOINT_CLOSE
)
LEADER2FOLLOWER_JOINT_FN = lambda x: FOLLOWER_GRIPPER_JOINT_UNNORMALIZE_FN(
    LEADER_GRIPPER_JOINT_NORMALIZE_FN(x)
)

LEADER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (
    LEADER_GRIPPER_POSITION_OPEN - LEADER_GRIPPER_POSITION_CLOSE
)
FOLLOWER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (
    FOLLOWER_GRIPPER_POSITION_OPEN - FOLLOWER_GRIPPER_POSITION_CLOSE
)

LEADER_POS2JOINT = (
    lambda x: LEADER_GRIPPER_POSITION_NORMALIZE_FN(x)
    * (LEADER_GRIPPER_JOINT_OPEN - LEADER_GRIPPER_JOINT_CLOSE)
    + LEADER_GRIPPER_JOINT_CLOSE
)
LEADER_JOINT2POS = lambda x: LEADER_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - LEADER_GRIPPER_JOINT_CLOSE)
    / (LEADER_GRIPPER_JOINT_OPEN - LEADER_GRIPPER_JOINT_CLOSE)
)
FOLLOWER_POS2JOINT = (
    lambda x: FOLLOWER_GRIPPER_POSITION_NORMALIZE_FN(x)
    * (FOLLOWER_GRIPPER_JOINT_OPEN - FOLLOWER_GRIPPER_JOINT_CLOSE)
    + FOLLOWER_GRIPPER_JOINT_CLOSE
)
FOLLOWER_JOINT2POS = lambda x: FOLLOWER_GRIPPER_POSITION_UNNORMALIZE_FN(
    (x - FOLLOWER_GRIPPER_JOINT_CLOSE)
    / (FOLLOWER_GRIPPER_JOINT_OPEN - FOLLOWER_GRIPPER_JOINT_CLOSE)
)

LEADER_GRIPPER_JOINT_MID = (LEADER_GRIPPER_JOINT_OPEN + LEADER_GRIPPER_JOINT_CLOSE) / 2

### Real hardware task configurations

# TASK_CONFIGS = {
#     ### Template
#     # 'aloha_template':{
#     #     'dataset_dir': [
#     #         DATA_DIR + '/aloha_template',
#     #         DATA_DIR + '/aloha_template_subtask',
#     #         DATA_DIR + '/aloha_template_other_subtask',
#     #     ], # only the first entry in dataset_dir is used for eval
#     #     'stats_dir': [
#     #         DATA_DIR + '/aloha_template',
#     #     ],
#     #     'sample_weights': [6, 1, 1],
#     #     'train_ratio': 0.99, # ratio of train data from the first dataset_dir
#     #     'episode_len': 1500,
#     #     'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
#     # },
#     "aloha_mobile_hello_aloha": {
#         "dataset_dir": DATA_DIR + "/aloha_mobile_hello_aloha",
#         "episode_len": 800,
#         "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
#     },
#     "aloha_mobile_dummy": {
#         "dataset_dir": DATA_DIR + "/aloha_mobile_dummy",
#         "episode_len": 1000,
#         "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
#     },
#     "aloha_stationary_hello_aloha": {
#         "dataset_dir": DATA_DIR + "/aloha_stationary_hello_aloha",
#         "episode_len": 800,
#         "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
#     },
#     "aloha_stationary_dummy": {
#         "dataset_dir": DATA_DIR + "/aloha_stationary_dummy",
#         "episode_len": 800,
#         "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
#     },
# }

TASK_CONFIGS = {
    "aloha_test_move_water": {
        "dataset_dir": DATA_DIR + "/aloha_test_move_water",
        "episode_len": 600,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    },
    "aloha_test": {
        "dataset_dir": DATA_DIR + "/aloha_test",
        "episode_len": 600,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    },
    "aloha_test_pick_and_place_sweets": {
        "dataset_dir": DATA_DIR + "/aloha_test_pick_and_place_sweets",
        "episode_len": 500,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    },
    "aloha_test_pick_and_place_sweets_2": {
        "dataset_dir": DATA_DIR + "/aloha_test_pick_and_place_sweets_2",
        "episode_len": 2000,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    },
    "aloha_put_bottle_on_circle": {
        "dataset_dir": DATA_DIR + "/aloha_put_bottle_on_circle",
        "episode_len": 1000,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    },
    "aloha_put_bottle_on_circle_switch_hand": {
        "dataset_dir": DATA_DIR + "/aloha_put_bottle_on_circle_switch_hand",
        "episode_len": 1000,
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    },
}
