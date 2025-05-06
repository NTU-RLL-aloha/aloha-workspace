# flake8: noqa

import os

# Try to import ALOHA package's DATA_DIR, else default to ~/aloha_data
try:
    from aloha.constants import DATA_DIR
except ImportError:
    DATA_DIR = os.path.expanduser('~/aloha_data')

# TASK_CONFIGS = {
#     'aloha_mobile_hello_aloha':{
#         'dataset_dir': DATA_DIR + '/aloha_mobile_hello_aloha',
#         'episode_len': 800,
#         'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
#     },
# }
TASK_CONFIGS = {
    "aloha_test_move_water": {
        "dataset_dir": DATA_DIR + "/aloha_test_move_water",
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
