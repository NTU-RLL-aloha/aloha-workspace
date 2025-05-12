from typing import Dict

import os
import time
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
import warnings
import random

import quest.utils.utils as utils
from quest.utils.logger import Logger

import sys

sys.path.insert(0, "../")
# from utils.flow_viz import flow_to_images, get_wandb_video


OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(config_path="config", version_base=None)
def main(cfg):
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    train_cfg = cfg.training
    input_type = (
        cfg.task["dataset"]["keys"]
        if "keys" in cfg.task["dataset"]
        else cfg.task["dataset"]["dataset_keys"]
    )

    # create model
    model = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta)
