import numpy as np
import cv2
import time
import warnings

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FrameHealthChecker:
    def __init__(self, mse_threshold=5.0, max_static_count=5, min_fps=20):
        self.prev_frame = None
        self.prev_stamp = None
        self.static_count = 0
        self.mse_threshold = mse_threshold
        self.max_static_count = max_static_count
        self.min_fps = min_fps
        self.prev_time = None

    def is_duplicate(self, img1, img2):
        if img1.shape != img2.shape:
            return False
        mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
        return mse < self.mse_threshold

    def check(self, frame: np.ndarray, stamp=None, name=None):
        # now = time.time()
        # interval = now - self.prev_time
        # self.prev_time = now
        # str_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        # use time stamp to check if the fps

        now = stamp.sec + stamp.nanosec * 1e-9
        if self.prev_time is None:
            self.prev_time = now
        interval = now - self.prev_time
        self.prev_time = now
        # 1. 檢查 FPS 是否掉落
        if interval > 1.0 / self.min_fps:
            logger.warning(f"{name} FPS drop detected: {1/interval:.2f} fps")

        # 2. 檢查是否 Timestamp 沒變
        if self.prev_stamp is not None and stamp == self.prev_stamp:
            logger.warning(f"{name} Duplicate timestamp detected: {stamp}")

        self.prev_stamp = stamp

        # 3. 檢查是否畫面重複（幀凍結）
        # if self.prev_frame is not None and self.is_duplicate(self.prev_frame, frame):
        #     self.static_count += 1
        #     if self.static_count >= self.max_static_count:
        #         logger.warning(
        #             f"{name} Frozen frame detected ({self.static_count} times)"
        #         )
        # else:
        #     self.static_count = 0

        self.prev_frame = frame.copy()
