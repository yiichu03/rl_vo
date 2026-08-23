"""EuRoC loader used by the official RL-VO environment.

The upstream loader only supported validation, assigned one trajectory to each
environment, and returned no timestamps to :class:`VecSVOEnv`. This adapter
keeps the official V1_01 crop and camera/ground-truth convention while adding
the minimum mechanics needed for single-sequence overfit.
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

parent = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
svo_lib_path = os.path.join(os.path.dirname(parent), "svo-lib/build/svo_env")
sys.path.append(svo_lib_path)
import svo_env


test_split = {
    "MH_01_easy": [1000, 3500],
    "MH_02_easy": [500, 3000],
    "MH_03_medium": [400, 2645],
    "MH_04_difficult": [340, 1900],
    "MH_05_difficult": [350, 2245],
    "V1_01_easy": [0, 2860],
    "V1_02_medium": [130, 1685],
    "V1_03_difficult": [0, 2050],
    "V2_01_easy": [0, 2190],
    "V2_02_medium": [100, 2290],
    "V2_03_difficult": [0, 1875],
}


class EurocLoader:
    def __init__(self, root_path, mode, num_envs, val_traj_ids=None, traj_name=None):
        if mode not in {"train", "val"}:
            raise ValueError(f"Unsupported EuRoC mode: {mode}")

        self.root_path = str(Path(root_path).resolve())
        self.mode = mode
        self.num_envs = int(num_envs)
        self.val_traj_ids = val_traj_ids
        self.test_split = list(test_split.keys())
        self.img_h, self.img_w = 480, 752

        if traj_name:
            print(f"[EuRoC Dataloader] Loading trajectory: {traj_name}")
            self.extract_trajectory(traj_name)
        else:
            print("[EuRoC Dataloader] Loading trajectories")
            self.extract_trajectories()

        self.extract_poses_images()
        self.set_up_running_indices()

    @staticmethod
    def _trajectory_name(path: str) -> str:
        return Path(path).name

    def is_test_scene(self, scene):
        return Path(scene).name in self.test_split

    def extract_trajectories(self):
        trajectories = glob.glob(os.path.join(self.root_path, "*"))
        trajectories = sorted(traj for traj in trajectories if self.is_test_scene(traj))
        if self.val_traj_ids != -1 and self.val_traj_ids is not None:
            trajectories = [trajectories[i] for i in self.val_traj_ids]
        if not trajectories:
            raise FileNotFoundError(f"No EuRoC trajectories found under {self.root_path}")
        if self.mode != "train" or len(trajectories) != 1:
            if self.num_envs != len(trajectories):
                raise ValueError(
                    f"num_envs={self.num_envs} but {len(trajectories)} EuRoC trajectories were selected"
                )
        self.trajectories_paths = trajectories

    def extract_trajectory(self, traj_name):
        if traj_name not in test_split:
            raise ValueError(f"The requested trajectory ({traj_name}) is not available")
        trajectory_path = os.path.join(self.root_path, traj_name)
        if not os.path.isdir(trajectory_path):
            raise FileNotFoundError(trajectory_path)
        self.trajectories_paths = [trajectory_path]

    @staticmethod
    def matching_time_indices(
        stamps_1: np.ndarray,
        stamps_2: np.ndarray,
        max_diff: float = 0.01,
        offset_2: float = 0.0,
    ):
        """Return nearest timestamp pairs within ``max_diff``."""
        matching_indices_1 = []
        matching_indices_2 = []
        stamps_2 = stamps_2.copy() + offset_2
        for index_1, stamp_1 in enumerate(stamps_1):
            diffs = np.abs(stamps_2 - stamp_1)
            index_2 = int(np.argmin(diffs))
            if diffs[index_2] <= max_diff:
                matching_indices_1.append(index_1)
                matching_indices_2.append(index_2)
        return matching_indices_1, matching_indices_2

    def extract_poses_images(self):
        self.poses = {}
        self.image_filenames = {}
        self.timestamps_nsec = {}
        self.absolute_timestamps_nsec = {}

        for traj in self.trajectories_paths:
            traj_name = self._trajectory_name(traj)
            with open(os.path.join(traj, "mav0", "cam0", "data.csv"), "r", encoding="utf-8") as file:
                image_data = list(csv.reader(file, delimiter=","))
            image_timestamps = np.asarray([int(row[0]) for row in image_data[1:]], dtype=np.int64)
            seq_image_filenames = [row[1] for row in image_data[1:]]

            start_idx, end_idx = test_split[traj_name]
            image_timestamps = image_timestamps[start_idx:end_idx]
            seq_image_filenames = seq_image_filenames[start_idx:end_idx]

            gt_path = os.path.join(traj, "mav0", "state_groundtruth_estimate0", "data.csv")
            gt_data = np.genfromtxt(gt_path, delimiter=",")[:, :8]
            matching_image, matching_gt = self.matching_time_indices(
                image_timestamps, gt_data[:, 0], max_diff=100000
            )
            if not matching_image:
                raise RuntimeError(f"No synchronized image/GT samples for {traj_name}")
            expected = matching_image[-1] - matching_image[0] + 1
            if expected != len(matching_image):
                raise RuntimeError(f"Non-contiguous synchronized image range for {traj_name}")

            matched_timestamps = image_timestamps[matching_image]
            gt_data = gt_data[matching_gt, :]
            seq_image_filenames = [seq_image_filenames[i] for i in matching_image]
            if gt_data.shape[0] != len(seq_image_filenames):
                raise RuntimeError(f"Image/pose count mismatch for {traj_name}")

            with open(os.path.join(traj, "mav0", "cam0", "sensor.yaml"), "r", encoding="utf-8") as stream:
                sensor_data = yaml.safe_load(stream)
            T_B_cam = np.asarray(sensor_data["T_BS"]["data"]).reshape([4, 4])
            T_w_B = np.zeros([gt_data.shape[0], 4, 4])
            T_w_B[:, 3, 3] = 1
            T_w_B[:, :3, 3] = gt_data[:, 1:4]
            T_w_B[:, :3, :3] = Rotation.from_quat(gt_data[:, [5, 6, 7, 4]]).as_matrix()

            T_w_cam_matrix = np.matmul(T_w_B, T_B_cam)
            T_w_cam = np.zeros([gt_data.shape[0], 7])
            T_w_cam[:, :3] = T_w_cam_matrix[:, :3, 3]
            T_w_cam[:, 3:] = Rotation.from_matrix(T_w_cam_matrix[:, :3, :3]).as_quat()

            self.image_filenames[traj_name] = seq_image_filenames
            self.poses[traj_name] = T_w_cam
            self.absolute_timestamps_nsec[traj_name] = matched_timestamps
            self.timestamps_nsec[traj_name] = matched_timestamps - matched_timestamps[0]

    def set_up_running_indices(self):
        if len(self.trajectories_paths) == 1:
            self.traj_idx = np.zeros(self.num_envs, dtype=int)
        else:
            self.traj_idx = np.arange(self.num_envs, dtype=int)
        self.cursor = np.zeros(self.num_envs, dtype=int)
        self.nr_samples_per_traj = np.asarray(
            [len(self.image_filenames[self._trajectory_name(path)]) for path in self.trajectories_paths],
            dtype=int,
        )
        self.nr_samples = int(self.nr_samples_per_traj.sum())
        self.last_batch_indices = np.zeros(self.num_envs, dtype=int)
        self.last_batch_timestamps_nsec = np.zeros(self.num_envs, dtype=np.float64)

    def get_image_timestamps_in_sec(self, traj_name):
        if traj_name.startswith("EuRoC_"):
            traj_name = traj_name[len("EuRoC_") :]
        if traj_name not in self.timestamps_nsec:
            raise KeyError(f"{traj_name} is not loaded")
        return self.timestamps_nsec[traj_name] * 1e-9

    def get_timestamps_nsec(self, traj_name):
        return self.timestamps_nsec[traj_name].copy()

    def sequence_length(self, env_id=0):
        return int(self.nr_samples_per_traj[self.traj_idx[int(env_id)]])

    def _load_env_ids(self, env_ids: np.ndarray):
        env_ids = np.asarray(env_ids, dtype=int).reshape(-1)
        sample_indices = self.cursor[env_ids].copy()
        pose_shape = [len(env_ids), 2, 7] if self.mode == "train" else [len(env_ids), 7]
        poses = np.zeros(pose_shape)
        timestamps = np.zeros(len(env_ids), dtype=np.float64)
        new_sequence_mask = sample_indices == 0
        image_paths = []

        for out_idx, env_id in enumerate(env_ids):
            traj_idx = self.traj_idx[env_id]
            traj_path = self.trajectories_paths[traj_idx]
            traj_name = self._trajectory_name(traj_path)
            sample_idx = int(sample_indices[out_idx])
            next_idx = min(sample_idx + 1, len(self.poses[traj_name]) - 1)
            if self.mode == "train":
                poses[out_idx, 0, :] = self.poses[traj_name][sample_idx]
                poses[out_idx, 1, :] = self.poses[traj_name][next_idx]
            else:
                poses[out_idx, :] = self.poses[traj_name][sample_idx]
            timestamps[out_idx] = float(self.timestamps_nsec[traj_name][sample_idx])
            image_paths.append(
                os.path.join(
                    traj_path,
                    "mav0",
                    "cam0",
                    "data",
                    self.image_filenames[traj_name][sample_idx],
                )
            )

        images = svo_env.load_image_batch(image_paths, len(env_ids), self.img_h, self.img_w)
        self.last_batch_indices[env_ids] = sample_indices
        self.last_batch_timestamps_nsec[env_ids] = timestamps
        lengths = self.nr_samples_per_traj[self.traj_idx[env_ids]]
        self.cursor[env_ids] = (sample_indices + 1) % lengths
        return images, poses, new_sequence_mask, timestamps

    def reset_envs(self, env_mask):
        env_ids = np.flatnonzero(np.asarray(env_mask, dtype=bool))
        self.cursor[env_ids] = 0
        return self._load_env_ids(env_ids)

    def __getitem__(self, idx):
        del idx
        return self._load_env_ids(np.arange(self.num_envs, dtype=int))

    def __len__(self):
        return int(self.nr_samples_per_traj.max())
