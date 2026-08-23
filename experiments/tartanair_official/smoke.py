#!/usr/bin/env python3
"""Small compute-node integration smoke for the official TartanAir environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from env.svo_wrapper import VecSVOEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()

    reward = OmegaConf.create(
        {"align_reward": 0.01, "keyframe_reward": 0.0001, "traj_length": 5, "nr_points_for_align": 3}
    )
    env = VecSVOEnv(
        str(args.params),
        str(args.calibration),
        str(args.dataset_root),
        2,
        mode="train",
        reward_config=reward,
        initialize_glog=True,
    )
    try:
        observation = env.reset(use_gt_initialization=True)
        if observation.shape != (2, env.obs_dim) or not np.isfinite(observation).all():
            raise RuntimeError(f"invalid reset observation: {observation.shape}")
        valid_count = 0
        failure_count = 0
        for step in range(args.steps):
            action = np.asarray([[step % 2, step % 5], [(step + 1) % 2, (step + 2) % 5]])
            observation, reward_value, done, _, valid = env.step(action, use_gt_initialization=True)
            if observation.shape != (2, env.obs_dim) or not np.isfinite(observation).all():
                raise RuntimeError(f"invalid step observation at {step}")
            if not np.isfinite(reward_value).all():
                raise RuntimeError(f"non-finite reward at {step}")
            valid_count += int(valid.sum())
            failure_count += int(done.sum())
        print(
            json.dumps(
                {
                    "status": "passed",
                    "num_envs": 2,
                    "steps": args.steps,
                    "valid_transitions": valid_count,
                    "terminations": failure_count,
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
