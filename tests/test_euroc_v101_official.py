from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from env import svo_wrapper
from experiments.euroc_v101_official.assessment import (
    build_scientific_assessment,
    comparison_eligibility,
)
from experiments.euroc_v101_official.metrics import estimate_sim3, sim3_trajectory_metrics


def test_sim3_metrics_recover_scale_rotation_and_translation():
    rng = np.random.default_rng(8)
    pred = rng.normal(size=(40, 3))
    pred_rotation = Rotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    scale = 2.4
    translation = np.asarray([1.0, -0.5, 2.0])
    gt = scale * (pred_rotation @ pred.T).T + translation
    estimated_scale, estimated_rotation, estimated_translation = estimate_sim3(pred, gt)
    assert np.isclose(estimated_scale, scale)
    assert np.allclose(estimated_rotation, pred_rotation)
    assert np.allclose(estimated_translation, translation)

    pred_quaternions = Rotation.identity(len(pred)).as_quat()
    gt_quaternions = Rotation.from_matrix(
        np.repeat(pred_rotation[None, :, :], len(pred), axis=0)
    ).as_quat()
    metrics = sim3_trajectory_metrics(
        pred,
        pred_quaternions,
        gt,
        gt_quaternions,
        np.arange(len(pred)),
    )
    assert metrics["sim3_ate_translation_rmse_m"] < 1e-10
    assert metrics["sim3_rpe_translation_rmse_m"] < 1e-10
    assert metrics["sim3_rpe_rotation_rmse_deg"] < 1e-10


class FakeLoader:
    img_h = 4
    img_w = 6

    def __init__(self, root_path, mode, num_envs, val_traj_ids=None, traj_name=None):
        del root_path, val_traj_ids, traj_name
        self.mode = mode
        self.num_envs = num_envs
        self.trajectories_paths = ["V1_01_easy"]
        self.set_up_running_indices()

    def __iter__(self):
        return self

    def __next__(self):
        return self._batch(np.arange(self.num_envs))

    def set_up_running_indices(self):
        self.cursor = np.zeros(self.num_envs, dtype=int)
        self.last_batch_indices = np.zeros(self.num_envs, dtype=int)

    def _batch(self, env_ids):
        env_ids = np.asarray(env_ids, dtype=int)
        frame_indices = self.cursor[env_ids].copy()
        self.last_batch_indices[env_ids] = frame_indices
        self.cursor[env_ids] = (frame_indices + 1) % 4
        images = np.zeros([len(env_ids), self.img_h, self.img_w, 1], dtype=np.uint8)
        images[:, 0, 0, 0] = frame_indices
        poses = np.zeros([len(env_ids), 7], dtype=np.float64)
        poses[:, 0] = frame_indices
        poses[:, 6] = 1
        if self.mode == "train":
            next_poses = poses.copy()
            next_poses[:, 0] += 1
            poses = np.stack([poses, next_poses], axis=1)
        return images, poses, frame_indices == 0, frame_indices.astype(np.float64) * 20_000_000

    def reset_envs(self, env_mask):
        env_ids = np.flatnonzero(env_mask)
        self.cursor[env_ids] = 0
        return self._batch(env_ids)


class FakeBackend:
    def __init__(self, params, calib, num_envs, initialize_glog):
        del params, calib, initialize_glog
        self.num_envs = num_envs
        self.fail_next = False
        self.last_env_step_scalar_shapes = None

    def reset(self, indices):
        del indices

    def setSeed(self, seed):
        del seed

    @staticmethod
    def _fill_outputs(action, poses, observations, dones, stages, fail):
        poses[:] = np.eye(4, dtype=np.float64).reshape(1, 16)
        observations[:] = 0
        observations[:, 0] = action[:, 1]
        observations[:, 1] = 0
        dones[:] = 1 if fail else 0
        stages[:] = 3 if fail else 2

    def step(
        self,
        images,
        timestamps,
        action,
        use_rl_actions,
        poses,
        observations,
        dones,
        stages,
        runtime,
        use_gt_init,
        gt_init,
    ):
        del images, timestamps, use_rl_actions, runtime, use_gt_init, gt_init
        self._fill_outputs(action, poses, observations, dones, stages, self.fail_next)
        self.fail_next = False

    def env_step(
        self,
        indices,
        images,
        timestamps,
        action,
        use_rl_actions,
        poses,
        observations,
        dones,
        stages,
        runtime,
        use_gt_init,
        gt_init,
    ):
        del indices, images, use_rl_actions, runtime, use_gt_init, gt_init
        self.last_env_step_scalar_shapes = (timestamps.shape, dones.shape, stages.shape)
        self._fill_outputs(action, poses, observations, dones, stages, False)
        stages[:] = 1


def make_fake_env(monkeypatch):
    monkeypatch.setattr(svo_wrapper, "EurocLoader", FakeLoader)
    monkeypatch.setattr(svo_wrapper.svo_env, "SVOEnv", FakeBackend)
    config = SimpleNamespace(
        align_reward=0.01,
        keyframe_reward=0.0001,
        traj_length=5,
        nr_points_for_align=3,
    )
    return svo_wrapper.VecSVOEnv(
        "params.yaml",
        "calib.yaml",
        "unused",
        1,
        "val",
        config,
        dataset="euroc",
        dataset_traj_name="V1_01_easy",
        restart_failed_from_sequence_start=True,
    )


def test_action_timing_and_exact_requested_executed_telemetry(monkeypatch):
    env = make_fake_env(monkeypatch)
    env.reset()
    _, _, dones, infos, valid = env.step(np.asarray([[1, 2]]), use_RL_actions_bool=True)
    info = infos[0]
    assert not dones[0]
    assert valid[0]
    assert info["frame_index"] == 1
    assert info["timestamp_nsec"] == 20_000_000
    assert info["requested_action"] == [1, 2]
    assert info["requested_action_scaled"] == [1.0, 30.0]
    assert info["executed_action_scaled"] == [1.0, 30.0]
    assert info["actual_feature_count"] == 30


def test_failure_action_is_logged_and_restart_is_frame_zero(monkeypatch):
    env = make_fake_env(monkeypatch)
    env.reset()
    env.env.fail_next = True
    _, _, dones, infos, valid = env.step(
        np.asarray([[0, 4]]), use_RL_actions_bool=True, use_gt_initialization=True
    )
    info = infos[0]
    assert dones[0]
    assert not valid[0]  # Preserved official masked-credit behavior, made explicit in telemetry.
    assert info["tracking_failure"]
    assert info["requested_action"] == [0, 4]
    assert info["executed_action_scaled"] == [0.0, 40.0]
    assert info["reset_to_sequence_start"]
    assert info["reset_frame_index"] == 0
    assert info["reset_timestamp_nsec"] == 0
    assert env.env.last_env_step_scalar_shapes == ((1, 1), (1, 1), (1, 1))


def _assessment_row(label, ate, *, full_route, coverage, failures=0, terminations=0):
    return {
        "label": label,
        "controller": "policy",
        "sim3_ate_translation_rmse_m": ate,
        "full_route_completed": full_route,
        "tracking_coverage": coverage,
        "tracking_failure_count": failures,
        "termination_count": terminations,
    }


def test_truncated_low_ate_is_not_scientifically_comparable():
    eligible, reasons = comparison_eligibility(
        _assessment_row(
            "short_false_positive",
            0.003,
            full_route=False,
            coverage=0.02,
            failures=1,
            terminations=1,
        )
    )
    assert not eligible
    assert "route_not_completed" in reasons
    assert "coverage_below_0p95" in reasons


def test_assessment_uses_only_full_route_fixed_and_policy_results():
    bad_fixed = _assessment_row("bad_fixed", 0.001, full_route=False, coverage=0.02)
    bad_fixed["controller"] = "fixed"
    good_fixed = _assessment_row("good_fixed", 0.15, full_route=True, coverage=0.97)
    good_fixed["controller"] = "fixed"
    raw = {
        "controls": [bad_fixed, good_fixed],
        "checkpoint_evaluations": {
            "23": [
                _assessment_row("seed23_iter0", 0.20, full_route=True, coverage=0.97),
                _assessment_row(
                    "seed23_iter100", 0.002, full_route=False, coverage=0.01
                ),
            ]
        },
        "seed_conclusions": {
            "23": {"reward_curve": {"delta_last_minus_first": -0.1}}
        },
    }
    assessment = build_scientific_assessment(raw)
    assert assessment["best_eligible_fixed_by_sim3_ate"]["label"] == "good_fixed"
    assert not assessment["any_trained_checkpoint_better_than_best_fixed"]
    assert not assessment["scientific_success"]
    assert assessment["verdict"] == "no_viable_trained_policy"
