"""Canonical full-trajectory Sim(3) metrics for the EuRoC diagnostic."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def rms_checksum(path: str | Path) -> str:
    np_file = np.load(path)
    digest = hashlib.sha256()
    digest.update(np.asarray(np_file["mean"], dtype=np.float64).tobytes())
    digest.update(np.asarray(np_file["var"], dtype=np.float64).tobytes())
    return digest.hexdigest()


def estimate_sim3(pred_positions: np.ndarray, gt_positions: np.ndarray):
    """Estimate ``gt = scale * rotation @ pred + translation``."""
    pred_positions = np.asarray(pred_positions, dtype=np.float64)
    gt_positions = np.asarray(gt_positions, dtype=np.float64)
    if pred_positions.shape != gt_positions.shape or pred_positions.ndim != 2 or pred_positions.shape[1] != 3:
        raise ValueError("Expected matching Nx3 position arrays")
    if pred_positions.shape[0] < 3:
        raise ValueError("At least three poses are required for Sim(3) alignment")

    pred_mean = pred_positions.mean(axis=0)
    gt_mean = gt_positions.mean(axis=0)
    pred_centered = pred_positions - pred_mean
    gt_centered = gt_positions - gt_mean
    covariance = (gt_centered.T @ pred_centered) / pred_positions.shape[0]
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u_matrix) * np.linalg.det(vt_matrix) < 0:
        sign[-1] = -1
    rotation = u_matrix @ np.diag(sign) @ vt_matrix
    pred_variance = np.mean(np.sum(pred_centered**2, axis=1))
    if pred_variance <= np.finfo(np.float64).eps:
        raise ValueError("Predicted trajectory has zero variance")
    scale = float(np.sum(singular_values * sign) / pred_variance)
    translation = gt_mean - scale * (rotation @ pred_mean)
    return scale, rotation, translation


def sim3_trajectory_metrics(
    pred_positions: np.ndarray,
    pred_quaternions: np.ndarray,
    gt_positions: np.ndarray,
    gt_quaternions: np.ndarray,
    frame_indices: np.ndarray,
) -> dict:
    pred_positions = np.asarray(pred_positions, dtype=np.float64)
    pred_quaternions = np.asarray(pred_quaternions, dtype=np.float64)
    gt_positions = np.asarray(gt_positions, dtype=np.float64)
    gt_quaternions = np.asarray(gt_quaternions, dtype=np.float64)
    frame_indices = np.asarray(frame_indices, dtype=int)
    if len(pred_positions) < 3:
        return {
            "sim3_scale": None,
            "sim3_ate_translation_rmse_m": None,
            "sim3_rpe_translation_rmse_m": None,
            "sim3_rpe_rotation_rmse_deg": None,
            "metric_pose_count": int(len(pred_positions)),
            "rpe_pair_count": 0,
        }

    scale, align_rotation, translation = estimate_sim3(pred_positions, gt_positions)
    aligned_positions = scale * (align_rotation @ pred_positions.T).T + translation
    pred_rotations = Rotation.from_quat(pred_quaternions).as_matrix()
    gt_rotations = Rotation.from_quat(gt_quaternions).as_matrix()
    aligned_rotations = np.einsum("ij,njk->nik", align_rotation, pred_rotations)

    ate_errors = np.linalg.norm(aligned_positions - gt_positions, axis=1)
    translation_rpe = []
    rotation_rpe = []
    for idx in range(len(frame_indices) - 1):
        if frame_indices[idx + 1] != frame_indices[idx] + 1:
            continue
        gt_relative_rotation = gt_rotations[idx].T @ gt_rotations[idx + 1]
        pred_relative_rotation = aligned_rotations[idx].T @ aligned_rotations[idx + 1]
        gt_relative_translation = gt_rotations[idx].T @ (gt_positions[idx + 1] - gt_positions[idx])
        pred_relative_translation = aligned_rotations[idx].T @ (
            aligned_positions[idx + 1] - aligned_positions[idx]
        )
        translation_rpe.append(np.linalg.norm(pred_relative_translation - gt_relative_translation))
        rotation_error = gt_relative_rotation.T @ pred_relative_rotation
        rotation_rpe.append(np.linalg.norm(Rotation.from_matrix(rotation_error).as_rotvec()))

    return {
        "sim3_scale": scale,
        "sim3_ate_translation_rmse_m": float(np.sqrt(np.mean(np.square(ate_errors)))),
        "sim3_rpe_translation_rmse_m": (
            float(np.sqrt(np.mean(np.square(translation_rpe)))) if translation_rpe else None
        ),
        "sim3_rpe_rotation_rmse_deg": (
            float(np.rad2deg(np.sqrt(np.mean(np.square(rotation_rpe))))) if rotation_rpe else None
        ),
        "metric_pose_count": int(len(pred_positions)),
        "rpe_pair_count": int(len(translation_rpe)),
    }
