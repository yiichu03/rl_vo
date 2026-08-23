#!/usr/bin/env python3
"""Official RL-VO on EuRoC V1_01: gated controls, bounded PPO, Sim(3) eval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf
from stable_baselines3.common.utils import get_linear_fn, obs_as_tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.euroc_loader import EurocLoader
from env.svo_wrapper import VecSVOEnv
from experiments.euroc_v101_official.metrics import rms_checksum, sim3_trajectory_metrics
from policies.attention_policy import CustomActorCriticPolicy
from rl_algorithms.ppo import PPO


OFFICIAL = {
    "n_envs": 100,
    "n_steps": 250,
    "batch_size": 25000,
    "n_epochs": 10,
    "gamma": 0.6,
    "gae_lambda": 0.95,
    "ent_coef": 0.0025,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "clip_range": 0.2,
    "total_timesteps": 2_500_000,
    "eval_iterations": [0, 10, 50, 100],
}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    temporary.replace(path)


def git_state() -> dict:
    if shutil.which("git") is None:
        return {
            "commit": os.environ.get("RLVO_EXPECTED_COMMIT", "unavailable-in-container"),
            "branch": os.environ.get("RLVO_EXPECTED_BRANCH", "research/rlvo-official-euroc-v101"),
            "status_porcelain": os.environ.get("RLVO_GIT_STATUS", ""),
        }

    def run(*args):
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_porcelain": run("status", "--short"),
    }


def reward_config():
    return SimpleNamespace(
        align_reward=0.01,
        keyframe_reward=0.0001,
        traj_length=5,
        nr_points_for_align=3,
    )


def policy_kwargs(env):
    return dict(
        encoder_kwargs=dict(
            variable_feature_dim=3,
            obs_dim_variable=env.agent_obs_dim_variable,
            obs_dim_fixed=env.agent_obs_dim_fixed,
            critique_dim=env.critique_dim,
        ),
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        log_std_init=-0.0,
    )


def make_env(args, mode: str, num_envs: int, initialize_glog: bool = False):
    return VecSVOEnv(
        args.params_yaml,
        args.calib_yaml,
        args.dataset_dir,
        num_envs,
        reward_config=reward_config(),
        mode=mode,
        initialize_glog=initialize_glog,
        dataset="euroc",
        dataset_traj_name=args.sequence,
        restart_failed_from_sequence_start=True,
    )


def create_policy(env, seed: int, checkpoint: str | Path | None = None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy = CustomActorCriticPolicy(
        env.observation_space,
        env.action_space,
        lr_schedule=lambda _: 3e-4,
        **policy_kwargs(env),
    )
    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        policy.load_state_dict(state_dict, strict=True)
    policy.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    policy.set_training_mode(False)
    return policy


def _policy_action(policy, obs, deterministic: bool):
    obs_tensor = obs_as_tensor(obs, policy.device)
    with torch.no_grad():
        distribution = policy.get_distribution(obs_tensor)
        action = distribution.get_actions(deterministic=deterministic)
    head_probabilities = [head.probs.detach().cpu().numpy()[0] for head in distribution.distribution]
    margins = []
    for probabilities in head_probabilities:
        sorted_probabilities = np.sort(probabilities)
        margins.append(float(sorted_probabilities[-1] - sorted_probabilities[-2]))
    return action.detach().cpu().numpy(), head_probabilities, margins


def evaluate_controller(
    args,
    output_dir: str | Path,
    label: str,
    controller: str,
    fixed_action=None,
    seed: int = 23,
    deterministic: bool = True,
    checkpoint: str | Path | None = None,
    rms_path: str | Path | None = None,
    gt_initialize_at_frame0: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = make_env(args, mode="val", num_envs=1, initialize_glog=False)
    if rms_path is not None:
        env.load_rms(str(rms_path))
    policy = create_policy(env, seed, checkpoint) if controller == "policy" else None
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs = env.reset(use_gt_initialization=gt_initialize_at_frame0)
    sequence_length = env.dataloader.sequence_length(0)
    expected_timestamps = env.dataloader.get_timestamps_nsec(args.sequence)
    records = []
    pred_positions = []
    pred_quaternions = []
    gt_positions = []
    gt_quaternions = []
    metric_frame_indices = []
    action_counter = Counter()
    probability_margins = []
    action_probabilities = []
    tracking_failure_count = 0
    action_mismatch_count = 0
    timestamp_mismatch_count = 0
    restart_contract_violation_count = 0
    full_route_completed = True

    try:
        for _ in range(sequence_length - 1):
            if controller == "native":
                action = np.zeros([1, 2], dtype=np.int64)
                use_rl_actions = False
                head_probabilities = []
                margins = []
            elif controller == "fixed":
                action = np.asarray(fixed_action, dtype=np.int64).reshape(1, 2)
                use_rl_actions = True
                head_probabilities = []
                margins = []
            elif controller == "policy":
                action, head_probabilities, margins = _policy_action(policy, obs, deterministic)
                action = action.astype(np.int64)
                use_rl_actions = True
            else:
                raise ValueError(controller)

            obs, rewards, dones, infos, valid_mask = env.step(
                action, use_RL_actions_bool=use_rl_actions, use_gt_initialization=True
            )
            info = infos[0]
            frame_index = int(info["frame_index"])
            timestamp_nsec = float(info["timestamp_nsec"])
            if frame_index < 0 or frame_index >= len(expected_timestamps):
                timestamp_mismatch_count += 1
            elif not np.isclose(timestamp_nsec, float(expected_timestamps[frame_index]), rtol=0, atol=0.5):
                timestamp_mismatch_count += 1

            if info["action_applied"]:
                requested = tuple(int(value) for value in info["requested_action"])
                action_counter[f"{requested[0]}_{requested[1]}"] += 1
                expected_scaled = [float(requested[0]), float(20 + 5 * requested[1])]
                if info["executed_action_scaled"] is None or not np.allclose(
                    info["executed_action_scaled"], expected_scaled
                ):
                    action_mismatch_count += 1

            is_tracking = int(info["action_stage"]) == 2
            if is_tracking:
                pred_positions.append(np.asarray(info["position"], dtype=np.float64).reshape(3))
                pred_quaternions.append(np.asarray(info["rotation"], dtype=np.float64).reshape(-1, 4)[-1])
                gt_positions.append(np.asarray(info["gt_position"], dtype=np.float64).reshape(3))
                gt_quaternions.append(np.asarray(info["gt_rotation"], dtype=np.float64).reshape(4))
                metric_frame_indices.append(frame_index)

            probability_margins.extend(margins)
            if head_probabilities:
                action_probabilities.append(np.concatenate(head_probabilities))
            record = {
                "frame_index": frame_index,
                "timestamp_nsec": timestamp_nsec,
                "reward": float(rewards[0]),
                "valid": bool(valid_mask[0]),
                "stage": int(info["action_stage"]),
                "tracking_failure": bool(info["tracking_failure"]),
                "termination": bool(info["termination"]),
                "requested_action": info["requested_action"],
                "executed_action_scaled": info["executed_action_scaled"],
                "action_applied": bool(info["action_applied"]),
                "actual_feature_count": float(info["actual_feature_count"]),
                "actual_keyframe_selected": bool(info["actual_keyframe_selected"]),
                "position_reward": float(info["position_reward"]),
                "keyframe_reward": float(info["keyframe_reward"]),
            }
            records.append(record)
            if info["tracking_failure"]:
                tracking_failure_count += 1
                if not info.get("reset_to_sequence_start", False) or info.get("reset_frame_index") != 0:
                    restart_contract_violation_count += 1
                full_route_completed = False
                break
            if bool(info["new_seq"]):
                full_route_completed = False
                break
    finally:
        env.close()

    metric_values = sim3_trajectory_metrics(
        np.asarray(pred_positions).reshape(-1, 3),
        np.asarray(pred_quaternions).reshape(-1, 4),
        np.asarray(gt_positions).reshape(-1, 3),
        np.asarray(gt_quaternions).reshape(-1, 4),
        np.asarray(metric_frame_indices, dtype=int),
    )
    valid_count = sum(int(row["valid"]) for row in records)
    actual_features = [row["actual_feature_count"] for row in records if np.isfinite(row["actual_feature_count"])]
    actual_keyframes = sum(int(row["actual_keyframe_selected"]) for row in records)
    reward_sum = float(sum(row["reward"] for row in records))
    valid_reward_sum = float(sum(row["reward"] for row in records if row["valid"]))
    applied_count = sum(int(row["action_applied"]) for row in records)

    summary = {
        "status": "complete",
        "label": label,
        "controller": controller,
        "fixed_action": fixed_action,
        "seed": seed,
        "deterministic": deterministic,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "rms_path": str(rms_path) if rms_path is not None else None,
        "rms_checksum": rms_checksum(rms_path) if rms_path is not None else None,
        "initialization_mode": "gt_at_frame0" if gt_initialize_at_frame0 else "cold_mono",
        "sequence": args.sequence,
        "sequence_length": sequence_length,
        "processed_steps_after_reset": len(records),
        "full_route_completed": bool(full_route_completed and len(records) == sequence_length - 1),
        "tracking_pose_count": len(pred_positions),
        "tracking_coverage": len(pred_positions) / float(sequence_length),
        "valid_transition_count": valid_count,
        "valid_transition_ratio": valid_count / float(max(len(records), 1)),
        "reward_sum": reward_sum,
        "reward_mean_all_steps": reward_sum / float(max(len(records), 1)),
        "reward_sum_valid": valid_reward_sum,
        "reward_mean_per_valid": valid_reward_sum / float(max(valid_count, 1)),
        "tracking_failure_count": tracking_failure_count,
        "termination_count": sum(int(row["termination"]) for row in records),
        "action_applied_count": applied_count,
        "requested_executed_action_mismatch_count": action_mismatch_count,
        "timestamp_mismatch_count": timestamp_mismatch_count,
        "failure_restart_contract_violation_count": restart_contract_violation_count,
        "mean_actual_feature_count": float(np.mean(actual_features)) if actual_features else None,
        "actual_keyframe_ratio": actual_keyframes / float(max(len(records), 1)),
        "action_counts_when_applied": dict(sorted(action_counter.items())),
        "unique_actions_when_applied": len(action_counter),
        "policy_probability_margin_mean": (
            float(np.mean(probability_margins)) if probability_margins else None
        ),
        "policy_probability_margin_min": (
            float(np.min(probability_margins)) if probability_margins else None
        ),
        "policy_mean_head_probabilities": (
            np.mean(np.asarray(action_probabilities), axis=0).tolist() if action_probabilities else None
        ),
        "wall_seconds": time.time() - started,
        **metric_values,
    }
    write_json(output_dir / "summary.json", summary)
    np.savez_compressed(
        output_dir / "trace.npz",
        frame_indices=np.asarray([row["frame_index"] for row in records], dtype=int),
        timestamps_nsec=np.asarray([row["timestamp_nsec"] for row in records], dtype=np.float64),
        rewards=np.asarray([row["reward"] for row in records], dtype=np.float64),
        valid=np.asarray([row["valid"] for row in records], dtype=bool),
        stages=np.asarray([row["stage"] for row in records], dtype=int),
        requested_actions=np.asarray([row["requested_action"] for row in records], dtype=int),
        actual_feature_counts=np.asarray([row["actual_feature_count"] for row in records], dtype=np.float64),
        actual_keyframe_selected=np.asarray([row["actual_keyframe_selected"] for row in records], dtype=bool),
        pred_positions=np.asarray(pred_positions, dtype=np.float64),
        pred_quaternions=np.asarray(pred_quaternions, dtype=np.float64),
        gt_positions=np.asarray(gt_positions, dtype=np.float64),
        gt_quaternions=np.asarray(gt_quaternions, dtype=np.float64),
        metric_frame_indices=np.asarray(metric_frame_indices, dtype=int),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return summary


def validate_dataset(args) -> dict:
    loader = EurocLoader(args.dataset_dir, "train", 2, traj_name=args.sequence)
    images, poses, new_sequence, timestamps = loader[0]
    second_images, second_poses, second_new_sequence, second_timestamps = loader[1]
    del second_images
    timestamps_all = loader.get_timestamps_nsec(args.sequence)
    checks = {
        "image_shape": list(images.shape),
        "train_pose_shape": list(poses.shape),
        "second_train_pose_shape": list(second_poses.shape),
        "first_new_sequence_all": bool(np.all(new_sequence)),
        "second_new_sequence_none": bool(not np.any(second_new_sequence)),
        "first_timestamp_zero": bool(np.all(timestamps == 0)),
        "second_timestamp_exact": bool(np.all(second_timestamps == timestamps_all[1])),
        "timestamp_strictly_increasing": bool(np.all(np.diff(timestamps_all) > 0)),
        "sequence_length": int(len(timestamps_all)),
        "replicated_env_pose_equal": bool(np.allclose(poses[0], poses[1])),
        "next_pose_contract": bool(np.allclose(poses[0, 1], second_poses[0, 0])),
    }
    checks["passed"] = bool(
        checks["image_shape"] == [2, 480, 752, 1]
        and checks["train_pose_shape"] == [2, 2, 7]
        and checks["first_new_sequence_all"]
        and checks["second_new_sequence_none"]
        and checks["first_timestamp_zero"]
        and checks["second_timestamp_exact"]
        and checks["timestamp_strictly_increasing"]
        # The official [0, 2860) crop contains 2,839 camera frames with a GT
        # match inside the upstream 100 microsecond association tolerance.
        and checks["sequence_length"] == 2839
        and checks["replicated_env_pose_equal"]
        and checks["next_pose_contract"]
    )
    return checks


def command_validate(args) -> int:
    payload = {
        "status": "passed",
        "git": git_state(),
        "dataset": validate_dataset(args),
        "official_constants": OFFICIAL,
    }
    if not payload["dataset"]["passed"]:
        payload["status"] = "failed"
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0 if payload["status"] == "passed" else 2


def command_gate(args) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_checks = validate_dataset(args)
    cold_native = evaluate_controller(
        args,
        output_dir / "cold_start_diagnostic/native",
        "cold_start_native",
        "native",
        gt_initialize_at_frame0=False,
    )
    summaries = []
    summaries.append(
        evaluate_controller(
            args,
            output_dir / "controls/native",
            "native",
            "native",
            gt_initialize_at_frame0=True,
        )
    )
    for keyframe in range(2):
        for grid_idx, grid_size in enumerate(range(20, 41, 5)):
            label = f"fixed_kf{keyframe}_grid{grid_size}"
            summaries.append(
                evaluate_controller(
                    args,
                    output_dir / "controls" / label,
                    label,
                    "fixed",
                    fixed_action=[keyframe, grid_idx],
                    gt_initialize_at_frame0=True,
                )
            )
    summaries.append(
        evaluate_controller(
            args,
            output_dir / "controls/untrained_seed23_deterministic",
            "untrained_seed23_deterministic",
            "policy",
            seed=23,
            deterministic=True,
            gt_initialize_at_frame0=True,
        )
    )
    summaries.append(
        evaluate_controller(
            args,
            output_dir / "controls/untrained_seed23_stochastic",
            "untrained_seed23_stochastic",
            "policy",
            seed=23,
            deterministic=False,
            gt_initialize_at_frame0=True,
        )
    )

    fixed_summaries = [summary for summary in summaries if summary["controller"] == "fixed"]
    gate_checks = {
        "dataset_adapter": bool(dataset_checks["passed"]),
        "all_13_controls_completed": len(summaries) == 13 and all(s["status"] == "complete" for s in summaries),
        "timestamps_exact": all(s["timestamp_mismatch_count"] == 0 for s in summaries),
        "fixed_requested_executed_exact": all(
            s["requested_executed_action_mismatch_count"] == 0 and s["action_applied_count"] > 0
            for s in fixed_summaries
        ),
        "failure_restart_from_frame_zero": all(
            s["failure_restart_contract_violation_count"] == 0 for s in summaries
        ),
        "finite_reward_telemetry": all(np.isfinite(s["reward_sum"]) for s in summaries),
    }
    gate_passed = all(gate_checks.values())
    native_summary = summaries[0]
    finite_ates = [
        s for s in fixed_summaries if s["sim3_ate_translation_rmse_m"] is not None
    ]
    best_fixed = (
        min(finite_ates, key=lambda item: item["sim3_ate_translation_rmse_m"]) if finite_ates else None
    )
    payload = {
        "status": "passed" if gate_passed else "failed",
        "scientific_quality_does_not_control_gate": True,
        "git": git_state(),
        "dataset_checks": dataset_checks,
        "gate_checks": gate_checks,
        "native": native_summary,
        "cold_start_native_diagnostic": cold_native,
        "best_fixed_by_sim3_ate": best_fixed,
        "controls": summaries,
        "contract": str(REPO_ROOT / "experiments/euroc_v101_official/contract.yaml"),
    }
    write_json(output_dir / "gate.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0 if gate_passed else 3


def _experiment_config():
    return OmegaConf.create(
        {
            "agent": {
                "n_epochs": OFFICIAL["n_epochs"],
                "gae_lambda": OFFICIAL["gae_lambda"],
                "gamma": OFFICIAL["gamma"],
                "n_steps": OFFICIAL["n_steps"],
                "ent_coef": OFFICIAL["ent_coef"],
                "vf_coef": OFFICIAL["vf_coef"],
                "max_grad_norm": OFFICIAL["max_grad_norm"],
                "batch_size": OFFICIAL["batch_size"],
                "reward": vars(reward_config()),
            }
        }
    )


def _read_reward_curve(metrics_path: Path):
    rows = []
    if metrics_path.is_file():
        for line in metrics_path.read_text().splitlines():
            payload = json.loads(line)
            if payload.get("event") == "rollout":
                rows.append(payload)
    values = [float(row["rollout/reward_per_valid"]) for row in rows]
    window = min(10, len(values))
    return {
        "rollout_count": len(rows),
        "first_window_mean_reward_per_valid": float(np.mean(values[:window])) if window else None,
        "last_window_mean_reward_per_valid": float(np.mean(values[-window:])) if window else None,
        "delta_last_minus_first": (
            float(np.mean(values[-window:]) - np.mean(values[:window])) if window else None
        ),
    }


def command_train(args) -> int:
    gate = json.loads(Path(args.gate_json).read_text())
    if gate.get("status") != "passed":
        raise RuntimeError(f"P1 gate did not pass: {args.gate_json}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = git_state()
    if state["status_porcelain"]:
        raise RuntimeError("Training requires a clean source worktree")
    if args.expected_commit and state["commit"] != args.expected_commit:
        raise RuntimeError(f"Expected commit {args.expected_commit}, found {state['commit']}")

    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    train_env = make_env(args, mode="train", num_envs=OFFICIAL["n_envs"], initialize_glog=True)
    val_env = make_env(args, mode="val", num_envs=1, initialize_glog=False)
    train_env.seed(args.seed)
    model = PPO(
        tensorboard_log=None,
        log_dir=str(output_dir),
        policy=CustomActorCriticPolicy,
        policy_kwargs=policy_kwargs(train_env),
        env=train_env,
        n_epochs=OFFICIAL["n_epochs"],
        gae_lambda=OFFICIAL["gae_lambda"],
        gamma=OFFICIAL["gamma"],
        n_steps=OFFICIAL["n_steps"],
        ent_coef=OFFICIAL["ent_coef"],
        vf_coef=OFFICIAL["vf_coef"],
        max_grad_norm=OFFICIAL["max_grad_norm"],
        batch_size=OFFICIAL["batch_size"],
        learning_rate=get_linear_fn(3e-4, 3e-5, 1.0),
        clip_range=OFFICIAL["clip_range"],
        use_sde=False,
        verbose=1,
        seed=args.seed,
        wandb_logging=args.wandb_mode != "disabled",
        wandb_tag=f"seed-{args.seed}",
        wandb_group=args.wandb_group,
        config=_experiment_config(),
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )
    policy_dir = output_dir / "Policy"
    policy_dir.mkdir(parents=True, exist_ok=True)
    model.policy.save(str(policy_dir / "iter_00000.pth"))
    train_env.save_rms(str(policy_dir / "iter_00000_rms.npz"))
    manifest = {
        "status": "training",
        "seed": args.seed,
        "git": state,
        "gate_json": str(Path(args.gate_json).resolve()),
        "official_constants": OFFICIAL,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "params_yaml": str(Path(args.params_yaml).resolve()),
        "calib_yaml": str(Path(args.calib_yaml).resolve()),
        "wandb_mode": os.environ.get("WANDB_MODE"),
        "started_unix": time.time(),
    }
    write_json(output_dir / "manifest.json", manifest)

    try:
        model.learn(
            total_timesteps=OFFICIAL["total_timesteps"],
            log_interval=100,
            eval_interval=10,
            val_env=val_env,
        )
    finally:
        if getattr(model, "wandb_logging", False) and hasattr(model, "wandb_run"):
            model.wandb_run.finish()
        train_env.close()
        val_env.close()

    evaluations = []
    for iteration in OFFICIAL["eval_iterations"]:
        checkpoint = policy_dir / f"iter_{iteration:05d}.pth"
        checkpoint_rms = policy_dir / f"iter_{iteration:05d}_rms.npz"
        if not checkpoint.is_file() or not checkpoint_rms.is_file():
            raise FileNotFoundError(f"Missing preregistered checkpoint pair at iteration {iteration}")
        evaluations.append(
            evaluate_controller(
                args,
                output_dir / "full_v101_eval" / f"iter_{iteration:05d}",
                f"seed{args.seed}_iter{iteration}",
                "policy",
                seed=args.seed,
                deterministic=True,
                checkpoint=checkpoint,
                rms_path=checkpoint_rms,
                gt_initialize_at_frame0=True,
            )
        )

    reward_curve = _read_reward_curve(output_dir / "metrics.jsonl")
    initial_eval = evaluations[0]
    final_eval = evaluations[-1]
    aggregate = {
        "status": "complete",
        "seed": args.seed,
        "git": state,
        "official_constants": OFFICIAL,
        "reward_curve": reward_curve,
        "evaluations": evaluations,
        "reward_improved": (
            reward_curve["delta_last_minus_first"] is not None
            and reward_curve["delta_last_minus_first"] > 0
        ),
        "final_ate_better_than_initial": (
            final_eval["sim3_ate_translation_rmse_m"] is not None
            and initial_eval["sim3_ate_translation_rmse_m"] is not None
            and final_eval["sim3_ate_translation_rmse_m"]
            < initial_eval["sim3_ate_translation_rmse_m"]
        ),
        "finished_unix": time.time(),
    }
    write_json(output_dir / "summary.json", aggregate)
    manifest["status"] = "complete"
    manifest["finished_unix"] = aggregate["finished_unix"]
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(aggregate, indent=2, sort_keys=True, default=_json_default))
    return 0


def _parse_seed_run(value: str):
    seed_text, separator, run_dir = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Expected SEED=/absolute/run/directory")
    seed = int(seed_text)
    if seed not in {23, 47, 71}:
        raise argparse.ArgumentTypeError(f"Unexpected seed: {seed}")
    path = Path(run_dir).resolve()
    return seed, path


def command_posthoc(args) -> int:
    """Re-evaluate completed old-commit runs with the corrected frame-0 contract."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    controls = []
    controls.append(
        evaluate_controller(
            args,
            output_dir / "controls/native",
            "native_gt_at_frame0",
            "native",
            gt_initialize_at_frame0=True,
        )
    )
    for keyframe in range(2):
        for grid_idx, grid_size in enumerate(range(20, 41, 5)):
            label = f"fixed_kf{keyframe}_grid{grid_size}_gt_at_frame0"
            controls.append(
                evaluate_controller(
                    args,
                    output_dir / "controls" / label,
                    label,
                    "fixed",
                    fixed_action=[keyframe, grid_idx],
                    gt_initialize_at_frame0=True,
                )
            )

    checkpoint_evaluations = {}
    source_runs = {}
    for seed, run_dir in args.seed_run:
        source_summary_path = run_dir / "summary.json"
        if not source_summary_path.is_file():
            raise FileNotFoundError(source_summary_path)
        source_summary = json.loads(source_summary_path.read_text())
        if source_summary.get("status") != "complete":
            raise RuntimeError(f"Training run is incomplete: {source_summary_path}")
        source_runs[str(seed)] = source_summary
        seed_evaluations = []
        for iteration in OFFICIAL["eval_iterations"]:
            checkpoint = run_dir / "Policy" / f"iter_{iteration:05d}.pth"
            checkpoint_rms = run_dir / "Policy" / f"iter_{iteration:05d}_rms.npz"
            seed_evaluations.append(
                evaluate_controller(
                    args,
                    output_dir / "checkpoints" / f"seed{seed}" / f"iter_{iteration:05d}",
                    f"seed{seed}_iter{iteration}_gt_at_frame0",
                    "policy",
                    seed=seed,
                    deterministic=True,
                    checkpoint=checkpoint,
                    rms_path=checkpoint_rms,
                    gt_initialize_at_frame0=True,
                )
            )
        checkpoint_evaluations[str(seed)] = seed_evaluations

    fixed_controls = [item for item in controls if item["controller"] == "fixed"]
    finite_fixed = [item for item in fixed_controls if item["sim3_ate_translation_rmse_m"] is not None]
    best_fixed = min(finite_fixed, key=lambda item: item["sim3_ate_translation_rmse_m"]) if finite_fixed else None
    seed_conclusions = {}
    for seed_text, evaluations in checkpoint_evaluations.items():
        initial = evaluations[0]
        final = evaluations[-1]
        best_fixed_ate = best_fixed["sim3_ate_translation_rmse_m"] if best_fixed else None
        seed_conclusions[seed_text] = {
            "reward_curve": source_runs[seed_text].get("reward_curve"),
            "initial_ate": initial["sim3_ate_translation_rmse_m"],
            "final_ate": final["sim3_ate_translation_rmse_m"],
            "final_better_than_initial": (
                initial["sim3_ate_translation_rmse_m"] is not None
                and final["sim3_ate_translation_rmse_m"] is not None
                and final["sim3_ate_translation_rmse_m"] < initial["sim3_ate_translation_rmse_m"]
            ),
            "final_better_than_best_fixed": (
                best_fixed_ate is not None
                and final["sim3_ate_translation_rmse_m"] is not None
                and final["sim3_ate_translation_rmse_m"] < best_fixed_ate
            ),
        }

    payload = {
        "status": "complete",
        "git": git_state(),
        "boundary": "frame-0 GT initialization only; no mid-sequence fresh initialization or trajectory stitching",
        "controls": controls,
        "best_fixed_by_sim3_ate": best_fixed,
        "checkpoint_evaluations": checkpoint_evaluations,
        "seed_conclusions": seed_conclusions,
    }
    write_json(output_dir / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    return 0


def add_common_arguments(parser):
    parser.add_argument(
        "--dataset-dir",
        default=os.environ.get(
            "RLVO_EUROC_DATASET_DIR",
            "/scratch/e1538633/liuyi/vio_rl_project/vio_rl/data",
        ),
    )
    parser.add_argument("--sequence", default="V1_01_easy")
    parser.add_argument(
        "--params-yaml",
        default=str(REPO_ROOT / "svo-lib/svo_env/param/euroc.yaml"),
    )
    parser.add_argument(
        "--calib-yaml",
        default=str(REPO_ROOT / "svo-lib/svo_env/param/calib/euroc_mono.yaml"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    add_common_arguments(validate_parser)
    validate_parser.add_argument("--output")
    validate_parser.set_defaults(func=command_validate)

    gate_parser = subparsers.add_parser("gate")
    add_common_arguments(gate_parser)
    gate_parser.add_argument("--output-dir", required=True)
    gate_parser.set_defaults(func=command_gate)

    train_parser = subparsers.add_parser("train")
    add_common_arguments(train_parser)
    train_parser.add_argument("--seed", type=int, choices=[23, 47, 71], required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--gate-json", required=True)
    train_parser.add_argument("--expected-commit")
    train_parser.add_argument("--torch-threads", type=int, default=1)
    train_parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    train_parser.add_argument("--wandb-project", default="rl-vo-euroc-v101")
    train_parser.add_argument("--wandb-group", default="official-ppo-v101-overfit")
    train_parser.set_defaults(func=command_train)

    posthoc_parser = subparsers.add_parser("posthoc")
    add_common_arguments(posthoc_parser)
    posthoc_parser.add_argument("--output-dir", required=True)
    posthoc_parser.add_argument(
        "--seed-run",
        action="append",
        type=_parse_seed_run,
        required=True,
        help="Repeat three times as SEED=/absolute/training/run/directory",
    )
    posthoc_parser.set_defaults(func=command_posthoc)
    return parser.parse_args()


def main():
    args = parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
