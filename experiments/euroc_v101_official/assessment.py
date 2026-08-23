"""Coverage-aware scientific assessment for the EuRoC V101 diagnostic."""

from __future__ import annotations

import math
from typing import Any


MIN_FULL_ROUTE_COVERAGE = 0.95


def comparison_eligibility(result: dict[str, Any]) -> tuple[bool, list[str]]:
    """Return whether an evaluation is comparable on the full V101 route."""

    reasons: list[str] = []
    if not bool(result.get("full_route_completed")):
        reasons.append("route_not_completed")
    if int(result.get("tracking_failure_count") or 0) != 0:
        reasons.append("tracking_failure")
    if int(result.get("termination_count") or 0) != 0:
        reasons.append("terminated_early")

    coverage = result.get("tracking_coverage")
    if coverage is None or not math.isfinite(float(coverage)):
        reasons.append("coverage_missing_or_nonfinite")
    elif float(coverage) < MIN_FULL_ROUTE_COVERAGE:
        reasons.append("coverage_below_0p95")

    ate = result.get("sim3_ate_translation_rmse_m")
    if ate is None or not math.isfinite(float(ate)):
        reasons.append("sim3_ate_missing_or_nonfinite")
    return not reasons, reasons


def annotated_result(result: dict[str, Any]) -> dict[str, Any]:
    eligible, reasons = comparison_eligibility(result)
    return {
        **result,
        "scientific_comparison_eligible": eligible,
        "scientific_comparison_exclusion_reasons": reasons,
    }


def _compact(result: dict[str, Any]) -> dict[str, Any]:
    eligible, reasons = comparison_eligibility(result)
    keys = (
        "label",
        "controller",
        "seed",
        "full_route_completed",
        "tracking_coverage",
        "tracking_failure_count",
        "termination_count",
        "sim3_ate_translation_rmse_m",
        "sim3_rpe_translation_rmse_m",
        "sim3_rpe_rotation_rmse_deg",
        "sim3_scale",
        "mean_actual_feature_count",
        "actual_keyframe_ratio",
        "action_counts_when_applied",
        "unique_actions_when_applied",
        "policy_probability_margin_mean",
        "policy_probability_margin_min",
    )
    compact = {key: result.get(key) for key in keys}
    compact["scientific_comparison_eligible"] = eligible
    compact["scientific_comparison_exclusion_reasons"] = reasons
    return compact


def build_scientific_assessment(raw_summary: dict[str, Any]) -> dict[str, Any]:
    """Derive a non-cherry-picked assessment without rerunning the estimator."""

    controls = [annotated_result(item) for item in raw_summary.get("controls", [])]
    eligible_fixed = [
        item
        for item in controls
        if item.get("controller") == "fixed"
        and item["scientific_comparison_eligible"]
    ]
    best_fixed = (
        min(eligible_fixed, key=lambda item: item["sim3_ate_translation_rmse_m"])
        if eligible_fixed
        else None
    )
    best_fixed_ate = (
        float(best_fixed["sim3_ate_translation_rmse_m"])
        if best_fixed is not None
        else None
    )

    checkpoint_evaluations: dict[str, list[dict[str, Any]]] = {}
    seed_conclusions: dict[str, dict[str, Any]] = {}
    any_trained_better = False
    for seed_text, raw_evaluations in raw_summary.get(
        "checkpoint_evaluations", {}
    ).items():
        evaluations = [annotated_result(item) for item in raw_evaluations]
        checkpoint_evaluations[str(seed_text)] = evaluations
        initial = evaluations[0]
        final = evaluations[-1]
        trained = evaluations[1:]
        eligible_trained = [
            item for item in trained if item["scientific_comparison_eligible"]
        ]
        best_trained = (
            min(
                eligible_trained,
                key=lambda item: item["sim3_ate_translation_rmse_m"],
            )
            if eligible_trained
            else None
        )
        trained_better = bool(
            best_trained is not None
            and best_fixed_ate is not None
            and float(best_trained["sim3_ate_translation_rmse_m"])
            < best_fixed_ate
        )
        any_trained_better = any_trained_better or trained_better
        old_conclusion = raw_summary.get("seed_conclusions", {}).get(
            str(seed_text), {}
        )
        reward_curve = old_conclusion.get("reward_curve")
        reward_delta = (
            reward_curve.get("delta_last_minus_first")
            if isinstance(reward_curve, dict)
            else None
        )
        seed_conclusions[str(seed_text)] = {
            "reward_curve": reward_curve,
            "reward_improved": reward_delta is not None and reward_delta > 0,
            "initial": _compact(initial),
            "final": _compact(final),
            "eligible_trained_checkpoint_labels": [
                item.get("label") for item in eligible_trained
            ],
            "best_eligible_trained_checkpoint": (
                _compact(best_trained) if best_trained is not None else None
            ),
            "any_trained_checkpoint_better_than_best_fixed": trained_better,
        }

    reward_improved_all_seeds = bool(seed_conclusions) and all(
        item["reward_improved"] for item in seed_conclusions.values()
    )
    return {
        "schema": "rl_vo.euroc_v101_official.scientific_assessment.v1",
        "status": "complete",
        "comparison_contract": {
            "primary_alignment": "Sim(3)",
            "minimum_tracking_coverage": MIN_FULL_ROUTE_COVERAGE,
            "requires_full_route_completed": True,
            "requires_zero_tracking_failures": True,
            "requires_zero_terminations": True,
            "truncated_prefix_ate_is_ineligible": True,
            "checkpoint_schedule": [0, 10, 50, 100],
        },
        "native": next(
            (_compact(item) for item in controls if item.get("controller") == "native"),
            None,
        ),
        "best_eligible_fixed_by_sim3_ate": (
            _compact(best_fixed) if best_fixed is not None else None
        ),
        "eligible_fixed_count": len(eligible_fixed),
        "checkpoint_evaluations": {
            seed: [_compact(item) for item in items]
            for seed, items in checkpoint_evaluations.items()
        },
        "seed_conclusions": seed_conclusions,
        "reward_improved_all_seeds": reward_improved_all_seeds,
        "any_trained_checkpoint_better_than_best_fixed": any_trained_better,
        "scientific_success": reward_improved_all_seeds and any_trained_better,
        "verdict": (
            "official_loop_viable"
            if reward_improved_all_seeds and any_trained_better
            else "no_viable_trained_policy"
        ),
    }
