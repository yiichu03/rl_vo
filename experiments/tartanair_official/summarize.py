#!/usr/bin/env python3
"""Summarize preregistered official RL-VO TartanAir learning signals."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List


def _median(records: List[Dict[str, object]], key: str) -> float:
    return float(statistics.median(float(record[key]) for record in records))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics_path = args.run_dir / "metrics.jsonl"
    records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    eval_records = [record for record in records if record.get("event") == "official_partial_eval"]
    eval_records.sort(key=lambda record: int(record["iteration"]))
    early = [record for record in eval_records if 0 <= int(record["iteration"]) <= 100]
    late = [record for record in eval_records if 900 <= int(record["iteration"]) <= 1000]
    if not early or not late:
        raise RuntimeError(f"incomplete evaluation windows: early={len(early)}, late={len(late)}")

    early_reward = _median(early, "eval/mean_summed_reward")
    late_reward = _median(late, "eval/mean_summed_reward")
    reward_relative_gain = (late_reward - early_reward) / max(abs(early_reward), 1e-3)
    early_ate = _median(early, "eval/mean_ate")
    late_ate = _median(late, "eval/mean_ate")
    ate_relative_change = (late_ate - early_ate) / max(abs(early_ate), 1e-6)
    early_coverage = _median(early, "eval/ratio_valid_stages")
    late_coverage = _median(late, "eval/ratio_valid_stages")
    coverage_change = late_coverage - early_coverage
    passed = reward_relative_gain >= 0.05 and ate_relative_change <= 0.05 and coverage_change >= -0.05

    payload = {
        "status": "complete" if int(eval_records[-1]["iteration"]) >= 1000 else "incomplete",
        "seed": args.seed,
        "evaluation_count": len(eval_records),
        "last_iteration": int(eval_records[-1]["iteration"]),
        "early_window_count": len(early),
        "late_window_count": len(late),
        "early_reward_median": early_reward,
        "late_reward_median": late_reward,
        "reward_relative_gain": reward_relative_gain,
        "early_ate_median": early_ate,
        "late_ate_median": late_ate,
        "ate_relative_change": ate_relative_change,
        "early_coverage_median": early_coverage,
        "late_coverage_median": late_coverage,
        "coverage_change": coverage_change,
        "preregistered_seed_vote": bool(passed),
        "metrics_path": str(metrics_path.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
