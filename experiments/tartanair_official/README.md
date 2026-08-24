# Official RL-VO TartanAir Reproduction Diagnostic

> **Completed result (2026-08-24):** both 25M-step seeds finished, and the
> preregistered deterministic-validation vote was **0/2**. See
> [RESULTS.md](RESULTS.md) for the full result, RMS audit, limitations, and
> implications for RL-VIO.

This experiment asks whether the official RL-VO PPO loop exhibits a reproducible
learning signal on its native TartanAir domain. It deliberately does not test an
RL-VIO action or modify the published action, observation, reward, PPO, or online
RMS schedule.

## Data audit

The source consists only of the inputs consumed by official RL-VO: 640x480
left-camera grayscale JPEG images, `pose_left.txt`, and the official pinhole
calibration. The four physical roots contain:

| split | trajectories | frames |
|---|---:|---:|
| Easy train | 189 | 172,061 |
| Hard train | 148 | 107,926 |
| Easy validation | 17 | 17,592 |
| Hard validation | 15 | 9,058 |
| Total | 369 | 306,637 |

All 369 image counts equal their pose-row counts and manifest counts. The 32
validation trajectories exactly match `dataloader/tartan_loader.py:test_split`.
Because official `train.py` expects one dataset root, `dataset_view.py` creates a
zero-copy symlink view; source data is never rewritten or duplicated.

## Frozen scientific contract

Both PBS jobs run the same official 25M-step configuration and differ only by
seed (`0` and `23`): 100 environments, 250 steps per rollout, PPO, the published
asymmetric actor/critic, `MultiDiscrete([2,5])` keyframe/grid action, local
Sim(3)-aligned position reward, keyframe penalty, and the official online RMS
mechanism (nominal promotion every ten iterations). The completed-run audit found
that object aliasing makes the active RMS drift between promotions; see
`RESULTS.md`. See `contract.yaml` for the frozen values.

Measurement-only additions are an iteration-0 deterministic validation, a local
JSONL mirror of scalar W&B metrics, RMS checksums, data validation, and pybind
array-contiguity/reset-shape corrections required by the project container.
Tracking-failure transitions remain masked exactly as in official RL-VO.

## Preregistered interpretation

For each seed, compare the median deterministic validation metrics from
iterations 0-100 with iterations 900-1000. A seed votes positive when reward
improves by at least 5%, first-subtrajectory ATE does not regress by more than
5%, and valid-stage coverage falls by no more than five percentage points.

- 2/2 positive: reproducible official-domain learning signal.
- 1/2 positive: suggestive but seed-sensitive.
- 0/2 positive: no reproduced learning signal under the official contract.

The official validation ATE is a partial, first-subtrajectory Sim(3) diagnostic;
it is not by itself a complete benchmark metric. Reward, ATE, and coverage must
be read together.

W&B defaults to offline mode so network availability cannot fail training. The
local run directory contains `metrics.jsonl`, `Policy/` checkpoints/RMS files,
the W&B run, the data audit, and a preregistered `summary.json` after completion.

## Result status

Jobs `580495` (seed 0) and `580496` (seed 23) both completed 25M steps and 101
deterministic evaluations. Neither seed passed the frozen gate, so this branch's
conclusion is `no_reproducible_learning_signal` under the tested official-style
contract. This result does not isolate a single cause and does not claim that the
published method can never learn; the precise evidence boundary is documented in
[RESULTS.md](RESULTS.md).
