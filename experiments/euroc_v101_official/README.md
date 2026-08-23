# Official RL-VO EuRoC V1_01 Overfit Diagnostic

This experiment asks one narrow question: does the published RL-VO PPO loop
show a rising reward signal and a corresponding full-route monocular VO
improvement when trained repeatedly on EuRoC `V1_01_easy` camera 0?

It is not a reproduction of the paper's training result. The paper trained on
TartanAir; this branch deliberately performs a single-sequence EuRoC overfit
diagnostic.

## Frozen scientific contract

The following are unchanged from official RL-VO:

- PPO and the asymmetric actor/privileged-critic network;
- the 564-dimensional actor observation and 7 critic-only GT/future values;
- `MultiDiscrete([2,5])`: keyframe `{0,1}` and grid
  `{20,25,30,35,40}`;
- local Sim(3)-aligned position reward and keyframe penalty;
- RMS update behavior and PPO hyperparameters.

The bounded run is 100 official updates (`2.5M` raw environment steps) for
seeds 23, 47, and 71. Full V1_01 deterministic evaluations are preregistered
at iterations 0, 10, 50, and 100. No result-driven hyperparameter retry is
allowed. See `contract.yaml` for exact values.

## Engineering-only changes

- allow the official EuRoC loader to run in training mode and replicate V1_01
  over 100 vector slots;
- pass real camera timestamps to SVO instead of a synthetic 30 Hz clock;
- on tracking failure, start the next episode at V1_01 frame zero rather than
  fresh-initializing from the current middle frame;
- make the existing pybind reset arrays contiguous and shape-correct;
- add local JSONL/W&B telemetry and canonical full-route Sim(3) ATE/RPE,
  coverage, action, feature-count, tracking, and scale reports.

The official `valid_mask` behavior is intentionally preserved: the action that
causes tracking failure is absent from PPO updates. The action and failure are
now logged explicitly so a reward-only success cannot hide this limitation.

## P1 gate and controls

P1 runs the loader/timestamp/action tests and then one V1_01 evaluation for:

- native SVO keyframe/grid heuristic;
- all ten fixed legal RL actions;
- the untrained seed-23 policy, deterministic and stochastic.

The scientific controls and trained-policy evaluations use GT only to set the
pose of sequence frame zero. A separate cold-start native diagnostic is kept.
This separation is necessary because V1_01 cold monocular initialization can
fail before the policy receives a valid action; no evaluation may initialize
from a later frame or stitch reset subtrajectories.

The gate checks mechanics only. Weak control accuracy is a scientific result,
not an infrastructure failure. Seeds are submitted with `afterok` on P1.

## Result interpretation

- Reward and Sim(3) accuracy improve across seeds: the official loop is viable
  on V1_01 and our VIO difficulty is likely downstream of integration/design.
- Reward improves but Sim(3) accuracy does not: the official local reward is a
  proxy mismatch on EuRoC.
- Neither improves: EuRoC single-sequence data, terminal credit, or official
  action/reward itself is a plausible bottleneck; do not infer that PPO alone
  is the cause.

Local `metrics.jsonl`, checkpoint RMS files/checksums, per-frame compressed
traces, and summary JSON files are the source of truth. W&B runs default to
offline mode so network or credentials cannot invalidate an experiment.
