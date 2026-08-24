# RL-VO TartanAir 25M Two-Seed Results

## Executive conclusion

The official-style RL-VO PPO loop did **not** reproduce a positive deterministic
validation learning signal on the available TartanAir subset. Both 25M-step runs
completed, but both failed the preregistered gate: deterministic validation reward
fell by 9.68% for seed 0 and 7.84% for seed 23. The resulting vote is **0/2**, which
maps to `no_reproducible_learning_signal` in `contract.yaml`.

This is not evidence that the optimizer did nothing. Training-rollout reward rose
slightly, policy entropy collapsed, and action usage changed substantially. It is
evidence that those changes did not translate into a reproducible improvement on
the frozen deterministic validation readout.

## Material Passport

| field | value |
|---|---|
| Origin Skill | experiment-agent |
| Origin Mode | validate |
| Origin Date | 2026-08-24 |
| Verification Status | ANALYZED |
| Version Label | `rlvo_tartanair_25m_two_seed_v1` |
| Source branch | `research/rlvo-official-tartanair-repro` |
| Experiment implementation | `ff59162`, with PBS preflight fix `5672e18` |
| PBS jobs | seed 0: `580495`; seed 23: `580496` |
| Scientific variable | random seed only (`0`, `23`) |
| Total budget | 25M environment steps per seed; 50M combined |
| Confidence | CAUTION |

## Preregistered readout

The frozen rule compares medians over deterministic evaluations at iterations
0-100 and 900-1000. A seed votes positive only if validation reward improves by
at least 5%, first-subtrajectory Sim(3) ATE regresses by no more than 5%, and
valid-stage coverage drops by no more than five percentage points.

| seed / job | eval reward early -> late | reward change | Sim(3) ATE early -> late | ATE change | coverage early -> late | vote |
|---|---:|---:|---:|---:|---:|---:|
| 0 / 580495 | 0.7651 -> 0.6911 | -9.68% | 2.5802 -> 2.1420 | -16.98% (better) | 92.24% -> 91.43% | no |
| 23 / 580496 | 0.7989 -> 0.7363 | -7.84% | 2.1168 -> 2.3018 | +8.74% (worse) | 91.34% -> 91.21% | no |

Both runs reached iteration 1000 and produced 101 deterministic evaluations,
including iteration 0. Seed 0's partial ATE improved, but its validation reward
still declined; seed 23 declined on reward and worsened on partial ATE. Coverage
was broadly stable in both runs.

## What happened during training

The following is a supplementary diagnostic, not a replacement for the
preregistered deterministic gate. Values are medians over the first and last 100
training rollouts.

| seed | rollout reward | valid-stage coverage | keyframe-action fraction | entropy proxy `-entropy_loss` | approximate KL |
|---|---:|---:|---:|---:|---:|
| 0 | 22.9009 -> 23.2761 (+1.64%) | 87.90% -> 91.18% | 0.4910 -> 0.2437 | 1.7452 -> 0.1158 | 0.00353 -> 0.00037 |
| 23 | 23.7541 -> 23.8123 (+0.25%) | 89.08% -> 90.34% | 0.7486 -> 0.3209 | 1.7053 -> 0.1351 | 0.00458 -> 0.00040 |

The deterministic evaluation keyframe-action fraction also moved strongly:
0.6045 to 0.2682 for seed 0 and 0.9578 to 0.3462 for seed 23. Therefore the
final policy was not still the initial random policy. PPO learned a lower-entropy,
less-keyframe behavior, but the late policy updates became small and the behavior
did not improve deterministic validation reward.

Across the 101 evaluation checkpoints, Pearson correlation between validation
reward and partial ATE was approximately 0.03 for seed 0 and +0.45 for seed 23.
Since lower ATE is better, the positive seed-23 correlation is contrary to using
reward as a direct accuracy proxy. These checkpoint series are autocorrelated, so
the correlations are descriptive diagnostics only; no independence or causal
claim is made.

## RMS implementation audit

The official wrapper creates separate `obs_rms` and `obs_rms_new` objects, but
`update_rms()` performs the assignment below rather than copying arrays:

```python
self.obs_rms = self.obs_rms_new
```

After the first promotion, the two names alias the same mutable object. Subsequent
training calls to `obs_rms_new.update(...)` therefore also mutate the active
normalizer between nominal ten-iteration promotions. This is visible in both
runs: 991 distinct rollout RMS checksums were observed over 1000 rollouts. The
experiment preserved this behavior because its purpose was an official-contract
diagnostic.

There was no EuRoC-style late catastrophic reward cliff in these two runs: the
largest normalization-related discontinuity was near the first promotion, after
which validation reward declined more gradually. RMS aliasing is therefore an
important confound, but these runs do **not** isolate it as the sole cause of the
negative result.

## Metric and inference limits

- The official validation ATE is Sim(3)-aligned, which is appropriate for
  monocular VO, but it covers only the first recovered subtrajectory. It is not a
  complete-trajectory benchmark score.
- Tracking-failure transitions remain removed by the official `valid_mask`.
  Consequently, reward and ATE are conditioned on surviving valid stages and can
  under-represent actions that cause failure.
- Two seeds support the frozen 0/2 decision rule, but do not establish a universal
  claim about RL-VO, all TartanAir data, or the published paper.
- The experiment changes domain size and diversity together and does not identify
  whether RMS, reward alignment, action choice, terminal credit, or another
  mechanism caused the outcome.
- W&B ran in offline mode. Checkpoints, RMS snapshots, JSONL metrics, and offline
  W&B runs remain on Hopper and are intentionally not committed to Git. The local
  run IDs are `g65xlahk` (seed 0) and `3h5obpb3` (seed 23).

## Fallacy scan

All 11 validation checks were reviewed. The main cautions are survivorship/collider
bias from `valid_mask`, construct validity of partial first-subtrajectory ATE,
autocorrelation/non-independence across checkpoints, and correlation-versus-
causation for RMS and reward diagnostics. The preregistered windows and vote rule
limit garden-of-forking-paths and look-elsewhere risk; supplemental diagnostics
were not used to redefine the gate after seeing the results.

## Implication for RL-VIO

The useful conclusion is narrow: substantially more and more diverse monocular
training data did not, by itself, make this official-style PPO contract show a
reproducible validation improvement. It does not justify another unbounded PPO
scan. The RL-VIO project should first finish the V14 action-headroom experiment:
if different states genuinely prefer different actions for final ATE, the next
step is a reward/credit pipeline explicitly validated against that final metric;
if V14 finds no headroom, reward redesign cannot rescue that action space.

## Artifact provenance

Cluster artifact roots:

```text
/scratch/e1538633/liuyi/vio_rl_project/vio_rl/outputs/hpc/rl_vo/tartanair_official/seed0_580495
/scratch/e1538633/liuyi/vio_rl_project/vio_rl/outputs/hpc/rl_vo/tartanair_official/seed23_580496
```

SHA-256 checksums:

```text
83bb056c0a02810cc39c31d80ac6be02e25ad9b821274c91a8e1e86012a73b03  seed0_580495/summary.json
9a4cfc6ff0ba3c04d57a64b848ef7dbdb3b848f54d7e9956a0ecb2319cb70c61  seed23_580496/summary.json
9cf16ab993ead6d44f21dba3b6f39f6134ee197a6a7765b116c93ef3198c3162  seed0_580495/train/metrics.jsonl
270b52ede938000917325c3dacec224aa062909d660721afb206d0001fe80e63  seed23_580496/train/metrics.jsonl
```
