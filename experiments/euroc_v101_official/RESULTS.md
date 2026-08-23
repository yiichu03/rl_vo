# Official RL-VO on EuRoC V101: results

## Verdict

The bounded three-seed diagnostic is an engineering pass and a scientific
failure.  The official RL-VO PPO loop did not produce a trained checkpoint
that both completed V101 and beat the best complete fixed action.  Training
reward decreased for all three seeds.

This is a EuRoC single-sequence overfit diagnostic, not a reproduction of the
paper's TartanAir training result.

## Execution record

| Role | PBS job | Exit | Source commit | Wall time |
|---|---:|---:|---|---:|
| Mechanics and cold-start gate | 580170 | 0 | `93cf188` | 00:04:40 |
| PPO seed 47 | 580171 | 0 | `93cf188` | 01:04:00 |
| PPO seed 23 | 580172 | 0 | `93cf188` | 01:48:21 |
| PPO seed 71 | 580173 | 0 | `93cf188` | 01:04:46 |
| Frame-zero full-route post-hoc evaluation | 580188 | 0 | `8514f16` | 00:07:48 |

Each seed ran the registered 100 PPO updates: 100 environments times 250
steps per update, for 2.5 million raw environment steps.  Checkpoints were
evaluated only at iterations 0, 10, 50, and 100.

## Training reward

| Seed | First-window reward/valid | Last-window reward/valid | Delta | Improved |
|---:|---:|---:|---:|:---:|
| 23 | 0.001921383 | 0.001902688 | -0.000018694 | no |
| 47 | 0.001901704 | 0.001888974 | -0.000012730 | no |
| 71 | 0.001887678 | 0.001782585 | -0.000105093 | no |

The first cold-start rollout is included in the runner's registered summary.
Excluding it does not reverse the sign of the early-versus-late comparison.

## Full-route controls

Only rows that completed the route with at least 0.95 tracking coverage and
without a tracking failure are scientifically comparable.  Sim(3) is primary
because this is monocular VO.

| Controller | Coverage | Sim(3) ATE m | RPE-t m | RPE-r deg | Mean features | Keyframe ratio |
|---|---:|---:|---:|---:|---:|---:|
| Native | 0.968651 | 0.342733 | 0.005496 | 0.075819 | 120.354 | 0.233968 |
| fixed keyframe1/grid20 | 0.968651 | **0.157251** | 0.003803 | 0.061813 | 131.105 | 0.968992 |
| fixed keyframe1/grid25 | 0.968651 | 0.268511 | 0.003933 | 0.042352 | 130.621 | 0.968992 |
| fixed keyframe1/grid30 | 0.968651 | 0.295661 | 0.005392 | 0.082944 | 129.965 | 0.968992 |
| fixed keyframe1/grid35 | 0.968651 | 0.362553 | 0.004663 | 0.044885 | 128.881 | 0.968992 |
| fixed keyframe1/grid40 | 0.968651 | 0.404355 | 0.005684 | 0.066217 | 125.982 | 0.968992 |

All five fixed `keyframe=0` controls failed after roughly 2--3% coverage and
are excluded from ATE ranking.  The action setter nevertheless had a clear
effect on feature count and requested/executed action mismatches were zero.

## Policy checkpoints

| Seed | Iteration | Coverage | End | Sim(3) ATE m | Keyframe ratio | Comparable |
|---:|---:|---:|---|---:|---:|:---:|
| 23 | 0 | 0.020077 | tracking failure | 0.051040 | 0.006024 | no |
| 23 | 10 | 0.912998 | tracking failure | 0.264517 | 0.923820 | no |
| 23 | 50 | 0.152518 | tracking failure | 0.784211 | 0.020446 | no |
| 23 | 100 | 0.025361 | tracking failure | 0.004306 | 0.005814 | no |
| 47 | 0 | 0.968651 | full route | 0.202455 | 0.968992 | yes, untrained |
| 47 | 10 | 0.457203 | tracking failure | 0.876043 | 0.403571 | no |
| 47 | 50 | 0.022895 | tracking failure | 0.003632 | 0.006098 | no |
| 47 | 100 | 0.016907 | tracking failure | 0.003351 | 0.006803 | no |
| 71 | 0 | 0.968651 | full route | 0.268511 | 0.968992 | yes, untrained |
| 71 | 10 | 0.968651 | full route | 0.268511 | 0.968992 | yes, fixed action |
| 71 | 50 | 0.020430 | tracking failure | 0.021972 | 0.006024 | no |
| 71 | 100 | 0.018669 | tracking failure | 0.003427 | 0.006579 | no |

Seed 71 iteration 10 executed only `keyframe=1, grid25`; it is worse than the
best fixed action and does not demonstrate adaptive behavior.  No trained
checkpoint beats fixed `keyframe=1, grid20`.

## Failure mode and interpretation boundary

All three policies eventually moved toward nearly never selecting a keyframe,
after which V101 tracking failed.  This is consistent with a reward/credit
loophole: the official implementation masks the transition that causes a
tracking failure, while keyframes carry a direct cost.  The experiment
preserved that behavior deliberately, so the evidence does not isolate PPO,
RMS, data diversity, and terminal credit as independent causes.

There is no evidence of the earlier abrupt train/evaluation RMS-alias failure:
each checkpoint has its own saved RMS and checksum, and the rollout reward
decline is gradual.  Frozen or redesigned RMS could still affect another
algorithm, but it is not the leading explanation for this result.

The first version of job 580188's aggregate compared truncated-prefix ATE and
therefore produced false-positive booleans.  The raw per-controller summaries
are retained.  `scientific_assessment.json`, produced by the coverage-aware
assessment code, is the authoritative conclusion.

## Source artifacts

The local source of truth is under:

```text
/scratch/e1538633/liuyi/vio_rl_project/vio_rl/outputs/hpc/rl_vo/euroc_v101_official/
  p1_580170/gate.json
  seed23_580172/{metrics.jsonl,summary.json,Policy/,wandb/}
  seed47_580171/{metrics.jsonl,summary.json,Policy/,wandb/}
  seed71_580173/{metrics.jsonl,summary.json,Policy/,wandb/}
  posthoc_frame0_eval_580188/{summary.json,scientific_assessment.json}
```

PBS logs are under `outputs/hpc/logs/rlvo_v101_*.hopper-m-02.log`.  The offline
runs were synchronized after all jobs finished:

- [seed 23 / b8m1zaho](https://wandb.ai/yiichu03-nus/rl-vo-euroc-v101/runs/b8m1zaho)
- [seed 47 / 7r1y1n14](https://wandb.ai/yiichu03-nus/rl-vo-euroc-v101/runs/7r1y1n14)
- [seed 71 / uq9gfcrg](https://wandb.ai/yiichu03-nus/rl-vo-euroc-v101/runs/uq9gfcrg)

## Stop decision

Do not add official-RL-VO EuRoC seeds or tune this branch.  Return to the
RL-VIO project and require causal action headroom, observation predictability,
terminal credit, and authentic prefix-state sampling before another training
run.
