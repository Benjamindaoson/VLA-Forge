# M06 Independent Robot Policy Regression & Release Evaluation

**Engineering acceptance: PASS.** All four frozen policy identities have 400 real FP32 Eager LIBERO/MuJoCo rollouts each (1,600 total). The registered algorithm-quality decision is separate from engineering completeness.

Protocol SHA-256: `d70bed88abf3cd4835140d3567f596cf38baf7053b9bcf9ee447a3a53a1b17b1`. Local VLA code commit `3f7ed1a8f5e20871a39b205e7822ec006cef6583`; local ModelOps commit `fafcd9e24e38c1fd2a911222ba85ecb00f0ee42f`. GPU: `NVIDIA GeForce RTX 3090, 24576 MiB, 580.173.02`.
Published source: [VLA-Forge M06](https://github.com/Benjamindaoson/VLA-Forge/commit/0872811c1453d0f5a76121c05acc016c8ae870f3) and [ModelOps M06](https://github.com/Benjamindaoson/ModelOps-Platform/commit/92d9ee614253053b347e065b36b18db8bc201a0f). GitHub connector publication commits differ from the cloud-local commit IDs; the key source blob hashes were independently checked.

## 40-task benchmark

| Policy | Spatial | Object | Goal | Long | Overall | ModelOps decision |
|---|---:|---:|---:|---:|---:|---|
| frozen | 61/100 | 66/100 | 77/100 | 43/100 | 247/400 | PROMOTE |
| standard | 59/100 | 68/100 | 78/100 | 42/100 | 247/400 | INSUFFICIENT_EVIDENCE |
| corrective | 39/100 | 38/100 | 41/100 | 12/100 | 130/400 | REJECT |
| replay | 54/100 | 62/100 | 68/100 | 42/100 | 226/400 | INSUFFICIENT_EVIDENCE |

Every task has ten locked init states; task-level Wilson intervals are in `task_metrics.csv`. All failure episodes remain in the denominator and retain original video/action/physics evidence.

## Target repair and old-skill retention

Target: Goal task 6, ten matched states. One-sided exact paired discordance tests use Holm adjustment across three candidates; percent changes alone are not proof.

| Candidate | Target Frozen→Candidate | Paired difference | Improved / worsened | Holm p | Legacy Δ | Task-cluster lower 95% | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| standard | 4/10 → 5/10 | +0.100 | 1/0 | 1 | -0.003 | -0.026 | INSUFFICIENT_EVIDENCE |
| corrective | 4/10 → 0/10 | -0.400 | 0/4 | 1 | -0.290 | -0.356 | REJECT |
| replay | 4/10 → 5/10 | +0.100 | 3/2 | 1 | -0.056 | -0.097 | INSUFFICIENT_EVIDENCE |

The legacy pool has 39 task clusters, not 390 independent tasks. The 3-point noninferiority margin, target significance rule, latency ceiling and safety rule were frozen before test opening. Worst task regressions and newly failed cases are in `skill_retention.json`; complete paired action differences are in `paired_outcomes.csv`.

Historical M05 development scores (Frozen 3/8, Standard 6/8, Corrective 1/8, Replay 5/8) are exploratory and were never substituted for the M06 results.

## Runtime and cost

| Policy | Episodes | Episode GPU-hours | Chunk p50/p95/p99 (ms) | Peak allocated GPU (GiB) | Mean sampled GPU util |
|---|---:|---:|---:|---:|---:|
| frozen | 400 | 5.290 | 294.2/329.5/371.9 | 1.75 | 1.1% |
| standard | 400 | 5.317 | 341.4/381.1/430.1 | 3.41 | 1.3% |
| corrective | 400 | 6.391 | 341.7/382.8/433.9 | 3.40 | 1.2% |
| replay | 400 | 5.414 | 337.3/367.4/427.1 | 3.40 | 1.0% |

Measured episode-wall total: 22.411 GPU-hours; this excludes startup, model loading, idle and failed retries. Model loading and pure encoder CPU time were not timed separately, so those phase values remain N/A; `performance.json` contains measured preprocessing, policy, MuJoCo, control-cycle and resource samples plus an explicitly labeled video/loop residual upper bound. The corrected worker wall consumed 22.617 GPU-hours, including 0.207 hours outside measured episodes. The quarantined native-BF16 technical attempt consumed 0.357 additional GPU wall-hours and contributes zero formal episodes. Actual rental invoice/rate is unavailable; monetary cost remains N/A. Error/retry count: 0.

## Visual and reproducibility evidence

- Private film: `demo/M06_demo_200s.mp4` (193.1s, 1920×1080 H.264, 54.6 MiB, SHA-256 `483ad5daddb3e534fd277efdbdc5806d8069742b99b9f576043bb2de10b6d382`).
- Offline dashboard: `visualization/index.html`. Source episode IDs, source time windows and hashes: `video_sources.json`.
- `formal/receipts/`, `formal/attempts.jsonl`, `formal/registry/` retain real videos, executed actions, 8D observations, full available simulator state and resource events. `validation.json` binds all 1,600 artifacts.
- Reproduce on the authorized GPU host: `/data/VLA-Forge/Evidence/M06/reproduce.sh`. The runner skips only receipts that still match the frozen model/protocol/artifact hashes.
- Revalidate/aggregate: `PYTHONPATH=benchmarks/robot-vla-engine/src .venv/bin/python benchmarks/robot-vla-engine/scripts/finalize_m06.py`. ModelOps admission: `.venv/bin/python -m pipelines.robotics.ingest_m06_release --evidence /data/VLA-Forge/Evidence/M06 --registry /data/VLA-Forge/Evidence/M05/modelops/registry --output /data/VLA-Forge/Evidence/M06/modelops/evaluations` from `/data/ModelOps-Platform`.

## Boundaries

- The locked states are disjoint from recorded M02–M05 project development groups and exclude six retired contaminated source groups. LIBERO expert episode IDs have no verified mapping to fixed physical init IDs, and foundation-model pretraining exposure is unknown. Thus this is an independent project post-training regression test, not proof that the base model never saw related states.
- All video and asset files remain private under the M04 source-license review. No public multimedia redistribution or real-robot deployment is asserted.
- A candidate marked `INSUFFICIENT_EVIDENCE` did not prove noninferiority merely because an interval includes zero. Only `PROMOTE` is eligible for the authorized simulation service; ModelOps records gate decisions separately.

## Resume bullet, based on measured evidence

> Built a hash-bound, resumable SmolVLA→LIBERO regression pipeline on RTX 3090; executed 1,600 matched MuJoCo rollouts across 40 tasks and four frozen policy versions, retained dual-camera video/action/physical-state traces, and connected paired repair and task-cluster retention analysis to a conservative ModelOps simulation release gate (standard: INSUFFICIENT_EVIDENCE, corrective: REJECT, replay: INSUFFICIENT_EVIDENCE).

M06 stops here. New training or M07 requires a separate instruction.
