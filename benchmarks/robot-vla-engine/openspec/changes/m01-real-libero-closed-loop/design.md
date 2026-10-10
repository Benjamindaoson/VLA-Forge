## Context

The project already has a pinned LeRobot integration, `libero_batch`, an observable episode recorder, registry/event recording, latency profiling, and GPU telemetry. Its main evaluator deliberately requires a checkpoint owned by a registered training run, while the official `lerobot/smolvla_libero` checkpoint is a standalone external baseline. Remote code and the local working branch have matching hashes for the modules being reused, but the remote Git checkout is older and contains pre-existing untracked qualification reports that must be preserved.

## Goals / Non-Goals

**Goals:** Run a smoke-gated, official-checkpoint evaluation on one fixed task per LIBERO suite, save per-episode trajectories and results, encode genuine simulator videos, and produce self-contained reproducibility evidence.

**Non-Goals:** Modify previous experiment records; alter the formal 40-task benchmark; fine-tune the model; run another training queue; claim capability or benchmark results from software tests.

## Decisions

- Add a dedicated M01 entry point instead of weakening the training-ownership checks in `evaluate_policy.py`.
- Use the checkpoint's own LeRobot config and pre/post processors, and adapt the pinned LIBERO environment through the existing `libero_batch` conversion and `record_episode` code. Confirm dimensions from the loaded config and actual tensor/action values at runtime.
- Freeze suite/task selections in the M01 manifest, with 3 smoke episodes followed by 10 episodes per suite. Persist evidence after each episode so an interrupted run can resume only missing episodes.
- Keep model weights on the server's data disk and keep lightweight source/spec/report artifacts in Git.

## Risks / Trade-offs

- The official checkpoint's processor schema may differ from training-owned checkpoints → load processor assets directly from the resolved official snapshot and fail closed on missing/shape mismatch.
- Model failures can produce low or zero success → preserve all observed failures and report the actual success rate without adjusting thresholds.
- Simulator or video errors can interrupt episodes → record the error, retain partial trajectories, and mark the milestone incomplete unless all required episodes are present.
- The remote checkout is not the current local branch → transfer only committed M01 source files and verify their hashes; never reset the server checkout or remove its untracked reports.

## Migration Plan

Run unit/contract checks locally, commit the M01 source and OpenSpec change on the current local branch, transfer only the committed runner to the verified remote checkout, run three smoke episodes, then run/resume the 40 formal episodes. Copy the resulting evidence bundle back without model-weight binaries. Rollback requires only removing the new script/feature commit; prior registries, checkpoints, and reports remain untouched.

## Open Questions

- Which exact LIBERO task per suite should be selected? Resolve deterministically from the installed pinned benchmark task registry and freeze the result before the first rollout.
- Whether the official checkpoint includes all required processor assets and whether the installed LeRobot version can load them will be resolved by the real load/inference smoke gate.
