## Context

M01's runner already strictly loads the frozen `lerobot/smolvla_libero` checkpoint, uses LeRobot processors, records the exact action passed to `env.step`, writes SQLite provenance, and captures video. It intentionally limits coverage to one task per suite. M02 needs the full 40-task matrix without modifying M01's PASS semantics or reusing its smaller sample as a benchmark.

## Goals / Non-Goals

**Goals:** Execute 400 unique fixed-state episodes serially on the authorized RTX 3090; persist paired camera evidence and robot-state/action trajectories; support interruption-safe resume; summarize failures and statistical intervals.

**Non-Goals:** Policy training, correction/repair, intervention experiments, evaluation-based tuning, multi-seed claims, or running M03.

## Decisions

- Add a separate M02 runner and pure analysis/validation module instead of widening M01's task contract. This preserves M01's already-accepted experiment identity and avoids applying its 40-row validator to 400 rows.
- Enumerate the pinned official benchmark API in frozen suite order and require exactly ten tasks per suite. Initial-state IDs are 0–9; seeds are deterministic from suite/task/initial-state identity.
- Run one environment at a time. Persist a row and append-only attempt record after each Episode. Registry runs and a dedicated lock isolate M02 from historical matrices.
- Keep videos and registry trajectories on the data disk; the manifest and per-Episode CSV carry absolute artifact paths and SHA-256 hashes.
- Store both camera views side by side in each Episode video. Store the complete eight-dimensional observed robot state and the exact seven-dimensional executed action trajectory.
- Categorize only directly observed timeout/termination or human-reviewed causes. Unclassified failures remain Unknown; the analysis selects a deterministic 20-failure review sample.
- Compute two-sided 95% Wilson score intervals per task, suite, and overall; no cross-seed generalization claim is made.

## Risks / Trade-offs

- 400 videos and trajectories consume data-disk capacity → preflight available space, monitor it, and keep output outside Git.
- LIBERO API ordering or initial-state behavior can drift → pin package/data identity, assert 10 tasks per suite, record task names and explicit state/seed IDs.
- A long run can be interrupted → atomic CSV/JSON updates, append-only attempt log, one-episode checkpoints, and verified resume.
- Image capture might expose only one camera → fail the Episode unless both checkpoint-mapped camera observations are present.
- Visual failure labels may be ambiguous → retain Unknown and a written evidence basis; do not infer grasp/placement causes from success=False alone.

## Migration Plan

1. Add and test M02 contracts and runner locally.
2. Transfer only required source files to `/data/vla-forge`; retain existing files and use a separate output root.
3. Validate checkpoint hashes, process/lock state, and a real first Episode before continuing the same frozen 400-Episode queue.
4. Resume missing identities until coverage, hashes, and manual review gates pass.
5. Rollback means stopping only the M02 process and preserving all attempts; M01 and historical registries remain untouched.

## Open Questions

- None for the frozen M02 protocol. The benchmark remains INCOMPLETE if any infrastructure episode cannot be completed within bounded retries.
