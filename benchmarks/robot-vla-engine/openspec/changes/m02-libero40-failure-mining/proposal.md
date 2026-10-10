## Why

M01 proved the pinned SmolVLA policy can control LIBERO/MuJoCo in real closed-loop simulation, but its four fixed tasks do not establish standard LIBERO 40-task performance or provide broad, auditable failure evidence. M02 extends the verified runner into a resumable full-suite benchmark and creates a provenance-linked failure dataset for later diagnosis.

## What Changes

- Evaluate all 40 tasks, ten fixed initial states per task, with the frozen M01 policy and environment contract.
- Persist Episode-level trajectories, paired camera video, seeds, runtime, latency, model identity, attempts, and hashes; resume only missing/invalid episodes.
- Generate task/suite/global metrics with sample-size-appropriate Wilson intervals and an evidence-backed failure database.
- Keep unverified failure causes as Unknown; add a deterministic sample for manual review and M03 candidate references.

## Capabilities

### New Capabilities
- `libero40-benchmark`: Reproducible full LIBERO benchmark execution, analysis, and failure evidence.

### Modified Capabilities

## Impact

Adds an M02 runner, analysis utilities, validation tests, and M02 evidence under `reports/vla_forge_2_transition_20261010/milestones/M02_libero40_failure_mining/`. Uses the M01 LeRobot/MuJoCo environment and frozen checkpoint without training or changing the policy. Rollout binaries remain on the authorized external data disk; source, manifests, and summarized evidence remain in Git.
