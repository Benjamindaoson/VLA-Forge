## Why

The existing evaluator requires a training-run-owned checkpoint and cannot represent the user's requested independent official `lerobot/smolvla_libero` baseline. A separate, provenance-bound runner is needed to exercise the official policy through real LIBERO/MuJoCo rollouts and save episode-level results and videos without relabeling historical experiments.

## What Changes

- Add an M01-only evaluator for a pinned official SmolVLA LIBERO checkpoint and four explicitly selected LIBERO suite tasks.
- Record smoke and formal rollouts, action-chunk execution, timings, device memory, success signals, videos, source/config/model hashes, and a reproducible invocation.
- Keep the existing training-owned formal evaluator and previous Registry records unchanged.

## Capabilities

### New Capabilities
- `official-libero-baseline`: Independently reproduce a pinned official SmolVLA checkpoint on real LIBERO/MuJoCo tasks and retain verifiable rollout evidence.

### Modified Capabilities

## Impact

Adds a script and tests under `scripts/` and `tests/`, an M01 evidence bundle under `reports/vla_forge_2_transition_20261010/milestones/M01_real_libero/`, and a project-local OpenSpec capability. Reuses the pinned LeRobot 0.6.2 / LIBERO / MuJoCo runtime and existing observation, rollout, metrics, and telemetry modules. It does not modify training recipes, official benchmark protocols, prior run records, or large model assets.
