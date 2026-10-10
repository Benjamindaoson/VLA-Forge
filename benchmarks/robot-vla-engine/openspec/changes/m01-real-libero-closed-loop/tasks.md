## 1. Reproducible Runner Contract

- [x] 1.1 Add tests for strict M01 suite/task selection, smoke gate, resumption identity, and episode evidence aggregation.
- [x] 1.2 Implement an independent official-checkpoint evaluator using pinned LeRobot policy and processor APIs with actual LIBERO action chunks.
- [x] 1.3 Add deterministic task selection, per-episode persistence/resume, video logging, resource/timing capture, and evidence manifest generation.

## 2. Real GPU Experiment

- [x] 2.1 Resolve and hash the official checkpoint revision and verify no conflicting GPU run or lock is active.
- [x] 2.2 Run three real MuJoCo smoke episodes on one task and verify model output, action execution, success signal, and video before continuing.
- [x] 2.3 Run or resume exactly ten real episodes for one task in each of LIBERO Spatial, Object, Goal, and Long.
- [x] 2.4 Verify episode coverage, metrics, logs, videos, model/dependency/source hashes, and resource measurements.

## 3. Delivery

- [x] 3.1 Run relevant project tests and the complete existing test suite; record exact results.
- [x] 3.2 Commit the M01 source and OpenSpec artifacts on the current working branch and verify the commit hash.
- [x] 3.3 Finalize the M01 report and reproduction command with PASS/INCOMPLETE based only on observed evidence.
