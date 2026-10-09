# Capabilities and proof levels

## Implemented code and verified CI

On 2026-10-09, GitHub Actions completed successfully: 47 CPU tests passed, 3 tests requiring PyTorch were skipped in the minimal CI environment, CLI smoke passed and Ruff basic lint passed. These are software-unit results, not robot-policy experiment metrics.

Strict model validation, Bayesian BIR decision logic, synthetic decision example, simulator adapter interface, correction admission, split leakage guard, SQLite case events, paired statistical gate, numerical Action Chunk checks, loss helper, API and command builders.

## Requires real external testing

- Actual LIBERO state restoration, perturbation and success oracle.
- Verified expert continuation and source-aligned action dataset.
- SmolVLA/LeRobot weighted SFT integration and GPU training.
- LIBERO standard / LIBERO-Plus real paired tests and non-regression evidence.
- Isaac Lab PPO, ROS2 simulation, hardware-safe policy runtime, TensorRT, web UI, customer trial.

Synthetic demo and fake adapter tests are explicitly not robot-policy performance results.

## Next engineering gate

Pin LeRobot and simulator versions on a Linux GPU host. Run a real baseline under frozen observation, control and state-reset protocol. Implement LIBERO adapter, capture first real RepairCase, verify replay, and measure baseline vs identical checkpoint and execution settings. Then introduce expert correction and actual checkpoint training.
