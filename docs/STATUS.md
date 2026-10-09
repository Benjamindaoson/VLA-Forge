# Capabilities and proof levels

## Implemented code (subject to CI/unit verification)

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
