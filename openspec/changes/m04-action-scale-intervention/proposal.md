# M04 action-scale intervention

## Intent
Add an auditable action-execution scale as a second, physically meaningful M04 intervention factor alongside camera brightness. Keep the frozen SmolVLA policy, source-prefix checkpoint, task and relative-control mode fixed.

## Acceptance
- Scale defaults to 1.0 and is range-validated before a rollout.
- Only the environment command is scaled; raw policy output remains recorded.
- The executed action is clipped using the existing environment contract.
- Result/event receipts record the requested scale and effective command.
- Tests cover identity, scaling and invalid values.
- A real RTX 3090 MuJoCo attempt is run and retained regardless of success.
