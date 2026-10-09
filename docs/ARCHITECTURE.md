# Architecture

## Evidence-oriented workflow

1. A failure enters the system with policy version, task ID, simulator revision and source episode ID.
2. Related source episodes, scene families, corrections and interventions share a split-group ID.
3. Optional simulation intervention experiments restore the same snapshot and compare controlled changes.
4. Repair decisions compare calibrated outcome estimates against probe/repair cost, regression risk and safety risk.
5. Verified expert rollouts become eligible for positive correction supervision.
6. An updated checkpoint enters paired target and old-skill evaluation with locked, independent test episodes.
7. The release gate checks improvement confidence, retention noninferiority, unsafe events and latency.

## Boundaries and accountability

The belief planner assumes explicitly supplied outcome likelihoods and repair-effect tables. It must not infer causal validity from observational trajectory similarity. The simulator adapter Protocol defines restore, fingerprint, apply_intervention and rollout; no real LIBERO implementation ships at this stage.

The corrections module rejects positive labels without replay and outcome verification and links them to a source case. The split guard detects ID-level leakage but cannot infer all semantic near duplicates. The training loss helper must be wired to actual SmolVLA velocity targets and independently verified data before training.

The action scheduler validates age and numeric limits, but real robot safety must be independently implemented below it. The FastAPI service is local only and lacks identity/authentication/fleet commands.

## Persistence

Case IDs cannot be overwritten with altered data. Experiments append audited events with canonical JSON hashes. Checkpoints and large multimedia artifacts must live in an external controlled file or object store with matching checksums.

## Planned integrations

- Pin LeRobot LIBERO dependencies, implement real snapshot/restore and true paired policy rollouts.
- Implement verified expert correction from restored recoverable failure states.
- Integrate correction manifests/weights into real SmolVLA training and save checkpoints.
- Ingest LIBERO and LIBERO-Plus results with exact scene-state references.
- Add ROS2 simulator execution with action/observation correspondence and safety supervisor.
- Add Isaac Lab and learning-based value model only after strong baseline comparisons.
- Add secure commercial data/workflow control plane after external pilot validation.
