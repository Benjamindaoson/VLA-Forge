## ADDED Requirements

### Requirement: Preserve official checkpoint identity
The runner MUST resolve the official LIBERO SmolVLA checkpoint to an immutable repository revision and verify downloaded file hashes before inference.

#### Scenario: Reproducible official model load
- **WHEN** the runner starts from a model repository identifier
- **THEN** it records the resolved revision, configuration, processor assets, and checkpoint hashes and refuses incomplete downloads

### Requirement: Execute real closed-loop rollouts
The runner MUST use the loaded SmolVLA policy, LIBERO/MuJoCo physics, the environment success signal, and the configured action chunk; it MUST record every requested episode outcome, including failures and runtime errors.

#### Scenario: Smoke gate
- **WHEN** three smoke episodes are requested for one valid LIBERO task
- **THEN** each episode resets the environment, performs policy inference and physical simulation, records termination and success, and the formal batch is gated on all three completing without infrastructure errors

#### Scenario: Four-suite evaluation
- **WHEN** the smoke gate passes and formal evaluation is requested
- **THEN** exactly ten episodes execute for each of one fixed task in Spatial, Object, Goal, and Long, with episode-level logs and at least one genuine simulator video

### Requirement: Emit verifiable M01 evidence
Each M01 run MUST retain episode records, task metrics, configuration and dependency provenance, model/source hashes, logs, videos, and a runnable reproduction command; missing or partial episodes MUST be reported as incomplete.

#### Scenario: Evidence verification
- **WHEN** evidence is finalized
- **THEN** the manifest hashes match the recorded files, success rates derive from saved episode records, and all requested episodes are accounted for without synthetic results
