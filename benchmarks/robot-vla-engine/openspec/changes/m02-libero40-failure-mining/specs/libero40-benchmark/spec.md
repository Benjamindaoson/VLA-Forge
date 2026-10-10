## ADDED Requirements

### Requirement: Complete frozen LIBERO 40 coverage
The evaluator SHALL run ten distinct initial-state episodes for each of the 40 tasks in Spatial, Object, Goal, and Long (`libero_10`) using one frozen SmolVLA checkpoint, its fitted processors, the M01 camera/state/action mapping, and the frozen MuJoCo configuration.

#### Scenario: Exact benchmark coverage
- **WHEN** the benchmark is accepted
- **THEN** each of the 40 suite/task identities has exactly ten unique completed Episode identities, with seed and initial-state ID recorded

#### Scenario: Infrastructure failure is not a policy failure
- **WHEN** an Episode raises an environment or runtime error
- **THEN** the attempt is preserved as an error and the Episode remains incomplete until a valid rollout is recorded

### Requirement: Durable and resumable rollout evidence
The evaluator SHALL atomically persist each completed Episode's measured result, policy/action metadata, robot-state and action trajectory, paired camera video, and content hashes; resume SHALL skip only identities whose completed evidence validates.

#### Scenario: Resume after interruption
- **WHEN** the evaluator restarts with partially completed results
- **THEN** validated completed identities are reused and only missing or invalid identities are executed

#### Scenario: Evidence corruption
- **WHEN** a completed Episode's required video or trajectory hash fails verification
- **THEN** that Episode is rejected as complete and is scheduled for an auditable retry

### Requirement: Evidence-bounded failure mining
Failure records SHALL link to the real Episode, video, state/action prefix, final termination evidence, and provenance. A cause SHALL be assigned only from a direct environment signal or an explicitly recorded human review; otherwise the category SHALL be Unknown/Insufficient Evidence.

#### Scenario: Timeout classification
- **WHEN** a failed Episode reaches the frozen maximum horizon without a success signal
- **THEN** its category is Timeout with the observed step/horizon evidence

#### Scenario: Uncertain cause
- **WHEN** a failed Episode has no reliable stage/cause signal
- **THEN** it is retained as Unknown and remains available for human review and M03 analysis

### Requirement: Statistically bounded benchmark reporting
The analysis SHALL report task, suite, and overall success counts, sample-size-matched uncertainty intervals, failure categories, and reproducibility metadata without interpreting a single fixed-state run as multi-seed stability.

#### Scenario: Full result summary
- **WHEN** all Episode evidence validates
- **THEN** the report contains 40 task rows, four suite summaries, a 400-Episode aggregate, Wilson intervals, costs, and traceable failure candidates

#### Scenario: Incomplete matrix
- **WHEN** any required Episode is missing, invalid, or a runtime error
- **THEN** the report marks the benchmark INCOMPLETE and does not claim a 400-Episode success rate

### Requirement: Manual review sample
The failure mining deliverable SHALL identify at least 20 failed Episode clips for manual review, or all failures when fewer than 20 exist, and record the review disposition and evidence.

#### Scenario: Review completeness
- **WHEN** failure review is complete
- **THEN** every selected review item has an explicit reviewer disposition, evidence location, and uncertainty label
