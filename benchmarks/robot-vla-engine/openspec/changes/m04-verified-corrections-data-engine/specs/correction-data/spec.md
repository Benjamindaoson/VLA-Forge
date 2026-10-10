# Correction Data Requirements

## Requirement: Verified correction admission
A record MUST be rejected when source provenance, task-success oracle, physics reset/prefix evidence, action trace, or checkpoint identity is missing.

### Scenario: Ordinary demonstration without simulator receipt
- **WHEN** a demonstration lacks a replay-matched initial state and two independent successful verification replays
- **THEN** it MUST remain an unverified source episode and MUST NOT enter the positive correction dataset.

### Scenario: Replayed correction
- **WHEN** both verification replays use the frozen policy/control contract and task oracle reports success
- **THEN** the system records both replay identities, state-match evidence, action hashes, and dataset round-trip receipt before admitting the row.
