# Design

## Data path
Immutable source episode -> candidate checkpoint/prefix -> state-match receipt -> expert or controlled correction -> independent replay A and B -> correction record -> LeRobotDataset v3 round-trip audit.

## Failure behavior
Missing physics state, task oracle, action provenance, or independent replay confirmation fails closed. Preserve rejection receipts and never convert ordinary demonstration episodes into corrections.

## Runtime measurements
Freeze policy/checkpoint/config and episode set. Compare baseline against at least two optimization candidates using the same inputs, action chunking, control mode, warmup, environment count, and success oracle. Record p50/p95 latency, throughput, peak allocated/reserved GPU memory, and task outcomes.

## Storage and provenance
Project code, reports, and M04 deliverables are kept under /data/VLA-Forge; large immutable source assets may be referenced by path and hash from /data/vla-forge-artifacts until consolidated. Secrets remain outside the repository.
