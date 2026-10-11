# M06 independent policy regression and release evaluation

## Problem
M05 development results cover eight states and cannot establish independent target repair or legacy skill retention. The M04 replacement test freeze contains 40 tasks x init 40–49; it has not been used for M05 selection.

## Scope and acceptance
Freeze the 400 case identities, all four model/processor/adapter hashes, FP32 Eager control contract, paired statistics and release rule before any formal rollout. Use one development-only technical preflight, then execute 1600 actual LIBERO/MuJoCo episodes with per-episode video, actions, full available simulation states, resource samples, hashes and resumable receipts. All cases, including failures, enter task/suite/target/legacy summaries. Use paired target analysis, task-cluster uncertainty, Holm correction and a preregistered 3 percentage point legacy noninferiority boundary. Feed validated results to ModelOps for immutable version decisions. Produce a 180–240 second private demo, dashboard, code commits and reproducible evidence. M06 does not modify weights or open M07.

## Boundaries
The test is independent of recorded project development source groups; whether the foundation model saw these states in pretraining is unknown. HF expert episode IDs lack verified mapping to LIBERO fixed init-state IDs. Treat that uncertainty as a provenance limitation, never as proof of complete physical-state independence. Do not publish restricted visual assets. Currency cost requires an invoice.

## Implementation order
1. Validate source freeze, checkpoint/config hashes and excluded groups; persist immutable protocol and gate.
2. Prove one development-only technical rollout, reset fidelity, video and resume contract.
3. Run all frozen cases with fail-closed receipts and shared GPU lock.
4. Independently validate coverage and artifacts; compute paired statistics, ModelOps decisions, demo and reports.
