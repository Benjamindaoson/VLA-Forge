# Proposed research experiment protocol

## Central hypothesis

Under equal expert and compute budgets, selecting which diagnostic tests to perform and which repair method to use improves held-out robot-policy success per unit cost compared with universal tuning, universal fine-tuning and static failure classification.

## Preregister before locked evaluation

Freeze hypotheses, baselines, outcome metrics, sample size/power, intervention levels, task and model versions, safety margins, cost units, statistical intervals and multiplicity policy.

## Units, data and splits

- A RepairCase is a source failure plus related probes, verified corrective trials and candidate repairs.
- Split by parent episodes and scene family, not individual contiguous frames.
- Hold out task-preserving perturbation instances and clearly separate task-changing perturbations.
- Never tune diagnosis policies on final locked-test outcomes.

## Paired evaluation and contact repeatability

- Pin checkpoint, cameras, proprioception mapping, action normalization, control mode and executed action steps.
- Reset to the *same recorded initial physical state*. Matching seeds alone are insufficient in vectorized LIBERO environments.
- Verify state fingerprints before each intervention. Contact physics may still be nondeterministic; repeat matched runs and report variance.
- Distinguish model-weight gains from runtime-only gains.
- Perform target-failure success evaluation and old-skill retention evaluation separately.
- Estimate paired differences with task-cluster uncertainty; do not interpret an inconclusive CI as noninferiority.

## Mandatory baselines

1. Frozen original policy.
2. Best pre-registered fixed runtime tuning.
3. Random additional expert demonstrations with matched cost.
4. Failure-targeted corrective SFT plus Replay.
5. Static diagnosis with fixed routing.
6. BIR with no active diagnostics.
7. Full BIR with active information-value diagnostics.

A full hindsight search is permissible as an explicitly labeled offline upper bound, never an information source for online routing.

## Costs and outcomes

Report target success, legacy-skill success, repair cost (GPU-hours and expert seconds reported separately), mean/risk-adjusted utility, unsafe events, intervention reproducibility and P95 observation-to-command latency. Independent robot safety evaluation is mandatory before physical deployment.

## Standards and third-party references

LeRobot official LIBERO integration: https://huggingface.co/docs/lerobot/libero

Physical robot results, scientific novelty and customer value are NOT established by the unit tests or this document.
