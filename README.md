# VLA-Forge

**Intervention-Guided Robot Policy Repair & Continuous Improvement**

VLA-Forge is an evidence-first robot policy reliability and research toolkit. It diagnoses failure mechanisms, compares controlled interventions, builds validated corrective-data manifests, recommends cost-aware repairs and gates releases using paired evaluation.

**Status: experimental CPU-tested core. Latest checked CI: 47 passed, 3 optional torch tests skipped; basic lint/CLI passed (2026-10-09). Not a trained robot policy, not a safety-certified controller, and not an operational multi-robot product.**

## Research question

When is a diagnostic intervention worth its cost, and when should a robot learning team use runtime calibration, expert correction, VLA fine-tuning or human escalation? The initial algorithm is a transparent Bayesian budgeted repair baseline, not a claimed new causal discovery model.

## Implemented

- Strict models and cross-field validation for failure cases, interventions and correction data.
- Group-level train/validation/test assignments and cross-split lineage guards.
- Explicit Bayesian failure beliefs, diagnostic posterior updates, repair utility and net value of information.
- Paired, simulator-neutral intervention protocol with restoration fingerprint checks.
- Positive training examples admitted only with verified replay and successful outcomes.
- SQLite case repository with immutable identifiers/checksums and append-only experiment events.
- Task-cluster bootstrap paired evaluation, locked-test skill-retention/safety/latency release gate.
- Fresh-observation Action Chunk scheduler with input validation; NOT a physical safety layer.
- PyTorch flow-matching loss helper for weighted corrected examples; not integrated with LeRobot yet.
- FastAPI local research API; CLI to generate standard LeRobot evaluation/SFT commands.

## Not implemented or validated yet

No real LIBERO simulator state replay/perturbation adapter, expert recovery, weighted SmolVLA training loop, checkpoint or GPU result. No Isaac Lab PPO, ROS2 hardware-control integration, TensorRT performance result, secure production service, customer deployment or published academic experiment. Demo values are labeled synthetic.

## First real GPU milestone: LIBERO smoke and frozen baseline

Do **not** treat a training command as proof of policy improvement. First run a real
rollout with [lerobot/smolvla_libero](https://huggingface.co/lerobot/smolvla_libero),
record actual failures, and validate environment/control settings. The generic
smolvla_base checkpoint is not a LIBERO-trained baseline.

See [P0 runbook](docs/P0_RUNBOOK.md) and run
python -m vla_forge.preflight --require-gpu on the Linux GPU host.
A short SFT can then check the training stack; research training still requires verified corrections.

## Quick start

Python 3.11+:

~~~bash
python -m pip install -e ".[dev,serve]"
pytest -q
vla-forge demo-planner
vla-forge eval-command --policy lerobot/smolvla_libero --output-dir outputs/eval
vla-forge train-command --dataset YOUR_LEROBOT_DATASET --checkpoint lerobot/smolvla_base --output-dir outputs/sft
uvicorn vla_forge.api:app --host 127.0.0.1 --port 8000
~~~

LeRobot commands are printed for inspection rather than executed. For actual LIBERO execution on Linux, install matching LeRobot extra dependencies and confirm checkpoint observation/action contracts, control mode and action horizon. Reference: https://huggingface.co/docs/lerobot/libero .

## Source layout

| Location | Function |
| --- | --- |
| src/vla_forge/models.py | Shared evidence, failure and correction schemas |
| src/vla_forge/belief.py | BIR belief updates and diagnostic value |
| src/vla_forge/interventions.py | Paired simulator protocol |
| src/vla_forge/corrections.py | Verified positive labels and lineage |
| src/vla_forge/splits.py | Leakage-resistant source-family split checks |
| src/vla_forge/training.py | Weighted flow-matching loss building block |
| src/vla_forge/evaluation.py | Paired statistics and conservative release gate |
| src/vla_forge/runtime.py | Fresh-observation Action Chunk scheduler |
| src/vla_forge/storage.py | Case/event persistence and content hashes |
| src/vla_forge/api.py | Local research API |
| src/vla_forge/adapters | Opt-in external LeRobot command builders |

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Verified development status](docs/STATUS.md)
- [Safety boundaries](docs/SAFETY.md)

## Research integrity

Improvements due to changed model weights and changes to runtime/action scheduling are evaluated separately. Matching random seeds do not prove matching physical initial states. Lack of a statistically significant performance drop is not proof of non-regression. A locked-test release gate is intentionally conservative.

This repository does not currently include a software license; redistribution and third-party model/data rights require explicit review.
