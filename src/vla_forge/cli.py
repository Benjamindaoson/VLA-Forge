"""Inspectable command-line entry points. Example inputs are synthetic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_forge.adapters.lerobot_commands import (
    LiberoEvalConfig, SmolVLATrainConfig, command_as_shell,
    libero_eval_command, smolvla_sft_command,
)
from vla_forge.belief import (
    BudgetedRepairPlanner, DiagnosticProbe, FailureBelief, PlannerSettings, RepairOption,
)
from vla_forge.evaluation import ReleaseThresholds, evaluate_release, summarize_pairs
from vla_forge.models import PairOutcome, RepairKind


def synthetic_planner_example() -> dict[str, object]:
    """Synthetic illustrative inputs only, never robot observations."""
    belief = FailureBelief(probabilities={"timing": 0.5, "geometry": 0.5})
    options = [
        RepairOption(
            option_id="adjust_horizon", kind=RepairKind.RUNTIME_FIX,
            cost=1, safety_risk=0, regression_risk=0,
            estimated_gain_by_hypothesis={"timing": 0.45, "geometry": 0.0},
        ),
        RepairOption(
            option_id="verified_corrective_sft", kind=RepairKind.POLICY_UPDATE,
            cost=3, safety_risk=0, regression_risk=0.01,
            estimated_gain_by_hypothesis={"timing": 0.0, "geometry": 0.5},
        ),
    ]
    probes = [
        DiagnosticProbe(
            probe_id="horizon_sweep", cost=0.2, safety_risk=0,
            likelihoods={
                "timing": {"improves": 0.9, "no_change": 0.1},
                "geometry": {"improves": 0.1, "no_change": 0.9},
            },
        )
    ]
    planner = BudgetedRepairPlanner(
        options, probes, PlannerSettings(budget=5, cost_weight=0.01)
    )
    return {
        "data_kind": "synthetic_illustration",
        "recommendation": planner.recommend(belief).model_dump(),
    }


def _read_outcomes(path: str) -> list[PairOutcome]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Expected JSON array of paired outcomes.")
    return [PairOutcome.model_validate(row) for row in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VLA-Forge policy repair evidence tools")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo-planner", help="Print a labeled synthetic BIR demonstration.")

    ev = commands.add_parser("eval-command", help="Print a LIBERO evaluation command.")
    ev.add_argument("--policy", required=True)
    ev.add_argument("--output-dir", required=True)
    ev.add_argument("--suite", default="libero_spatial")
    ev.add_argument("--episodes-per-task", type=int, default=10)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--control-mode", default="relative")
    ev.add_argument("--action-steps", type=int)

    train = commands.add_parser("train-command", help="Print a vanilla SmolVLA SFT command.")
    train.add_argument("--dataset", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument("--batch-size", type=int, default=2)

    gate = commands.add_parser("release-check", help="Evaluate locked-test paired JSON.")
    gate.add_argument("--target", required=True)
    gate.add_argument("--retention", required=True)
    gate.add_argument("--p95-ms", type=float, required=True)
    gate.add_argument("--min-pairs", type=int, default=100)
    gate.add_argument("--min-tasks", type=int, default=4)

    args = parser.parse_args(argv)
    if args.command == "demo-planner":
        print(json.dumps(synthetic_planner_example(), indent=2))
        return 0
    if args.command == "eval-command":
        cfg = LiberoEvalConfig(
            policy_path=args.policy, output_dir=args.output_dir,
            suite=args.suite, episodes_per_task=args.episodes_per_task,
            seed=args.seed, control_mode=args.control_mode, action_steps=args.action_steps,
        )
        print(command_as_shell(libero_eval_command(cfg)))
        return 0
    if args.command == "train-command":
        cfg = SmolVLATrainConfig(
            dataset_repo_id=args.dataset, checkpoint=args.checkpoint,
            output_dir=args.output_dir, steps=args.steps, batch_size=args.batch_size,
        )
        print(command_as_shell(smolvla_sft_command(cfg)))
        return 0
    if args.command == "release-check":
        target = summarize_pairs(_read_outcomes(args.target))
        retention = summarize_pairs(_read_outcomes(args.retention))
        decision = evaluate_release(
            target=target, retention=retention, p95_latency_ms=args.p95_ms,
            thresholds=ReleaseThresholds(min_pairs=args.min_pairs, min_tasks=args.min_tasks),
        )
        print(json.dumps(decision.model_dump(), indent=2))
        return 0 if decision.approved else 2
    parser.error("Unknown command.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
