"""Pinned-protocol LeRobot commands; no hidden model training or simulated metrics.

The command builder avoids implicit parallel-autoreset pairing, but users must
verify initial-state identities in actual output before statistical pairing.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


@dataclass(frozen=True)
class LiberoEvalConfig:
    policy_path: str
    output_dir: str
    suite: str = "libero_spatial"
    episodes_per_task: int = 10
    seed: int = 42
    control_mode: str = "relative"
    action_steps: int | None = None

    def __post_init__(self) -> None:
        if not self.policy_path or not self.output_dir:
            raise ValueError("Policy and output paths must be nonempty.")
        if self.suite not in SUITES:
            raise ValueError("Unknown LIBERO suite.")
        if self.episodes_per_task < 1:
            raise ValueError("Evaluation episodes must be positive.")
        if self.control_mode not in ("relative", "absolute"):
            raise ValueError("Unsupported control mode.")
        if self.action_steps is not None and self.action_steps < 1:
            raise ValueError("Action steps must be positive.")


def libero_eval_command(config: LiberoEvalConfig) -> list[str]:
    command = [
        "lerobot-eval",
        "--policy.path=" + config.policy_path,
        "--output_dir=" + config.output_dir,
        "--env.type=libero",
        "--env.task=" + config.suite,
        "--env.control_mode=" + config.control_mode,
        "--env.init_states=true",
        "--env.hard_reset=true",
        "--env.max_parallel_tasks=1",
        "--eval.batch_size=" + str(config.episodes_per_task),
        "--eval.n_episodes=" + str(config.episodes_per_task),
        "--seed=" + str(config.seed),
    ]
    if config.action_steps is not None:
        command.append("--policy.n_action_steps=" + str(config.action_steps))
    return command


@dataclass(frozen=True)
class SmolVLATrainConfig:
    dataset_repo_id: str
    checkpoint: str
    output_dir: str
    steps: int = 5000
    batch_size: int = 2

    def __post_init__(self) -> None:
        if not all((self.dataset_repo_id, self.checkpoint, self.output_dir)):
            raise ValueError("Dataset, checkpoint and output directory are required.")
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("Training steps and batch size must be positive.")


def smolvla_sft_command(config: SmolVLATrainConfig) -> list[str]:
    """Standard LeRobot SFT only. Weighted correction loss requires further integration."""
    return [
        "lerobot-train",
        "--policy.path=" + config.checkpoint,
        "--policy.push_to_hub=false",
        "--dataset.repo_id=" + config.dataset_repo_id,
        "--output_dir=" + config.output_dir,
        "--steps=" + str(config.steps),
        "--batch_size=" + str(config.batch_size),
    ]


def command_as_shell(command: list[str]) -> str:
    return shlex.join(command)


def execute_explicitly(command: list[str], *, confirmed: bool = False) -> None:
    if not confirmed:
        raise PermissionError("Execution requires explicit confirmed=True.")
    subprocess.run(command, check=True)
