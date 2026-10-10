"""Contracts and evidence-bounded analysis for the full LIBERO 40 benchmark."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .schema import file_hash

M02_SUITES = {
    "spatial": "libero_spatial",
    "object": "libero_object",
    "goal": "libero_goal",
    "long": "libero_10",
}
TASKS_PER_SUITE = 10
EPISODES_PER_TASK = 10
Z_95 = 1.959963984540054
FAILURE_CATEGORIES = {
    "Reach / Alignment",
    "Grasp Failure",
    "Object Drop",
    "Placement / Goal Failure",
    "Sequence / Long-Horizon Failure",
    "Timeout",
    "Unknown / Insufficient Evidence",
}


def expected_episode_identities(
    *, tasks_per_suite: int = TASKS_PER_SUITE, episodes_per_task: int = EPISODES_PER_TASK
):
    return {
        (suite, task_id, episode_id)
        for suite in M02_SUITES
        for task_id in range(tasks_per_suite)
        for episode_id in range(episodes_per_task)
    }


def episode_seed(seed_base: int, suite: str, task_id: int, init_state_id: int):
    if suite not in M02_SUITES or not 0 <= task_id < TASKS_PER_SUITE or not 0 <= init_state_id < EPISODES_PER_TASK:
        raise ValueError("seed identity is outside the frozen M02 matrix")
    suite_ordinal = list(M02_SUITES).index(suite)
    return seed_base + suite_ordinal * 1000 + task_id * 10 + init_state_id


def validate_m02_coverage(rows):
    expected = expected_episode_identities()
    actual = [
        (row.get("suite"), int(row.get("task_id", -1)), int(row.get("episode_id", -1)))
        for row in rows
    ]
    if len(actual) != len(set(actual)):
        raise ValueError("duplicate Episode identity in M02 matrix")
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("M02 coverage must contain exactly 400 expected Episodes")
    if any(row.get("status") != "complete" for row in rows):
        raise ValueError("M02 coverage contains incomplete or error Episodes")
    if any(type(row.get("success")) is not bool for row in rows):
        raise ValueError("completed M02 Episode requires a measured boolean success signal")
    for row in rows:
        episode_id = int(row["episode_id"])
        if int(row.get("init_state_id", -1)) != episode_id:
            raise ValueError("Episode identity does not match the frozen initial-state ID")
        if row.get("seed") in (None, ""):
            raise ValueError("Episode is missing its deterministic seed")
    return {suite: TASKS_PER_SUITE * EPISODES_PER_TASK for suite in M02_SUITES}


def wilson_interval(successes: int, total: int, z: float = Z_95):
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    if total == 0:
        return None
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    radius = z * ((p * (1 - p) / total + z2 / (4 * total * total)) ** 0.5) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def aggregate_binary(rows):
    completed = [row for row in rows if row.get("status") == "complete"]
    if any(type(row.get("success")) is not bool for row in completed):
        raise ValueError("completed Episode has no measured success signal")
    successes = sum(row["success"] for row in completed)
    total = len(completed)
    return {
        "successes": successes,
        "episodes": total,
        "failures": total - successes,
        "success_rate": successes / total if total else None,
        "wilson_95": wilson_interval(successes, total),
    }


def task_and_suite_metrics(rows):
    task_groups = defaultdict(list)
    suite_groups = defaultdict(list)
    for row in rows:
        if row.get("status") != "complete":
            continue
        task_groups[(row["suite"], int(row["task_id"]))].append(row)
        suite_groups[row["suite"]].append(row)
    tasks = {}
    for (suite, task_id), values in sorted(task_groups.items()):
        tasks[(suite, task_id)] = aggregate_binary(values)
    suites = {suite: aggregate_binary(values) for suite, values in suite_groups.items()}
    return tasks, suites, aggregate_binary(rows)


def classify_failure(row, manual_review=None):
    if row.get("success") is not False:
        raise ValueError("failure classification requires a measured failed Episode")
    if manual_review is not None:
        category = manual_review.get("category")
        evidence = str(manual_review.get("evidence", "")).strip()
        confidence = manual_review.get("confidence", "medium")
        if category not in FAILURE_CATEGORIES:
            raise ValueError("manual review uses an unsupported failure category")
        if not evidence:
            raise ValueError("manual review requires visible/video/trajectory evidence")
        if confidence not in {"high", "medium", "low", "uncertain"}:
            raise ValueError("manual review confidence is invalid")
        failure_step = manual_review.get("failure_step")
        if failure_step is not None and (not isinstance(failure_step, int) or failure_step < 0):
            raise ValueError("manual review failure_step must be a non-negative integer or null")
        return {
            "category": category,
            "basis": "human_review",
            "evidence": evidence,
            "confidence": confidence,
            "failure_step": failure_step,
        }
    reason = row.get("termination_reason")
    steps = int(row.get("total_action_steps") or 0)
    horizon = int(row.get("horizon") or 0)
    if reason == "timeout" and horizon > 0 and steps >= horizon:
        return {
            "category": "Timeout",
            "basis": "environment success signal remained false through the frozen horizon",
            "confidence": "high",
        }
    return {
        "category": "Unknown / Insufficient Evidence",
        "basis": f"no reliable failure-stage signal; termination_reason={reason!r}",
        "confidence": "uncertain",
    }


def artifact_is_valid(path, expected_sha256):
    candidate = Path(path) if path else None
    return bool(
        candidate
        and candidate.is_file()
        and expected_sha256
        and file_hash(candidate) == expected_sha256
    )


def select_manual_review(rows, *, minimum: int = 20):
    failed = sorted(
        (row for row in rows if row.get("status") == "complete" and row.get("success") is False),
        key=lambda row: (row["suite"], int(row["task_id"]), int(row["episode_id"])),
    )
    return failed[:minimum] if len(failed) >= minimum else failed
