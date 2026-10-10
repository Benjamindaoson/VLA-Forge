"""Outcome analysis with exact coverage and paired, task-cluster uncertainty."""

from collections import defaultdict

import numpy as np

from .metrics import aggregate_rollouts


def trial_key(row):
    return str(row["suite"]), str(row["task_id"]), int(row["seed"]), str(row["episode_id"])


def validate_coverage(rows, protocol, seed):
    mapping = {(v["suite"], str(v["task_id"])): (k, v) for k, v in protocol["task_mapping"].items()}
    settings = protocol["evaluation"]
    expected = {
        (suite, task, seed, str(episode))
        for suite, task in mapping
        for episode in settings["initial_state_indices"]
    }
    keys = [trial_key(row) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("exact task/seed/initial-state coverage mismatch")
    for row in rows:
        if type(row["success"]) is not bool:
            raise ValueError("success is not a measured boolean")
        language, task = mapping[(str(row["suite"]), str(row["task_id"]))]
        if row["instruction"] != language:
            raise ValueError("task language mismatch")
        env = row["environment_configuration"]
        required = dict(
            task_name=task["task_name"],
            init_state_index=int(row["episode_id"]),
            **{k: settings[k] for k in ["control_freq", "num_steps_wait", "hard_reset"]},
        )
        if any(env.get(k) != value for k, value in required.items()):
            raise ValueError("environment protocol mismatch")
        length, reason = row["episode_length"], row["termination_reason"]
        horizon = settings["horizons"][row["suite"]]
        if type(length) is not int or not 1 <= length <= horizon:
            raise ValueError("episode length outside protocol")
        if (
            row["success"] != (reason == "success")
            or reason
            not in ["success", "timeout", "environment_terminated", "environment_truncated"]
            or (reason == "timeout" and length != horizon)
        ):
            raise ValueError("inconsistent termination semantics")


def summarize_condition(rows, seeds):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["seed"])].append(row)
    if set(grouped) - set(seeds):
        raise ValueError("unexpected training seed")
    seed_results = []
    for seed, values in sorted(grouped.items()):
        metrics = aggregate_rollouts(values)
        seed_results.append(
            dict(
                seed=seed,
                task_macro_sr=metrics["observed_task_macro_sr"],
                suite_sr=metrics["suite_sr"],
                tasks=metrics["tasks"],
                n=len(values),
            )
        )
    complete = set(grouped) == set(seeds)
    rates = [r["task_macro_sr"] for r in seed_results]
    lengths = [r["episode_length"] for r in rows if r["success"]]
    return dict(
        status="COMPLETE" if complete else "INCOMPLETE",
        seed_results=seed_results,
        missing_seeds=sorted(set(seeds) - set(grouped)),
        mean_sr=float(np.mean(rates)) if complete else None,
        std_sr=float(np.std(rates, ddof=1)) if complete and len(rates) > 1 else None,
        seed_range=[min(rates), max(rates)] if complete else None,
        rollouts=len(rows),
        failures=sum(not r["success"] for r in rows),
        timeouts=sum(r["termination_reason"] == "timeout" for r in rows),
        successful_episode_length_mean=float(np.mean(lengths)) if lengths else None,
        uncertainty_scope="Seed sample standard deviation is descriptive; per-task Wilson intervals conditional on the evaluated trials",
    )


def paired_difference(reference, candidate, seeds, tasks=None, draws=10000):
    if draws < 1:
        raise ValueError("bootstrap draws must be positive")
    selected = set(tasks) if tasks is not None else None

    def index(rows):
        value = {}
        for row in rows:
            key = trial_key(row)
            if selected is not None and key[:2] not in selected:
                continue
            if key in value or type(row["success"]) is not bool:
                raise ValueError("invalid paired trial")
            value[key] = int(row["success"])
        return value

    left, right = index(reference), index(candidate)
    if not left or not right:
        return dict(status="INCOMPLETE", delta_pp=None, task_cluster_ci95_pp=None)
    seed_sets = [{k[2] for k in value} for value in [left, right]]
    if any(observed - set(seeds) for observed in seed_sets):
        raise ValueError("unexpected paired seed")
    if any(observed != set(seeds) for observed in seed_sets):
        return dict(status="INCOMPLETE", delta_pp=None, task_cluster_ci95_pp=None)
    if left.keys() != right.keys():
        raise ValueError("paired trial identities differ")
    grouped = defaultdict(list)
    for key in left:
        grouped[key[:2]].append(right[key] - left[key])
    if selected is not None and set(grouped) != selected:
        raise ValueError("paired task group incomplete")
    differences = np.array([np.mean(grouped[k]) for k in sorted(grouped)]) * 100
    rng = np.random.default_rng(20261007)
    samples = differences[rng.integers(0, len(differences), size=(draws, len(differences)))].mean(
        axis=1
    )
    return dict(
        status="COMPLETE",
        delta_pp=float(differences.mean()),
        task_cluster_ci95_pp=np.percentile(samples, [2.5, 97.5]).tolist(),
        tasks=len(grouped),
        paired_trials=len(left),
        bootstrap_draws=draws,
        bootstrap_seed=20261007,
        sign="candidate minus reference",
        interval_scope="Task-cluster percentile bootstrap conditional on these trained seeds and fixed initial states; not training-population uncertainty",
    )
