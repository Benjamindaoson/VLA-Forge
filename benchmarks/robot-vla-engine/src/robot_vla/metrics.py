"""Metrics that distinguish missing coverage from measured failures."""

from collections import defaultdict
from math import sqrt

import numpy as np


def summarize_latency(samples):
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("latencies must be finite nonnegative milliseconds")
    keys = ["mean_ms", "p50_ms", "p95_ms", "p99_ms"]
    if not len(values):
        return dict(n=0, **dict.fromkeys(keys))
    return dict(
        n=len(values),
        **dict(zip(keys, [float(values.mean()), *map(float, np.percentile(values, [50, 95, 99]))])),
    )


def wilson(successes, total):
    if not total:
        return [None, None]
    z = 1.959963984540054
    p = successes / total
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    radius = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return [max(0.0, center - radius), min(1.0, center + radius)]


def aggregate_rollouts(rows, expected_tasks=None, expected_episodes_per_task=None):
    grouped, identities = defaultdict(list), set()
    for row in rows:
        if type(row["success"]) is not bool:
            raise ValueError("success must be a measured boolean")
        task = (str(row["suite"]), str(row["task_id"]))
        identity = (*task, str(row.get("seed", "")), str(row["episode_id"]))
        if identity in identities:
            raise ValueError("duplicate rollout identity")
        identities.add(identity)
        grouped[task].append(row["success"])
    expected = set(map(tuple, expected_tasks)) if expected_tasks is not None else set(grouped)
    missing = sorted(expected - set(grouped))
    unexpected = sorted(set(grouped) - expected)
    incomplete = sorted(
        k
        for k in expected
        if expected_episodes_per_task is not None
        and len(grouped.get(k, [])) != expected_episodes_per_task
    )
    tasks = [
        dict(
            suite=s,
            task_id=t,
            n=len(v),
            successes=sum(v),
            sr=sum(v) / len(v),
            wilson95=wilson(sum(v), len(v)),
        )
        for (s, t), v in sorted(grouped.items())
    ]
    macro = float(np.mean([t["sr"] for t in tasks])) if tasks else None
    suites = defaultdict(list)
    for task in tasks:
        suites[task["suite"]].append(task["sr"])
    return dict(
        tasks=tasks,
        missing_tasks=[list(x) for x in missing],
        unexpected_tasks=[list(x) for x in unexpected],
        incomplete_tasks=[list(x) for x in incomplete],
        suite_sr={s: float(np.mean(v)) for s, v in suites.items()},
        observed_task_macro_sr=macro,
        formal_overall_sr=macro
        if expected_tasks
        and expected_episodes_per_task
        and not missing
        and not unexpected
        and not incomplete
        else None,
        pooled_sr=sum(sum(v) for v in grouped.values()) / len(identities) if identities else None,
        total_rollouts=len(identities),
        interval_scope="descriptive per-task Wilson 95%",
    )
