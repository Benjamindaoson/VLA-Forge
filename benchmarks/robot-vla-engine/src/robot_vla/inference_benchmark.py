"""Pure selection and decision rules for paired inference engineering experiments."""

from collections import defaultdict

import numpy as np


def choose_cases(rows, validation_episodes):
    validation = set(map(int, validation_episodes))
    tasks = {int(row["task_index"]) for row in rows}
    groups = defaultdict(list)
    for row in rows:
        if int(row["episode_index"]) in validation:
            groups[int(row["task_index"])].append(row)
    if set(groups) != tasks:
        raise ValueError("held-out observations do not cover every task")
    selected = []
    for task in sorted(groups):
        episode = min(int(row["episode_index"]) for row in groups[task])
        frames = sorted(
            [row for row in groups[task] if int(row["episode_index"]) == episode],
            key=lambda row: int(row["frame_index"]),
        )
        if len({row["frame_index"] for row in frames}) != len(frames):
            raise ValueError("duplicate held-out frame")
        selected.append(frames[len(frames) // 2])
    return selected


def compare_measurements(reference, candidate):
    def index(rows):
        output = {}
        for row in rows:
            key = (row["case_id"], row["repeat"])
            value = row["chunk_forward_ms"]
            if key in output or not np.isfinite(value) or value <= 0:
                raise ValueError("invalid measurement pairing or latency")
            output[key] = value
        return output

    left, right = index(reference), index(candidate)
    if not left or left.keys() != right.keys():
        raise ValueError("measurement pairing mismatch")
    baseline = float(np.median(list(left.values())))
    optimized = float(np.median(list(right.values())))
    reduction = 100 * (1 - optimized / baseline)
    return dict(
        paired_measurements=len(left),
        reference_median_ms=baseline,
        candidate_median_ms=optimized,
        median_latency_reduction_percent=reduction,
        throughput_gain_percent=100 * (baseline / optimized - 1),
        materially_faster=reduction >= 10,
        quality_preserved=None,
        scope="Offline paired input timing only; success-rate preservation requires closed-loop trials",
    )


def quality_gate(observed_seeds, expected_seeds, baseline_successes, delta_ci95_pp):
    if len(observed_seeds) != len(set(observed_seeds)) or baseline_successes < 0:
        raise ValueError("invalid quality evidence")
    if (
        set(observed_seeds) != set(expected_seeds)
        or baseline_successes < 20
        or delta_ci95_pp is None
    ):
        return dict(
            status="UNVERIFIED",
            reason="Require all paired seeds, >=20 baseline successes and a measured interval",
        )
    if (
        len(delta_ci95_pp) != 2
        or not np.isfinite(delta_ci95_pp).all()
        or delta_ci95_pp[0] > delta_ci95_pp[1]
    ):
        raise ValueError("invalid quality interval")
    return dict(
        status="PASS_CONDITIONAL" if delta_ci95_pp[0] >= -5 else "FAIL",
        margin_pp=5,
        scope="Conditional task-cluster interval for these checkpoints/seeds/initial states; not universal equivalence",
    )
