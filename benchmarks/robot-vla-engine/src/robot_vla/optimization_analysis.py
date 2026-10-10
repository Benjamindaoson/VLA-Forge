"""Fail-closed pairing for optimized policy quality, distinct from latency."""

from .analysis import paired_difference, trial_key, validate_coverage
from .inference_benchmark import quality_gate


def verify_trajectory_bijection(registry, run_id, rows):
    from pathlib import Path

    with registry.connect() as db:
        registered = [
            str(Path(r[0]).resolve())
            for r in db.execute(
                "SELECT path FROM artifacts WHERE run_id=? AND role='rollout_trajectory'", (run_id,)
            )
        ]
    referenced = [str(Path(row["trajectory_path"]).resolve()) for row in rows]
    if (
        len(registered) != len(set(registered))
        or len(referenced) != len(set(referenced))
        or set(registered) != set(referenced)
    ):
        raise ValueError("trajectory references must be one-to-one with registered artifacts")


def compare_quality(reference, candidate, protocol):
    expected_seeds = protocol["evaluation"]["seeds"]
    observed = sorted({row["seed"] for row in candidate})
    if set(observed) - set(expected_seeds):
        raise ValueError("unexpected optimized seed")
    for seed in observed:
        for rows in [reference, candidate]:
            validate_coverage([row for row in rows if row["seed"] == seed], protocol, seed)
    left, right = (
        {trial_key(row): row for row in reference},
        {trial_key(row): row for row in candidate},
    )
    if len(left) != len(reference) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError("quality trial pairing mismatch")
    for key in left:
        if left[key]["environment_configuration"] != right[key]["environment_configuration"]:
            raise ValueError("paired environment configuration differs")
    difference = paired_difference(reference, candidate, expected_seeds)
    successes = sum(row["success"] for row in reference)
    return dict(
        paired_seeds=observed,
        baseline_successes=successes,
        baseline_trials=len(reference),
        optimized_successes=sum(row["success"] for row in candidate),
        optimized_trials=len(candidate),
        difference=difference,
        gate=quality_gate(observed, expected_seeds, successes, difference["task_cluster_ci95_pp"]),
    )
