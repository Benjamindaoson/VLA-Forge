"""Registered analysis to plot values; missing measurements never become zero."""

import json
from pathlib import Path

import numpy as np

from .schema import content_hash, file_hash


def _config(registry, run_id, kind):
    record = registry.get(run_id)
    value = json.loads(record["config_json"])
    stored = json.loads((registry.root / "runs" / run_id / "config.json").read_text())
    if (
        record["status"] != "COMPLETE"
        or value.get("experiment_type") != kind
        or value != stored
        or content_hash(value) != record["config_hash"]
        or registry.verify_artifacts(run_id)
    ):
        raise ValueError(f"invalid complete {kind} evidence: {run_id}")
    return value


def registered_analysis(registry, path, compute_identity, data_identity, analysis_identity):
    """Require owned immutable report and still-valid referenced raw artifacts."""
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    run_id = report["analysis_run_id"]
    config = _config(registry, run_id, "analysis_report")
    with registry.connect() as db:
        matched = db.execute(
            "SELECT path FROM artifacts WHERE run_id=? AND role='analysis_report' AND sha256=?",
            (run_id, file_hash(path)),
        ).fetchall()
    if len(matched) != 1:
        raise ValueError("input is not the exact registered analysis report")
    if (
        report["compute_identity"] != compute_identity
        or report["data_identity"] != data_identity
        or report["analysis_identity"] != analysis_identity
        or config.get("compute_protocol_identity") != compute_identity
        or config.get("analysis_identity") != analysis_identity
    ):
        raise ValueError("analysis identity mismatch")
    ids = [r["run_id"] for r in report["verified_evaluations"]]
    if len(ids) != len(set(ids)) or sorted(ids) != sorted(config["verified_evaluation_ids"]):
        raise ValueError("analysis evaluation binding mismatch")
    checked_parents = set()
    for run_id in ids:
        evaluation = _config(registry, run_id, "formal_eval")
        if evaluation.get("compute_protocol_identity") != compute_identity:
            raise ValueError("referenced evaluation identity mismatch")
        parent_id = evaluation["parent_run_id"]
        if parent_id not in checked_parents:
            parent = _config(registry, parent_id, "formal_training")
            if parent.get("compute_protocol_identity") != compute_identity:
                raise ValueError("referenced training identity mismatch")
            checked_parents.add(parent_id)
    return report


def _finite(value, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("expected numeric measurement")
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError("measurement outside bounds")
    return float(value)


def _equal(actual, expected):
    if actual is None or not np.isclose(_finite(actual, -100, 100), expected, atol=1e-10, rtol=0):
        raise ValueError("inconsistent measured summary")


def _diversity_evidence(value, rows, data, seeds, episodes):
    if value.get("reference") != ["smolvla", "diversity_low"] or value.get("candidate") != [
        "smolvla",
        "diversity_high",
    ]:
        raise ValueError("diversity direction identity mismatch")
    if value["status"] != "COMPLETE":
        return
    selected = sorted(
        {
            (data["task_mapping"][n]["suite"], str(data["task_mapping"][n]["task_id"]))
            for n in data[value["name"]]
        }
    )
    if (
        not selected
        or value.get("tasks") != len(selected)
        or value.get("paired_trials") != len(selected) * len(seeds) * episodes
        or value.get("sign") != "candidate minus reference"
        or value.get("bootstrap_draws") != 10000
        or value.get("bootstrap_seed") != 20261007
    ):
        raise ValueError("diversity group or bootstrap identity mismatch")
    summaries = {}
    for condition in ["diversity_low", "diversity_high"]:
        row = next(r for r in rows if r["policy"] == "smolvla" and r["condition"] == condition)
        summaries[condition] = {
            (r["seed"], t["suite"], str(t["task_id"])): t["sr"]
            for r in row["seed_results"]
            for t in r["tasks"]
        }
    # Frozen task-cluster estimator needs per-task marginal rates, not episode ordering.
    differences = (
        np.array(
            [
                np.mean(
                    [
                        summaries["diversity_high"][(s, *task)]
                        - summaries["diversity_low"][(s, *task)]
                        for s in seeds
                    ]
                )
                for task in selected
            ]
        )
        * 100
    )
    rng = np.random.default_rng(20261007)
    samples = differences[rng.integers(0, len(selected), size=(10000, len(selected)))].mean(axis=1)
    interval = np.percentile(samples, [2.5, 97.5]).tolist()
    if not np.isclose(
        value["delta_pp"], float(differences.mean()), rtol=0, atol=1e-10
    ) or not np.allclose(value["task_cluster_ci95_pp"], interval, rtol=0, atol=1e-10):
        raise ValueError("diversity effect or interval differs from verified group counts")


def plot_data(report, compute, data):
    seeds = compute["seeds"]
    expected = {(p, c) for p, s in compute["policies"].items() for c in s["conditions"]}
    rows = report["conditions"]
    keys = [(r["policy"], r["condition"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("planned condition coverage mismatch")
    declared = [(r["policy"], r["condition"], r["seed"]) for r in report["verified_evaluations"]]
    if len(declared) != len(set(declared)):
        raise ValueError("duplicate evaluation binding")
    task_keys = {(t["suite"], str(t["task_id"])) for t in data["task_mapping"].values()}
    suites = sorted({s for s, _ in task_keys})
    episodes = len(data["evaluation"]["initial_state_indices"])
    converted, observed = [], []
    for row in rows:
        key = row["policy"], row["condition"]
        budget = data["conditions"][key[1]]
        scope = "supplementary_all_40" if key[1].startswith("diversity_") else "primary_all_40"
        if (
            row["demonstrations"] != budget["episodes"]
            or row["frames"] != budget["frames"]
            or row["tasks_in_training"] != len(budget["tasks"])
            or row["evaluation_scope"] != scope
        ):
            raise ValueError("condition budget or scope mismatch")
        results = row["seed_results"]
        actual_seeds = [r["seed"] for r in results]
        if len(actual_seeds) != len(set(actual_seeds)) or set(actual_seeds) - set(seeds):
            raise ValueError("invalid condition seeds")
        missing = sorted(set(seeds) - set(actual_seeds))
        complete = not missing
        if row["missing_seeds"] != missing or row["status"] != (
            "COMPLETE" if complete else "INCOMPLETE"
        ):
            raise ValueError("condition completeness mismatch")
        rates, suite_rates = [], {s: [] for s in suites}
        for result in results:
            tasks = result["tasks"]
            task_ids = [(t["suite"], str(t["task_id"])) for t in tasks]
            if len(task_ids) != len(set(task_ids)) or set(task_ids) != task_keys:
                raise ValueError("exact task coverage mismatch")
            if result["n"] != len(task_keys) * episodes or set(result["suite_sr"]) != set(suites):
                raise ValueError("suite/episode coverage mismatch")
            for task in tasks:
                if task["n"] != episodes or type(task["successes"]) is not int:
                    raise ValueError("invalid task trial counts")
                _equal(task["sr"], _finite(task["successes"], 0, episodes) / episodes)
            rate = float(np.mean([t["sr"] for t in tasks]))
            _equal(result["task_macro_sr"], rate)
            rates.append(rate)
            for suite in suites:
                value = float(np.mean([t["sr"] for t in tasks if t["suite"] == suite]))
                _equal(result["suite_sr"][suite], value)
                suite_rates[suite].append(value)
            observed.append((*key, result["seed"]))
        if complete:
            _equal(row["mean_sr"], float(np.mean(rates)))
            _equal(row["std_sr"], float(np.std(rates, ddof=1)))
            if len(row["seed_range"]) != 2:
                raise ValueError("invalid seed range")
            for actual, target in zip(row["seed_range"], [min(rates), max(rates)]):
                _equal(actual, target)
        elif any(row[k] is not None for k in ["mean_sr", "std_sr", "seed_range"]):
            raise ValueError("incomplete condition cannot have a final estimate")

        def summary(values):
            return dict(
                mean_percent=100 * float(np.mean(values)) if complete else None,
                std_pp=100 * float(np.std(values, ddof=1)) if complete else None,
                seed_points=[dict(seed=s, percent=100 * v) for s, v in zip(actual_seeds, values)],
            )

        converted.append(
            dict(
                policy=key[0],
                condition=key[1],
                status=row["status"],
                demonstrations=budget["episodes"],
                frames=budget["frames"],
                evaluation_scope=scope,
                planned_seeds=seeds,
                **summary(rates),
                suites={s: summary(suite_rates[s]) for s in suites},
            )
        )
    if set(observed) != set(declared):
        raise ValueError("seed summary differs from bound evaluations")
    diversity = []
    for name in ["diversity_common_seen_tasks", "diversity_unseen_tasks"]:
        matches = [r for r in report["contrasts"] if r["name"] == name]
        if len(matches) != 1:
            raise ValueError("primary diversity contrast missing or duplicate")
        value = matches[0]
        if value["status"] == "COMPLETE":
            if any(
                r["status"] != "COMPLETE"
                for r in converted
                if r["condition"].startswith("diversity_")
            ):
                raise ValueError("diversity contrast has incomplete conditions")
            _finite(value["delta_pp"], -100, 100)
            interval = value["task_cluster_ci95_pp"]
            if len(interval) != 2 or _finite(interval[0], -100, 100) > _finite(
                interval[1], -100, 100
            ):
                raise ValueError("invalid diversity interval")
        elif (
            value["status"] != "INCOMPLETE"
            or value["delta_pp"] is not None
            or value["task_cluster_ci95_pp"] is not None
        ):
            raise ValueError("incomplete contrast cannot have an effect")
        _diversity_evidence(value, rows, data, seeds, episodes)
        diversity.append(
            dict(
                name=name,
                status=value["status"],
                delta_pp=value["delta_pp"],
                task_cluster_ci95_pp=value["task_cluster_ci95_pp"],
            )
        )
    return dict(conditions=converted, diversity=diversity, suites=suites)
