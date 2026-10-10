"""Qualification of demonstration candidates, never implicit expert certification."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .schema import content_hash, file_hash


def inventory_episodes(root, selected_task_ids, train_ids, validation_ids, episode_lengths):
    """Aggregate across files BEFORE filtering tasks, preserving mixed-task rejection."""
    import pyarrow.parquet as pq

    episodes = defaultdict(list)
    columns = ["episode_index", "frame_index", "task_index", "action", "observation.state"]
    paths = sorted((Path(root) / "data").glob("**/*.parquet"))
    if not paths:
        raise ValueError("no numeric source files")
    for path in paths:
        for row in pq.read_table(path, columns=columns).to_pylist():
            episodes[row["episode_index"]].append(row)
    if set(episodes) != set(episode_lengths):
        raise ValueError("episode metadata/source identities differ")
    result = []
    for episode_id, rows in sorted(episodes.items()):
        observed_tasks = {r["task_index"] for r in rows}
        selected = observed_tasks & selected_task_ids
        if not selected:
            continue
        rows.sort(key=lambda r: r["frame_index"])
        item = audit_episode(
            [r["action"] for r in rows],
            [r["observation.state"] for r in rows],
            [r["frame_index"] for r in rows],
            [r["task_index"] for r in rows],
            episode_id=episode_id,
            expected_task_id=min(selected),
            train_ids=train_ids,
            validation_ids=validation_ids,
        )
        if len(rows) != episode_lengths[episode_id]:
            item["issues"].append("metadata_length")
            item["candidate_eligible"] = False
        result.append(item)
    return result


def audit_episode(
    actions,
    states,
    frame_indices,
    task_indices,
    *,
    episode_id,
    expected_task_id,
    train_ids,
    validation_ids,
):
    issues = []
    arrays = {}
    for name, raw in [("action", actions), ("state", states)]:
        try:
            arrays[name] = np.asarray(raw)
        except (TypeError, ValueError):
            arrays[name] = np.asarray(None)
        if arrays[name].dtype.kind not in "iuf":
            issues.append(name + "_type")
    a, s = arrays["action"], arrays["state"]
    n = len(a) if a.ndim else 0
    for name, array, width in [("action", a, 7), ("state", s, 8)]:
        if n == 0 or array.shape != (n, width):
            issues.append(name + "_shape")
        if array.dtype.kind in "iuf" and not np.isfinite(array).all():
            issues.append(name + "_nonfinite")
    if a.dtype.kind in "iuf" and np.isfinite(a).all() and np.any((a < -1) | (a > 1)):
        issues.append("action_bounds")
    frames, tasks = np.asarray(frame_indices), np.asarray(task_indices)
    if frames.dtype.kind not in "iu" or not np.array_equal(frames, np.arange(n)):
        issues.append("frame_sequence")
    if tasks.dtype.kind not in "iu" or tasks.shape != (n,) or not np.all(tasks == expected_task_id):
        issues.append("task_identity")
    # Frozen protocol IDs are strings; Parquet episode_index values are integers.
    if str(episode_id) not in {str(x) for x in train_ids}:
        issues.append("not_training_split")
    if str(episode_id) in {str(x) for x in validation_ids}:
        issues.append("validation_overlap")
    return dict(
        episode_id=int(episode_id),
        frames=n,
        task_index=int(expected_task_id),
        candidate_eligible=not issues,
        issues=issues,
        verified_correction=False,
    )


def load_gate0_evidence(folder):
    """Check archived summary bytes; does not verify today's remote processes/files."""
    folder = Path(folder)
    receipt = json.loads((folder / "verification.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS":
        raise ValueError("Gate 0 archive was not independently verified")
    records = receipt["verified_runs"]
    if len(records) != 2 or len({r["run_id"] for r in records}) != 2:
        raise ValueError("ambiguous verification records")
    result = []
    for name in ["summary_v2.json", "prefix_replay.json"]:
        path = folder / name
        value = json.loads(path.read_text(encoding="utf-8"))
        record = next((r for r in records if r["run_id"] == value["run_id"]), None)
        if record is None or file_hash(path) != record["summary_sha256"]:
            raise ValueError("Gate 0 summary hash mismatch")
        result.append(value)
    return tuple(result)


def bind_cases(baseline, prefix):
    expected = {(suite, 0, k) for suite in ("libero_10", "libero_goal") for k in (20, 60)}
    rows = prefix["rows"]
    if prefix["parent_run_id"] != baseline["run_id"]:
        raise ValueError("wrong parent run")
    if len(rows) != 4 or {(r["suite"], r["task_id"], r["prefix"]) for r in rows} != expected:
        raise ValueError("missing or duplicate reconstruction cases")
    cases = []
    for row in rows:
        key = (row["suite"], row["task_id"])
        pairs = [p for p in baseline["pairs"] if (p["suite"], p["task_id"]) == key]
        if len(pairs) != 1 or pairs[0]["successes"] != [False, False]:
            raise ValueError("case must have two observed baseline failures")
        references = [
            r
            for r in baseline["replays"]
            if (r["suite"], r["task_id"], r["prefix"]) == (*key, row["prefix"])
        ]
        if len(references) != 1 or references[0]["raw_path"] != row["reference_path"]:
            raise ValueError("reconstruction reference mismatch")
        metrics = row["comparisons"]
        required = {"sim", "observation", "success", "pixels_image", "pixels_image2"}
        if (
            row["qualified"] is not True
            or set(metrics) != required
            or any(
                m["within_tolerance"] is not True or m["max_abs_error"] != 0
                for m in metrics.values()
            )
        ):
            raise ValueError("reconstruction is not qualified")
        identity = dict(
            suite=row["suite"],
            task_id=row["task_id"],
            prefix=row["prefix"],
            baseline_run_id=baseline["run_id"],
            prefix_run_id=prefix["run_id"],
            reference_sha256=references[0]["raw_sha256"],
            reconstruction_sha256=row["raw_sha256"],
        )
        cases.append(
            dict(
                **identity,
                case_id=content_hash(identity),
                correction_status="NOT_RUN",
                verified_correction=False,
                live_remote_verification="NOT_RUN",
            )
        )
    return sorted(cases, key=lambda x: (x["suite"], x["prefix"]))
