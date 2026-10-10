"""Streaming numeric checks over every actual frame, independent of advertised counts."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def audit_numeric(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    episodes = {}
    total = bad_action = bad_state = outside = 0
    action_min = state_min = float("inf")
    action_max = state_max = float("-inf")
    for file in sorted((root / "data").rglob("*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(
            batch_size=4096,
            columns=[
                "episode_index",
                "task_index",
                "timestamp",
                "frame_index",
                "action",
                "observation.state",
            ],
        ):
            for row in batch.to_pylist():
                e = episodes.setdefault(
                    row["episode_index"],
                    dict(
                        frames=0,
                        tasks=set(),
                        last_t=None,
                        first_t=None,
                        timestamp_valid=True,
                        index_valid=True,
                        hash=hashlib.sha256(),
                    ),
                )
                t = row["timestamp"]
                e["timestamp_valid"] &= bool(
                    np.isfinite(t) and (e["last_t"] is None or t > e["last_t"])
                )
                e["index_valid"] &= row["frame_index"] == e["frames"]
                e["first_t"] = t if e["first_t"] is None else e["first_t"]
                e["last_t"] = t
                e["frames"] += 1
                e["tasks"].add(row["task_index"])
                a = np.asarray(row["action"], dtype=np.float32)
                s = np.asarray(row["observation.state"], dtype=np.float32)
                bad_action += int((~np.isfinite(a)).sum())
                bad_state += int((~np.isfinite(s)).sum())
                outside += int((np.abs(a[np.isfinite(a)]) > 1.00001).sum())
                for values, label in [(a, "action"), (s, "state")]:
                    finite = values[np.isfinite(values)]
                    if len(finite):
                        lo, hi = float(finite.min()), float(finite.max())
                        if label == "action":
                            action_min, action_max = min(action_min, lo), max(action_max, hi)
                        else:
                            state_min, state_max = min(state_min, lo), max(state_max, hi)
                e["hash"].update(a.tobytes() + s.tobytes())
                total += 1
    duplicates = defaultdict(list)
    records = []
    for index, e in sorted(episodes.items()):
        digest = e["hash"].hexdigest()
        duplicates[digest].append(index)
        records.append(
            dict(
                episode_id=index,
                frames=e["frames"],
                task_indices=sorted(e["tasks"]),
                timestamps_strict=e["timestamp_valid"],
                frame_index_contiguous=e["index_valid"],
                first_timestamp=e["first_t"],
                last_timestamp=e["last_t"],
                numeric_content_hash=digest,
            )
        )
    invalid_times = [e["episode_id"] for e in records if not e["timestamps_strict"]]
    invalid_indices = [e["episode_id"] for e in records if not e["frame_index_contiguous"]]
    ambiguous = [e["episode_id"] for e in records if len(e["task_indices"]) != 1]
    tasks = Counter(t for e in records for t in e["task_indices"])
    counts_match = (
        total == info["total_frames"]
        and len(episodes) == info["total_episodes"]
        and len(tasks) == info["total_tasks"]
    )
    return dict(
        episode_count=len(episodes),
        frame_count=total,
        task_count=len(tasks),
        metadata_counts_match=counts_match,
        episodes=records,
        episodes_per_task=dict(tasks),
        nonfinite_action_values=bad_action,
        nonfinite_state_values=bad_state,
        action_outside_unit_range=outside,
        action_range=[action_min, action_max] if np.isfinite(action_min) else None,
        state_range=[state_min, state_max] if np.isfinite(state_min) else None,
        invalid_timestamp_episodes=invalid_times,
        invalid_frame_index_episodes=invalid_indices,
        ambiguous_task_episodes=ambiguous,
        numeric_duplicate_groups=[v for v in duplicates.values() if len(v) > 1],
        numeric_quality_pass=counts_match
        and not any((bad_action, bad_state, outside, invalid_times, invalid_indices, ambiguous)),
        success="N/A: no success labels in this schema",
        termination="N/A",
        timestamp_semantics="source index/fps timeline; not proven acquisition wall time",
        image_integrity="N/A: separate decode audit required",
        physical_state_limits="N/A: not source-certified",
    )
