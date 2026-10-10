"""Stable nested episode selection; content/pair leakage is a hard error."""

from collections import defaultdict

from .schema import content_hash


def nested_subset(rows, per_task, seed):
    if not isinstance(per_task, int) or per_task < 1:
        raise ValueError("per_task must be a positive integer")
    groups = defaultdict(list)
    identities = set()
    for row in rows:
        identity = row["episode_id"]
        if identity in identities:
            raise ValueError("duplicate episode identity")
        identities.add(identity)
        groups[row["task"]].append(row)
    selected = []
    for task, records in sorted(groups.items()):
        if len(records) < per_task:
            raise ValueError(f"insufficient episodes for {task}: {len(records)} < {per_task}")
        ordered = sorted(records, key=lambda r: content_hash([seed, task, r["episode_id"]]))
        selected.extend(ordered[:per_task])
    return selected


def assert_no_leakage(train, test):
    for key in ("episode_id", "group_id", "pair_id", "content_hash"):
        left = {r[key] for r in train if r.get(key) is not None}
        right = {r[key] for r in test if r.get(key) is not None}
        overlap = left & right
        if overlap:
            raise ValueError(f"leakage via {key}: {sorted(overlap)}")
