"""Deterministic family-level splits and leakage checks."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Literal

Split = Literal["train", "validation", "test"]


class LeakageError(ValueError):
    pass


def assign_split(
    group_id: str, *, salt: str = "vla-forge-v1",
    train_fraction: float = 0.8, validation_fraction: float = 0.1
) -> Split:
    if not group_id.strip() or not salt.strip():
        raise ValueError("Group ID and salt must be nonempty.")
    if not (0 < train_fraction < 1 and 0 <= validation_fraction < 1):
        raise ValueError("Invalid split fractions.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Test set must have nonzero probability.")
    digest = hashlib.sha256((salt + ":" + group_id).encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def assert_no_leakage(
    records: list[dict[str, object]],
    *,
    group_key: str = "split_group_id",
    split_key: str = "split",
    linked_keys: tuple[str, ...] = ("source_episode_id", "correction_episode_id"),
) -> None:
    """Fail closed when an exact family or linked episode appears in two splits.

    The upstream dataset builder must assign scene/parent-derived records the same
    split_group_id. This guard cannot infer semantic near-duplicates from images.
    """
    appearances: dict[str, set[str]] = defaultdict(set)
    for record in records:
        split = record.get(split_key)
        group = record.get(group_key)
        if split not in ("train", "validation", "test"):
            raise ValueError("Missing or invalid split.")
        if not isinstance(group, str) or not group:
            raise ValueError("Missing lineage split group.")
        appearances["group:" + group].add(split)
        for key in linked_keys:
            value = record.get(key)
            if value is not None:
                if not isinstance(value, str) or not value:
                    raise ValueError("Invalid lineage episode ID.")
                appearances["episode:" + value].add(split)
        parents = record.get("parent_episode_ids", [])
        if not isinstance(parents, list) or any(not isinstance(x, str) or not x for x in parents):
            raise ValueError("Invalid parent_episode_ids.")
        for parent in parents:
            appearances["episode:" + parent].add(split)
    clashes = {key: sorted(splits) for key, splits in appearances.items() if len(splits) > 1}
    if clashes:
        raise LeakageError("Cross-split lineage collisions: " + repr(clashes))
