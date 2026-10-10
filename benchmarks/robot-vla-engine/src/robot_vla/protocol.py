"""Outcome-independent, source-mapped splits and fixed-budget conditions."""

import json
import re
from collections import defaultdict
from pathlib import Path

from .sampling import assert_no_leakage, nested_subset
from .schema import content_hash


def load_protocol(path):
    value = json.loads(Path(path).read_text())
    declared = value.get("identity")
    if (
        not declared
        or content_hash({k: v for k, v in value.items() if k != "identity"}) != declared
    ):
        raise ValueError("protocol identity mismatch")
    return value


def build_protocol(rows, task_map, seed=42):
    mapping = {}
    for suite, names in sorted(task_map.items()):
        for task_id, name in enumerate(names):
            language = re.sub(r"^.*?SCENE\d+_", "", name).replace("_", " ")
            if language in mapping:
                raise ValueError(f"ambiguous task mapping: {language}")
            mapping[language] = dict(suite=suite, task_id=task_id, task_name=name)
    if set(r["task"] for r in rows) != set(mapping):
        raise ValueError("dataset task mapping does not exactly match suites")
    hashes = [r["content_hash"] for r in rows if r.get("content_hash")]
    if len(hashes) != len(set(hashes)):
        raise ValueError("duplicate content requires grouping before split")
    validation = nested_subset(rows, 3, seed)
    valid_ids = {r["episode_id"] for r in validation}
    train = [r for r in rows if r["episode_id"] not in valid_ids]
    assert_no_leakage(train, validation)
    conditions = {}

    def condition(name, selected):
        conditions[name] = dict(
            episode_ids=sorted(r["episode_id"] for r in selected),
            episodes=len(selected),
            frames=sum(r["frames"] for r in selected),
            tasks=sorted({r["task"] for r in selected}),
        )

    condition("baseline_full", train)
    for count in [6, 12, 24]:
        condition(f"scale_{count}", nested_subset(train, count, seed + 1))
    by_suite = defaultdict(list)
    for language, identity in mapping.items():
        by_suite[identity["suite"]].append(language)
    low, high, unseen = [], [], []
    for suite, tasks in sorted(by_suite.items()):
        if len(tasks) != 10:
            raise ValueError("expected 10 tasks per suite")
        ordered = sorted(tasks, key=lambda task: content_hash([seed, "diversity", suite, task]))
        unseen.extend(ordered[:2])
        low.extend(ordered[2:6])
        high.extend(ordered[2:])
    condition("diversity_low", nested_subset([r for r in train if r["task"] in low], 24, seed + 1))
    condition(
        "diversity_high", nested_subset([r for r in train if r["task"] in high], 12, seed + 1)
    )
    return dict(
        seed=seed,
        task_order_index=0,
        task_mapping=mapping,
        validation_episode_ids=sorted(valid_ids),
        conditions=conditions,
        diversity_common_seen_tasks=sorted(low),
        diversity_unseen_tasks=sorted(unseen),
        comparison_scope="diversity conditions only: no pretrained weights finetuned on any LIBERO task; base pretraining overlap unknown",
        budget_matching="same optimizer updates and effective batch; episode count matched for diversity, frame counts reported, not matched",
        checkpoint_rule="last scheduled update; no evaluation-based selection",
        formal_results=None,
    )
