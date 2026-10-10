"""CPU-only workload partitioning; never launches an evaluator."""

import re
from collections import Counter

from .schema import canonical_json, content_hash


def _episodes(protocol, seed, checkpoint_sha256):
    unsigned = {k: v for k, v in protocol.items() if k != "identity"}
    if content_hash(unsigned) != protocol.get("identity"):
        raise ValueError("data protocol identity mismatch")
    settings = protocol["evaluation"]
    if type(seed) is not int or seed not in settings["seeds"]:
        raise ValueError("seed is outside frozen evaluation")
    if not isinstance(checkpoint_sha256, str) or not re.fullmatch(
        "[0-9a-f]{64}", checkpoint_sha256
    ):
        raise ValueError("invalid checkpoint SHA256")
    initial = settings["initial_state_indices"]
    if initial != list(range(10)) or settings["episodes_per_task"] != 10:
        raise ValueError("expected frozen 10 initial states")
    tasks = protocol["task_mapping"]
    keys = [(t["suite"], t["task_id"]) for t in tasks.values()]
    expected = {(s, i) for s in settings["suites"] for i in range(10)}
    if len(expected) != 40 or len(keys) != 40 or set(keys) != expected:
        raise ValueError("expected unique 40-task coverage")
    episodes = []
    for language, task in sorted(tasks.items(), key=lambda x: (x[1]["suite"], x[1]["task_id"])):
        horizon = settings["horizons"][task["suite"]]
        if type(horizon) is not int or horizon <= 0:
            raise ValueError("invalid horizon")
        for initial_state in initial:
            episodes.append(
                dict(
                    suite=task["suite"],
                    task_id=task["task_id"],
                    task_name=task["task_name"],
                    language=language,
                    initial_state=initial_state,
                    seed=seed,
                    horizon=horizon,
                    torch_seed=seed * 10000 + task["task_id"] * 100 + initial_state,
                )
            )
    if settings["formal_episodes_per_seed"] != len(episodes):
        raise ValueError("declared episode count mismatch")
    return episodes


def make_plan(protocol, *, seed, workers, checkpoint_sha256):
    episodes = _episodes(protocol, seed, checkpoint_sha256)
    if type(workers) is not int or not 1 <= workers <= len(episodes):
        raise ValueError("workers must be an integer between 1 and episode count")
    shards = [dict(worker_id=i, episodes=[], horizon_budget=0) for i in range(workers)]
    for episode in sorted(
        episodes, key=lambda e: (-e["horizon"], e["suite"], e["task_id"], e["initial_state"])
    ):
        target = min(shards, key=lambda s: (s["horizon_budget"], s["worker_id"]))
        target["episodes"].append(episode)
        target["horizon_budget"] += episode["horizon"]
    plan = dict(
        status="PLANNED_NOT_EXECUTED",
        data_protocol_identity=protocol["identity"],
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_bytes_verified=False,
        seed=seed,
        environment={
            k: protocol["evaluation"][k] for k in ["control_freq", "num_steps_wait", "hard_reset"]
        },
        scheduling="deterministic longest-horizon-first; horizon is not measured runtime",
        workers=shards,
    )
    plan["identity"] = content_hash(plan)
    return plan


def validate_plan(protocol, plan, *, seed, checkpoint_sha256):
    expected = _episodes(protocol, seed, checkpoint_sha256)
    unsigned = {k: v for k, v in plan.items() if k != "identity"}
    if content_hash(unsigned) != plan.get("identity"):
        raise ValueError("plan identity mismatch")
    if (
        plan.get("data_protocol_identity") != protocol["identity"]
        or plan.get("checkpoint_sha256") != checkpoint_sha256
        or plan.get("seed") != seed
    ):
        raise ValueError("plan source binding mismatch")
    shards = plan.get("workers", [])
    if not shards or len(shards) > len(expected):
        raise ValueError("invalid shard count")
    actual = []
    for i, shard in enumerate(shards):
        if type(shard.get("worker_id")) is not int or shard["worker_id"] != i:
            raise ValueError("worker identity mismatch")
        rows = shard.get("episodes", [])
        if not rows or sum(e["horizon"] for e in rows) != shard["horizon_budget"]:
            raise ValueError("empty shard or invalid budget")
        actual.extend(rows)
    if Counter(map(canonical_json, actual)) != Counter(map(canonical_json, expected)):
        raise ValueError("episode coverage or semantics mismatch")
    if plan != make_plan(
        protocol, seed=seed, workers=len(shards), checkpoint_sha256=checkpoint_sha256
    ):
        raise ValueError("plan does not match deterministic assignment")
    return len(actual)
