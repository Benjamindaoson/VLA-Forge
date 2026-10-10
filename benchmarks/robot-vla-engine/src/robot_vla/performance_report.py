"""Measured training and rollout costs, preserving timing and sampling boundaries."""

import json
from collections import Counter

import numpy as np


def _number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("missing numeric measurement")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("invalid measured value")
    return value


def distribution(values):
    values = [_number(v) for v in values]
    if not values:
        return dict(count=0, mean=None, median=None, p95=None, p99=None, max=None)
    return dict(
        count=len(values),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
        max=max(values),
    )


def resource_summary(events):
    columns = {
        k: [] for k in ("utilization_percent", "device_memory_mib", "temperature_c", "power_watts")
    }
    resources = [e["payload"] for e in events if e["kind"] == "resources"]
    valid = 0
    moments, rss = [], []
    for row in resources:
        if row.get("process_rss_bytes") is not None:
            rss.append(_number(row["process_rss_bytes"]))
        if row.get("monotonic_seconds") is not None:
            moments.append(_number(row["monotonic_seconds"]))
        gpu = row.get("gpu") or {}
        if gpu.get("returncode") != 0 or not gpu.get("stdout"):
            continue
        lines = gpu["stdout"].strip().splitlines()
        if len(lines) != 1:
            raise ValueError("ambiguous multi-device resource sample")
        parts = lines[0].split(",")
        if len(parts) != 4:
            raise ValueError("unexpected GPU telemetry shape")
        parsed = 0
        for key, value in zip(columns, parts):
            try:
                numeric = float(value.strip())
            except ValueError:
                continue
            numeric = _number(numeric)
            if key == "utilization_percent" and numeric > 100:
                raise ValueError("GPU utilization outside percentage range")
            columns[key].append(numeric)
            parsed += 1
        valid += int(parsed > 0)
    gaps = np.diff(moments)
    if (gaps <= 0).any():
        raise ValueError("resource sample clock is not increasing")
    return dict(
        **{k: distribution(v) for k, v in columns.items()},
        samples=len(resources),
        valid_samples=valid,
        sample_gap_seconds_max=float(gaps.max()) if len(gaps) else None,
        process_rss_bytes=distribution(rss),
        scope="Unweighted observed samples; device memory max is sampled, not an exact peak; process RSS excludes DataLoader workers",
    )


def training_cost(events, *, expected_steps, effective_batch):
    steps = [e["payload"] for e in events if e["kind"] == "training"]
    ends = [e["payload"] for e in events if e["kind"] == "training_complete"]
    if [v["global_step"] for v in steps] != list(range(1, expected_steps + 1)) or len(ends) != 1:
        raise ValueError(
            "incomplete/duplicated training sequence; recovery lineage needs explicit handling"
        )
    end = ends[0]
    if end["steps"] != expected_steps:
        raise ValueError("training completion budget mismatch")
    for row in steps:
        if (
            row["effective_batch_size"] != effective_batch
            or row["examples_seen"] != row["global_step"] * effective_batch
        ):
            raise ValueError("training example count mismatch")
        _number(row["loss"])
        _number(row["grad_norm"])
        _number(row["step_ms"], positive=True)
        if _number(row["data_loading_ms"]) > row["step_ms"] + 1e-5:
            raise ValueError("nested loading timer exceeds timed update")
        if _number(row["gpu_peak_bytes"]) > _number(row["gpu_peak_reserved_bytes"]):
            raise ValueError("allocated peak exceeds reserved peak")
    wall = _number(end["wall_seconds"], positive=True)
    timed = sum(v["step_ms"] for v in steps) / 1000
    if timed > wall + 0.001 or not np.isclose(
        _number(end["training_loop_gpu_hours"]), wall / 3600, rtol=1e-9
    ):
        raise ValueError("inconsistent training wall time or GPU-hours")
    examples = expected_steps * effective_batch
    return dict(
        steps=expected_steps,
        examples=examples,
        loop_wall_seconds=wall,
        training_loop_gpu_hours=wall / 3600,
        timed_step_seconds=timed,
        loop_examples_per_second=examples / wall,
        timed_step_examples_per_second=examples / timed,
        step_ms=distribution([v["step_ms"] for v in steps]),
        data_loading_ms=distribution([v["data_loading_ms"] for v in steps]),
        data_loading_fraction_of_timed_steps=sum(v["data_loading_ms"] for v in steps)
        / (timed * 1000),
        peak_allocated_bytes=max(v["gpu_peak_bytes"] for v in steps),
        peak_reserved_bytes=max(v["gpu_peak_reserved_bytes"] for v in steps),
        gpu_resources=resource_summary(events),
        monitor_errors=sum(e["kind"] == "monitor_error" for e in events),
        timing_scope="Training optimizer loop includes logging/checkpoints and batch loading, excludes initialization/export; not cloud billed GPU-hours",
    )


def rollout_cost(trials):
    if not trials:
        raise ValueError("empty completed evaluation")
    steps = sum(_number(t["episode_length"], positive=True) for t in trials)
    wall = sum(_number(t["wall_seconds"], positive=True) for t in trials)
    successful = [t["episode_length"] for t in trials if t["success"]]
    return dict(
        trials=len(trials),
        control_steps=int(steps),
        rollout_wall_seconds=wall,
        achieved_serial_control_hz=steps / wall,
        mean_episode_length=steps / len(trials),
        successful_episode_length_mean=float(np.mean(successful)) if successful else None,
        timeout_rate=sum(t["termination_reason"] == "timeout" for t in trials) / len(trials),
        termination_counts=dict(Counter(t["termination_reason"] for t in trials)),
        timing_scope="Sum of rollout timers excludes reset/initialization and final encoding flush; includes step observation/video work; not a deployed robot rate",
    )


def finalized_events(registry, run_id):
    if registry.get(run_id)["status"] != "COMPLETE":
        raise ValueError("performance source must be COMPLETE")
    path = registry.root / "runs" / run_id / "events.jsonl"
    exported = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    with registry.connect() as db:
        rows = [
            dict(r)
            for r in db.execute("SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,))
        ]
    expected = [
        dict(
            id=r["id"],
            run_id=run_id,
            timestamp=r["timestamp"],
            kind=r["kind"],
            payload=json.loads(r["payload_json"]),
        )
        for r in rows
    ]
    if exported != expected:
        raise ValueError("finalized event export differs from registry")
    return exported, path
