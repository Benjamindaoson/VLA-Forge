"""Bounded training-input identities; timings never establish policy quality."""

import hashlib
import json

import numpy as np

RECIPE = dict(
    policy="smolvla",
    condition="baseline_full",
    seed=42,
    updates=400,
    warmup_updates=100,
    batch_size=8,
    workers=4,
    precision="bf16",
    accumulation=1,
    lr=1e-4,
    gpu_memory_gib=23.0,
)


def dataset_query(config):
    return dict(
        input_features={
            k: dict(type=v.type.value, shape=list(v.shape))
            for k, v in config.input_features.items()
        },
        output_features={
            k: dict(type=v.type.value, shape=list(v.shape))
            for k, v in config.output_features.items()
        },
        action_delta_indices=list(config.action_delta_indices),
        observation_delta_indices=None
        if config.observation_delta_indices is None
        else list(config.observation_delta_indices),
    )


def batch_hash(batch):
    def encode(value):
        if isinstance(value, dict):
            return {k: encode(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, str):
            return dict(text=value)
        if hasattr(value, "detach"):
            if value.device.type != "cpu":
                raise ValueError("qualification expects CPU tensors")
            value = value.detach().numpy()
        array = np.asarray(value)
        if array.dtype.kind not in "buif" or not np.isfinite(array).all():
            raise ValueError("unsupported/nonfinite batch field")
        return dict(
            dtype=str(array.dtype),
            shape=list(array.shape),
            sha256=hashlib.sha256(array.tobytes()).hexdigest(),
        )

    return hashlib.sha256(json.dumps(encode(batch), sort_keys=True).encode()).hexdigest()


def validate_qualification(result, binding, planned):
    if result.get("binding") != binding or result.get("exact") is not True:
        raise ValueError("qualification binding or exactness differs")
    rows = result.get("batches", [])
    if len(rows) != len(planned):
        raise ValueError("qualification batch coverage incomplete")
    for step, (row, indices) in enumerate(zip(rows, planned, strict=True)):
        digest = row.get("pyav", "")
        if (
            row.get("step") != step
            or row.get("indices") != indices
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or digest != row.get("torchcodec")
        ):
            raise ValueError("qualification order, indices or tensor hashes differ")


def compare_profiles(profiles):
    if [p["backend"] for p in profiles] != ["pyav", "torchcodec", "torchcodec", "pyav"]:
        raise ValueError("four independent ABBA profiles required")
    first = profiles[0]
    recipe = first["binding"]["recipe"]
    updates, warm = recipe["updates"], recipe["warmup_updates"]
    expected = (updates - warm) * recipe["batch_size"]
    for p in profiles:
        for key in ["initial_model_sha256", "final_model_sha256"]:
            value = p.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("missing/invalid model hash")
        for key in ["peak_allocated_bytes", "peak_reserved_bytes"]:
            value = p.get(key)
            if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                raise ValueError("invalid memory measurement")
        if p["peak_allocated_bytes"] > p["peak_reserved_bytes"]:
            raise ValueError("allocated memory exceeds reserved peak")
        for key in [
            "binding",
            "qualification_run_id",
            "initial_model_sha256",
            "hardware_identity",
            "gpu_budget",
        ]:
            if p[key] != first[key]:
                raise ValueError(f"profile identity mismatch: {key}")
        if (
            not p["hardware_identity"]
            or p["environment"]["hostname"] != first["environment"]["hostname"]
            or len(p["measurements"]) != updates
            or p["measured_examples"] != expected
        ):
            raise ValueError("profile host or measured coverage mismatch")
        seconds = p["measured_loop_seconds"]
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("invalid measured duration")
        if not np.isclose(p["examples_per_loop_second"], expected / seconds, rtol=1e-12):
            raise ValueError("throughput denominator mismatch")
        for step, row in enumerate(p["measurements"]):
            if row["step"] != step or row["measured"] != (step >= warm):
                raise ValueError("step coverage mismatch")
            if any(
                not np.isfinite(row[k]) or row[k] < 0
                for k in ["loss", "grad_norm", "step_ms", "loading_ms"]
            ):
                raise ValueError("invalid training measurement")
            if row["loading_ms"] > row["step_ms"]:
                raise ValueError("loading timer exceeds enclosing step")
        if sum(r["step_ms"] for r in p["measurements"][warm:]) > seconds * 1000 + 1:
            raise ValueError("step time exceeds loop duration")
    pooled = {}
    for backend in ["pyav", "torchcodec"]:
        selected = [p for p in profiles if p["backend"] == backend]
        step_times = [r["step_ms"] for p in selected for r in p["measurements"][warm:]]
        pooled[backend] = dict(
            examples_per_loop_second=sum(p["measured_examples"] for p in selected)
            / sum(p["measured_loop_seconds"] for p in selected),
            step_p50_ms=float(np.percentile(step_times, 50)),
            step_p95_ms=float(np.percentile(step_times, 95)),
            peak_allocated_bytes=max(p["peak_allocated_bytes"] for p in selected),
            peak_reserved_bytes=max(p["peak_reserved_bytes"] for p in selected),
        )
    differences = []
    for a, b in [(0, 1), (3, 2), (0, 3), (1, 2)]:
        left = np.array([r["loss"] for r in profiles[a]["measurements"]])
        right = np.array([r["loss"] for r in profiles[b]["measurements"]])
        differences.append(
            dict(
                jobs=[a, b],
                loss_max_abs=float(np.max(np.abs(left - right))),
                loss_rmse=float(np.sqrt(np.mean((left - right) ** 2))),
            )
        )
    return dict(
        pooled=pooled,
        numerical_diagnostics=differences,
        throughput_gain_percent=100
        * (
            pooled["torchcodec"]["examples_per_loop_second"]
            / pooled["pyav"]["examples_per_loop_second"]
            - 1
        ),
        final_model_hashes_equal=len({p["final_model_sha256"] for p in profiles}) == 1,
        quality_preserved=None,
        scope="Bounded 300 measured updates per job, warm-cache full-model throughput; no convergence or closed-loop quality claim",
    )
