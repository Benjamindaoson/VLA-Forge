"""Bounded RTX3090 inference matrix on a real, private M04 dual-camera observation."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from robot_vla.m05_training import EVIDENCE, atomic_json, sha256

MODES = ("eager", "cuda_graph", "bf16", "fp16", "tf32", "compile")
INPUT = EVIDENCE / "training_perf/data_pipeline/observation-corrective-0.npz"
OUT = EVIDENCE / "inference"


def percentiles(values):
    return {
        "p50_s": float(np.percentile(values, 50)),
        "p95_s": float(np.percentile(values, 95)),
        "p99_s": float(np.percentile(values, 99)),
        "mean_s": float(np.mean(values)),
    }


def child(mode, calls, adapter):
    from robot_vla.m05_serving import PolicyEngine

    with np.load(INPUT, allow_pickle=False) as data:
        image1, image2 = data["image1"], data["image2"]
        state, task = data["state"], str(data["task"])
    engine = PolicyEngine("m05-corrective-cam180-s5501", adapter, mode)
    warm = []
    for _ in range(100):
        warm.append(engine.predict(image1, image2, state, task, seed=991))
    measured = []
    for _ in range(calls):
        measured.append(engine.predict(image1, image2, state, task, seed=991))
    action = np.asarray(measured[0]["actions"], dtype=np.float32)
    raw_action = np.asarray(measured[0]["raw_actions"], dtype=np.float32)
    if action.shape != (50, 7) or not np.isfinite(action).all():
        raise ValueError("Invalid continuous action output")
    if any(
        not np.array_equal(action, np.asarray(row["actions"], dtype=np.float32))
        for row in measured[1:]
    ):
        raise ValueError("Identical seeded observations produced nondeterministic actions")
    result = {
        "mode": mode,
        "calls": calls,
        "warm_calls": 100,
        "input_sha256": sha256(INPUT),
        "adapter_sha256": sha256(Path(adapter) / "adapter_model.safetensors"),
        "model_load_s": engine.load_s,
        "chunk_latency": percentiles([row["chunk_s"] for row in measured]),
        "end_to_end_latency": percentiles([row["predict_s"] for row in measured]),
        "throughput_calls_s": calls / sum(row["predict_s"] for row in measured),
        "gpu_peak_allocated": max(row["gpu_allocated_peak"] for row in measured),
        "gpu_peak_reserved": max(row["gpu_reserved_peak"] for row in measured),
        "clipped_values": max(row["clipped_values"] for row in measured),
        "actions": action.tolist(),
        "raw_actions": raw_action.tolist(),
    }
    atomic_json(OUT / (mode + ".json"), result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("actions", "raw_actions")}))


def parent(calls, timeout):
    OUT.mkdir(parents=True, exist_ok=True)
    adapter = Path(
        json.loads(
            (
                EVIDENCE / "modelops/jobs/m05-corrective-cam180-s5501/artifacts/result.json"
            ).read_text()
        )["adapter"]
    )
    with (EVIDENCE / "gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipts = {}
        for mode in MODES:
            receipt = OUT / (mode + ".json")
            if receipt.exists():
                prior = json.loads(receipt.read_text())
                if (
                    prior.get("calls") != calls
                    or prior.get("input_sha256") != sha256(INPUT)
                    or prior.get("adapter_sha256") != sha256(adapter / "adapter_model.safetensors")
                ):
                    raise ValueError(
                        f"Existing inference receipt has different frozen inputs: {mode}"
                    )
            else:
                log = OUT / (mode + ".log")
                with log.open("w") as f:
                    try:
                        subprocess.run(
                            [
                                sys.executable,
                                __file__,
                                "--mode",
                                mode,
                                "--calls",
                                str(calls),
                                "--adapter",
                                str(adapter),
                            ],
                            env={k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"},
                            stdout=f,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=timeout,
                        )
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                        receipts[mode] = {"status": "failed", "error": repr(exc), "log": str(log)}
                        atomic_json(OUT / "matrix.json", receipts)
                        continue
            receipts[mode] = {"status": "completed", **json.loads(receipt.read_text())}
            atomic_json(OUT / "matrix.json", receipts)
    if receipts.get("eager", {}).get("status") != "completed":
        raise RuntimeError("Eager baseline unavailable")
    baseline = np.asarray(receipts["eager"]["actions"])
    baseline_raw = np.asarray(receipts["eager"]["raw_actions"])
    for mode, row in receipts.items():
        if row["status"] != "completed":
            continue
        delta = np.abs(np.asarray(row["actions"]) - baseline)
        row["action_linf_vs_eager"] = float(delta.max())
        row["action_mae_vs_eager"] = float(delta.mean())
        raw_delta = np.abs(np.asarray(row["raw_actions"]) - baseline_raw)
        row["raw_action_linf_vs_eager"] = float(raw_delta.max())
        row["raw_action_mae_vs_eager"] = float(raw_delta.mean())
        row["correctness_gate"] = row["action_linf_vs_eager"] <= (
            1e-6 if mode == "cuda_graph" else 0.02
        ) and row["action_mae_vs_eager"] <= (1e-6 if mode == "cuda_graph" else 0.005)
        row.pop("actions", None)
        row.pop("raw_actions", None)
    atomic_json(
        OUT / "matrix.json",
        {
            "scope": "one real M04 observation; static inference correctness and latency, not closed-loop improvement",
            "hardware": "RTX3090",
            "modes": receipts,
            "eager_chunk_p50_s": receipts["eager"]["chunk_latency"]["p50_s"],
            "completed_modes": sum(v["status"] == "completed" for v in receipts.values()),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.calls < 1 or args.calls > 100:
        raise ValueError("Frozen profiling budget is 1..100 calls")
    if args.mode:
        child(args.mode, args.calls, args.adapter)
    else:
        parent(args.calls, args.timeout)


if __name__ == "__main__":
    main()
