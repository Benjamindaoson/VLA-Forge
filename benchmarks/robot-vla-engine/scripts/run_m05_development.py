"""Resumeable M05-only SmolVLA development rollouts in real LIBERO/MuJoCo."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import random
import time
import traceback
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# Reuse the validated project-owned LIBERO assets and initial-state files.
os.environ["LIBERO_CONFIG_PATH"] = str(Path(__file__).resolve().parents[3] / "work/libero")

import av
import numpy as np
import torch
from lerobot.envs.libero import LiberoEnv
from libero.libero import benchmark

from robot_vla.instrumentation import ChunkProfiler, ResourceMonitor
from robot_vla.m01 import fit_libero_state_to_checkpoint
from robot_vla.m05_training import EVIDENCE, atomic_json, load_policy, sha256
from robot_vla.observations import libero_batch
from robot_vla.preflight import create_libero_env
from robot_vla.protocol import load_protocol
from robot_vla.registry import Registry
from robot_vla.rollout import record_episode

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("frozen", "standard", "corrective", "replay")
SUITES = {"goal": "libero_goal", "object": "libero_object"}


def validate_plan(plan):
    if plan.get("scope") != "M05-development-only":
        raise ValueError("Expected frozen M05 development protocol")
    seen = set()
    for item in plan["development"]:
        key = (item["suite"], int(item["task"]), int(item["init"]))
        if item["suite"] not in SUITES or key in seen:
            raise ValueError("Unknown or repeated development case")
        if (
            int(item["init"]) in plan["formal_test_forbidden_init_ids"]
            or not 10 <= int(item["init"]) <= 13
        ):
            raise ValueError("M06 or unfrozen initial state is forbidden")
        if int(item["seed"]) < 1:
            raise ValueError("Missing frozen seed")
        seen.add(key)
    if len(seen) != 8:
        raise ValueError("Frozen M05 development protocol requires eight cases")
    return plan["development"]


def adapter_for(condition):
    if condition == "frozen":
        return None, sha256(Path("/data/VLA-Forge/Assets/models/smolvla_libero/model.safetensors"))
    results = {
        "standard": EVIDENCE / "training/standard/result.json",
        "corrective": EVIDENCE / "modelops/jobs/m05-corrective-cam180-s5501/artifacts/result.json",
        "replay": EVIDENCE / "training/replay_cam180/result.json",
    }
    result = json.loads(results[condition].read_text())
    if (
        result["status"] != "COMPLETED"
        or result["optimizer_updates"] != 500
        or not result["finite_gradients"]
        or result["reload_linf"] != 0
    ):
        raise ValueError(f"Unqualified training result: {condition}")
    adapter = Path(result["adapter"])
    if sha256(adapter / "adapter_model.safetensors") != result["adapter_sha256"]:
        raise ValueError(f"Adapter hash mismatch: {condition}")
    return adapter, result["adapter_sha256"]


def receipt_valid(receipt, digest):
    if (
        receipt.get("status") != "complete"
        or receipt.get("checkpoint_sha256") != digest
        or type(receipt.get("success")) is not bool
    ):
        return False
    for kind in ("video", "trajectory", "action_trace"):
        entry = receipt.get(kind, {})
        path = Path(entry.get("path", ""))
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            return False
    return True


def write_summary(output):
    rows = [json.loads(p.read_text()) for p in sorted((output / "receipts").glob("*.json"))]
    keys = [
        "condition",
        "suite",
        "task",
        "init",
        "seed",
        "status",
        "success",
        "steps",
        "wall_s",
        "chunk_mean_ms",
        "gpu_peak_allocated",
        "checkpoint_sha256",
        "error",
    ]
    tmp = output / "episodes.csv.tmp"
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(output / "episodes.csv")
    metrics = {}
    for condition in CONDITIONS:
        group = [r for r in rows if r["condition"] == condition and r["status"] == "complete"]
        metrics[condition] = {
            "success": sum(bool(r["success"]) for r in group),
            "total": len(group),
            "errors": sum(r["condition"] == condition and r["status"] != "complete" for r in rows),
        }
    atomic_json(
        output / "metrics.json",
        {"scope": "M05-development-only", "conditions": metrics, "updated_unix": time.time()},
    )


def execute(condition, cases, output, max_new, backend="eager"):
    adapter, digest = adapter_for(condition)
    policy, pre, post = load_policy(adapter, train=False, lora=False)
    base = policy.get_base_model() if adapter else policy
    if backend == "cuda_graph":
        from robot_vla.m04_acceleration import CudaGraphSampler

        base.model.sample_actions = CudaGraphSampler(base.model)
    elif backend != "eager":
        raise ValueError("Unknown closed-loop backend")
    measured = ChunkProfiler(base, torch.cuda.synchronize)
    protocol = load_protocol(ROOT / "configs/data_protocol_v1.json")
    protocol = json.loads(json.dumps(protocol))
    protocol["evaluation"]["initial_state_indices"] = [10, 11, 12, 13]
    protocol["evaluation"]["episodes_per_task"] = 4
    suite_classes = benchmark.get_benchmark_dict()
    suites = {
        k: suite_classes[v](task_order_index=protocol["task_order_index"])
        for k, v in SUITES.items()
    }
    registry = Registry(output / "registry")
    with (
        registry.run(
            {
                "scope": "M05-development-only",
                "condition": condition,
                "backend": backend,
                "checkpoint_sha256": digest,
                "cases": cases,
            }
        ) as run,
        ResourceMonitor(run, interval=2.0, gpu=True),
        torch.inference_mode(),
    ):
        new = 0
        for case in cases:
            name = f"{condition}_{case['suite']}_t{case['task']:02d}_i{case['init']:02d}"
            receipt_path = output / "receipts" / (name + ".json")
            if receipt_path.exists():
                prior = json.loads(receipt_path.read_text())
                if receipt_valid(prior, digest):
                    continue
                raise ValueError(f"Existing receipt cannot be silently replaced: {receipt_path}")
            if max_new is not None and new >= max_new:
                break
            new += 1
            suite_name = SUITES[case["suite"]]
            task = suites[case["suite"]].get_task(case["task"])
            seed = int(case["seed"])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
            horizon = int(protocol["evaluation"]["horizons"][suite_name])
            video_path = run.directory / "videos" / (name + ".mp4")
            video_path.parent.mkdir(parents=True, exist_ok=True)
            env = container = stream = None
            action_trace = []
            started = time.perf_counter()
            try:
                env = create_libero_env(
                    LiberoEnv,
                    suites[case["suite"]],
                    case["task"],
                    suite_name,
                    protocol,
                    case["init"],
                    256,
                    horizon,
                )
                container = av.open(str(video_path), "w")

                def process(obs):
                    batch = libero_batch(obs)
                    batch["observation.state"], _ = fit_libero_state_to_checkpoint(
                        batch["observation.state"], expected_dim=6, normalizer_dim=8
                    )
                    batch["task"] = [task.language]
                    return pre(batch)

                def output_action(action):
                    raw = action.detach().float().cpu().numpy()[0]
                    command = post(action).detach().float().cpu().numpy()[0]
                    clipped = np.clip(command, -1, 1).astype(np.float32)
                    action_trace.append(
                        {
                            "raw": raw.tolist(),
                            "command": command.tolist(),
                            "executed": clipped.tolist(),
                        }
                    )
                    return clipped

                def video_sink(obs, step):
                    nonlocal stream
                    frames = [
                        np.flip(obs["pixels"][key], axis=(0, 1)).copy()
                        for key in ("image", "image2")
                    ]
                    frame = np.concatenate(frames, axis=1)
                    if stream is None:
                        stream = container.add_stream(
                            "libx264", rate=protocol["evaluation"]["control_freq"]
                        )
                        stream.width, stream.height, stream.pix_fmt = (
                            frame.shape[1],
                            frame.shape[0],
                            "yuv420p",
                        )
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                        container.mux(packet)

                result = record_episode(
                    env,
                    measured,
                    run,
                    suite=case["suite"],
                    task_id=case["task"],
                    episode_id=case["init"],
                    seed=seed,
                    instruction=task.language,
                    max_steps=horizon,
                    preprocess=process,
                    postprocess=output_action,
                    synchronize=torch.cuda.synchronize,
                    state_extract=lambda obs: libero_batch(obs)["observation.state"][0].numpy(),
                    observation_sink=video_sink,
                    environment_config={
                        "control_mode": "relative",
                        "initial_state": case["init"],
                        "suite": suite_name,
                    },
                    policy_metadata={"condition": condition, "backend": backend, "checkpoint_sha256": digest},
                )
                if stream is None or not result.get("chunk_calls"):
                    raise ValueError("No measured chunk inference or frames")
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                container = None
                trace_path = run.directory / (name + "_actions.json")
                atomic_json(trace_path, action_trace)
                with np.load(result["trajectory_path"]) as trace:
                    actions = trace["actions"]
                    states = trace["states"]
                if (
                    not np.array_equal(
                        actions, np.asarray([a["executed"] for a in action_trace], dtype=np.float32)
                    )
                    or states.shape[1] != 8
                ):
                    raise ValueError("Action/state trace integrity mismatch")
                chunks = [x["latency_ms"] for x in result["chunk_calls"]]
                receipt = {
                    **case,
                    "condition": condition,
                    "backend": backend,
                    "status": "complete",
                    "success": bool(result["success"]),
                    "steps": len(actions),
                    "wall_s": time.perf_counter() - started,
                    "chunk_mean_ms": float(np.mean(chunks)),
                    "chunk_p95_ms": float(np.percentile(chunks, 95)),
                    "gpu_peak_allocated": torch.cuda.max_memory_allocated(),
                    "checkpoint_sha256": digest,
                    "registry_run_id": run.run_id,
                    "instruction": task.language,
                    "termination_reason": result["termination_reason"],
                    "video": {"path": str(video_path), "sha256": sha256(video_path)},
                    "trajectory": {
                        "path": result["trajectory_path"],
                        "sha256": sha256(result["trajectory_path"]),
                    },
                    "action_trace": {"path": str(trace_path), "sha256": sha256(trace_path)},
                }
                atomic_json(receipt_path, receipt)
                run.artifact(video_path, "dev_rollout_video")
                run.artifact(trace_path, "executed_action_trace")
                print(
                    json.dumps(
                        {"case": name, "success": receipt["success"], "steps": receipt["steps"]}
                    ),
                    flush=True,
                )
            except Exception as exc:
                atomic_json(
                    receipt_path,
                    {
                        **case,
                        "condition": condition,
                        "backend": backend,
                        "status": "error",
                        "success": None,
                        "checkpoint_sha256": digest,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                print(json.dumps({"case": name, "error": repr(exc)}), flush=True)
                raise
            finally:
                if container is not None:
                    container.close()
                if env is not None:
                    env.close()
                write_summary(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--max-new-per-condition", type=int)
    parser.add_argument("--backend", choices=("eager", "cuda_graph"), default="eager")
    parser.add_argument("--output", type=Path, default=EVIDENCE / "development")
    args = parser.parse_args()
    output = args.output
    if args.backend != "eager" and output.resolve() == (EVIDENCE / "development").resolve():
        raise ValueError("Graph rollouts require a separate evidence directory")
    (output / "receipts").mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT.parents[1] / "configs/m05/formal_protocol.json").read_text())
    cases = validate_plan(plan)
    with (EVIDENCE / "gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for condition in args.conditions:
            execute(condition, cases, output, args.max_new_per_condition, args.backend)
    write_summary(output)


if __name__ == "__main__":
    main()
