"""Fresh pinned SmolVLA inference from a verified M02 failure checkpoint."""

from __future__ import annotations

import hashlib
import json
import os

os.environ.pop(
    "LD_LIBRARY_PATH", None
)  # avoid incompatible base-conda cuDNN shadowing the project wheel
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.libero import LiberoEnv
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from libero.libero import benchmark

from robot_vla.instrumentation import ChunkProfiler
from robot_vla.m01 import fit_libero_state_to_checkpoint
from robot_vla.m04_interventions import (
    benchmark_suite_name,
    brightness_camera_keys,
    intervention_is_active,
    scale_environment_action,
)
from robot_vla.observations import libero_batch

ROOT = Path("/data/VLA-Forge")
CHECKPOINT = ROOT / "Assets/models/smolvla_libero"
VLM = ROOT / "Assets/models/smolvlm2/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
SOURCE_SUITE = os.environ.get("M04_SUITE", "goal")
SOURCE_TASK_ID = int(os.environ.get("M04_TASK_ID", "6"))
SOURCE_EPISODE_ID = int(os.environ.get("M04_SOURCE_EPISODE", "7"))
M02_EPISODE_DIR = (
    ROOT
    / "Evidence/M02/source/M02_libero40/registry/runs/20261010T030315-982cacc8515f"
    / "episodes"
    / f"{SOURCE_SUITE}_task{SOURCE_TASK_ID:02d}_episode{SOURCE_EPISODE_ID:02d}_attempt1"
)
M02_TRACE = M02_EPISODE_DIR / "action_trace.json"
M02_EPISODE = json.loads((M02_EPISODE_DIR / "episode.json").read_text())
ENV_SEED = int(M02_EPISODE["seed"])
POLICY_SEED = int(os.environ.get("M04_POLICY_SEED", str(ENV_SEED)))
M03 = ROOT / "Evidence/M03/source/M03_interventional_diagnosis"
SOURCE = M03 / "episodes/m03-g09-goal-t06-init07_baseline.npz"
OUT = ROOT / "Evidence/M04/artifacts/policy_attempts"
OUT.mkdir(parents=True, exist_ok=True)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resource_sample():
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "cpu_percent": psutil.cpu_percent(),
        "rss_bytes": psutil.Process().memory_info().rss,
        "gpu": gpu,
    }


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    torch.set_num_threads(1)
    torch.manual_seed(84067)
    np.random.seed(84067)
    assert CHECKPOINT.is_dir() and VLM.is_dir()
    source_trace = json.loads(M02_TRACE.read_text())
    prefix = [
        np.asarray(x["env_step_argument"], dtype=np.float32) for x in source_trace["steps"][:100]
    ]
    if len(prefix) != 100:
        raise RuntimeError(f"M02 source has fewer than 100 prefix actions: {len(prefix)}")
    if SOURCE_SUITE == "goal" and SOURCE_TASK_ID == 6 and SOURCE_EPISODE_ID == 7:
        with np.load(SOURCE) as z:
            expected_qpos, expected_qvel, expected_time = (
                z["sim_qpos"][0].copy(),
                z["sim_qvel"][0].copy(),
                float(z["sim_time"][0]),
            )
    else:
        expected_qpos = expected_qvel = None
        expected_time = None
    import libero.libero as libero_api

    libero_api._assets_path_cache = str(ROOT / "Assets/LIBERO-assets")

    cfg = PreTrainedConfig.from_pretrained(CHECKPOINT)
    cfg.device = "cuda"
    cfg.vlm_model_name = str(VLM)
    state_dim = cfg.input_features["observation.state"].shape[0]
    policy_load_start = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained(
        CHECKPOINT, config=cfg, strict=True, local_files_only=True
    )
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=CHECKPOINT,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": str(VLM)},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    normalizer_stats = pre.steps[-1].stats.get("observation.state", {})
    normalizer_dim = int(np.asarray(normalizer_stats["mean"]).shape[-1])
    if cfg.output_features["action"].shape != (7,) or state_dim != 6 or normalizer_dim != 8:
        raise RuntimeError(
            f"checkpoint feature mismatch: action={cfg.output_features['action'].shape}, state={state_dim}, normalizer={normalizer_dim}"
        )
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - policy_load_start
    policy_hash = digest(CHECKPOINT / "model.safetensors")
    vlm_hash = digest(VLM / "model.safetensors")
    torch.cuda.reset_peak_memory_stats()
    chunk_meter = ChunkProfiler(policy, torch.cuda.synchronize)
    benchmark_suite = benchmark_suite_name(SOURCE_SUITE)
    suite = benchmark.get_benchmark_dict()[benchmark_suite]()
    task = suite.get_task(SOURCE_TASK_ID)
    result_rows = []
    condition_specs = {
        "baseline": ("m04-g09-fresh-baseline-cudnnfix-01", "baseline_rgb", 1.0, 0, 1.0),
        "brightness_all120": (
            "m04-g09-fresh-brightness-115-cudnnfix-01",
            "input_brightness_gain_1.15_all120",
            1.15,
            120,
            1.0,
        ),
        "brightness_first50": (
            "m04-g09-fresh-brightness-115-first50-01",
            "input_brightness_gain_1.15_first50",
            1.15,
            50,
            1.0,
        ),
        "action_scale": (
            "m04-g09-action-scale-090-01",
            "action_scale_0.9",
            1.0,
            0,
            float(os.environ.get("M04_ACTION_SCALE", "0.9")),
        ),
    }
    selected = [
        x.strip()
        for x in os.environ.get("M04_POLICY_CONDITIONS", "baseline,brightness_all120").split(",")
        if x.strip()
    ]
    for condition_key in selected:
        attempt_id, condition, gain, intervention_steps, action_scale = condition_specs[
            condition_key
        ]
        attempt_id = os.environ.get("M04_RUN_ID", attempt_id)
        # Every condition has an explicit default no-op observation window.
        # Brightness interventions override it; action-scale conditions do not.
        intervention_start = 0
        brightness_camera = "both"
        if condition_key == "brightness_first50":
            brightness_camera = os.environ.get("M04_BRIGHTNESS_CAMERA", "both")
            brightness_camera_keys(brightness_camera)
            gain = float(os.environ.get("M04_BRIGHTNESS_GAIN", str(gain)))
            intervention_steps = int(
                os.environ.get("M04_INTERVENTION_STEPS", str(intervention_steps))
            )
            intervention_start = int(os.environ.get("M04_INTERVENTION_START", "0"))
            if not 0 <= intervention_steps <= 120:
                raise ValueError(
                    "M04_INTERVENTION_STEPS must be within the 120-step correction horizon"
                )
            if intervention_start < 0 or intervention_start + intervention_steps > 120:
                raise ValueError("brightness intervention window must fit the 120-step horizon")
            condition = (
                f"input_brightness_gain_{gain:g}_steps{intervention_start}-"
                f"{intervention_start + intervention_steps}_camera_{brightness_camera}"
            )
        elif condition_key == "action_scale":
            action_scale = float(os.environ.get("M04_ACTION_SCALE", str(action_scale)))
            # Validate before any expensive model or environment work begins.
            scale_environment_action(np.zeros(7, dtype=np.float32), action_scale)
            condition = f"action_scale_{action_scale:g}"
        env = LiberoEnv(
            suite,
            SOURCE_TASK_ID,
            benchmark_suite,
            episode_length=300,
            obs_type="pixels_agent_pos",
            observation_width=256,
            observation_height=256,
            episode_index=SOURCE_EPISODE_ID,
            init_states=True,
            hard_reset=True,
            control_freq=20,
            num_steps_wait=10,
            control_mode="relative",
        )
        env_start = time.perf_counter()
        obs, _ = env.reset(seed=ENV_SEED)
        # For M02 failures without an M03 checkpoint, establish a physical reference
        # by replaying the same frozen M02 prefix twice from independent hard resets.
        if expected_qpos is None:
            reference_obs = obs
            for action in prefix:
                reference_obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    raise RuntimeError("M02 reference prefix terminated before step 100")
            ref = env._env.sim.data
            expected_qpos = np.asarray(ref.qpos).copy()
            expected_qvel = np.asarray(ref.qvel).copy()
            expected_time = float(ref.time)
            # LiberoEnv.reset advances init_state_id; pin it back before the independent replay.
            env.init_state_id = SOURCE_EPISODE_ID
            obs, _ = env.reset(seed=ENV_SEED)
        reset_s = time.perf_counter() - env_start
        event_path = OUT / f"{attempt_id}_events.jsonl"

        def log_event(event):
            with event_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
                f.flush()
                os.fsync(f.fileno())

        log_event(
            {
                "event": "episode_started",
                "episode_id": attempt_id,
                "condition": condition,
                "brightness_gain": gain,
                "brightness_camera": brightness_camera,
                "seed": ENV_SEED,
                "init_state_id": SOURCE_EPISODE_ID,
                "policy_seed": POLICY_SEED,
                "utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        frames = []
        action_trace, qpos_trace, qvel_trace, state_trace, inference_ms, simulation_ms = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        def pair(o):
            return np.concatenate([o["pixels"]["image"], o["pixels"]["image2"]], axis=1).astype(
                np.uint8, copy=True
            )

        def capture(o):
            frames.append(pair(o))
            sim = env._env.sim.data
            qpos_trace.append(np.asarray(sim.qpos).copy())
            qvel_trace.append(np.asarray(sim.qvel).copy())
            rs = o["robot_state"]
            state_trace.append(
                {
                    "eef_pos": np.asarray(rs["eef"]["pos"]).copy(),
                    "eef_quat": np.asarray(rs["eef"]["quat"]).copy(),
                    "joints_pos": np.asarray(rs["joints"]["pos"]).copy(),
                    "joints_vel": np.asarray(rs["joints"]["vel"]).copy(),
                    "gripper_qpos": np.asarray(rs["gripper"]["qpos"]).copy(),
                }
            )

        capture(obs)
        for action in prefix:
            obs, _, term, trunc, _ = env.step(action)
            action_trace.append({"phase": "source_prefix", "env_step_argument": action.tolist()})
            capture(obs)
            log_event(
                {
                    "event": "prefix_step_completed",
                    "step": len(action_trace),
                    "action": action.tolist(),
                    "qpos": env._env.sim.data.qpos.tolist(),
                    "qvel": env._env.sim.data.qvel.tolist(),
                    "sim_time": float(env._env.sim.data.time),
                }
            )
            if term or trunc:
                raise RuntimeError("source prefix unexpectedly ended before checkpoint")
        sim = env._env.sim.data
        recovery = {
            "qpos_linf": float(np.max(np.abs(sim.qpos - expected_qpos))),
            "qvel_linf": float(np.max(np.abs(sim.qvel - expected_qvel))),
            "time_abs": abs(float(sim.time) - expected_time),
            "ncon": int(sim.ncon),
        }
        if (
            recovery["qpos_linf"] > 1e-5
            or recovery["qvel_linf"] > 1e-4
            or recovery["time_abs"] > 1e-8
        ):
            raise RuntimeError(f"checkpoint recovery gate failed: {recovery}")
        log_event(
            {
                "event": "checkpoint_recovered",
                **recovery,
                "utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        # Match M03's episode-local stochastic reset after the exact source-prefix replay.
        import random

        random.seed(POLICY_SEED)
        np.random.seed(POLICY_SEED)
        torch.manual_seed(POLICY_SEED)
        torch.cuda.manual_seed_all(POLICY_SEED)
        policy.reset()
        chunk_meter.reset()
        torch.cuda.reset_peak_memory_stats()
        resource_rows = [resource_sample()]
        log_event({"event": "resource_sample", **resource_rows[-1]})
        t0 = time.perf_counter()
        successes = []
        raw_actions = []
        clipped_count = 0
        with torch.inference_mode():
            for step in range(120):
                batch = libero_batch(obs)
                batch["observation.state"], state_semantics = fit_libero_state_to_checkpoint(
                    batch["observation.state"],
                    expected_dim=state_dim,
                    normalizer_dim=normalizer_dim,
                )
                batch["task"] = [task.language]
                # Treatment acts only on the current two camera inputs; the physical state is unchanged.
                step_gain = (
                    gain
                    if intervention_is_active(step, intervention_start, intervention_steps)
                    else 1.0
                )
                if step_gain != 1.0:
                    for key in brightness_camera_keys(brightness_camera):
                        if key in batch:
                            batch[key] = torch.clamp(batch[key] * step_gain, 0.0, 1.0)
                processed = pre(batch)
                log_event(
                    {
                        "event": "inference_started",
                        "step": step,
                        "sim_time": float(env._env.sim.data.time),
                        "qpos": env._env.sim.data.qpos.tolist(),
                        "qvel": env._env.sim.data.qvel.tolist(),
                    }
                )
                torch.cuda.synchronize()
                chunk_count_before = len(chunk_meter.chunk_calls)
                infer0 = time.perf_counter()
                raw = policy.select_action(processed)
                torch.cuda.synchronize()
                if len(chunk_meter.chunk_calls) > chunk_count_before:
                    log_event({"event": "action_chunk_inference", **chunk_meter.chunk_calls[-1]})
                infer_ms = (time.perf_counter() - infer0) * 1000
                raw_np = raw.detach().float().cpu().numpy()[0]
                commanded = post(raw).detach().float().cpu().numpy()[0]
                scaled_command = scale_environment_action(commanded, action_scale)
                clipped = scaled_command
                clipped_count += int(np.count_nonzero(clipped != commanded * action_scale))
                raw_actions.append(raw_np.copy())
                log_event(
                    {
                        "event": "policy_action_ready",
                        "step": step,
                        "policy_output": raw_np.tolist(),
                        "env_step_argument": clipped.tolist(),
                        "inference_ms": infer_ms,
                        "brightness_gain": step_gain,
                        "action_scale": action_scale,
                        "intervention_start": intervention_start
                        if condition_key == "brightness_first50"
                        else None,
                    }
                )
                sim_t = time.perf_counter()
                obs, _, terminated, truncated, info = env.step(clipped)
                sim_ms = (time.perf_counter() - sim_t) * 1000
                inference_ms.append(infer_ms)
                simulation_ms.append(sim_ms)
                action_trace.append(
                    {
                        "phase": "policy",
                        "step": step,
                        "policy_output": raw_np.tolist(),
                        "env_step_argument": clipped.tolist(),
                        "brightness_gain": gain,
                        "action_scale": action_scale,
                        "inference_ms": infer_ms,
                        "simulation_step_ms": sim_ms,
                    }
                )
                capture(obs)
                success = bool(info.get("is_success", False))
                successes.append(success)
                log_event(
                    {
                        "event": "step_completed",
                        "step": step,
                        "success": success,
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "qpos": env._env.sim.data.qpos.tolist(),
                        "qvel": env._env.sim.data.qvel.tolist(),
                        "sim_time": float(env._env.sim.data.time),
                        "inference_ms": infer_ms,
                        "simulation_step_ms": sim_ms,
                    }
                )
                if success or terminated or truncated:
                    break
                if (step + 1) % 20 == 0:
                    resource_rows.append(resource_sample())
                    log_event({"event": "resource_sample", **resource_rows[-1]})
        elapsed = time.perf_counter() - t0
        resource_rows.append(resource_sample())
        log_event({"event": "resource_sample", **resource_rows[-1]})
        result = {
            "episode_id": attempt_id,
            "source_m02_episode": M02_EPISODE_DIR.name,
            "source_trace_sha256": digest(M02_TRACE),
            "source_checkpoint": f"M02_{M02_EPISODE_DIR.name}_prefix100",
            "source_checkpoint_action_trace_sha256": digest(M02_TRACE),
            "suite": SOURCE_SUITE,
            "task_id": SOURCE_TASK_ID,
            "task_name": task.name,
            "instruction": task.language,
            "init_state_id": SOURCE_EPISODE_ID,
            "seed": ENV_SEED,
            "policy_seed": POLICY_SEED,
            "prefix_steps": 100,
            "inference_action_steps": len(inference_ms),
            "success": bool(successes and successes[-1]),
            "termination_reason": "success" if successes and successes[-1] else "budget_or_timeout",
            "policy_repo": "lerobot/smolvla_libero",
            "policy_revision": "31d453f7edd78c839a8bbc39744a292686daf0de",
            "policy_model_sha256": policy_hash,
            "vlm_repo": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
            "vlm_revision": "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
            "vlm_model_sha256": vlm_hash,
            "condition": condition,
            "brightness_gain": gain,
            "brightness_camera": brightness_camera,
            "intervention_observation_steps": intervention_steps,
            "intervention_start": intervention_start
            if condition_key == "brightness_first50"
            else None,
            "action_scale": action_scale,
            "control_mode": "relative",
            "action_chunk_size": int(cfg.chunk_size),
            "n_action_steps": int(cfg.n_action_steps),
            "checkpoint_recovery": recovery,
            "state_semantics": state_semantics,
            "reset_seconds": reset_s,
            "model_load_seconds": model_load_s,
            "policy_inference_ms": {
                "p50": float(np.percentile(inference_ms, 50)),
                "p95": float(np.percentile(inference_ms, 95)),
                "p99": float(np.percentile(inference_ms, 99)),
            },
            "chunk_inference_ms": {
                "p50": float(np.percentile([x["latency_ms"] for x in chunk_meter.chunk_calls], 50))
                if chunk_meter.chunk_calls
                else None,
                "p95": float(np.percentile([x["latency_ms"] for x in chunk_meter.chunk_calls], 95))
                if chunk_meter.chunk_calls
                else None,
                "p99": float(np.percentile([x["latency_ms"] for x in chunk_meter.chunk_calls], 99))
                if chunk_meter.chunk_calls
                else None,
                "calls": len(chunk_meter.chunk_calls),
            },
            "simulation_step_ms": {
                "p50": float(np.percentile(simulation_ms, 50)),
                "p95": float(np.percentile(simulation_ms, 95)),
                "p99": float(np.percentile(simulation_ms, 99)),
            },
            "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "gpu_hours_allocated": elapsed / 3600.0,
            "gpu_hours_inference_estimate": elapsed / 3600.0,
            "action_components_clipped": clipped_count,
            "cpu_gpu_samples": resource_rows,
            "video_fps": 20,
            "video_frames": len(frames),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        stem = OUT / attempt_id
        npz = stem.with_suffix(".npz")
        video = stem.with_name(stem.name + "_dual_camera.mp4")
        trace_path = stem.with_suffix(".json")
        arrays = {k: np.asarray([x[k] for x in state_trace]) for k in state_trace[0]}
        np.savez_compressed(
            npz,
            actions=np.asarray([x["env_step_argument"] for x in action_trace], dtype=np.float32),
            qpos=np.asarray(qpos_trace),
            qvel=np.asarray(qvel_trace),
            robot_state=arrays["eef_pos"],
            dual_camera_frames=np.asarray(frames),
            eef_pos=arrays["eef_pos"],
            eef_quat=arrays["eef_quat"],
            joint_pos=arrays["joints_pos"],
            joint_vel=arrays["joints_vel"],
            gripper_qpos=arrays["gripper_qpos"],
        )
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        if not writer.isOpened():
            raise RuntimeError("video writer failed")
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        result["trajectory"] = {
            "path": str(npz),
            "sha256": digest(npz),
            "bytes": npz.stat().st_size,
        }
        result["video"] = {
            "path": str(video),
            "sha256": digest(video),
            "bytes": video.stat().st_size,
            "fps": 20,
            "width": w,
            "height": h,
            "cameras": ["agentview_image", "robot0_eye_in_hand_image"],
        }
        result["action_trace"] = {"path": str(trace_path), "sha256": None}
        trace_path.write_text(
            json.dumps({"steps": action_trace, "success": result["success"]}, indent=2),
            encoding="utf-8",
        )
        result["action_trace"]["sha256"] = digest(trace_path)
        log_event(
            {
                "event": "episode_complete",
                "success": result["success"],
                "result_sha256": digest(trace_path),
            }
        )
        (stem.with_name(stem.name + "_result.json")).write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        with (OUT / "attempts.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        result_rows.append(result)
        env.close()
        print(
            json.dumps(
                {
                    "episode_id": attempt_id,
                    "success": result["success"],
                    "condition": condition,
                    "steps": len(inference_ms),
                    "inference_ms": result["policy_inference_ms"],
                    "chunk_ms": result["chunk_inference_ms"],
                    "gpu_peak_bytes": result["gpu_peak_allocated_bytes"],
                    "recovery": recovery,
                    "trajectory": result["trajectory"],
                    "video": result["video"],
                }
            ),
            flush=True,
        )
    print(
        "RESULTS",
        json.dumps(
            [
                {
                    "episode_id": x["episode_id"],
                    "success": x["success"],
                    "condition": x["condition"],
                }
                for x in result_rows
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one real SmolVLA + LIBERO M04 policy intervention episode."
    )
    parser.parse_args()
    main()
