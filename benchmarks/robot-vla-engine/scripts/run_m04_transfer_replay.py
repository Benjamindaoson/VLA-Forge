"""Test a saved SmolVLA correction on another real M02 failure state; no model inference."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/data/VLA-Forge")
EVID = ROOT / "Evidence/M04"
RUN = "20261010T030315-982cacc8515f"
M03 = (
    ROOT
    / "Evidence/M03/source/M03_interventional_diagnosis/episodes/m03-g09-goal-t06-init07_brightness.npz"
)
M03_META = M03.with_suffix(".json")
OUT = EVID / "artifacts/transfer_candidates"
OUT.mkdir(parents=True, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import libero.libero as libero_api
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark

    ep = int(os.environ.get("M04_SOURCE_EPISODE", "0"))
    edir = (
        ROOT
        / "Evidence/M02/source/M02_libero40/registry/runs"
        / RUN
        / "episodes"
        / f"goal_task06_episode{ep:02d}_attempt1"
    )
    source_meta = json.loads((edir / "episode.json").read_text())
    trace = json.loads((edir / "action_trace.json").read_text())
    if source_meta["success"] is not False:
        raise RuntimeError("source must be an M02 failure")
    seed = int(source_meta["seed"])
    prefix = np.asarray([x["env_step_argument"] for x in trace["steps"][:100]], np.float32)
    if len(prefix) != 100:
        raise RuntimeError("source prefix must have 100 actions")
    with np.load(M03) as z:
        correction = z["actions"].copy()
    action_scale = float(os.environ.get("M04_ACTION_SCALE", "1.0"))
    action_axis = os.environ.get("M04_ACTION_AXIS", "xyz")
    axes = {"x": (0,), "y": (1,), "z": (2,), "rotation": (3, 4, 5), "xyz": (0, 1, 2)}
    if action_axis not in axes:
        raise ValueError(f"unsupported correction axis: {action_axis}")
    correction[:, axes[action_axis]] *= action_scale
    scale_token = f"{action_axis}{int(round(action_scale * 1000)):04d}"
    libero_api._assets_path_cache = str(ROOT / "Assets/LIBERO-assets")
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    env = LiberoEnv(
        suite,
        6,
        "libero_goal",
        episode_length=300,
        obs_type="pixels_agent_pos",
        observation_width=256,
        observation_height=256,
        episode_index=ep,
        init_states=True,
        hard_reset=True,
        control_freq=20,
        num_steps_wait=10,
        control_mode="relative",
    )
    attempt = f"m04-goal-t06-e{ep:02d}-m03-success-transfer-{scale_token}"
    out = OUT / attempt
    if out.with_suffix(".json").exists():
        raise FileExistsError(out.with_suffix(".json"))
    event_path = Path(str(out) + ".events.jsonl")

    def log_event(event):
        with event_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
            f.flush()
            os.fsync(f.fileno())

    log_event(
        {
            "event": "started",
            "episode_id": attempt,
            "source_m02": edir.name,
            "action_axis": action_axis,
            "action_scale": action_scale,
            "utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    t0 = time.perf_counter()
    obs, _ = env.reset(seed=seed)

    def prefix_replay():
        nonlocal obs
        for a in prefix:
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                raise RuntimeError("M02 prefix terminated before 100")

    prefix_replay()
    d = env._env.sim.data
    refq = np.array(d.qpos).copy()
    refv = np.array(d.qvel).copy()
    reft = float(d.time)
    env.init_state_id = ep
    obs, _ = env.reset(seed=seed)
    prefix_replay()
    d = env._env.sim.data
    recovery = {
        "qpos_linf": float(np.max(np.abs(d.qpos - refq))),
        "qvel_linf": float(np.max(np.abs(d.qvel - refv))),
        "time_abs": abs(float(d.time) - reft),
        "ncon": int(d.ncon),
    }
    if recovery["qpos_linf"] > 1e-5 or recovery["qvel_linf"] > 1e-4 or recovery["time_abs"] > 1e-8:
        raise RuntimeError(f"prefix checkpoint recovery failed {recovery}")
    log_event({"event": "checkpoint_recovered", **recovery})

    def pair(o):
        return np.concatenate([o["pixels"]["image"], o["pixels"]["image2"]], axis=1).astype(
            np.uint8, copy=True
        )

    frames = [pair(obs)]
    qpos = [np.array(d.qpos).copy()]
    qvel = [np.array(d.qvel).copy()]
    states = []
    acts = []
    succ = []

    def capture(o):
        x = env._env.sim.data
        qpos.append(np.array(x.qpos).copy())
        qvel.append(np.array(x.qvel).copy())
        r = o["robot_state"]
        states.append(
            {
                "eef_pos": np.array(r["eef"]["pos"]).copy(),
                "eef_quat": np.array(r["eef"]["quat"]).copy(),
                "joint_pos": np.array(r["joints"]["pos"]).copy(),
                "joint_vel": np.array(r["joints"]["vel"]).copy(),
                "gripper_qpos": np.array(r["gripper"]["qpos"]).copy(),
            }
        )

    capture(obs)
    sim_ms = []
    success = False
    for a in correction:
        t = time.perf_counter()
        obs, _, term, trunc, info = env.step(a)
        sim_ms.append((time.perf_counter() - t) * 1000)
        acts.append(a.copy())
        frames.append(pair(obs))
        capture(obs)
        success = bool(info.get("is_success", False))
        succ.append(success)
        log_event(
            {
                "event": "correction_step",
                "step": len(acts) - 1,
                "action": np.asarray(a).tolist(),
                "success": success,
                "qpos": env._env.sim.data.qpos.tolist(),
                "qvel": env._env.sim.data.qvel.tolist(),
                "sim_time": float(env._env.sim.data.time),
                "utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        if success or term or trunc:
            break
    elapsed = time.perf_counter() - t0
    npz = out.with_suffix(".npz")
    vid = Path(str(out) + "_dual_camera.mp4")
    np.savez_compressed(
        npz,
        actions=np.asarray(acts),
        dual_camera_frames=np.asarray(frames),
        qpos=np.asarray(qpos),
        qvel=np.asarray(qvel),
        joint_pos=np.asarray([x["joint_pos"] for x in states]),
        joint_vel=np.asarray([x["joint_vel"] for x in states]),
        eef_pos=np.asarray([x["eef_pos"] for x in states]),
        eef_quat=np.asarray([x["eef_quat"] for x in states]),
        gripper_qpos=np.asarray([x["gripper_qpos"] for x in states]),
    )
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    if not wr.isOpened():
        raise RuntimeError("OpenCV video writer failed")
    for f in frames:
        wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    wr.release()
    result = {
        "episode_id": attempt,
        "source_m02_episode": edir.name,
        "source_m02_episode_json_sha256": sha(edir / "episode.json"),
        "source_m02_action_trace_sha256": sha(edir / "action_trace.json"),
        "source_was_measured_failure": True,
        "suite": "libero_goal",
        "task_id": 6,
        "task_name": source_meta["task_name"],
        "instruction": source_meta["instruction"],
        "init_state_id": ep,
        "seed": seed,
        "prefix_steps": len(prefix),
        "prefix_source": "exact M02 env.step arguments",
        "recovery_method": "fixed LIBERO reset + independent duplicate source-prefix replay with init_state_id pinned after reset; compare qpos/qvel/simulation time",
        "recovery": recovery,
        "correction_source_m03_episode": "m03-g09-goal-t06-init07_brightness",
        "correction_source_m03_npz_sha256": sha(M03),
        "correction_source_m03_episode_json_sha256": sha(M03_META),
        "correction_actions_available": len(correction),
        "correction_actions_executed": len(acts),
        "action_sha256": hashlib.sha256(np.asarray(acts, dtype=np.float32).tobytes()).hexdigest(),
        "action_source": f"saved M03 SmolVLA treatment env.step arguments with {action_axis} scale={action_scale}; no new policy inference; controlled replay variant",
        "correction_transform": {"axis": action_axis, "scale": action_scale},
        "success": success,
        "termination_reason": "success"
        if success
        else ("terminated" if term else "truncated" if trunc else "correction_budget_exhausted"),
        "simulator_wall_seconds": elapsed,
        "policy_inference_seconds": 0.0,
        "gpu_hours_policy_inference": 0.0,
        "simulation_step_ms": {
            "p50": float(np.percentile(sim_ms, 50)),
            "p95": float(np.percentile(sim_ms, 95)),
            "p99": float(np.percentile(sim_ms, 99)),
        },
        "trajectory": {"path": str(npz), "sha256": sha(npz), "bytes": npz.stat().st_size},
        "video": {
            "path": str(vid),
            "sha256": sha(vid),
            "bytes": vid.stat().st_size,
            "frames": len(frames),
            "fps": 20,
            "width": w,
            "height": h,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    withopen = OUT / "attempts.jsonl"
    with withopen.open("a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
