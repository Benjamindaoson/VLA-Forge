"""M04 physical prefix restore and independent replay of an M03 policy correction.

This is explicitly an action-sequence replay, not a new policy inference result.
It is useful to qualify source-state recovery and preserve an independently
reproduced successful corrected trajectory before expanding data collection.
"""

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
PROJECT = ROOT / "project/robot-vla"
EVIDENCE = ROOT / "Evidence/M04"
M02 = (
    ROOT
    / "Evidence/M02/source/M02_libero40/registry/runs/20261010T030315-982cacc8515f/episodes/goal_task06_episode07_attempt1/action_trace.json"
)
M03 = ROOT / "Evidence/M03/source/M03_interventional_diagnosis"
GROUP = "m03-g09-goal-t06-init07"
POLICY_RUN = M03 / "episodes" / f"{GROUP}_brightness.npz"
POLICY_META = M03 / "episodes" / f"{GROUP}_brightness.json"
OUT = EVIDENCE / "artifacts/first_correction"
OUT.mkdir(parents=True, exist_ok=True)


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def sim_data(env):
    return env._env.sim.data


def rgb_pair(obs):
    a = np.asarray(obs["pixels"]["image"], dtype=np.uint8)
    b = np.asarray(obs["pixels"]["image2"], dtype=np.uint8)
    return np.concatenate([a, b], axis=1)


def run_one(index, prefix, actions, expected, trace_hash, policy_hash):
    import libero.libero as libero_api
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark

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
        episode_index=7,
        init_states=True,
        hard_reset=True,
        control_freq=20,
        num_steps_wait=10,
        control_mode="relative",
    )
    t0 = time.perf_counter()
    obs, info = env.reset(seed=84067)
    reset_s = time.perf_counter() - t0
    frames = [rgb_pair(obs)]
    actions_log = []
    states = []
    qpos = []
    qvel = []
    simtime = []
    success_flags = []

    def capture(o):
        d = sim_data(env)
        qpos.append(np.asarray(d.qpos).copy())
        qvel.append(np.asarray(d.qvel).copy())
        simtime.append(float(d.time))
        rs = o["robot_state"]
        states.append(
            {
                "joint_pos": np.asarray(rs["joints"]["pos"]).copy(),
                "joint_vel": np.asarray(rs["joints"]["vel"]).copy(),
                "eef_pos": np.asarray(rs["eef"]["pos"]).copy(),
                "eef_quat": np.asarray(rs["eef"]["quat"]).copy(),
                "gripper_qpos": np.asarray(rs["gripper"]["qpos"]).copy(),
            }
        )

    capture(obs)
    try:
        for a in prefix:
            obs, _, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
            actions_log.append(np.asarray(a, dtype=np.float32))
            frames.append(rgb_pair(obs))
            capture(obs)
            if term or trunc:
                raise RuntimeError(f"prefix ended early: terminated={term}, truncated={trunc}")
        d = sim_data(env)
        qe = np.max(np.abs(np.asarray(d.qpos) - np.asarray(expected["qpos"])))
        ve = np.max(np.abs(np.asarray(d.qvel) - np.asarray(expected["qvel"])))
        te = abs(float(d.time) - float(expected["time"]))
        replay_ok = bool(qe <= 1e-5 and ve <= 1e-4 and te <= 1e-8)
        if not replay_ok:
            raise RuntimeError(
                f"prefix recovery outside frozen tolerance: qpos_linf={qe}, qvel_linf={ve}, time_abs={te}"
            )
        prefix_metrics = {
            "qpos_linf": float(qe),
            "qvel_linf": float(ve),
            "time_abs": float(te),
            "contact_count": int(d.ncon),
            "state_match": True,
        }
        started = time.perf_counter()
        inference_times = []
        for a in actions:
            step_t = time.perf_counter()
            obs, _, term, trunc, info = env.step(np.asarray(a, dtype=np.float32))
            inference_times.append(time.perf_counter() - step_t)
            actions_log.append(np.asarray(a, dtype=np.float32))
            frames.append(rgb_pair(obs))
            capture(obs)
            is_success = bool(info.get("is_success", False))
            success_flags.append(is_success)
            if is_success or term or trunc:
                break
        exec_s = time.perf_counter() - started
        final_success = bool(info.get("is_success", False))
        result = {
            "episode_id": f"m04-first-correction-replay-{index:02d}",
            "source_m02_episode": "goal_task06_episode07_attempt1",
            "source_m02_trace_sha256": trace_hash,
            "correction_source_m03_episode": f"{GROUP}_brightness",
            "correction_source_m03_action_sha256": policy_hash,
            "suite": "libero_goal",
            "task_id": 6,
            "task_name": "put_the_cream_cheese_in_the_bowl",
            "instruction": "put the cream cheese in the bowl",
            "source_init_state_id": 7,
            "seed": 84067,
            "prefix_steps": len(prefix),
            "correction_actions_executed": len(actions_log) - len(prefix),
            "success": final_success,
            "termination_reason": "success"
            if final_success
            else (
                "terminated" if term else "truncated" if trunc else "correction_budget_exhausted"
            ),
            "control_mode": "relative",
            "action_dim": 7,
            "action_source": "replayed M03 SmolVLA brightness-treatment env.step arguments; no new inference in this M04 replay",
            "checkpoint_hash": "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8",
            "prefix_recovery": prefix_metrics,
            "reset_seconds": reset_s,
            "execution_seconds": exec_s,
            "policy_inference_seconds_in_this_run": 0.0,
            "sim_step_latency_ms": {
                "p50": float(np.percentile(inference_times, 50) * 1000),
                "p95": float(np.percentile(inference_times, 95) * 1000),
                "p99": float(np.percentile(inference_times, 99) * 1000),
            },
            "gpu_hours_policy_inference_in_this_run": 0.0,
            "gpu_cost_attributed": "N/A: replayed prior saved policy actions; upstream inference cost not present in M03 action episode receipt",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        npz = OUT / f"{result['episode_id']}.npz"
        vid = OUT / f"{result['episode_id']}_dual_camera.mp4"
        packed = {k: np.asarray([s[k] for s in states]) for k in states[0]}
        np.savez_compressed(
            npz,
            actions=np.asarray(actions_log),
            qpos=np.asarray(qpos),
            qvel=np.asarray(qvel),
            sim_time=np.asarray(simtime),
            dual_camera_frames=np.asarray(frames),
            **packed,
        )
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        if not writer.isOpened():
            raise RuntimeError("OpenCV MP4 writer failed")
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        result["trajectory"] = {"path": str(npz), "sha256": sha(npz), "bytes": npz.stat().st_size}
        result["video"] = {
            "path": str(vid),
            "sha256": sha(vid),
            "bytes": vid.stat().st_size,
            "fps": 20,
            "frames": len(frames),
            "width": w,
            "height": h,
            "cameras": ["agentview_image", "robot0_eye_in_hand_image"],
        }
        result_path = OUT / f"{result['episode_id']}.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        (OUT / "episodes.jsonl").open("a").write(json.dumps(result) + "\n")
        return result
    finally:
        env.close()


def main():
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    trace = json.loads(M02.read_text())
    source = trace["steps"][:100]
    assert len(source) == 100
    prefix = [s["env_step_argument"] for s in source]
    trace_hash = sha(M02)
    with np.load(POLICY_RUN) as z:
        actions = z["actions"].copy()
        expected = {
            "qpos": z["sim_qpos"][0].copy(),
            "qvel": z["sim_qvel"][0].copy(),
            "time": float(z["sim_time"][0]),
        }
    policy_hash = sha(POLICY_RUN)
    meta = json.loads(POLICY_META.read_text())
    assert (
        meta["success"] is True
        and meta["condition"] == "brightness"
        and meta["prefix_steps"] == 100
    )
    # Repeat source-prefix recovery and apply the saved, policy-produced correction twice independently.
    for i in (1, 2):
        result = run_one(i, prefix, actions, expected, trace_hash, policy_hash)
        print(json.dumps(result), flush=True)
    summary = {
        "protocol": "reset fixed init state + replay exact source M02 env.step prefix + replay saved M03 treatment action sequence; independent physical executions",
        "source_m02_trace_sha256": trace_hash,
        "m03_treatment_action_sha256": policy_hash,
        "m03_treatment_metadata_sha256": sha(POLICY_META),
        "checkpoint_sha256": "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8",
        "recovery_method": "fixed init state and exact action prefix replay; compare qpos/qvel/time to M03 reference snapshot",
        "recovery_tolerances": {"qpos_linf": 1e-5, "qvel_linf": 1e-4, "time_abs": 1e-8},
        "episodes": [
            {
                "episode_id": f"m04-first-correction-replay-{i:02d}",
                "success": json.loads(
                    (OUT / f"m04-first-correction-replay-{i:02d}.json").read_text()
                )["success"],
            }
            for i in (1, 2)
        ],
        "note": "Both are M04 independent physical replays of M03-saved correction actions, not two newly inferred or novel corrections.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
