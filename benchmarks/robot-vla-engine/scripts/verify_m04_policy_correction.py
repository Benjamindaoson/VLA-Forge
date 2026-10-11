"""Independently replay a fresh SmolVLA action trace from its M02 failure checkpoint."""

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
SOURCE_EP = int(os.environ.get("M04_SOURCE_EPISODE", "7"))
RUN_ID = os.environ["M04_POLICY_RUN_ID"]
POLICY_NPZ = ROOT / "Evidence/M04/artifacts/policy_attempts" / f"{RUN_ID}.npz"
POLICY_META = ROOT / "Evidence/M04/artifacts/policy_attempts/attempts.jsonl"
M03_BASELINE = (
    ROOT
    / "Evidence/M03/source/M03_interventional_diagnosis/episodes/m03-g09-goal-t06-init07_baseline.json"
)
RUN_BASE = "20261010T030315-982cacc8515f"
EP_DIR = (
    ROOT
    / "Evidence/M02/source/M02_libero40/registry/runs"
    / RUN_BASE
    / "episodes"
    / f"goal_task06_episode{SOURCE_EP:02d}_attempt1"
)
TRACE_PATH = EP_DIR / "action_trace.json"
EP_JSON = EP_DIR / "episode.json"
OUT = ROOT / "Evidence/M04/artifacts/verified_corrections" / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)
if any(OUT.iterdir()):
    raise FileExistsError(f"refusing to overwrite nonempty replay evidence: {OUT}")
CHECKPOINT_SHA = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    import libero.libero as libero_api
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark

    trace = json.loads(TRACE_PATH.read_text())
    ep_meta = json.loads(EP_JSON.read_text())
    source_prefix = np.asarray(
        [x["env_step_argument"] for x in trace["steps"][:100]], dtype=np.float32
    )
    with np.load(POLICY_NPZ) as z:
        all_actions = z["actions"].astype(np.float32)
        expected_qpos = z["qpos"][100].copy()
        expected_qvel = z["qvel"][100].copy()
        recorded_qpos = z["qpos"][100].copy()
        recorded_qvel = z["qvel"][100].copy()
    baseline_reference = json.loads(M03_BASELINE.read_text())["checkpoint_state"]
    expected_qpos = np.asarray(baseline_reference["qpos"], dtype=np.float64)
    expected_qvel = np.asarray(baseline_reference["qvel"], dtype=np.float64)
    expected_time = float(baseline_reference["time"])
    recorded_state_error = {
        "qpos_linf": float(np.max(np.abs(recorded_qpos - expected_qpos))),
        "qvel_linf": float(np.max(np.abs(recorded_qvel - expected_qvel))),
    }
    if recorded_state_error["qpos_linf"] > 1e-5 or recorded_state_error["qvel_linf"] > 1e-4:
        raise RuntimeError(
            f"policy attempt stored a checkpoint different from M03 independent baseline: {recorded_state_error}"
        )
    if len(source_prefix) != 100 or all_actions.shape[0] <= 100:
        raise RuntimeError("prefix/correction segment is incomplete")
    prefix_error = float(np.max(np.abs(all_actions[:100] - source_prefix)))
    if prefix_error != 0.0:
        raise RuntimeError(f"policy attempt prefix differs from source M02 actions: {prefix_error}")
    correction = all_actions[100:].copy()
    policy_row = next(
        (
            json.loads(line)
            for line in POLICY_META.read_text().splitlines()
            if json.loads(line).get("episode_id") == RUN_ID
        ),
        None,
    )
    if policy_row is None or policy_row.get("success") is not True:
        raise RuntimeError("source policy inference attempt is not Success=True")
    if policy_row["policy_model_sha256"] != CHECKPOINT_SHA:
        raise RuntimeError("policy checkpoint differs from frozen M04 identity")
    source_trace_sha = sha(TRACE_PATH)
    policy_npz_sha = sha(POLICY_NPZ)
    libero_api._assets_path_cache = str(ROOT / "Assets/LIBERO-assets")
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    summaries = []
    for replicate in (1, 2):
        env = LiberoEnv(
            suite,
            6,
            "libero_goal",
            episode_length=300,
            obs_type="pixels_agent_pos",
            observation_width=256,
            observation_height=256,
            episode_index=SOURCE_EP,
            init_states=True,
            hard_reset=True,
            control_freq=20,
            num_steps_wait=10,
            control_mode="relative",
        )
        try:
            obs, _ = env.reset(seed=int(ep_meta["seed"]))
            frames, actions, qpos, qvel, times, states, success_flags = [], [], [], [], [], [], []

            def capture(o):
                data = env._env.sim.data
                frames.append(
                    np.concatenate([o["pixels"]["image"], o["pixels"]["image2"]], axis=1).astype(
                        np.uint8
                    )
                )
                qpos.append(np.asarray(data.qpos).copy())
                qvel.append(np.asarray(data.qvel).copy())
                times.append(float(data.time))
                state = o["robot_state"]
                states.append(
                    {
                        "eef_pos": np.asarray(state["eef"]["pos"]).copy(),
                        "eef_quat": np.asarray(state["eef"]["quat"]).copy(),
                        "joint_pos": np.asarray(state["joints"]["pos"]).copy(),
                        "joint_vel": np.asarray(state["joints"]["vel"]).copy(),
                        "gripper_qpos": np.asarray(state["gripper"]["qpos"]).copy(),
                    }
                )

            capture(obs)
            for action in source_prefix:
                obs, _, term, trunc, _ = env.step(action)
                actions.append(action.copy())
                capture(obs)
                if term or trunc:
                    raise RuntimeError("source M02 prefix terminated early")
            data = env._env.sim.data
            recovery = {
                "qpos_linf": float(np.max(np.abs(data.qpos - expected_qpos))),
                "qvel_linf": float(np.max(np.abs(data.qvel - expected_qvel))),
                "time_abs": abs(float(data.time) - expected_time),
                "ncon": int(data.ncon),
            }
            if (
                recovery["qpos_linf"] > 1e-5
                or recovery["qvel_linf"] > 1e-4
                or recovery["time_abs"] > 1e-8
            ):
                raise RuntimeError(f"independent checkpoint recovery mismatch: {recovery}")
            started = time.perf_counter()
            for action in correction:
                obs, _, term, trunc, info = env.step(action)
                actions.append(action.copy())
                capture(obs)
                success_flags.append(bool(info.get("is_success", False)))
                if success_flags[-1] or term or trunc:
                    break
            run_seconds = time.perf_counter() - started
            success = bool(success_flags[-1]) if success_flags else False
            ep_id = f"{RUN_ID}-independent-replay-{replicate:02d}"
            npz = OUT / f"{ep_id}.npz"
            video = OUT / f"{ep_id}_dual_camera.mp4"
            state_arrays = {key: np.asarray([x[key] for x in states]) for key in states[0]}
            np.savez_compressed(
                npz,
                actions=np.asarray(actions, dtype=np.float32),
                qpos=np.asarray(qpos),
                qvel=np.asarray(qvel),
                sim_time=np.asarray(times),
                dual_camera_frames=np.asarray(frames),
                **state_arrays,
            )
            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("cannot create replay video")
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()
            receipt = {
                "schema": "vla-forge.m04.verified-policy-replay.v1",
                "episode_id": ep_id,
                "policy_attempt_id": RUN_ID,
                "source_m02_episode": ep_meta["episode_id"],
                "source_m02_trace_sha256": source_trace_sha,
                "m03_baseline_checkpoint_reference": str(M03_BASELINE),
                "policy_attempt_stored_checkpoint_error": recorded_state_error,
                "policy_attempt_npz_sha256": policy_npz_sha,
                "policy_action_segment_sha256": hashlib.sha256(correction.tobytes()).hexdigest(),
                "policy_checkpoint_sha256": CHECKPOINT_SHA,
                "policy_seed": policy_row["policy_seed"],
                "condition": policy_row["condition"],
                "brightness_gain": policy_row["brightness_gain"],
                "prefix_steps": 100,
                "correction_actions_executed": len(actions) - 100,
                "prefix_exact_action_match_linf": prefix_error,
                "recovery": recovery,
                "success": success,
                "termination_reason": "success" if success else "budget_or_timeout",
                "policy_inference_gpu_hours": policy_row["gpu_hours_inference_estimate"],
                "replay_policy_inference_gpu_hours": 0.0,
                "replay_wall_seconds": run_seconds,
                "trajectory": {"path": str(npz), "bytes": npz.stat().st_size, "sha256": sha(npz)},
                "video": {
                    "path": str(video),
                    "bytes": video.stat().st_size,
                    "sha256": sha(video),
                    "width": width,
                    "height": height,
                    "fps": 20,
                    "frames": len(frames),
                    "cameras": ["agentview_image", "robot0_eye_in_hand_image"],
                },
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            result_path = OUT / f"{ep_id}.json"
            result_path.write_text(json.dumps(receipt, indent=2) + "\n")
            with (OUT / "replays.jsonl").open("a") as f:
                f.write(json.dumps(receipt) + "\n")
            summaries.append(
                {
                    "episode_id": ep_id,
                    "success": success,
                    "recovery": recovery,
                    "trajectory_sha256": receipt["trajectory"]["sha256"],
                }
            )
            print(json.dumps(summaries[-1]), flush=True)
        finally:
            env.close()
    summary = {
        "schema": "vla-forge.m04.verified-policy-replay-summary.v1",
        "source_policy_run": RUN_ID,
        "source_action_sha256": hashlib.sha256(correction.tobytes()).hexdigest(),
        "replicates": summaries,
        "independent_replay_pass": len(summaries) == 2 and all(x["success"] for x in summaries),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not summary["independent_replay_pass"]:
        raise SystemExit("independent replay verification failed; retain all run artifacts")


if __name__ == "__main__":
    main()
