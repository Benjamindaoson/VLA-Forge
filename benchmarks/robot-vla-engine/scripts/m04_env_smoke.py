"""One real headless LIBERO reset/step smoke, isolated from M04 correction data."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from lerobot.envs.libero import LiberoEnv
from libero.libero import benchmark

OUT = Path(
    os.environ.get("M04_SMOKE_OUT", "/data/VLA-Forge/Evidence/M04/artifacts/env_smoke_goal6_init7")
)
OUT.mkdir(parents=True, exist_ok=True)


def main():
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(6)
    env = LiberoEnv(
        suite,
        6,
        "libero_goal",
        episode_length=20,
        obs_type="pixels_agent_pos",
        observation_width=128,
        observation_height=128,
        episode_index=7,
        init_states=True,
        hard_reset=True,
        control_freq=20,
        num_steps_wait=10,
        control_mode="relative",
    )
    started = time.perf_counter()
    try:
        obs, info = env.reset(seed=706)
        reset_seconds = time.perf_counter() - started
        actions, rewards, flags = [], [], []
        frames = {k: [np.asarray(v).copy()] for k, v in obs["pixels"].items()}
        states = [
            {
                "joints": np.asarray(obs["robot_state"]["joints"]["pos"]).copy(),
                "eef": np.asarray(obs["robot_state"]["eef"]["pos"]).copy(),
                "quat": np.asarray(obs["robot_state"]["eef"]["quat"]).copy(),
            }
        ]
        for _ in range(3):
            action = np.zeros(7, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            rewards.append(float(reward))
            flags.append(
                {
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success_signal": bool(info.get("is_success", False)),
                }
            )
            for k, v in obs["pixels"].items():
                frames[k].append(np.asarray(v).copy())
            states.append(
                {
                    "joints": np.asarray(obs["robot_state"]["joints"]["pos"]).copy(),
                    "eef": np.asarray(obs["robot_state"]["eef"]["pos"]).copy(),
                    "quat": np.asarray(obs["robot_state"]["eef"]["quat"]).copy(),
                }
            )
        np.savez_compressed(
            OUT / "rollout.npz",
            actions=np.asarray(actions),
            rewards=np.asarray(rewards),
            camera1=np.asarray(frames["image"]),
            camera2=np.asarray(frames["image2"]),
            joints=np.asarray([s["joints"] for s in states]),
            eef=np.asarray([s["eef"] for s in states]),
            quat=np.asarray([s["quat"] for s in states]),
        )
        receipt = {
            "kind": "environment_smoke_not_correction",
            "suite": "libero_goal",
            "task_id": 6,
            "task_name": task.name,
            "language": task.language,
            "init_state_id": 7,
            "seed": 706,
            "control_mode": "relative",
            "action_dim": 7,
            "action_steps": 3,
            "action": "zero-vector diagnostic only",
            "success": False,
            "success_claimed": False,
            "reset_seconds": reset_seconds,
            "steps": flags,
            "reward_sum": sum(rewards),
            "frames": {
                "camera1": list(frames["image"][0].shape),
                "camera2": list(frames["image2"][0].shape),
            },
            "artifact": str(OUT / "rollout.npz"),
        }
        (OUT / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        print(json.dumps(receipt))
    finally:
        env.close()


if __name__ == "__main__":
    main()
