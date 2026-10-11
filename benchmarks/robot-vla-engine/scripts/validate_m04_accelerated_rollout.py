"""Compare registered eager/graph continuations, including full physical trajectories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/VLA-Forge/Evidence/M04")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="m04-perf-cudagraph-init06-84182")
    parser.add_argument("--baseline", default="m04-diverse-goal-t06-init06-seed84182")
    args = parser.parse_args()

    def find(path, identifier):
        matches = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        matches = [x for x in matches if x["episode_id"] == identifier]
        if len(matches) != 1:
            raise ValueError(f"Expected one immutable result for {identifier}")
        item = matches[0]
        p = Path(item["trajectory"]["path"])
        if hashlib.sha256(p.read_bytes()).hexdigest() != item["trajectory"]["sha256"]:
            raise ValueError("Trajectory hash mismatch")
        return item

    baseline = find(ROOT / "artifacts/policy_attempts/attempts.jsonl", args.baseline)
    candidate = find(ROOT / "performance/rollouts/attempts.jsonl", args.candidate)
    keys = [
        "source_m02_episode",
        "source_trace_sha256",
        "suite",
        "task_id",
        "init_state_id",
        "seed",
        "policy_seed",
        "policy_model_sha256",
        "vlm_model_sha256",
        "condition",
        "control_mode",
        "action_chunk_size",
        "n_action_steps",
        "inference_action_steps",
    ]
    identity = {key: baseline[key] == candidate[key] for key in keys}
    errors = {}
    with (
        np.load(baseline["trajectory"]["path"]) as a,
        np.load(candidate["trajectory"]["path"]) as b,
    ):
        for key in ["actions", "qpos", "qvel", "eef_pos", "eef_quat", "dual_camera_frames"]:
            if a[key].shape != b[key].shape:
                raise ValueError(f"Shape differs for {key}")
            errors[key] = float(np.max(np.abs(a[key].astype(float) - b[key].astype(float))))
    result = dict(
        baseline=baseline["trajectory"],
        candidate=candidate["trajectory"],
        baseline_success=baseline["success"],
        candidate_success=candidate["success"],
        correction_steps=candidate["inference_action_steps"],
        identity_checks=identity,
        maximum_absolute_errors=errors,
        baseline_policy_loop_seconds=baseline["gpu_hours_allocated"] * 3600,
        candidate_policy_loop_seconds_including_capture=candidate["gpu_hours_allocated"] * 3600,
        candidate_cold_capture_seconds=candidate["cuda_graph_cold_seconds"],
        scope="Registered historical eager baseline and current candidate; exact physics check, not a contemporary paired end-to-end speed experiment.",
    )
    result["pass"] = (
        all(identity.values())
        and all(x == 0 for x in errors.values())
        and baseline["success"]
        and candidate["success"]
    )
    (ROOT / "performance/cuda_graph_physical_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
