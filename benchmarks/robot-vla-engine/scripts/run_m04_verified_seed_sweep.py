"""Bounded M04 policy-seed sweep for one frozen recovered Goal-6 checkpoint.

Each result remains an ordinary registered M04 rollout. Successful episodes are
counted only after two independent physical replays and unique correction-action
hash verification. This sweep does not imply task or initial-state diversity.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data/VLA-Forge")
REPO = ROOT / "project/robot-vla"
EVID = ROOT / "Evidence/M04"
ATTEMPTS = EVID / "artifacts/policy_attempts/attempts.jsonl"
VERIFIED = EVID / "artifacts/verified_corrections"
START_SEED = 84087
MAX_SEED = 84110
TARGET_FRESH = 10


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verified_policy_hashes() -> set[str]:
    result = set()
    for receipt in VERIFIED.glob("*/summary.json"):
        data = json.loads(receipt.read_text())
        if (
            data.get("source_policy_run")
            and data.get("independent_replay_pass") is True
            and len(data.get("replicates", [])) == 2
            and all(item.get("success") is True for item in data["replicates"])
        ):
            result.add(data["source_action_sha256"])
    return result


def run_logged(command: list[str], env: dict[str, str], log: Path) -> int:
    with log.open("w", encoding="utf-8") as stream:
        return subprocess.run(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                              stdout=stream, stderr=subprocess.STDOUT, check=False).returncode


def main() -> None:
    env_base = os.environ.copy()
    env_base.pop("LD_LIBRARY_PATH", None)
    env_base.update({
        "M04_SUITE": "goal",
        "M04_TASK_ID": "6",
        "M04_SOURCE_EPISODE": "7",
        "M04_POLICY_CONDITIONS": "brightness_first50",
        "M04_BRIGHTNESS_GAIN": "1.10",
        "M04_INTERVENTION_STEPS": "50",
        "M04_INTERVENTION_START": "0",
        "M04_BRIGHTNESS_CAMERA": "wrist",
        "M04_ACTION_SCALE": "1.0",
    })
    hashes = verified_policy_hashes()
    existing_fresh = len(hashes)
    result = {
        "schema": "vla-forge.m04.bounded-policy-seed-sweep.v1",
        "source_m02_episode": "goal_task06_episode07_attempt1",
        "condition": "wrist brightness gain 1.10 over first 50 policy observations",
        "target_verified_unique_fresh_policy_sequences": TARGET_FRESH,
        "starting_verified_unique_fresh_policy_sequences": existing_fresh,
        "attempts": [],
    }
    for seed in range(START_SEED, MAX_SEED + 1):
        if len(hashes) >= TARGET_FRESH:
            break
        run_id = f"m04-goal-t06-e07-wrist110-first50-seed{seed}"
        if any(row.get("episode_id") == run_id for row in read_rows(ATTEMPTS)):
            result["attempts"].append({"run_id": run_id, "status": "SKIP_EXISTING_ID"})
            continue
        attempt_env = {**env_base, "M04_RUN_ID": run_id, "M04_POLICY_SEED": str(seed)}
        attempt_log = EVID / "logs" / f"seed-sweep-{seed}-attempt.log"
        rc = run_logged(
            [sys.executable, "benchmarks/robot-vla-engine/scripts/run_m04_policy_attempt.py"],
            attempt_env,
            attempt_log,
        )
        rows = [row for row in read_rows(ATTEMPTS) if row.get("episode_id") == run_id]
        if rc != 0 or not rows:
            result["attempts"].append({"run_id": run_id, "status": "RUNNER_ERROR", "exit_code": rc, "log": str(attempt_log)})
            continue
        row = rows[-1]
        entry = {"run_id": run_id, "success": bool(row.get("success")), "trajectory_sha256": row.get("trajectory", {}).get("sha256"), "gpu_hours": row.get("gpu_hours_allocated"), "log": str(attempt_log)}
        if row.get("success") is not True:
            entry["status"] = "POLICY_FAILURE_RETAINED"
            result["attempts"].append(entry)
            continue
        verify_env = {**env_base, "M04_POLICY_RUN_ID": run_id}
        verify_log = EVID / "logs" / f"seed-sweep-{seed}-independent-replay.log"
        verify_rc = run_logged(
            [sys.executable, "benchmarks/robot-vla-engine/scripts/verify_m04_policy_correction.py"],
            verify_env,
            verify_log,
        )
        summary_path = VERIFIED / run_id / "summary.json"
        if verify_rc != 0 or not summary_path.is_file():
            entry.update({"status": "INDEPENDENT_REPLAY_FAILED", "verify_exit_code": verify_rc, "verify_log": str(verify_log)})
            result["attempts"].append(entry)
            continue
        summary = json.loads(summary_path.read_text())
        action_hash = summary.get("source_action_sha256")
        if summary.get("independent_replay_pass") is not True or len(summary.get("replicates", [])) != 2 or not all(x.get("success") is True for x in summary["replicates"]):
            entry["status"] = "REPLAY_RECEIPT_FAILED"
        elif action_hash in hashes:
            entry.update({"status": "DUPLICATE_ACTION_SEQUENCE", "action_sha256": action_hash})
        else:
            hashes.add(action_hash)
            entry.update({"status": "NEW_VERIFIED_UNIQUE_SEQUENCE", "action_sha256": action_hash, "verified_unique_fresh_total": len(hashes)})
        entry["verify_log"] = str(verify_log)
        result["attempts"].append(entry)
    result["final_verified_unique_fresh_policy_sequences"] = len(hashes)
    result["target_reached"] = len(hashes) >= TARGET_FRESH
    result["source_diversity"] = "all added runs retain the same Goal task 6 / init state 7 / M02 source episode; seed sweep is not generalization evidence"
    out = EVID / "logs" / "seed-sweep-summary.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": str(out), "attempts": len(result["attempts"]), "fresh_verified_unique": len(hashes), "target_reached": result["target_reached"]}, indent=2))
    if not result["target_reached"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
