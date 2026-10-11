"""Execute only the frozen, bounded M04 cross-task follow-up; never trains."""

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/VLA-Forge")
REPO = ROOT / "project/robot-vla"
OUT = ROOT / "Evidence/M04/finalization_20261011/cross_task"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ".controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = json.loads((OUT.parent / "protocol.json").read_text())["cross_task"]
        ledger = OUT / "controller.json"
        result = (
            json.loads(ledger.read_text())
            if ledger.exists()
            else {
                "attempts": [],
                "status": "RUNNING",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        if result["status"] in {"COMPLETED", "BUDGET_EXHAUSTED"}:
            print("Already terminal; no experiments restarted.")
            return
        spent = sum(x.get("wall_seconds", 0) for x in result["attempts"])
        deadline = time.monotonic() + max(0, protocol["maximum_total_seconds"] - spent)

        def save():
            ledger.write_text(json.dumps(result, indent=2) + "\n")

        def execute(command, env, logfile, limit):
            start = time.perf_counter()
            with logfile.open("w") as out:
                process = subprocess.Popen(
                    command,
                    env=env,
                    cwd=REPO,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    rc = process.wait(timeout=min(limit, max(1, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    rc = 124
            return rc, time.perf_counter() - start

        for condition in protocol["conditions"][: protocol["maximum_attempts"]]:
            run_id = f"m04-final-goal04-init04-seed{condition['seed']}"
            if any(x["run_id"] == run_id for x in result["attempts"]):
                continue
            if time.monotonic() >= deadline:
                break
            item = {
                "run_id": run_id,
                "condition": condition,
                "status": "STARTED",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            result["attempts"].append(item)
            save()
            env = {
                **os.environ,
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "M04_SUITE": "goal",
                "M04_TASK_ID": "4",
                "M04_SOURCE_EPISODE": "4",
                "M04_POLICY_SEED": str(condition["seed"]),
                "M04_RUN_ID": run_id,
                "M04_POLICY_CONDITIONS": condition["kind"],
                "M04_BRIGHTNESS_CAMERA": condition.get("camera", "wrist"),
                "M04_BRIGHTNESS_GAIN": str(condition.get("gain", 1)),
                "M04_ACTION_SCALE": str(condition.get("scale", 1)),
                "M04_INTERVENTION_STEPS": "50",
                "M04_INTERVENTION_START": "0",
                "M04_INFERENCE_BACKEND": "cuda_graph",
                "M04_ATTEMPT_DIR": str(OUT / "policy_attempts"),
                "M04_VERIFIED_DIR": str(OUT / "verified"),
            }
            env.pop("LD_LIBRARY_PATH", None)
            rc, seconds = execute(
                [sys.executable, "benchmarks/robot-vla-engine/scripts/run_m04_policy_attempt.py"],
                env,
                OUT / f"{run_id}.log",
                protocol["maximum_attempt_seconds"],
            )
            item.update(exit_code=rc, wall_seconds=seconds)
            path = OUT / "policy_attempts" / f"{run_id}_result.json"
            if rc or not path.exists():
                item["status"] = "TIMEOUT" if rc == 124 else "RUNNER_ERROR"
                save()
                continue
            row = json.loads(path.read_text())
            item.update(
                success=row["success"],
                phase_seconds=row.get("phase_seconds"),
                status="POLICY_FAILURE_RETAINED",
            )
            if row["success"] and time.monotonic() < deadline:
                rc, seconds = execute(
                    [
                        sys.executable,
                        "benchmarks/robot-vla-engine/scripts/verify_m04_policy_correction.py",
                    ],
                    {**env, "M04_POLICY_RUN_ID": run_id},
                    OUT / f"{run_id}-verify.log",
                    protocol["maximum_replay_seconds"],
                )
                item["wall_seconds"] += seconds
                item["independent_replay_wall_seconds"] = seconds
                item["replay_exit_code"] = rc
                summary = OUT / "verified" / run_id / "summary.json"
                if (
                    rc == 0
                    and summary.exists()
                    and json.loads(summary.read_text()).get("independent_replay_pass")
                ):
                    item["status"] = "VERIFIED_SUCCESS"
                    save()
                    break
                item["status"] = "REPLAY_FAILED_OR_TIMED_OUT"
            save()
        result["status"] = (
            "COMPLETED"
            if any(x["status"] == "VERIFIED_SUCCESS" for x in result["attempts"])
            else "BUDGET_EXHAUSTED"
        )
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        result["training_executed"] = False
        result["new_verified"] = sum(x["status"] == "VERIFIED_SUCCESS" for x in result["attempts"])
        save()
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
