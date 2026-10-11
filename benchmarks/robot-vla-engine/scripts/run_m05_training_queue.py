"""Run corrected M05 conditions sequentially under one GPU lock."""

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.pop("LD_LIBRARY_PATH", None)
sys.path.insert(0, "/data/ModelOps-Platform")
from pipelines.robotics.adapter import Jobs

from robot_vla.m05_training import atomic_json, sha256

root = Path("/data/VLA-Forge/Evidence/M05")
vla = Path("/data/VLA-Forge/project/robot-vla")
jobs = Jobs(root / "modelops/jobs", vla)
lock = (root / "gpu.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
base = json.loads((root / "training/configs/corrective.json").read_text())


def config(run_id, dataset, output, *, steps=500, stop_after=None, seed=5501):
    c = {
        **base,
        "run_id": run_id,
        "dataset": dataset,
        "output": str(output),
        "steps": steps,
        "seed": seed,
    }
    c.pop("resume", None)
    c.pop("stop_after", None)
    if stop_after is not None:
        c["stop_after"] = stop_after
    p = root / "training/configs" / (run_id + ".json")
    if p.exists() and json.loads(p.read_text()) != c:
        raise ValueError(f"Frozen config changed: {p}")
    atomic_json(p, c)
    return c, p


def native(c, p):
    result = Path(c["output"]) / "result.json"
    if result.exists():
        state = json.loads(result.read_text())["status"]
        if state == "COMPLETED":
            return
        raise ValueError(f"Existing native run requires explicit recovery: {state}")
    log = root / "logs" / (c["run_id"] + ".log")
    with log.open("ab") as f:
        subprocess.run(
            [
                str(vla / ".venv/bin/python"),
                str(vla / "benchmarks/robot-vla-engine/scripts/run_m05_training.py"),
                "--config",
                str(p),
            ],
            cwd=vla,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1200,
        )


def managed(c):
    d = jobs.directory(c["run_id"])
    if not d.exists():
        jobs.create(c)
    state = jobs.query(c["run_id"])
    if state["state"] == "CREATED":
        jobs.start(c["run_id"])
    elif state["state"] == "PAUSED":
        jobs.start(c["run_id"], resume=True)
    elif state["state"] == "COMPLETED":
        return
    elif state["state"] != "RUNNING":
        raise RuntimeError(state)
    start = time.time()
    while time.time() - start < 1200:
        state = jobs.query(c["run_id"])
        if state["state"] == "COMPLETED":
            return
        if state["state"] == "PAUSED":
            atomic_json(
                d / "pause_receipt.json",
                {"state": state, "resume_sha256": sha256(d / "artifacts/resume.pt")},
            )
            jobs.start(c["run_id"], resume=True)
        elif state["state"] == "FAILED":
            raise RuntimeError(state)
        time.sleep(2)
    jobs.stop(c["run_id"])
    raise TimeoutError("Managed run budget expired")


c, p = config(
    "m05-corrective-cam180-s5501", "corrective", root / "training/corrective_cam180", stop_after=250
)
managed(c)
c, p = config("m05-replay-cam180-s5501", "replay", root / "training/replay_cam180")
native(c, p)
c, p = config(
    "m05-ab-native-cam180-s5701",
    "corrective",
    root / "training/ab-native-cam180",
    steps=20,
    seed=5701,
)
native(c, p)
c, p = config(
    "m05-ab-managed-cam180-s5701",
    "corrective",
    root / "training/ab-managed-cam180",
    steps=20,
    seed=5701,
    stop_after=10,
)
managed(c)
atomic_json(
    root / "training/corrected_queue_complete.json",
    {"status": "COMPLETED", "finished": time.time()},
)
