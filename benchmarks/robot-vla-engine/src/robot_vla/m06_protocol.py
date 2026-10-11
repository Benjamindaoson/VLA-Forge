"""Fail-closed identity and preregistration contract for independent M06 rollouts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

SUITES = {"spatial": "libero_spatial", "object": "libero_object", "goal": "libero_goal", "long": "libero_10"}
CONDITIONS = ("frozen", "standard", "corrective", "replay")
EXCLUDED_GROUPS = frozenset({"goal:6:0", "goal:6:1", "goal:6:4", "goal:6:6", "object:6:1", "spatial:5:7"})
EVIDENCE = Path("/data/VLA-Forge/Evidence/M06")
FREEZE = Path("/data/VLA-Forge/Evidence/M04/finalization_20261011/independent_test_freeze.json")
ADMISSION = Path("/data/VLA-Forge/Evidence/M04/finalization_20261011/m05_admission.json")
M05 = Path("/data/VLA-Forge/Evidence/M05")
ROOT = Path("/data/VLA-Forge/project/robot-vla")
ENGINE = ROOT / "benchmarks/robot-vla-engine"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def validate_freeze_structure(freeze: dict) -> dict:
    if freeze.get("status") != "FROZEN_NOT_EXECUTED":
        raise ValueError("Replacement M06 test freeze is not unopened")
    if freeze.get("development_group_overlap") or freeze.get("old_M02_state_bytes_overlap"):
        raise ValueError("Frozen test overlaps recorded project development")
    rows = freeze.get("rows")
    if not isinstance(rows, list) or len(rows) != 400:
        raise ValueError("M06 requires exactly 400 frozen physical cases")
    seen, seeds, counts = set(), set(), Counter()
    for row in rows:
        suite, task, init = row["suite"], int(row["task_id"]), int(row["init_state_id"])
        key = (suite, task, init)
        group = f"{suite}:{task}:{init}"
        if (suite not in SUITES or not 0 <= task < 10 or not 40 <= init <= 49
                or row.get("source_group") != group or group in EXCLUDED_GROUPS
                or key in seen or type(row.get("seed")) is not int or row["seed"] in seeds):
            raise ValueError(f"Invalid/duplicate M06 case: {key}")
        seen.add(key)
        seeds.add(row["seed"])
        counts[suite] += 1
    if dict(counts) != {name: 100 for name in SUITES}:
        raise ValueError("M06 suite coverage is incomplete")
    return {"rows": len(rows), "tasks": len({(a, b) for a, b, _ in seen}),
            "per_suite": dict(counts), "init_ids": list(range(40, 50)),
            "excluded_source_groups": sorted(EXCLUDED_GROUPS)}


def validate_source_bytes(rows: list[dict]) -> dict:
    """Re-read official LIBERO init bytes and BDDL before opening a test case."""
    import numpy as np
    from libero.libero import benchmark, get_libero_path

    suite_cls = benchmark.get_benchmark_dict()
    checked = Counter()
    for short, full in SUITES.items():
        suite = suite_cls[full](task_order_index=0)
        for task_id in range(10):
            task = suite.get_task(task_id)
            states = suite.get_task_init_states(task_id)
            group = [r for r in rows if r["suite"] == short and r["task_id"] == task_id]
            if len(group) != 10 or len(states) < 50:
                raise ValueError(f"Missing LIBERO init states: {short}/{task_id}")
            init_file = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            init_sha, bddl_sha = sha256(init_file), sha256(bddl_file)
            for row in group:
                current_state = hashlib.sha256(np.asarray(states[row["init_state_id"]]).tobytes()).hexdigest()
                if (task.name != row["task_name"] or str(init_file) != row["init_states_file"]
                        or init_sha != row["init_states_file_sha256"]
                        or bddl_sha != row["bddl_sha256"]
                        or current_state != row["state_sha256"]):
                    raise ValueError(f"LIBERO source changed: {row['source_group']}")
                checked[short] += 1
    return {"verified_states": sum(checked.values()), "verified_by_suite": dict(checked),
            "init_files": 40, "bddl_files": 40}


def load_adapter_identities() -> dict:
    results = {
        "standard": M05 / "training/standard/result.json",
        "corrective": M05 / "modelops/jobs/m05-corrective-cam180-s5501/artifacts/result.json",
        "replay": M05 / "training/replay_cam180/result.json",
    }
    base = Path("/data/VLA-Forge/Assets/models/smolvla_libero")
    normalizer = base / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    identities = {"frozen": {"model_path": str(base), "model_sha256": sha256(base / "model.safetensors")}}
    for name, result_path in results.items():
        result = json.loads(result_path.read_text())
        adapter = Path(result["adapter"])
        digest = sha256(adapter / "adapter_model.safetensors")
        if (result["status"] != "COMPLETED" or result["optimizer_updates"] != 500
                or result["reload_linf"] != 0 or not result["finite_gradients"]
                or digest != result["adapter_sha256"]):
            raise ValueError(f"Training identity failed: {name}")
        identities[name] = {"adapter_path": str(adapter), "adapter_sha256": digest,
                            "result_path": str(result_path), "result_sha256": sha256(result_path)}
    identities["processor"] = {
        name: sha256(base / name) for name in
        ("policy_preprocessor.json", "policy_postprocessor.json")
    }
    identities["normalizer_sha256"] = sha256(normalizer)
    return identities


def frozen_rules() -> dict:
    return {
        "schema": "vla-forge.m06.independent.v1",
        "models": list(CONDITIONS), "backend": "fp32_eager", "case_count": 400,
        "target_task": {"suite": "goal", "task_id": 6},
        "legacy_tasks": 39, "action_dim": 7, "state_dim": 8,
        "action_chunk_size": 50, "control_mode": "relative",
        "camera_mapping": {"image": "camera1", "image2": "camera2"},
        "success_oracle": "LIBERO check_success via info.is_success; all failures count",
        "state_match_tolerance": {"qpos_max_abs": 1e-6, "qvel_max_abs": 1e-6,
                                  "sim_vector_max_abs": 1e-6, "time_abs_s": 1e-9},
        "statistics": {"rate_ci": "Wilson two-sided 95%", "paired_target_test": "one-sided exact discordant-pair sign test",
                       "multiple_comparisons": "Holm step-down across three target comparisons, family alpha 0.05",
                       "task_cluster_ci": "10,000 seeded task-cluster bootstrap replicates; two-sided 95% and one-sided 95% lower",
                       "bootstrap_seed": 606, "legacy_noninferiority_margin": -0.03},
        "release": {"target_positive_difference_required": True, "target_holm_p_max": 0.05,
                    "legacy_cluster_lower_gt": -0.03, "illegal_actions_max": 0,
                    "missing_or_corrupt_episodes_max": 0,
                    "latency_p95_ratio_max": 1.2,
                    "insufficient_evidence_if_noninferiority_unproven": True,
                    "deployment_scope": "authorized simulation only"},
        "claim_boundary": "Independent of recorded project M02-M05 groups; HF expert IDs lack verified physical init mapping; foundation pretraining exposure unknown",
    }
