"""Create M03 audit tables, reports, and an offline evidence gallery from receipts."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np

from robot_vla.schema import file_hash


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    half = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / d
    return [max(0.0, center - half), min(1.0, center + half)]

def sham_exact_groups(run_dir: Path, groups: list[dict]) -> list[str]:
    """Return groups whose baseline and sham action/physical traces match exactly."""
    exact = []
    for group in groups:
        group_id = group["group_id"]
        paths = [run_dir / "episodes" / f"{group_id}_{condition}.npz" for condition in ("baseline", "sham")]
        if not all(path.is_file() for path in paths):
            continue
        with np.load(paths[0], allow_pickle=False) as baseline, np.load(paths[1], allow_pickle=False) as sham:
            keys = ("actions", "states", "sim_qpos", "sim_qvel", "sim_time")
            if all(key in baseline.files and key in sham.files and np.array_equal(baseline[key], sham[key]) for key in keys):
                exact.append(group_id)
    return exact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--m02-report", type=Path, required=True)
    args = parser.parse_args()
    run_dir, report = args.run_dir.resolve(), args.report_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    m02_manifest = json.loads((args.m02_report / "manifest.json").read_text(encoding="utf-8"))
    manifest["frozen_m02_policy"] = {
        "model": m02_manifest.get("model"),
        "processor": m02_manifest.get("processor"),
        "protocol": m02_manifest.get("protocol"),
        "runtime": m02_manifest.get("runtime"),
        "baseline_manifest_sha256": file_hash(args.m02_report.parent / "M01_real_libero" / "manifest.json"),
    }
    rows = load_rows(run_dir / "paired_interventions.csv")
    for row in rows:
        baseline_path = run_dir / "episodes" / f"{row['group_id']}_baseline.npz"
        condition_path = run_dir / "episodes" / f"{row['group_id']}_{row['condition']}.npz"
        with np.load(baseline_path, allow_pickle=False) as baseline, np.load(
            condition_path, allow_pickle=False
        ) as condition:
            n = min(len(baseline["sim_qpos"]), len(condition["sim_qpos"]))
            row["sim_qpos_trajectory_mean_l2_vs_baseline"] = float(
                np.linalg.norm(baseline["sim_qpos"][:n] - condition["sim_qpos"][:n], axis=1).mean()
            )
            row["sim_qvel_trajectory_mean_l2_vs_baseline"] = float(
                np.linalg.norm(baseline["sim_qvel"][:n] - condition["sim_qvel"][:n], axis=1).mean()
            )
            contact_delta = np.abs(baseline["sim_ncon"][:n] - condition["sim_ncon"][:n])
            row["sim_contact_count_mean_abs_delta_vs_baseline"] = float(contact_delta.mean())
            row["sim_contact_count_max_abs_delta_vs_baseline"] = int(contact_delta.max(initial=0))
    groups = manifest.get("groups", [])
    exact_sham_groups = sham_exact_groups(run_dir, groups)
    object_pose_path = report / "object_poses.jsonl"
    object_pose_rows = []
    if object_pose_path.is_file():
        object_pose_rows = [json.loads(line) for line in object_pose_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = report / "visualization"
    for name in ("videos", "screenshots"):
        (output / name).mkdir(parents=True, exist_ok=True)

    integrity = []
    for row in rows:
        video, trajectory = Path(row["video_path"]), Path(row["trajectory_path"])
        checks = {
            "video_exists": video.is_file(),
            "video_hash_matches": video.is_file() and file_hash(video) == row["video_sha256"],
            "trajectory_exists": trajectory.is_file(),
            "trajectory_hash_matches": trajectory.is_file()
            and file_hash(trajectory) == row["trajectory_sha256"],
        }
        integrity.append({"group_id": row["group_id"], "condition": row["condition"], **checks})
        if all(checks.values()):
            shutil.copy2(video, output / "videos" / video.name)

    replay_rows = []
    for group in groups:
        matches = group.get("physical_state_pairwise_match", [])
        replay_rows.append(
            {
                "group_id": group["group_id"],
                "source_suite": group["source_suite"],
                "source_task_id": group["source_task_id"],
                "source_episode_id": group["source_episode_id"],
                "init_state_id": group["init_state_id"],
                "prefix_steps": group["prefix_steps"],
                "observation_state_linf_vs_m02": group["recovery_observation_state_linf_vs_m02"],
                "qpos_linf_max": max((m["qpos_linf"] for m in matches), default=None),
                "qvel_linf_max": max((m["qvel_linf"] for m in matches), default=None),
                "sim_time_abs_max": max((m["time_abs"] for m in matches), default=None),
                "all_exact": bool(matches) and all(m["exact"] for m in matches),
                "all_within_tolerance": bool(matches)
                and all(m["within_tolerance"] for m in matches),
            }
        )
    replay_path = report / "replay_fidelity.csv"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    with replay_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(replay_rows[0]) if replay_rows else ["group_id"]
        )
        writer.writeheader()
        writer.writerows(replay_rows)
    source_dir = Path(__file__).resolve().parents[1]
    manifest["m03_code"] = {
        name: file_hash(source_dir / name)
        for name in (
            "scripts/run_m03_interventions.py",
            "scripts/finalize_m03.py",
            "scripts/build_m03_demo.py",
            "scripts/extract_m03_poses.py",
            "src/robot_vla/m03.py",
        )
    }
    write_json(report / "manifest.json", manifest)
    for optional in ("run.log", "gpu_samples.txt"):
        if (run_dir / optional).is_file():
            shutil.copy2(run_dir / optional, report / optional)

    by_condition = {}
    for condition in ("baseline", "sham", "brightness", "occlusion"):
        subset = [r for r in rows if r["condition"] == condition and r["status"] == "complete"]
        success = sum(r["success"].lower() == "true" for r in subset)
        by_condition[condition] = {
            "episodes": len(subset),
            "successes": success,
            "failures": len(subset) - success,
            "success_rate": success / len(subset) if subset else None,
            "wilson_95": wilson(success, len(subset)),
            "inference_mean_ms": [
                float(r["inference_mean_ms"]) for r in subset if r["inference_mean_ms"]
            ],
            "wall_seconds": [float(r["wall_seconds"]) for r in subset if r["wall_seconds"]],
            "action_delta_mean_abs": [
                float(r["action_delta_mean_abs_vs_baseline"])
                for r in subset
                if r.get("action_delta_mean_abs_vs_baseline")
            ],
            "trajectory_delta_mean_l2": [
                float(r["trajectory_delta_mean_l2_vs_baseline"])
                for r in subset
                if r.get("trajectory_delta_mean_l2_vs_baseline")
            ],
            "sim_qpos_trajectory_mean_l2_vs_baseline": [
                float(r["sim_qpos_trajectory_mean_l2_vs_baseline"]) for r in subset
            ],
            "sim_qvel_trajectory_mean_l2_vs_baseline": [
                float(r["sim_qvel_trajectory_mean_l2_vs_baseline"]) for r in subset
            ],
            "sim_contact_count_mean_abs_delta_vs_baseline": [
                float(r["sim_contact_count_mean_abs_delta_vs_baseline"]) for r in subset
            ],
            "sim_contact_count_max_abs_delta_vs_baseline": [
                int(r["sim_contact_count_max_abs_delta_vs_baseline"]) for r in subset
            ],
            "action_delta_vs_baseline": [
                float(r["action_delta_mean_abs_vs_baseline"])
                for r in subset
                if r.get("action_delta_mean_abs_vs_baseline")
            ],
        }
        latency = by_condition[condition]["inference_mean_ms"]
        by_condition[condition]["inference_mean_ms_p50"] = (
            statistics.median(latency) if latency else None
        )
        by_condition[condition]["inference_mean_ms_p95"] = (
            sorted(latency)[max(0, int(0.95 * len(latency)) - 1)] if latency else None
        )
    report_csv = report / "paired_interventions.csv"
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    shutil.copy2(report_csv, report / "episodes.csv")
    treatment_rows = [
        r for r in rows if r["condition"] in ("brightness", "occlusion") and r["status"] == "complete"
    ]
    treatment_successes = sum(r["success"].lower() == "true" for r in treatment_rows)
    by_condition["treatment_combined"] = {
        "episodes": len(treatment_rows),
        "successes": treatment_successes,
        "failures": len(treatment_rows) - treatment_successes,
        "success_rate": treatment_successes / len(treatment_rows) if treatment_rows else None,
        "wilson_95": wilson(treatment_successes, len(treatment_rows)),
        "note": "descriptive pooled count across two declared visual factors, not a shared treatment estimand",
    }
    paired_success = {}
    for condition in ("sham", "brightness", "occlusion"):
        paired = []
        for group in groups:
            if condition not in group["condition_results"]:
                continue
            paired.append((
                bool(group["condition_results"]["baseline"]["success"]),
                bool(group["condition_results"][condition]["success"]),
            ))
        paired_success[condition] = {
            "matched_groups": len(paired),
            "both_success": sum(a and b for a, b in paired),
            "baseline_only_success": sum(a and not b for a, b in paired),
            "condition_only_success": sum(not a and b for a, b in paired),
            "both_failure": sum(not a and not b for a, b in paired),
            "note": "descriptive matched binary outcomes; sample too small for generalized causal claims",
        }
    metrics = {
        "group_count": len(groups),
        "matched_group_count": sum(bool(g.get("eligible_for_matched_comparison")) for g in groups),
        "condition_results": by_condition,
        "matched_success_outcomes": paired_success,
        "physical_replay": {
            "source_state_exact_matches": sum(
                float(g["recovery_observation_state_linf_vs_m02"]) == 0 for g in groups
            ),
            "physical_groups_exact_matches": sum(bool(r["all_exact"]) for r in replay_rows),
            "position_linf_max": max(
                (r["qpos_linf_max"] for r in replay_rows if r["qpos_linf_max"] is not None),
                default=None,
            ),
            "velocity_linf_max": max(
                (r["qvel_linf_max"] for r in replay_rows if r["qvel_linf_max"] is not None),
                default=None,
            ),
            "sim_time_abs_max": max(
                (r["sim_time_abs_max"] for r in replay_rows if r["sim_time_abs_max"] is not None),
                default=None,
            ),
            "method": "reset exact M02 init_state_id then replay stored env_step_argument prefix; not recovery of an unavailable original full-state snapshot",
        },
        "artifact_integrity": {
            "episode_condition_records": len(rows),
            "all_video_and_trajectory_hashes_match": bool(integrity)
            and all(
                all(
                    x[k]
                    for k in (
                        "video_exists",
                        "video_hash_matches",
                        "trajectory_exists",
                        "trajectory_hash_matches",
                    )
                )
                for x in integrity
            ),
            "checks": integrity,
        },
        "baseline_sham_exact_trace_groups": exact_sham_groups,
        "baseline_sham_exact_trace_group_count": len(exact_sham_groups),
        "object_pose_checkpoints": {
            "records": len(object_pose_rows),
            "all_free_joint_bodies_resolved": bool(object_pose_rows) and all(r.get("all_free_joint_bodies_resolved") for r in object_pose_rows),
            "named_free_joint_object_pose_records": sum(len(r.get("free_joint_objects", [])) for r in object_pose_rows),
            "file": str(object_pose_path),
            "sha256": file_hash(object_pose_path) if object_pose_path.is_file() else None,
        },
        "resource_usage": _resource_summary(run_dir, rows),
        "limitations": [
            "M02 did not store original mid-episode MuJoCo qpos/qvel snapshots; fidelity compares independent replays at the same prefix against one another and against saved policy observation state.",
            "Visual interventions alter policy input only; they do not change task object positions or task goals.",
            "Failure root causes are not inferred from timeout alone; source human classifications and uncertainty are retained.",
            "Episodes share fixed source initial states and therefore are paired diagnostics, not an estimate of cross-seed generalization.",
        ],
    }
    write_json(report / "intervention_metrics.json", metrics)

    source_failures = {}
    with (args.m02_report / "failure_cases.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            source = json.loads(line)
            source_failures[
                (source["suite"], int(source["task_id"]), int(source["episode_id"]))
            ] = source
    source_checks = []
    for group in groups:
        source = source_failures.get(
            (group["source_suite"], int(group["source_task_id"]), int(group["source_episode_id"])),
            {},
        )
        evidence = source.get("evidence", {})
        for key, path_key, hash_key in (
            ("action_trace", "action_trace_path", "action_trace_sha256"),
            ("trajectory", "trajectory_path", "trajectory_sha256"),
        ):
            path = Path(evidence.get(path_key, ""))
            valid = path.is_file() and file_hash(path) == evidence.get(hash_key)
            source_checks.append(
                {"group_id": group["group_id"], "source_kind": key, "valid": valid,
                 "path": str(path), "sha256": evidence.get(hash_key)}
            )
    case_path = report / "diagnosis_cases.jsonl"
    with case_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["success"].lower() == "true":
                continue
            source = source_failures.get(
                (row["suite"], int(row["task_id"]), int(row["source_episode_id"])), {}
            )
            record = {
                "group_id": row["group_id"],
                "condition": row["condition"],
                "suite": row["suite"],
                "task_id": int(row["task_id"]),
                "instruction": row["instruction"],
                "source_episode_id": int(row["source_episode_id"]),
                "init_state_id": int(row["init_state_id"]),
                "success": False,
                "termination_reason": row["termination_reason"],
                "followup_steps": int(row["followup_steps"]),
                "video_path": str(output / "videos" / Path(row["video_path"]).name),
                "video_sha256": row["video_sha256"],
                "trajectory_path": row["trajectory_path"],
                "trajectory_sha256": row["trajectory_sha256"],
                "state_action_prefix_ref": source.get("evidence", {}).get(
                    "state_action_prefix_ref"
                ),
                "source_video_path": source.get("evidence", {}).get("video_path"),
                "source_video_sha256": source.get("evidence", {}).get("video_sha256"),
                "source_failure_classification": source.get(
                    "failure_classification",
                    {
                        "category": "Unknown / Insufficient Evidence",
                        "basis": "not available",
                        "confidence": "unknown",
                    },
                ),
                "m03_post_checkpoint_failure_class": "Unknown / Insufficient Evidence",
                "m03_failure_basis": "no validated stage-specific physical predicate; retained as outcome only",
                "m03_root_cause_inferred": False,
            }
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

    expected_conditions = {
        g["group_id"]: {"baseline", "sham"}
        | {c for c in ("brightness", "occlusion") if any(
            r["group_id"] == g["group_id"] and r["condition"] == c for r in rows
        )}
        for g in groups
    }
    observed_conditions = {
        group_id: {r["condition"] for r in rows if r["group_id"] == group_id}
        for group_id in expected_conditions
    }
    validation = {
        "expected_matched_groups": 10,
        "actual_groups": len(groups),
        "actual_condition_records": len(rows),
        "unique_condition_records": len({(r["group_id"], r["condition"]) for r in rows}),
        "expected_conditions_per_group": 3,
        "conditions_complete": len(rows) == 30
        and len({(r["group_id"], r["condition"]) for r in rows}) == 30
        and all(observed_conditions[g] == expected_conditions[g] and len(expected_conditions[g]) == 3 for g in expected_conditions),
        "all_groups_physical_state_matched": len(groups) == 10
        and all(g.get("eligible_for_matched_comparison") for g in groups),
        "two_treatment_types": {"brightness", "occlusion"}.issubset({r["condition"] for r in rows}),
        "baseline_sham_exact_trace_groups": len(exact_sham_groups),
        "baseline_sham_exact_trace_exact": len(exact_sham_groups) == len(groups) == 10,
        "all_three_conditions_measured": all(r.get("status") == "complete" for r in rows),
        "physical_trajectory_deltas_complete": len(rows) == 30 and all("sim_qpos_trajectory_mean_l2_vs_baseline" in r and "sim_qvel_trajectory_mean_l2_vs_baseline" in r for r in rows),
        "contact_trajectory_deltas_complete": len(rows) == 30 and all("sim_contact_count_mean_abs_delta_vs_baseline" in r for r in rows),
        "object_pose_records_complete": len(object_pose_rows) == 30 and all(r.get("all_free_joint_bodies_resolved") for r in object_pose_rows),
        "all_episode_artifacts_valid": bool(integrity)
        and all(
            all(
                x[k]
                for k in (
                    "video_exists",
                    "video_hash_matches",
                    "trajectory_exists",
                    "trajectory_hash_matches",
                )
            )
            for x in integrity
        ),
        "all_m02_source_prefix_artifacts_valid": bool(source_checks)
        and all(x["valid"] for x in source_checks),
        "m02_source_prefix_checks": source_checks,
        "video_requirement_checked_after_demo_build": False,
        "status": "INCOMPLETE",
    }
    write_json(report / "validation.json", validation)
    source_dir = Path(__file__).resolve().parents[1]
    shutil.copy2(
        source_dir / "scripts/run_m03_interventions.py", report / "run_m03_interventions.py"
    )
    shutil.copy2(source_dir / "src/robot_vla/m03.py", report / "m03.py")
    (report / "reproduce.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "cd /data/vla-forge\nsource .venv-policy/bin/activate\nunset LD_LIBRARY_PATH\n"
        "export LIBERO_CONFIG_PATH=/data/vla-forge/work/libero MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1\n"
        "RUN=/data/vla-forge-artifacts/M03_interventional_diagnosis\n"
        "REPORT=/data/vla-forge/reports/vla_forge_2_transition_20261010/milestones/M03_interventional_diagnosis\n"
        "python scripts/run_m03_interventions.py --root /data/vla-forge --output \"$RUN\" --limit 10\n"
        "python scripts/extract_m03_poses.py --root /data/vla-forge --run-dir \"$RUN\" --output \"$REPORT/object_poses.jsonl\"\n"
        "python scripts/finalize_m03.py --run-dir \"$RUN\" --report-dir \"$REPORT\" --m02-report /data/vla-forge/reports/vla_forge_2_transition_20261010/milestones/M02_libero40_failure_mining\n"
        "python scripts/build_m03_demo.py --run-dir \"$RUN\" --m02-report /data/vla-forge/reports/vla_forge_2_transition_20261010/milestones/M02_libero40_failure_mining --output \"$REPORT\"\n"
    )
    for name in ("scripts/finalize_m03.py", "scripts/build_m03_demo.py", "scripts/extract_m03_poses.py"):
        shutil.copy2(source_dir / name, report / Path(name).name)

    counts = "\n".join(
        f"- {key}: {value['successes']}/{value['episodes']}" for key, value in by_condition.items()
    )
    (report / "REPORT.md").write_text(
        "# M03 Interventional Diagnosis\n\n"
        "Status is provisional until the 150-second demo is built and validated.\n\n"
        f"## Results\n\n{counts}\n\n"
        f"- matched groups: {metrics['matched_group_count']}/{metrics['group_count']}\n"
        f"- exact prefix observation-state replays: {metrics['physical_replay']['source_state_exact_matches']}/{metrics['group_count']}\n"
        f"- exact physical-state matched groups: {metrics['physical_replay']['physical_groups_exact_matches']}/{metrics['group_count']}\n"
        f"- maximum qpos/qvel/sim-time errors: {metrics['physical_replay']['position_linf_max']} / {metrics['physical_replay']['velocity_linf_max']} / {metrics['physical_replay']['sim_time_abs_max']}\n\n"
        f"- named task-object pose checkpoints: {metrics.get('object_pose_checkpoints', {}).get('named_free_joint_object_pose_records', 0)}\n\n"
        "## Evidence boundary\n\n"
        "The source trajectories contain policy observations and executed 7D actions, not an original mid-episode MuJoCo snapshot. The measured method resets to the exact LIBERO initial-state index and replays the logged action prefix. Matching physical snapshots across independent resets validate this reconstruction path; they do not prove equality with an unsaved historical physics snapshot. Timeout labels remain outcome labels, not root causes.\n\n"
        "## Reproduction\n\n`bash reproduce.sh` (verified Linux CUDA/EGL environment required).\n"
    )
    (output / "index.html").write_text(_gallery_html(rows, metrics), encoding="utf-8")
    write_json(report / "validation.json", validation)


def _gallery_html(rows: list[dict], metrics: dict) -> str:
    cards = []
    for row in rows:
        name = Path(row["video_path"]).name
        cards.append(
            f"<article><h3>{row['group_id']} · {row['condition']}</h3>"
            f"<p>{row['instruction']} · success={row['success']} · steps={row['followup_steps']} · latency={row['inference_mean_ms']} ms</p>"
            f"<video controls preload='none' src='videos/{name}'></video></article>"
        )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>VLA-Forge M03</title>"
        "<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;background:#f5f7fa;color:#182333}article{background:white;padding:16px;margin:16px 0;border:1px solid #dce2ea;border-radius:8px}video{width:min(100%,900px);background:#111}h1{color:#234}</style>"
        f"<h1>VLA-Forge M03 · matched LIBERO replay/interventions</h1><pre>{json.dumps(metrics, indent=2)}</pre>{''.join(cards)}</html>"
    )


def _resource_summary(run_dir: Path, rows: list[dict]) -> dict:
    memory_mib = []
    sample_times = []
    sample_path = run_dir / "gpu_samples.txt"
    if sample_path.is_file():
        for line in sample_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", line.strip()):
                sample_times.append(datetime.fromisoformat(line.strip().replace("Z", "+00:00")))
            if "%" not in line:
                continue
            match = re.search(r"(\d+)\s*MiB", line)
            if match:
                memory_mib.append(int(match.group(1)))
    latency = [float(row["inference_mean_ms"]) for row in rows if row.get("inference_mean_ms")]
    wall = [float(row["wall_seconds"]) for row in rows if row.get("wall_seconds")]
    return {
        "condition_rollout_wall_seconds_sum": sum(wall),
        "condition_wall_seconds_median": statistics.median(wall) if wall else None,
        "select_action_latency_episode_mean_ms_median": statistics.median(latency)
        if latency
        else None,
        "select_action_latency_episode_mean_ms_p95": (
            sorted(latency)[max(0, int(0.95 * len(latency)) - 1)] if latency else None
        ),
        "nvidia_smi_sample_count": len(memory_mib),
        "gpu_sampler_window_seconds": (sample_times[-1] - sample_times[0]).total_seconds() if len(sample_times) >= 2 else None,
        "device_memory_used_peak_mib_sampled": max(memory_mib) if memory_mib else None,
        "memory_metric_note": "nvidia-smi sampled device use; not PyTorch allocator peak",
    }


if __name__ == "__main__":
    main()
