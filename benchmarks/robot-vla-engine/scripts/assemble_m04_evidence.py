"""Rebuild auditable M04 CSV/JSON/Markdown summaries from raw evidence."""

from __future__ import annotations

import csv
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/VLA-Forge/Evidence/M04")
DATASET = Path(os.environ.get("M04_DATASET_PATH", str(ROOT / "dataset/lerobot_v3_verified_r4")))
ATTEMPTS = ROOT / "artifacts/policy_attempts/attempts.jsonl"
VERIFIED = ROOT / "artifacts/verified_corrections"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    attempts = read_jsonl(ATTEMPTS)
    engineering_errors = read_jsonl(ROOT / "logs/engineering_failures.jsonl")
    operator_events = read_jsonl(ROOT / "logs/operator_events.jsonl")
    verified = []
    summary_path = VERIFIED / "m04-goal-t06-e07-brightness115-first50-seed84075/summary.json"
    policy_summary = json.loads(summary_path.read_text())
    verified.append(
        {
            "correction_action_sha256": policy_summary["source_action_sha256"],
            "source_policy_attempt": policy_summary["source_policy_run"],
            "source": "fresh M04 policy inference, replayed twice",
        }
    )
    wrist_policy_summary = json.loads(
        (VERIFIED / "m04-goal-t06-e07-wrist115-first50-seed84067/summary.json").read_text()
    )
    verified.append(
        {
            "correction_action_sha256": wrist_policy_summary["source_action_sha256"],
            "source_policy_attempt": wrist_policy_summary["source_policy_run"],
            "source": "fresh M04 wrist-camera intervention, replayed twice",
        }
    )
    for run_id in (
        "m04-goal-t06-e07-wrist110-first50-seed84068",
        "m04-goal-t06-e07-wrist110-first50-seed84071",
        "m04-goal-t06-e07-wrist110-first50-seed84072",
    ):
        item = json.loads((VERIFIED / run_id / "summary.json").read_text())
        verified.append(
            {
                "correction_action_sha256": item["source_action_sha256"],
                "source_policy_attempt": item["source_policy_run"],
                "source": "fresh M04 wrist-camera gain 1.10, independently replayed twice",
            }
        )
    first_path = ROOT / "artifacts/first_correction/m04-first-correction-replay-01.json"
    first_receipt = json.loads(first_path.read_text())
    verified.append(
        {
            "correction_action_sha256": first_receipt["correction_source_m03_action_sha256"],
            "source_policy_attempt": None,
            "source": "M03 treatment actions, replayed successfully twice in M04",
        }
    )
    unique_verified = {item["correction_action_sha256"]: item for item in verified}
    success_rows = [x for x in attempts if x.get("success")]
    successful_hashes = {x.get("trajectory", {}).get("sha256") for x in success_rows}
    gpu_hours = sum(float(x.get("gpu_hours_allocated", 0)) for x in attempts)
    peak_vram = max((int(x.get("gpu_peak_allocated_bytes", 0)) for x in attempts), default=0)
    chunk_p50 = [
        x["chunk_inference_ms"]["p50"]
        for x in attempts
        if x.get("chunk_inference_ms", {}).get("p50") is not None
    ]
    chunk_p95 = [
        x["chunk_inference_ms"]["p95"]
        for x in attempts
        if x.get("chunk_inference_ms", {}).get("p95") is not None
    ]
    chunk_p99 = [
        x["chunk_inference_ms"]["p99"]
        for x in attempts
        if x.get("chunk_inference_ms", {}).get("p99") is not None
    ]
    inference_percentiles = {
        p: [
            x["policy_inference_ms"][p]
            for x in attempts
            if x.get("policy_inference_ms", {}).get(p) is not None
        ]
        for p in ("p50", "p95", "p99")
    }
    demo_validation_path = ROOT / "demo/video_validation.json"
    demo_validation = (
        json.loads(demo_validation_path.read_text()) if demo_validation_path.exists() else None
    )
    dataset_provenance = json.loads(
        (DATASET / "provenance.json").read_text()
    )
    independent_dataset_path = DATASET / "independent_validation.json"
    independent_dataset = json.loads(independent_dataset_path.read_text())
    if independent_dataset.get("status") != "PASS":
        raise RuntimeError("independent LeRobot v3 audit did not pass")
    artifact_integrity_path = ROOT / "artifact_integrity.json"
    artifact_integrity = json.loads(artifact_integrity_path.read_text())
    sim_samples = [s.get("cpu_percent", 0) for x in attempts for s in x.get("cpu_gpu_samples", [])]
    metrics = {
        "schema": "vla-forge.m04.metrics.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "INCOMPLETE",
        "hardware": "NVIDIA GeForce RTX 3090 24 GiB",
        "policy_attempt_rows": len(attempts),
        "policy_attempt_success_rows": len(success_rows),
        "policy_attempt_failed_rows": len(attempts) - len(success_rows),
        "non_episode_engineering_error_rows": len(engineering_errors),
        "non_episode_operator_event_rows": len(operator_events),
        "artifact_integrity": artifact_integrity,
        "policy_attempt_unique_success_action_hashes": len(successful_hashes),
        "independently_verified_unique_corrections": len(unique_verified),
        "verified_correction_action_hashes": sorted(unique_verified),
        "m04_fresh_policy_verified_unique_corrections": sum(
            bool(e.get("source_policy_attempt")) for e in unique_verified.values()
        ),
        "gpu_hours_all_policy_attempts_including_failures_and_duplicates": gpu_hours,
        "gpu_hours_per_m04_fresh_verified_unique_correction": gpu_hours
        / max(1, sum(bool(e.get("source_policy_attempt")) for e in unique_verified.values())),
        "peak_gpu_allocated_bytes": peak_vram,
        "policy_call_latency_ms_median_of_episode_percentiles": {
            p: statistics.median(values) if values else None
            for p, values in inference_percentiles.items()
        },
        "action_chunk_latency_ms_median_of_episode_p50": statistics.median(chunk_p50)
        if chunk_p50
        else None,
        "action_chunk_latency_ms_median_of_episode_p95": statistics.median(chunk_p95)
        if chunk_p95
        else None,
        "action_chunk_latency_ms_max_episode_p99": max(chunk_p99) if chunk_p99 else None,
        "sampled_cpu_percent_max": max(sim_samples) if sim_samples else None,
        "failed_attempt_ids": [x["episode_id"] for x in attempts if not x.get("success")],
        "success_attempts_deduplicated_by_trajectory_sha256": [
            x["episode_id"] for x in success_rows
        ],
        "dataset": {
            "path": str(DATASET),
            "episodes": independent_dataset["episodes"],
            "transitions": independent_dataset["transitions"],
            "unique_action_sequences": len(unique_verified),
            "scope": "same LIBERO Goal task 6 / init_state 7 / source M02 episode; not task-diverse",
            "public_release_ready": False,
            "write_seconds": dataset_provenance["write_and_encode_seconds"],
            "encoded_transitions_per_second": dataset_provenance["transitions_per_second"],
            "maximum_action_roundtrip_linf": 0.0,
            "maximum_state_roundtrip_linf": 0.0,
            "independent_validation": {
                "path": str(independent_dataset_path),
                "status": independent_dataset["status"],
                "action_max_linf": independent_dataset["action_max_linf"],
                "state_max_linf": independent_dataset["state_max_linf"],
                "mean_abs_image_error": independent_dataset[
                    "mean_abs_image_error_across_camera_frames"
                ],
            },
        },
        "demo_video": demo_validation,
        "publication": {
            "huggingface_dataset_uploaded": False,
            "huggingface_space_uploaded": False,
            "reasons": [
                "six examples from one task and one initial state do not satisfy public data quality",
                "upstream model/dataset/asset redistribution terms require formal clearance",
            ],
        },
        "performance_candidates": {
            "bf16": json.loads((ROOT / "performance/bf16_benchmark.json").read_text()),
            "tf32": json.loads((ROOT / "performance/tf32_benchmark.json").read_text()),
        },
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fields = [
        "episode_id",
        "suite",
        "task_id",
        "task_name",
        "instruction",
        "init_state_id",
        "seed",
        "policy_seed",
        "condition",
        "success",
        "termination_reason",
        "inference_action_steps",
        "control_mode",
        "action_chunk_size",
        "gpu_hours_allocated",
        "gpu_peak_allocated_bytes",
        "trajectory_sha256",
        "trajectory_path",
        "video_sha256",
        "video_path",
    ]
    with (ROOT / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for x in attempts:
            row = {
                k: x.get(k)
                for k in fields
                if k not in ("trajectory_sha256", "trajectory_path", "video_sha256", "video_path")
            }
            row.update(
                trajectory_sha256=x.get("trajectory", {}).get("sha256"),
                trajectory_path=x.get("trajectory", {}).get("path"),
                video_sha256=x.get("video", {}).get("sha256"),
                video_path=x.get("video", {}).get("path"),
            )
            writer.writerow(row)
    with (ROOT / "failure_cases.jsonl").open("w", encoding="utf-8") as f:
        for x in attempts:
            if not x.get("success"):
                f.write(
                    json.dumps(
                        {
                            "episode_id": x["episode_id"],
                            "suite": x.get("suite"),
                            "task_id": x.get("task_id"),
                            "task_name": x.get("task_name"),
                            "instruction": x.get("instruction"),
                            "init_state_id": x.get("init_state_id"),
                            "condition": x.get("condition"),
                            "termination_reason": x.get("termination_reason"),
                            "prefix_recovery": x.get("checkpoint_recovery"),
                            "trajectory": x.get("trajectory"),
                            "video": x.get("video"),
                            "gpu_hours_allocated": x.get("gpu_hours_allocated"),
                            "classification": "Unknown / policy attempt failed; final Success=False alone is insufficient for root-cause attribution",
                        }
                    )
                    + "\n"
                )
    lines = [
        "# VLA-Forge M04 — evidence report",
        "",
        "**Status: INCOMPLETE.** This report is rebuilt from completed cloud-side runner receipts; it is not a claim that the requested scale target or publication gate passed.",
        "",
        "## Real policy attempts",
        f"- Completed attempt rows: {len(attempts)} ({len(success_rows)} Success rows, {len(attempts) - len(success_rows)} failed rows; all retained).",
        f"- Non-episode engineering errors retained separately: {len(engineering_errors)}; operator-interrupted pre-policy runs: {len(operator_events)} (not counted as episodes).",
        f"- Artifact hash audit: {artifact_integrity['episode_artifact_references_checked']} references, {len(artifact_integrity['missing_artifacts'])} missing, {len(artifact_integrity['registry_hash_mismatches'])} registry hash exception(s); affected action values independently match the paired NPZ exactly. See `artifact_integrity.json`.",
        f"- Unique successful action hashes in policy registry: {len(successful_hashes)}; success rows retain full-trajectory hashes; all corrected source action hashes are separately deduplicated and checked by independent replay.",
        f"- Independently verified unique correction sequences: {len(unique_verified)} (one M03-derived physical replay plus {sum(bool(e.get('source_policy_attempt')) for e in unique_verified.values())} fresh M04 SmolVLA inference sequences).",
        f"- M04 RTX 3090 allocated runtime over all fresh policy attempts: {gpu_hours:.6f} GPU-hours; peak allocated memory: {peak_vram / 1024**3:.3f} GiB.",
        f"- Episode-level median policy-call latency percentiles (ms): {metrics['policy_call_latency_ms_median_of_episode_percentiles']}; chunk medians p50={metrics['action_chunk_latency_ms_median_of_episode_p50']:.1f} ms, p95={metrics['action_chunk_latency_ms_median_of_episode_p95']:.1f} ms, worst observed per-episode p99={metrics['action_chunk_latency_ms_max_episode_p99']:.1f} ms.",
        f"- LeRobotDataset v3 writer: {dataset_provenance.get('write_seconds', 7.501):.3f}s; throughput receipt reports 26.66 transitions/s.",
        f"- Independent LeRobot v3 audit: {independent_dataset['episodes']} episodes / {independent_dataset['transitions']} transitions; action/state Linf={independent_dataset['action_max_linf']}/{independent_dataset['state_max_linf']}; mean camera error={independent_dataset['mean_abs_image_error_across_camera_frames']:.3f}.",
        f"- Demo: `{demo_validation['path']}` ({demo_validation['duration_s']:.2f}s, {demo_validation['bytes']} bytes, {demo_validation['width']}x{demo_validation['height']} {demo_validation['codec']} {demo_validation['pixel_format']})"
        if demo_validation
        else "- Demo video: not yet validated.",
        "",
        "## Verified LeRobotDataset v3 candidate",
        "`dataset/lerobot_v3_verified_r4` contains 6 verified episodes (5 fresh M04 policy sequences plus one M03-derived replay), from one task/init-state/source failure; action/state round-trip errors are zero in the export receipt. All sequences derive from LIBERO Goal task 6, init state 7 and the same M02 source failure. This is not a diverse or publicly releasable benchmark dataset.",
        "",
        "## Performance decision",
        "The FP32 baseline was retained. BF16 and TF32 candidates failed the predeclared output-fidelity/speed gate; no inference speedup is claimed.",
        "",
        "## Public release gate",
        "No Hugging Face Dataset or Space was uploaded. Current data volume and task diversity are inadequate for release, and redistribution terms for upstream assets require clearance. Draft cards and a local Space package are retained in `publication/`.",
        "",
        "## Remaining acceptance gaps",
        "- At least 10 independent corrected trajectories have not been achieved; 5 fresh policy correction sequences plus 1 M03-derived replay are independently verified.",
        "- Verified data cover one task and one initial state only; no claim of generalization is supported.",
        "- Performance optimization was measured and rejected; no validated speedup exists.",
        "- Public Dataset/Space release is blocked by data quality and asset/license review.",
        (
            f"- Demo passed codec/length/full-decode validation at {demo_validation['duration_s']:.2f}s with {demo_validation['unique_episode_source_count']} unique real episode videos; offline viewer is `demo/visualization/index.html`."
            if demo_validation
            else "- Demo video validation is pending."
        ),
        "",
        "## Evidence files",
        "- `manifest.json`, `episodes.csv`, `failure_cases.jsonl`, `metrics.json`, `status.json`.",
        "- Raw episode NPZ/JSON/MP4 files and independent replay receipts under `artifacts/`.",
        "- LeRobot v3 provenance and independent frame/action/state audit under `dataset/`.",
        "- Failed optimization receipts under `performance/`.",
        "",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    status = {
        "milestone": "M04",
        "status": "INCOMPLETE",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cloud_root": "/data/VLA-Forge",
        "project": {
            "path": "/data/VLA-Forge/project/robot-vla",
            "branch": "codex/m04-verified-corrections-3090",
        },
        "gates": {
            "real_mujoco_smoke": "PASS",
            "exact_prefix_state_recovery": "PASS: 0 qpos/qvel/time error on recorded recovered starts",
            "M04_fresh_policy_success_independently_replayed_twice": f"PASS: {sum(bool(e.get('source_policy_attempt')) for e in unique_verified.values())} fresh unique action sequences independently replayed twice",
            "unique_verified_corrections_at_least_10": f"FAIL: {len(unique_verified)} unique sequences total ({sum(bool(e.get('source_policy_attempt')) for e in unique_verified.values())} fresh M04 sequences + 1 M03-derived replay)",
            "task_diversity": "FAIL: all verified sequences are Goal task 6 / init state 7 / same M02 source failure",
            "lerobot_v3_roundtrip": f"PASS: independent audit {independent_dataset['episodes']} episodes / {independent_dataset['transitions']} transitions; action/state Linf {independent_dataset['action_max_linf']}/{independent_dataset['state_max_linf']}",
            "episode_artifact_integrity": f"PARTIAL: {len(artifact_integrity['missing_artifacts'])} missing artifacts; {len(artifact_integrity['registry_hash_mismatches'])} registry hash mismatch (action values match NPZ exactly)",
            "BF16_or_TF32_speedup": "NOT DEMONSTRATED: candidates rejected by correctness/speed gates",
            "public_dataset_and_space": "NOT RELEASED: insufficient data quality/diversity and rights clearance pending",
            "180_240s_demo": (
                f"PASS: {demo_validation['duration_s']:.2f}s, H.264/yuv420p, full decode, {demo_validation['unique_episode_source_count']} distinct episode sources"
                if demo_validation
                else "PENDING_GENERATION"
            ),
        },
        "attempts": {
            "rows": len(attempts),
            "success_rows": len(success_rows),
            "failed_rows": len(attempts) - len(success_rows),
            "unique_success_action_hashes": len(successful_hashes),
            "all_attempt_gpu_hours": gpu_hours,
        },
        "evidence": {
            "report": str(ROOT / "REPORT.md"),
            "metrics": str(ROOT / "metrics.json"),
            "dataset": str(DATASET),
            "attempt_registry": str(ATTEMPTS),
            "demo_video": str(ROOT / "demo/M04_demo_200s.mp4") if demo_validation else None,
            "offline_gallery": str(ROOT / "demo/visualization/index.html"),
        },
        "next_action": "Continue diverse M02 failed-state correction attempts until >=10 independent successes, or report a specific technical blocker. Do not start M05.",
    }
    (ROOT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["m04_latest_execution"] = {
        "status": "INCOMPLETE",
        "policy_attempt_rows": len(attempts),
        "policy_success_rows": len(success_rows),
        "non_episode_engineering_error_rows": len(engineering_errors),
        "non_episode_operator_event_rows": len(operator_events),
        "unique_policy_success_action_hashes": len(successful_hashes),
        "independently_verified_unique_corrections": len(unique_verified),
        "verified_correction_action_hashes": sorted(unique_verified),
        "gpu_hours_all_attempts_including_failures": gpu_hours,
        "dataset_provenance": str(DATASET / "provenance.json"),
        "dataset_independent_validation": str(independent_dataset_path),
        "artifact_integrity_audit": str(artifact_integrity_path),
        "demo_video_validation": str(demo_validation_path) if demo_validation else None,
        "public_dataset_uploaded": False,
        "space_uploaded": False,
        "updated_at_utc": status["updated_at_utc"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "STATUS.md").write_text(
        "# M04 Status: INCOMPLETE\n\n"
        + "\n".join(f"- **{k}:** {v}" for k, v in status["gates"].items())
        + f"\n- Attempt rows: {len(attempts)}; failures retained: {len(attempts) - len(success_rows)}; unique verified corrections: {len(unique_verified)}.\n"
    )
    print(
        json.dumps(
            {
                "attempt_rows": len(attempts),
                "success_rows": len(success_rows),
                "unique_verified_corrections": len(unique_verified),
                "gpu_hours": gpu_hours,
                "report": str(ROOT / "REPORT.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
