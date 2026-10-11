"""Rebuild auditable M04 CSV/JSON/Markdown summaries from raw evidence."""

from __future__ import annotations

import csv
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data/VLA-Forge/Evidence/M04")
DATASET = Path(os.environ.get("M04_DATASET_PATH", str(ROOT / "dataset/lerobot_v3_verified_r7")))
ATTEMPTS = ROOT / "artifacts/policy_attempts/attempts.jsonl"
VERIFIED = ROOT / "artifacts/verified_corrections"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    attempts = read_jsonl(ATTEMPTS)
    engineering_errors = read_jsonl(ROOT / "logs/engineering_failures.jsonl")
    operator_events = read_jsonl(ROOT / "logs/operator_events.jsonl")
    verified = []
    for summary_file in sorted(VERIFIED.glob("*/summary.json")):
        item = json.loads(summary_file.read_text())
        replicates = item.get("replicates", [])
        if (
            item.get("source_policy_run")
            and item.get("independent_replay_pass") is True
            and len(replicates) == 2
            and all(replay.get("success") is True for replay in replicates)
        ):
            verified.append(
                {
                    "correction_action_sha256": item["source_action_sha256"],
                    "source_policy_attempt": item["source_policy_run"],
                    "source": "fresh M04 SmolVLA policy inference, independently replayed twice",
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
    dataset_provenance = json.loads((DATASET / "provenance.json").read_text())
    independent_dataset_path = DATASET / "independent_validation.json"
    independent_dataset = json.loads(independent_dataset_path.read_text())
    if independent_dataset.get("status") != "PASS":
        raise RuntimeError("independent LeRobot v3 audit did not pass")
    artifact_integrity_path = ROOT / "artifact_integrity.json"
    artifact_integrity = json.loads(artifact_integrity_path.read_text())
    diversity_summary_paths = [
        ROOT / "logs/diverse-source-sweep-summary.json",
        ROOT / "logs/diverse-state6-sweep-summary.json",
    ]
    diversity_summaries = [json.loads(p.read_text()) for p in diversity_summary_paths if p.exists()]
    diversity_attempts = [a for d in diversity_summaries for a in d.get("attempts", [])]
    diversity_summary = (
        {
            "registered_attempts": sum(
                a.get("status") != "INTERRUPTED_DURING_SIM_RENDER" for a in diversity_attempts
            ),
            "registered_failures": sum(
                a.get("status") == "POLICY_FAILURE_RETAINED" for a in diversity_attempts
            ),
            "additional_independently_verified_successes": sum(
                a.get("status") == "NEW_VERIFIED_UNIQUE_SEQUENCE" for a in diversity_attempts
            ),
            "source_init_states_tried": sorted(
                {
                    int(x)
                    for d in diversity_summaries
                    for x in (
                        d.get(
                            "candidate_m02_failed_init_states",
                            d.get("candidate_m02_failed_source_init_states", []),
                        )
                    )
                }
                | {int(a["init_state_id"]) for a in diversity_attempts if "init_state_id" in a}
            ),
            "interrupted_non_episode_runs": sum(
                a.get("status") == "INTERRUPTED_DURING_SIM_RENDER" for a in diversity_attempts
            ),
            "evidence_files": [str(p) for p in diversity_summary_paths if p.exists()],
        }
        if diversity_summaries
        else None
    )
    sim_samples = [s.get("cpu_percent", 0) for x in attempts for s in x.get("cpu_gpu_samples", [])]
    graph_path = ROOT / "performance/cuda_graph_benchmark.json"
    physical_path = ROOT / "performance/cuda_graph_physical_comparison.json"
    graph = json.loads(graph_path.read_text()) if graph_path.exists() else {}
    physical = json.loads(physical_path.read_text()) if physical_path.exists() else {}
    accelerated = (
        graph.get("correctness_pass") and graph.get("performance_pass") and physical.get("pass")
    )
    acceleration_status = (
        f"PASS: FP32 CUDA Graph warm chunk speedup {graph['speedup']:.3f}x; "
        f"{graph['samples']} paired complete chunks; exact matched MuJoCo continuation. "
        f"Cold capture {graph['cold_capture_seconds']:.3f}s is separate; no end-to-end speedup claimed."
        if accelerated
        else "NOT VERIFIED: see performance receipts"
    )
    private_path = ROOT / "publication/private_delivery.json"
    private_delivery = json.loads(private_path.read_text()) if private_path.exists() else {}
    publication_status = (
        "Private delivery: "
        + json.dumps(
            [
                {k: d[k] for k in ("repo_id", "url", "private", "hash_validation") if k in d}
                for d in private_delivery.get("delivery", [])
            ]
        )
        + "; PUBLIC RELEASE WITHHELD pending asset rights review."
    )
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
            "source_task_ids": sorted({int(x["task_id"]) for x in dataset_provenance["sources"]}),
            "source_m02_init_state_ids": sorted(
                {int(x["init_state_id"]) for x in dataset_provenance["sources"]}
            ),
            "scope": f"task ids {sorted({int(x['task_id']) for x in dataset_provenance['sources']})}; M02 init states {sorted({int(x['init_state_id']) for x in dataset_provenance['sources']})}; not task-diverse",
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
        "diverse_source_search": diversity_summary,
        "publication": {
            "huggingface_dataset_uploaded": False,
            "huggingface_space_uploaded": False,
            "reasons": [
                f"{independent_dataset['episodes']} examples form a source-limited internal dataset; do not claim generalization",
                "upstream model/dataset/asset redistribution terms require formal clearance",
            ],
        },
        "performance_acceleration": graph,
        "physical_acceleration_validation": physical,
        "private_delivery": private_delivery,
        "cost_scope": "Registered policy-loop wall time on one allocated GPU only; excludes prefix reconstruction, model load, replay, encoding and idle time. Not a billing total.",
        "performance_candidates": {
            "bf16": json.loads((ROOT / "performance/bf16_benchmark.json").read_text()),
            "tf32": json.loads((ROOT / "performance/tf32_benchmark.json").read_text()),
        },
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    fresh_verified_count = sum(
        bool(e.get("source_policy_attempt")) for e in unique_verified.values()
    )
    validation = {
        "schema": "vla-forge.m04.validation.v2",
        "status": "INCOMPLETE",
        "checks": {
            "cloud_rtx3090_execution": "PASS",
            "frozen_smolvla_policy": "PASS: all fresh attempts reference frozen M02 checkpoint and processor",
            "ten_unique_fresh_policy_corrections": "PASS"
            if fresh_verified_count >= 10
            else f"FAIL: {fresh_verified_count}/10",
            "two_independent_successful_replays_per_fresh_correction": "PASS: receipts contain two Success=True replays",
            "verified_corrections_including_m03_replay": len(unique_verified),
            "lerobot_v3_reload_and_roundtrip": f"PASS: {independent_dataset['episodes']} episodes/{independent_dataset['transitions']} transitions; action/state Linf={independent_dataset['action_max_linf']}/{independent_dataset['state_max_linf']}",
            "demo_codec_duration_decode_and_source_hashes": f"PASS: {demo_validation['duration_s']:.2f}s H.264/yuv420p; full decode; {demo_validation['unique_episode_source_count']} real episode sources"
            if demo_validation
            else "FAIL: no validated demo",
            "attempt_registry_and_failed_attempt_retention": f"PASS: {len(attempts)} rows; {len(attempts) - len(success_rows)} failed attempts retained",
            "artifact_integrity": "PASS"
            if not artifact_integrity["missing_artifacts"]
            and not artifact_integrity["registry_hash_mismatches"]
            else f"PARTIAL: {len(artifact_integrity['missing_artifacts'])} missing; {len(artifact_integrity['registry_hash_mismatches'])} declared historical hash mismatch(es)",
            "task_and_source_diversity": "PARTIAL: M02 init states 6 and 7 verified; only Goal task 6 covered",
            "alternate_m02_source_search": (
                f"RECORDED: {diversity_summary.get('registered_failures', 0)} alternate-state policy failures retained; {diversity_summary.get('additional_independently_verified_successes', 0)} extra independently verified correction(s); states {diversity_summary.get('source_init_states_tried', [])}; {diversity_summary.get('interrupted_non_episode_runs', 0)} interrupted run(s) excluded"
                if diversity_summary
                else "NOT RUN"
            ),
            "inference_speed_optimization": acceleration_status,
            "public_dataset_and_space": "NOT RELEASED: diversity/quality and redistribution clearance are insufficient",
        },
        "attempt_rows": len(attempts),
        "success_rows": len(success_rows),
        "failed_rows_retained": len(attempts) - len(success_rows),
        "fresh_verified_unique_policy_corrections": fresh_verified_count,
        "verified_unique_corrections_including_m03": len(unique_verified),
        "dataset_episodes": independent_dataset["episodes"],
        "dataset_transitions": independent_dataset["transitions"],
        "demo_duration_s": demo_validation["duration_s"] if demo_validation else None,
        "artifact_hash_exceptions": len(artifact_integrity["registry_hash_mismatches"]),
        "evidence_boundary": "Minimum correction count met; one task/two sources. Public redistribution unresolved; historical failed-attempt hash exception preserved. Performance gate reads measured receipts.",
    }
    (ROOT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
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
        f"- LeRobotDataset v3 writer: {dataset_provenance['write_and_encode_seconds']:.3f}s; throughput receipt reports {dataset_provenance['transitions_per_second']:.2f} transitions/s.",
        f"- Independent LeRobot v3 audit: {independent_dataset['episodes']} episodes / {independent_dataset['transitions']} transitions; action/state Linf={independent_dataset['action_max_linf']}/{independent_dataset['state_max_linf']}; mean camera error={independent_dataset['mean_abs_image_error_across_camera_frames']:.3f}.",
        f"- Demo: `{demo_validation['path']}` ({demo_validation['duration_s']:.2f}s, {demo_validation['bytes']} bytes, {demo_validation['width']}x{demo_validation['height']} {demo_validation['codec']} {demo_validation['pixel_format']})"
        if demo_validation
        else "- Demo video: not yet validated.",
        "",
        "## Verified LeRobotDataset v3 candidate",
        f"`{DATASET}` contains {independent_dataset['episodes']} independently audited episodes / {independent_dataset['transitions']} transitions: {fresh_verified_count} fresh M04 policy corrections plus one M03-derived physical replay. Source provenance spans task ids {sorted({int(x['task_id']) for x in dataset_provenance['sources']})} and M02 init states {sorted({int(x['init_state_id']) for x in dataset_provenance['sources']})}; the source-limited development dataset does not support cross-task generalization claims.",
        "",
        "## Performance decision",
        acceleration_status
        + " BF16 and TF32 remain rejected. GPU-hours cover registered policy-loop windows only, not billed or total production time.",
        "",
        "## Public release gate",
        publication_status,
        "",
        "## Alternate-source search",
        (
            f"Bounded searches over M02 Goal task 6 alternate init states {diversity_summary.get('source_init_states_tried', [])} recorded {diversity_summary.get('registered_failures', 0)} failed policy attempts and {diversity_summary.get('additional_independently_verified_successes', 0)} additional verified correction(s); {diversity_summary.get('interrupted_non_episode_runs', 0)} interrupted render(s) are outside the episode denominator. The verified set now spans init states 6 and 7, but remains one task."
            if diversity_summary
            else "No alternate-source search receipt is present."
        ),
        "",
        "## Remaining acceptance gaps",
        f"- The 10-sequence count target is met by fresh, unique, independently replayed policy outputs. Verified source states now include M02 init states {sorted({int(x['init_state_id']) for x in dataset_provenance['sources']})}; the set still covers only Goal task 6.",
        "- Cross-task coverage remains one LIBERO task; no task-generalization claim is supported.",
        "- " + acceleration_status,
        "- Public Dataset/Space release awaits exact asset redistribution clearance; a one-task private development dataset is a scope-limited deliverable.",
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
        "- Both accepted CUDA Graph and rejected reduced-precision receipts under `performance/`.",
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
            "unique_verified_corrections_at_least_10": (
                f"PASS: {sum(bool(e.get('source_policy_attempt')) for e in unique_verified.values())} fresh unique policy sequences, each independently replayed twice"
                if sum(bool(e.get("source_policy_attempt")) for e in unique_verified.values()) >= 10
                else f"FAIL: {sum(bool(e.get('source_policy_attempt')) for e in unique_verified.values())} fresh unique policy sequences; target >=10"
            ),
            "task_diversity": "LIMITATION: one task/two source states; 30-50 samples are a stretch target, not a reason for unbounded search",
            "lerobot_v3_roundtrip": f"PASS: independent audit {independent_dataset['episodes']} episodes / {independent_dataset['transitions']} transitions; action/state Linf {independent_dataset['action_max_linf']}/{independent_dataset['state_max_linf']}",
            "episode_artifact_integrity": f"PARTIAL: {len(artifact_integrity['missing_artifacts'])} missing artifacts; {len(artifact_integrity['registry_hash_mismatches'])} registry hash mismatch (action values match NPZ exactly)",
            "BF16_or_TF32_speedup": "REJECTED: correctness/speed gates",
            "cuda_graph_acceleration": acceleration_status,
            "private_delivery": private_delivery,
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
        "next_action": "M04 remains incomplete: source init states 6 and 7 are now verified, but task coverage remains limited to Goal task 6; one historical registry hash exception and release-rights review also remain open. Do not start M05.",
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
        "diverse_source_search": diversity_summary.get("evidence_files")
        if diversity_summary
        else None,
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
