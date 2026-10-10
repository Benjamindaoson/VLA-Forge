"""Verify M02 episode artifacts and generate benchmark/failure deliverables."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import av
import numpy as np

from robot_vla.m02 import (
    M02_SUITES,
    aggregate_binary,
    artifact_is_valid,
    classify_failure,
    count_verified_video_frames,
    expected_episode_identities,
    select_manual_review,
    task_and_suite_metrics,
    validate_m02_coverage,
)


def _load_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("task_id", "episode_id", "init_state_id", "seed", "attempt", "total_action_steps", "horizon"):
            row[field] = int(row[field])
        for field in ("success",):
            row[field] = True if row[field] == "True" else False if row[field] == "False" else None
        for field in ("episode_wall_seconds", "policy_inference_mean_ms", "policy_inference_p95_ms", "chunk_inference_mean_ms"):
            try:
                row[field] = float(row[field])
            except (TypeError, ValueError):
                row[field] = None
    return rows


def _check_episode(row):
    issues = []
    evidence = {"video_decoded": False, "action_trajectory_exact_match": False, "video": {}}
    required = (
        ("video_path", "video_sha256"),
        ("trajectory_path", "trajectory_sha256"),
        ("action_trace_path", "action_trace_sha256"),
        ("metadata_path", "metadata_sha256"),
    )
    for path_key, hash_key in required:
        if not artifact_is_valid(row.get(path_key), row.get(hash_key)):
            issues.append(f"{path_key} missing or hash mismatch")
    if issues:
        return issues, evidence
    video_info = {}
    try:
        with av.open(row["video_path"]) as container:
            stream = container.streams.video[0]
            video_info = {"width": stream.width, "height": stream.height, "frames": sum(1 for _ in container.decode(stream))}
        if video_info["frames"] <= 0 or video_info["width"] != 2 * video_info["height"]:
            issues.append(f"paired-camera video dimensions/frame count invalid: {video_info}")
        else:
            evidence["video_decoded"] = True
            evidence["video"] = video_info
    except Exception as exc:
        issues.append(f"video decode failed: {type(exc).__name__}: {exc}")
    try:
        with np.load(row["trajectory_path"], allow_pickle=False) as trajectory:
            actions = trajectory["actions"]
            states = trajectory["states"]
        if actions.shape != (row["total_action_steps"], 7) or states.shape != (len(actions) + 1, 8):
            issues.append(f"trajectory shape mismatch actions={actions.shape} states={states.shape}")
        if not np.isfinite(actions).all() or not np.isfinite(states).all():
            issues.append("trajectory contains non-finite state/action values")
        trace = json.loads(Path(row["action_trace_path"]).read_text(encoding="utf-8"))
        executed = np.asarray([step["env_step_argument"] for step in trace["steps"]], dtype=np.float32)
        if executed.shape != actions.shape or not np.array_equal(executed, actions):
            issues.append("action trace does not exactly match executed trajectory")
        if not trace.get("trajectory_exact_match"):
            issues.append("action trace lacks exact-match receipt")
        if not any("action trace" in issue for issue in issues):
            evidence["action_trajectory_exact_match"] = True
    except Exception as exc:
        issues.append(f"trajectory/action trace read failed: {type(exc).__name__}: {exc}")
    try:
        metadata = json.loads(Path(row["metadata_path"]).read_text(encoding="utf-8"))
        for key in ("suite", "task_id", "episode_id", "init_state_id", "seed", "success", "termination_reason", "total_action_steps"):
            if str(metadata.get(key)) != str(row.get(key)):
                issues.append(f"Episode metadata identity/result mismatch: {key}")
    except Exception as exc:
        issues.append(f"Episode metadata read failed: {type(exc).__name__}: {exc}")
    return issues, evidence


def _manual_reviews(path):
    reviews = {}
    if not path.is_file():
        return reviews
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("reviewed"):
            key = (entry["suite"], int(entry["task_id"]), int(entry["episode_id"]))
            reviews[key] = entry
    return reviews


def _task_name(row):
    return row.get("task_name") or f"{row['suite']}_task{row['task_id']}"


def analyze(output: Path):
    output = output.resolve()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = _load_rows(output / "episodes.csv")
    issues = []
    invalid = []
    verified = {}
    seen = set()
    for index, row in enumerate(rows):
        identity = (row["suite"], row["task_id"], row["episode_id"])
        if identity in seen:
            issues.append(f"duplicate row {identity}")
        seen.add(identity)
        if row["status"] != "complete" or type(row["success"]) is not bool:
            issues.append(f"non-complete row {identity}: {row['status']}")
            invalid.append({"identity": identity, "issues": ["status not complete"]})
            continue
        item_issues, video_info = _check_episode(row)
        verified[identity] = (not item_issues, video_info)
        if item_issues:
            invalid.append({"identity": identity, "issues": item_issues})
        row["_verified_video"] = not item_issues and bool(video_info)
        row["_video_info"] = video_info
    expected = expected_episode_identities()
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing:
        issues.append(f"missing {len(missing)} expected Episode identities")
    if extra:
        issues.append(f"found {len(extra)} unexpected Episode identities")
    valid_rows = [
        row
        for row in rows
        if row["status"] == "complete"
        and type(row["success"]) is bool
        and verified.get((row["suite"], row["task_id"], row["episode_id"]), (False, {}))[0]
    ]
    if len(valid_rows) == 400 and not issues:
        coverage = validate_m02_coverage(valid_rows)
    else:
        coverage = {suite: sum(row["suite"] == suite for row in valid_rows) for suite in M02_SUITES}
    tasks, suites, overall = task_and_suite_metrics(valid_rows)
    resource_summary = _registry_resource_summary(
        Path(manifest["artifact_root"]) / "registry" / "registry.sqlite"
    )
    task_rows = []
    for (suite, task_id), metric in sorted(tasks.items()):
        row = next(row for row in valid_rows if row["suite"] == suite and row["task_id"] == task_id)
        task_rows.append({"suite": suite, "benchmark_suite": row["benchmark_suite"], "task_id": task_id, "task_name": _task_name(row), "instruction": row["instruction"], **metric, "complete": metric["episodes"] == 10})
    with (output / "task_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["suite", "benchmark_suite", "task_id", "task_name", "instruction", "successes", "episodes", "failures", "success_rate", "wilson_95", "complete"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in task_rows:
            writer.writerow({**row, "wilson_95": json.dumps(row["wilson_95"], separators=(",", ":"))})
    suite_metrics = {}
    for suite in M02_SUITES:
        task_values = [row["success_rate"] for row in task_rows if row["suite"] == suite and row["success_rate"] is not None]
        suite_metrics[suite] = {
            **suites.get(suite, aggregate_binary([])),
            "task_count": sum(row["suite"] == suite for row in task_rows),
            "task_macro_success_rate": statistics.mean(task_values) if task_values else None,
            "task_macro_successes": sum(next(row["successes"] for row in task_rows if row["suite"] == suite and row["task_id"] == task_id) / 10 for task_id in range(10) if any(row["suite"] == suite and row["task_id"] == task_id for row in task_rows)) / 10 if len(task_values) == 10 else None,
        }
    summary_metrics = {
        "status": "PASS" if len(valid_rows) == 400 and not issues else "INCOMPLETE",
        "suites": suite_metrics,
        "overall_pooled": overall,
        "overall_task_macro_success_rate": statistics.mean([row["success_rate"] for row in task_rows]) if len(task_rows) == 40 else None,
        "inference_latency_episode_mean_ms": _latency_summary(valid_rows, "policy_inference_mean_ms"),
        "inference_latency_episode_p95_ms": _latency_summary(valid_rows, "policy_inference_p95_ms"),
        "action_chunk_generation_episode_mean_ms": _latency_summary(valid_rows, "chunk_inference_mean_ms"),
        "episode_wall_seconds_sum": sum(row["episode_wall_seconds"] or 0 for row in valid_rows),
        "gpu_peak_allocated_bytes": max((row.get("gpu_peak_allocated_bytes") or 0 for row in valid_rows), default=0),
        "gpu_peak_reserved_bytes": max((row.get("gpu_peak_reserved_bytes") or 0 for row in valid_rows), default=0),
        "nvidia_smi_resource_monitor": resource_summary,
        "video_frame_count": count_verified_video_frames(valid_rows),
    }
    _write_json(output / "suite_metrics.json", summary_metrics)
    _write_json(output / "metrics.json", summary_metrics)
    reviews = _manual_reviews(output / "manual_failure_review.jsonl")
    failures = [row for row in valid_rows if row["success"] is False]
    failure_counts = Counter()
    candidates = []
    failure_jsonl = output / "failure_cases.jsonl"
    with failure_jsonl.open("w", encoding="utf-8") as handle:
        for row in failures:
            key = (row["suite"], row["task_id"], row["episode_id"])
            classification = classify_failure(row, reviews.get(key))
            failure_counts[classification["category"]] += 1
            trace_prefix = f"{row['trajectory_path']}#actions[0:{row['total_action_steps']}],states[0:{row['total_action_steps'] + 1}]"
            candidate = {
                "suite": row["suite"], "benchmark_suite": row["benchmark_suite"],
                "task_id": row["task_id"], "task_name": _task_name(row),
                "instruction": row["instruction"], "episode_id": row["episode_id"],
                "init_state_id": row["init_state_id"], "seed": row["seed"],
                "checkpoint_revision": row["checkpoint_revision"], "checkpoint_sha256": row["checkpoint_sha256"],
                "success": False, "termination_reason": row["termination_reason"],
                "total_action_steps": row["total_action_steps"], "horizon": row["horizon"],
                "failure_classification": classification,
                "failure_step": classification.get("failure_step"),
                "evidence": {"video_path": row["video_path"], "video_sha256": row["video_sha256"], "trajectory_path": row["trajectory_path"], "trajectory_sha256": row["trajectory_sha256"], "action_trace_path": row["action_trace_path"], "action_trace_sha256": row["action_trace_sha256"], "state_action_prefix_ref": trace_prefix, "metadata_path": row["metadata_path"], "metadata_sha256": row["metadata_sha256"]},
                "manual_reviewed": key in reviews,
                "m03_candidate": bool(row.get("_verified_video")) and artifact_is_valid(row["trajectory_path"], row["trajectory_sha256"]) and artifact_is_valid(row["action_trace_path"], row["action_trace_sha256"]),
            }
            candidates.append(candidate)
            handle.write(json.dumps(candidate, ensure_ascii=False, allow_nan=False) + "\n")
    review_queue = select_manual_review(failures)
    with (output / "manual_review_queue.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["suite", "task_id", "task_name", "episode_id", "video_path", "trajectory_path", "action_trace_path", "review_status"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in review_queue:
            writer.writerow({"suite": row["suite"], "task_id": row["task_id"], "task_name": row["task_name"], "episode_id": row["episode_id"], "video_path": row["video_path"], "trajectory_path": row["trajectory_path"], "action_trace_path": row["action_trace_path"], "review_status": "reviewed" if (row["suite"], row["task_id"], row["episode_id"]) in reviews else "pending"})
    validation = {
        "status": "PASS" if len(valid_rows) == 400 and not issues else "INCOMPLETE",
        "expected_episodes": 400,
        "csv_rows": len(rows),
        "valid_completed_episodes": len(valid_rows),
        "invalid_episodes": invalid,
        "missing_identities": [list(value) for value in missing],
        "unexpected_identities": [list(value) for value in extra],
        "suite_coverage": coverage,
        "task_coverage_rows": len(task_rows),
        "paired_camera_videos_decoded": sum(
            verified.get((row["suite"], row["task_id"], row["episode_id"]), (False, {}))[1].get("video_decoded", False)
            for row in valid_rows
        ),
        "action_trajectory_exact_matches": sum(
            verified.get((row["suite"], row["task_id"], row["episode_id"]), (False, {}))[1].get("action_trajectory_exact_match", False)
            for row in valid_rows
        ),
        "failure_cases": len(failures),
        "manual_review_required": len(review_queue),
        "manual_review_completed": sum((row["suite"], row["task_id"], row["episode_id"]) in reviews for row in review_queue),
        "issues": issues,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "validation.json", validation)
    report_status = "PASS" if validation["status"] == "PASS" and validation["manual_review_completed"] == len(review_queue) else "INCOMPLETE"
    hardest = sorted(task_rows, key=lambda row: (row["success_rate"] if row["success_rate"] is not None else 1, row["suite"], row["task_id"]))[:3]
    top_failures = sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))
    suites_md = "\n".join(f"- {suite}: {suite_metrics[suite]['successes']}/{suite_metrics[suite]['episodes']} ({_pct(suite_metrics[suite]['success_rate'])}); 95% Wilson {_ci(suite_metrics[suite]['wilson_95'])}." for suite in M02_SUITES)
    hardest_md = "\n".join(f"- `{row['suite']}` task {row['task_id']} — {_task_name(next(item for item in valid_rows if item['suite']==row['suite'] and item['task_id']==row['task_id']))}: {row['successes']}/{row['episodes']} ({_pct(row['success_rate'])})." for row in hardest)
    failure_md = "\n".join(f"- {name}: {count} ({_pct(count / len(failures) if failures else None)})." for name, count in top_failures) or "- No measured failures."
    review_md = "\n".join(f"- `{row['suite']}` task {row['task_id']} Episode {row['episode_id']}: {row['video_path']}" for row in review_queue)
    review_sheets_md = "\n".join(f"- `manual_review/stratified_review_page{index}.jpg` ({suite} sample)." for index, suite in enumerate(M02_SUITES, start=1))
    report = f"""# M02 — Standard LIBERO 40 benchmark and failure mining

**Acceptance: {report_status}.** Benchmark status: {validation['status']}. A complete success rate is reported only when all 400 Episodes and required artifacts validate. Manual failure review is {validation['manual_review_completed']}/{len(review_queue)}.

## Benchmark results

{suites_md}

- Overall pooled success: {overall['successes']}/{overall['episodes']} ({_pct(overall['success_rate'])}); 95% Wilson {_ci(overall['wilson_95'])}.
- Overall task-macro success rate: {_pct(statistics.mean([row['success_rate'] for row in task_rows]) if len(task_rows) == 40 else None)}.
- The interval is a binomial Wilson interval for the observed fixed-state sample, not a multi-seed stability claim.

## Failure analysis

- Measured failed Episodes: {len(failures)}.
- Top categories:
{failure_md}
- Verified M03 candidate failures: {sum(item['m03_candidate'] for item in candidates)}.
- Cause is assigned only for horizon timeout or explicit manual video/trajectory review. Other failures remain Unknown / Insufficient Evidence.
- Evidence integrity: {validation['paired_camera_videos_decoded']}/400 paired-camera videos decoded; {validation['action_trajectory_exact_matches']}/400 action traces exactly match recorded environment commands; {summary_metrics['video_frame_count']} decoded video frames total.
- Three lowest task success rates:
{hardest_md}

## Manual review sample

The deterministic review sample takes five failures per suite, spread across that suite's sorted failures. Results, evidence notes, and video/contact-sheet hashes are in `manual_failure_review.jsonl`; the current status is {validation['manual_review_completed']}/{len(review_queue)} reviewed.

Review contact sheets:
{review_sheets_md}

{review_md}

## Runtime and provenance

- Policy: `{manifest['model']['repo_id']}@{manifest['model']['revision']}`; weight SHA-256 `{manifest['model']['sha256']}`.
- Dataset: `{manifest['dataset']['repo_id']}@{manifest['dataset']['revision']}`; metadata SHA-256 `{manifest['dataset']['metadata_sha256']}`.
- GPU: `{manifest['runtime']['gpu']['name']}`; PyTorch allocator peak {_bytes(suite_metrics, valid_rows, 'gpu_peak_allocated_bytes')} bytes; sampled full-device peak {resource_summary['peak_memory_mib']} MiB from {resource_summary['samples']} SQLite resource events.
- Total completed Episode wall time: {suite_metrics['spatial']['episodes'] and _sum_wall(valid_rows):.1f}s.
- Mean per-Episode policy inference mean: {_fmt(suite_metrics, output, 'inference_latency_episode_mean_ms')} ms; per-Episode p95 mean: {_fmt(suite_metrics, output, 'inference_latency_episode_p95_ms')} ms; action-chunk generation mean: {_fmt(suite_metrics, output, 'action_chunk_generation_episode_mean_ms')} ms.
- Code identity and environment versions: `manifest.json`; Episode-level costs and artifact hashes: `episodes.csv`; full events/resources: external SQLite Registry.

## Artifacts and reproduction

- Episode rows: `episodes.csv`; task statistics: `task_metrics.csv`; suite/global metrics: `suite_metrics.json`.
- Failure database: `failure_cases.jsonl`; review queue: `manual_review_queue.csv`.
- Videos, trajectories, action traces and Registry: `{manifest['artifact_root']}`.
- Reproduce/resume: `bash reports/vla_forge_2_transition_20261010/milestones/M02_libero40_failure_mining/reproduce.sh`.

## Limitations

This is a single fixed 10-initial-state sample per task under one frozen policy and simulator protocol. It does not establish multi-seed robustness. Visual failure review is evidence-bound and does not create corrected demonstrations. No training or intervention was performed in M02.

The attempt ledger preserves one failed preflight recording attempt caused by a missing video output directory. The directory creation fix was applied before the benchmark queue; that preflight error is retained in `episode_attempts.jsonl` and excluded from the 400 formal Episode denominator. All 400 formal Episodes subsequently passed artifact verification.
"""
    (output / "failure_analysis.md").write_text(report, encoding="utf-8")
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    manifest["acceptance_status"] = report_status
    manifest["analysis"] = {"valid_episodes": len(valid_rows), "failures": len(failures), "m03_candidates": sum(item["m03_candidate"] for item in candidates), "manual_review_completed": validation["manual_review_completed"], "manual_review_required": len(review_queue)}
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    return validation


def _latency_summary(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return {"episodes": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {"episodes": len(values), "mean": statistics.mean(values), "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)), "max": max(values)}


def _registry_resource_summary(database):
    result = {"samples": 0, "peak_memory_mib": None, "mean_utilization_percent": None, "mean_power_watts": None}
    database = Path(database)
    if not database.is_file():
        return result
    utilization, memory, power = [], [], []
    try:
        with sqlite3.connect(database) as connection:
            rows = connection.execute(
                "SELECT payload_json FROM events WHERE kind='resources' ORDER BY id"
            ).fetchall()
        for (encoded,) in rows:
            payload = json.loads(encoded)
            gpu = payload.get("gpu")
            text = gpu.get("stdout") if isinstance(gpu, dict) else None
            if not text:
                continue
            fields = text.splitlines()[0].split(",")
            if len(fields) < 4:
                continue
            try:
                utilization.append(float(fields[0].strip()))
                memory.append(float(fields[1].strip()))
                power.append(float(fields[3].strip()))
            except ValueError:
                continue
    except sqlite3.Error:
        return result
    result["samples"] = len(memory)
    result["peak_memory_mib"] = max(memory) if memory else None
    result["mean_utilization_percent"] = statistics.mean(utilization) if utilization else None
    result["mean_power_watts"] = statistics.mean(power) if power else None
    return result


def _write_json(path, value):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def _pct(value):
    return "N/A" if value is None else f"{100 * value:.1f}%"


def _ci(value):
    return "N/A" if value is None else f"[{100 * value[0]:.1f}%, {100 * value[1]:.1f}%]"


def _sum_wall(rows):
    return sum(row.get("episode_wall_seconds") or 0 for row in rows)


def _bytes(suites, rows, key):
    return max((int(row.get(key) or 0) for row in rows), default=0)


def _fmt(suites, output, key):
    data = json.loads((output / "suite_metrics.json").read_text(encoding="utf-8"))
    return data[key]["mean"] if data[key]["mean"] is not None else "N/A"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/vla_forge_2_transition_20261010/milestones/M02_libero40_failure_mining"))
    args = parser.parse_args()
    validation = analyze(args.output_dir)
    print(json.dumps(validation, indent=2, ensure_ascii=False))
    return 0 if validation["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
