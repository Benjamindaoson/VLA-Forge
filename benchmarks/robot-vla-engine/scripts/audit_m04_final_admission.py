"""Read-only audit of frozen M04 evidence and future evaluation isolation."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.processor.env_processor import LiberoProcessorStep

from robot_vla.m04_admission import source_group, validate_supervision

ROOT = Path("/data/VLA-Forge")
E = ROOT / "Evidence/M04"
OUT = E / "finalization_20261011"
DS = E / "dataset/lerobot_v3_verified_r7"
M02 = ROOT / "Evidence/M02/source/M02_libero40/registry/runs/20261010T030315-982cacc8515f/episodes"


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + "\n")


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def main():
    OUT.mkdir(exist_ok=True)
    prov = json.loads((DS / "provenance.json").read_text())
    frozen_before = {str(p.relative_to(DS)): sha(p) for p in sorted(DS.rglob("*")) if p.is_file()}
    ledger = E / "artifacts/policy_attempts/attempts.jsonl"
    attempts = [x for x in rows(ledger) if "episode_id" in x and "suite" in x]
    bad_id = "m04-g09-fresh-brightness-115-first50-01"
    quarantine = [
        {"row_index_zero_based": i, "row": x}
        for i, x in enumerate(rows(ledger))
        if x.get("episode_id") == bad_id
    ]
    save(
        "quarantine_manifest.json",
        {
            "status": "ISOLATED_NOT_REPAIRED",
            "episode_id": bad_id,
            "original_ledger_sha256": sha(ledger),
            "rows": quarantine,
            "current_trace_sha256": sha(E / f"artifacts/policy_attempts/{bad_id}.json"),
            "evidence": "Two ledger rows reuse an episode ID and trace path with different timestamps and SHA256. Current bytes match the later row, not the earlier. Both are excluded from training; both attempts remain in historical cost accounting.",
            "historical_files_modified": False,
            "included_in_r7": any(
                s.get("source_policy_attempt") == bad_id for s in prov["sources"]
            ),
        },
    )
    table = pq.read_table(sorted(DS.glob("data/**/*.parquet"))).to_pandas()
    proc = LiberoProcessorStep()
    audit = []
    replay_seconds = 0.0
    for idx, (s, ep) in enumerate(zip(prov["sources"], prov["episodes_index"])):
        src = M02 / f"{s['suite']}_task{s['task_id']:02d}_episode{s['init_state_id']:02d}_attempt1"
        origin = json.loads((src / "episode.json").read_text())
        prefix = np.array(
            [
                x["env_step_argument"]
                for x in json.loads((src / "action_trace.json").read_text())["steps"][:100]
            ],
            dtype=np.float32,
        )
        p = Path(s["source_trajectory"])
        meta_path = p.with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        n = s["correction_actions"]
        with np.load(p) as z:
            actions = z["actions"].astype(np.float32)
            start = len(actions) - n
            aa = proc._quat2axisangle(
                torch.from_numpy(z["eef_quat"][start : start + n].astype(np.float32))
            ).numpy()
            states = np.concatenate(
                [z["eef_pos"][start : start + n], aa, z["gripper_qpos"][start : start + n]], axis=-1
            ).astype(np.float32)
            shape_ok = len(z["dual_camera_frames"]) == len(actions) + 1
        actual = actions[start:]
        chunk = table.iloc[ep["global_start"] : ep["global_end_inclusive"] + 1]
        da = np.stack(chunk["action"].to_numpy())
        ds = np.stack(chunk["observation.state"].to_numpy())
        checks = {
            "source_failed": origin["success"] is False,
            "source_task_init_match": origin["suite"] == s["suite"]
            and origin["task_id"] == s["task_id"]
            and origin["init_state_id"] == s["init_state_id"],
            "source_trace_hash": sha(src / "action_trace.json") == s["source_m02_trace_sha256"],
            "trajectory_hash": sha(p) == s["source_trajectory_sha256"],
            "metadata_hash": sha(meta_path) == s["source_metadata_sha256"],
            "correction_action_hash": hashlib.sha256(actual.tobytes()).hexdigest()
            == s["correction_action_sha256"],
            "source_prefix_exact": start == 100 and np.array_equal(actions[:100], prefix),
            "terminal_success": meta["success"] is True,
            "T_Tplus1_alignment": shape_ok,
            "dataset_action_exact": np.array_equal(actual, da),
            "dataset_state_exact": np.array_equal(states, ds),
            "success_label_last_only": np.flatnonzero(
                np.array(chunk["next.success"].tolist()).ravel()
            ).tolist()
            == [n - 1],
            "done_label_last_only": np.flatnonzero(
                np.array(chunk["next.done"].tolist()).ravel()
            ).tolist()
            == [n - 1],
            "not_quarantined": s.get("source_policy_attempt") != bad_id,
        }
        validate_supervision(actual, states)
        replays = []
        if s.get("source_policy_attempt"):
            run = s["source_policy_attempt"]
            r = next(x for x in attempts if x["episode_id"] == run)
            checks["policy_success"] = r["success"] is True
            checks["policy_npz_hash"] = sha(r["trajectory"]["path"]) == r["trajectory"]["sha256"]
            log = json.loads(Path(r["action_trace"]["path"]).read_text())
            logged_actions = np.array(
                [v["env_step_argument"] for v in log["steps"]], dtype=np.float32
            )
            checks["executed_action_log_exact"] = np.array_equal(actions, logged_actions)
            summary = json.loads((p.parent / "summary.json").read_text())
            checks["two_independent_replays"] = (
                len(summary["replicates"]) == 2 and summary["independent_replay_pass"]
            )
            for rep in summary["replicates"]:
                receipt = json.loads((p.parent / (rep["episode_id"] + ".json")).read_text())
                npz_path = Path(receipt["trajectory"]["path"])
                with np.load(npz_path) as z:
                    rep_actions = z["actions"].astype(np.float32)
                ok = (
                    receipt["success"] is True
                    and sha(npz_path) == receipt["trajectory"]["sha256"]
                    and np.array_equal(rep_actions, actions)
                )
                recovery = receipt["recovery"]
                ok = (
                    ok
                    and recovery["qpos_linf"] <= 1e-5
                    and recovery["qvel_linf"] <= 1e-4
                    and recovery["time_abs"] <= 1e-8
                )
                checks[rep["episode_id"]] = ok
                replay_seconds += receipt["replay_wall_seconds"]
                replays.append(
                    {"episode_id": rep["episode_id"], "pass": bool(ok), "recovery": recovery}
                )
        audit.append(
            {
                "episode_index": idx,
                "episode_id": s["episode_id"],
                "source_group": source_group(s["suite"], s["task_id"], s["init_state_id"]),
                "checks": {k: bool(v) for k, v in checks.items()},
                "pass": all(checks.values()),
                "action_abs_max": float(np.abs(actual).max()),
                "replays": replays,
            }
        )
    assert frozen_before == {
        str(p.relative_to(DS)): sha(p) for p in DS.rglob("*") if p.is_file()
    }, "frozen data changed"
    save(
        "registry_audit.json",
        {
            "pass": all(a["pass"] for a in audit),
            "original_dataset_unchanged": True,
            "episodes": audit,
            "old_image_validation_receipt_sha256": sha(DS / "independent_validation.json"),
            "fresh_corrections": 11,
            "m03_derived": 1,
            "quarantined_rows": len(quarantine),
        },
    )
    manifest = json.loads((E / "manifest.json").read_text())
    groups = set(manifest["source_evidence"]["m03_development_groups"])
    extra = OUT / "cross_task/policy_attempts/attempts.jsonl"
    cross_rows = rows(extra) if extra.exists() else []
    for x in attempts + cross_rows:
        groups.add(source_group(x["suite"], x["task_id"], x["init_state_id"]))
    for p in (E / "artifacts/transfer_candidates").glob("attempts.jsonl"):
        for x in rows(p):
            if all(k in x for k in ["suite", "task_id", "init_state_id"]):
                groups.add(source_group(x["suite"], x["task_id"], x["init_state_id"]))
    contaminated = sorted(groups & set(manifest["source_evidence"]["m06_reserved_groups"]))
    save(
        "source_groups.json",
        {
            "all_development_groups": sorted(groups),
            "train_groups": sorted({a["source_group"] for a in audit}),
            "retired_previous_M06_groups": contaminated,
            "policy": "All descendants of every development source, including failed and quarantined attempts, inherit development-only status.",
        },
    )
    from libero.libero import benchmark, get_libero_path

    future = []
    state_hashes_dev = set()
    for suite_short, suite in [
        ("spatial", "libero_spatial"),
        ("object", "libero_object"),
        ("goal", "libero_goal"),
        ("long", "libero_10"),
    ]:
        bench = benchmark.get_benchmark_dict()[suite]()
        for task_id in range(10):
            task = bench.get_task(task_id)
            state_file = (
                Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            )
            bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            states = bench.get_task_init_states(task_id)
            assert len(states) >= 50
            for old_idx in range(10):
                state_hashes_dev.add(
                    hashlib.sha256(np.asarray(states[old_idx]).tobytes()).hexdigest()
                )
            for init in range(40, 50):
                future.append(
                    {
                        "suite": suite_short,
                        "task_id": task_id,
                        "task_name": task.name,
                        "init_state_id": init,
                        "seed": 95000 + len(future),
                        "source_group": source_group(suite_short, task_id, init),
                        "init_states_file": str(state_file),
                        "init_states_file_sha256": sha(state_file),
                        "state_sha256": hashlib.sha256(
                            np.asarray(states[init]).tobytes()
                        ).hexdigest(),
                        "bddl_sha256": sha(bddl_file),
                    }
                )
    overlap_groups = sorted({x["source_group"] for x in future} & groups)
    overlap_states = [x["source_group"] for x in future if x["state_sha256"] in state_hashes_dev]
    save(
        "independent_test_freeze.json",
        {
            "status": "FROZEN_NOT_EXECUTED",
            "rows": future,
            "development_group_overlap": overlap_groups,
            "old_M02_state_bytes_overlap": overlap_states,
            "claim_boundary": "Independent of this project's M02/M03/M04 development states; foundation model pretraining exposure is unknown. All derived scenes inherit parent group. No M05 training or heldout rollouts executed.",
        },
    )
    save(
        "training_freeze.json",
        {
            "dataset": "lerobot_v3_verified_r7",
            "episodes": 12,
            "transitions": 956,
            "files": frozen_before,
            "provenance_sha256": sha(DS / "provenance.json"),
            "source_groups_sha256": sha(OUT / "source_groups.json"),
            "independent_test_sha256": sha(OUT / "independent_test_freeze.json"),
            "excluded_episode_ids": [bad_id],
            "observation_semantics": "Canonical independently replayed camera frames and 8D pre-action state; corrections were proposed under declared brightness interventions. These are successful action supervision pairs, not unmodified-policy on-policy samples.",
            "normalization": "Fit training statistics only to the frozen development set; never fit to independent_test_freeze.",
            "task_coverage": 1,
            "source_state_coverage": 2,
        },
    )
    model_root = ROOT / "Assets/models/smolvla_libero"
    model_files = {
        p.name: sha(p)
        for p in model_root.iterdir()
        if p.is_file() and p.suffix in [".json", ".safetensors"]
    }
    assert model_files["model.safetensors"] == manifest["policy"]["checkpoint_sha256"]
    from safetensors import safe_open

    with safe_open(
        str(model_root / "policy_preprocessor_step_5_normalizer_processor.safetensors"),
        framework="np",
    ) as tensors:
        normalizer_shapes = {k: list(tensors.get_tensor(k).shape) for k in tensors.keys()}
    frozen = json.loads((OUT / "training_freeze.json").read_text())
    frozen["policy_identity"] = manifest["policy"]
    frozen["model_and_processor_files"] = model_files
    frozen["normalizer_tensor_shapes"] = normalizer_shapes
    frozen["training_adapter_condition"] = (
        "Preserve 8D LIBERO pose plus gripper state. The pinned feature schema declares 6D but its actual fitted normalizer is 8D; the verified runtime retains 8D and SmolVLA pads to max_state_dim. A future trainer must use this explicit contract and verify its batch processor; do not truncate to 6D or assume generic trainer defaults work. No training-forward/backward claim is made here."
    )
    save("training_freeze.json", frozen)
    costs = []
    for i, x in enumerate(attempts):
        costs.append(
            {
                "scope": "historical",
                "row_index": i,
                "episode_id": x["episode_id"],
                "success": x["success"],
                "policy_loop_seconds": x.get("gpu_hours_allocated", 0) * 3600,
                "model_load_seconds_reported": x.get("model_load_seconds"),
                "reset_seconds_reported": x.get("reset_seconds"),
                "quarantined": x["episode_id"] == bad_id,
            }
        )
    with (OUT / "cost_per_attempt.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=costs[0].keys())
        writer.writeheader()
        writer.writerows(costs)
    policy_seconds = sum(x["policy_loop_seconds"] for x in costs)

    # Repeated load/reset timings refer to shared model/env setup; value-based de-duplication is only a bound, not an exact process identity.
    def bounds(key):
        vals = [x[key] for x in costs if x[key] is not None]
        return {
            "reported_rows": len(vals),
            "unique_reported_duration_sum_seconds": sum(set(vals)),
            "per_row_sum_seconds": sum(vals),
            "exact_total": None,
            "reason": "Historical setup timing repeats across conditions without a unique process-load identity.",
        }

    controller = (
        json.loads((OUT / "cross_task/controller.json").read_text())
        if (OUT / "cross_task/controller.json").exists()
        else {}
    )
    lower = policy_seconds + replay_seconds + prov["write_and_encode_seconds"]
    cost = {
        "currency_billed": None,
        "rental_invoice_available": False,
        "hourly_rate": None,
        "historical_policy_search_seconds": policy_seconds,
        "historical_success_policy_seconds": sum(
            x["policy_loop_seconds"] for x in costs if x["success"]
        ),
        "historical_failed_policy_seconds": sum(
            x["policy_loop_seconds"] for x in costs if not x["success"]
        ),
        "historical_model_loading": bounds("model_load_seconds_reported"),
        "historical_reset": bounds("reset_seconds_reported"),
        "independent_replay_correction_segments_seconds_22_replays": replay_seconds,
        "replay_reset_prefix_model_initialization_seconds": None,
        "dataset_write_and_encode_seconds_r7": prov["write_and_encode_seconds"],
        "historical_raw_video_and_trajectory_encode_seconds": None,
        "demo_encoding_seconds": None,
        "idle_allocated_seconds": None,
        "whole_production_gpu_hours": None,
        "recorded_nonoverlap_wall_time_lower_bound_seconds": lower,
        "recorded_wall_time_lower_bound_hours_per_11_fresh_corrections": lower / 3600 / 11,
        "full_cost_per_trusted_fresh_correction": None,
        "estimate_formula": "(measured whole production allocation hours + missing setup/encode/idle hours) * actual rate / 11; no rate or invoice available",
        "cross_task_controller": controller,
        "boundary": "gpu_hours_allocated historically measures policy-loop wall time, not GPU-active time, end-to-end production time, or invoiced cost. Lower bound excludes missing phases and M03 inherited production. Dataset write is CPU encoding; do not call summed phase wall times GPU-active hours.",
    }
    transfer_rows = rows(E / "artifacts/transfer_candidates/attempts.jsonl")
    first_replays = [
        json.loads(p.read_text()) for p in (E / "artifacts/first_correction").glob("*replay-*.json")
    ]
    cost["additional_historical_failed_transfer_attempts"] = [
        {
            "episode_id": x["episode_id"],
            "success": x["success"],
            "simulator_wall_seconds": x["simulator_wall_seconds"],
        }
        for x in transfer_rows
    ]
    cost["inherited_correction_M04_replay_recorded_seconds"] = [
        {
            "episode_id": x["episode_id"],
            "reset_seconds": x.get("reset_seconds"),
            "execution_seconds": x.get("execution_seconds"),
        }
        for x in first_replays
    ]
    cost["optimization_benchmarks_and_engineering_failures_full_cost_seconds"] = None
    cost["all_known_stage_wall_time_lower_bound_seconds"] = (
        lower
        + sum(x["simulator_wall_seconds"] for x in transfer_rows)
        + sum(x.get("reset_seconds", 0) + x.get("execution_seconds", 0) for x in first_replays)
    )
    save("cost_breakdown.json", cost)
    admission = {
        "engineering_gate": all(a["pass"] for a in audit)
        and not overlap_groups
        and not overlap_states,
        "status": "CONDITIONAL_ENGINEERING_ADMISSION_ONLY",
        "formal_generalization_ready": False,
        "training_executed": False,
        "episodes": 12,
        "transitions": 956,
        "fresh_corrections": 11,
        "source_tasks": 1,
        "source_states": 2,
        "quarantine_excluded": not any(
            s.get("source_policy_attempt") == bad_id for s in prov["sources"]
        ),
        "prior_test_contamination_groups": contaminated,
        "replacement_test_count": len(future),
        "remaining_conditions": [
            "Use replacement frozen test conditions; retire contaminated old groups.",
            "Respect noncommercial/restricted visual asset provenance; no blanket commercial/public multimedia clearance.",
            "One task and two source states support a bounded feasibility pilot only; no cross-task improvement claim.",
            "Define a matched frozen baseline, training budget and evaluation protocol before a separately authorized M05 run.",
        ],
        "legal_action_supervision": "7D finite bounded relative commands exactly match executed actions; 8D pre-action state and terminal labels verified. License clearance is a separate gate.",
    }
    save("m05_admission.json", admission)
    print(
        json.dumps(
            {
                "audit_pass": all(a["pass"] for a in audit),
                "episodes": len(audit),
                "quarantined": len(quarantine),
                "contaminated": contaminated,
                "test_rows": len(future),
                "state_overlap": overlap_states,
                "cost_lower_bound_seconds": lower,
                "admission_gate": admission["engineering_gate"],
            }
        )
    )


if __name__ == "__main__":
    main()
