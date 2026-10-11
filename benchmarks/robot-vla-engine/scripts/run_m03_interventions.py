"""Matched, evidence-preserving replay/intervention experiment on frozen M02 assets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from robot_vla.m01 import fit_libero_state_to_checkpoint
from robot_vla.m02 import M02_SUITES
from robot_vla.m03 import action_delta, compare_sim_state, transform_observation
from robot_vla.observations import libero_batch
from robot_vla.preflight import create_libero_env
from robot_vla.schema import file_hash


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sim(env):
    node = env
    for _ in range(4):
        if hasattr(node, "sim") and hasattr(node.sim, "data"):
            return node.sim
        node = getattr(node, "_env", getattr(node, "env", None))
        if node is None:
            break
    raise RuntimeError("could not locate official LIBERO MuJoCo MjSim")


def _physical(env):
    sim = _sim(env)
    return {
        "qpos": sim.data.qpos.copy(),
        "qvel": sim.data.qvel.copy(),
        "time": float(sim.data.time),
        "ncon": int(sim.data.ncon),
    }


def _find_task(rows):
    # Frozen deterministic eligibility: M02-confirmed failures, trajectory + action
    # trace exist, and one source state per suite/task/init-state group. Prioritize
    # requested tasks; timeouts are candidates only, never treated as root causes.
    ranked = []
    preferred = {("spatial", 5): 0, ("long", 0): 1, ("long", 4): 2}
    for row in rows:
        if row.get("success") is not False or row.get("m03_candidate") is not True:
            continue
        evidence = row.get("evidence", {})
        if not all(
            Path(evidence.get(k, "")).is_file() for k in ("trajectory_path", "action_trace_path")
        ):
            continue
        key = (row["suite"], int(row["task_id"]), int(row["init_state_id"]))
        if key in {(x["suite"], int(x["task_id"]), int(x["init_state_id"])) for x in ranked}:
            continue
        category = (row.get("failure_classification") or {}).get("category", "Unknown")
        priority = (
            0
            if (row["suite"], int(row["task_id"])) in preferred
            else (1 if category == "Placement / Goal Failure" else 2)
        )
        ranked.append(row)
        row["_priority"] = priority
    ranked.sort(
        key=lambda r: (
            r["_priority"],
            preferred.get((r["suite"], int(r["task_id"])), 9),
            r["suite"],
            int(r["task_id"]),
            int(r["episode_id"]),
        )
    )
    return ranked


def _select_groups(rows, limit):
    """Freeze a mix across the requested source task families and failure types."""
    preferred = [("spatial", 5), ("long", 0), ("long", 4)]
    selected = []
    for task in preferred:
        selected.extend([row for row in rows if (row["suite"], int(row["task_id"])) == task][:2])
    placement = [
        row
        for row in rows
        if (row.get("failure_classification") or {}).get("category") == "Placement / Goal Failure"
        and (row["suite"], int(row["task_id"])) not in preferred
    ]
    selected.extend(placement[:4])
    used = {(r["suite"], int(r["task_id"]), int(r["init_state_id"])) for r in selected}
    for row in rows:
        key = (row["suite"], int(row["task_id"]), int(row["init_state_id"]))
        if key not in used:
            selected.append(row)
            used.add(key)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _record_hashes(paths):
    return {str(p): file_hash(p) for p in paths}


def run(args):
    import fcntl

    import av
    import torch
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark

    root, out = args.root.resolve(), args.output.resolve()
    sys.path.insert(0, str(root / "scripts"))
    from run_m02_libero40 import POLICY_ID, POLICY_REVISION, _load_policy

    out.mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(exist_ok=True)
    (out / "episodes").mkdir(exist_ok=True)
    lock_handle = (out / ".m03.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another M03 runner holds the experiment lock") from exc
    m02 = root / "reports/vla_forge_2_transition_20261010/milestones/M02_libero40_failure_mining"
    baseline = json.loads(
        (
            root
            / "reports/vla_forge_2_transition_20261010/milestones/M01_real_libero/manifest.json"
        ).read_text()
    )
    m02manifest = json.loads((m02 / "manifest.json").read_text())
    frozen_model_sha = next(
        item["sha256"]
        for item in baseline["model"]["files"]
        if Path(item["path"]).name == "model.safetensors"
    )
    if m02manifest.get("model", {}).get("sha256") != frozen_model_sha:
        raise RuntimeError("M02 model hash does not match frozen M01 policy")
    with (m02 / "failure_cases.jsonl").open() as f:
        failed = [json.loads(line) for line in f if line.strip()]
    eligible_candidates = _find_task(failed)
    candidates = eligible_candidates
    if len(candidates) < args.limit:
        raise RuntimeError(
            f"only {len(candidates)} eligible source groups for requested {args.limit}"
        )
    candidates = _select_groups(candidates, args.limit)
    protocol = json.loads((root / "configs/data_protocol_v1.json").read_text())

    # Reuse exact M02 model + processor factory; no new downloads or checkpoint edits.
    class A:
        pass

    a = A()
    a.seed_base = 731
    a.checkpoint = args.checkpoint
    a.vlm_root = args.vlm_root
    _, raw_policy, measured, pre, post, config, state_dim, normalizer_dim, rename, gpu, features = (
        _load_policy(a, baseline)
    )
    policy = measured
    names = benchmark.get_benchmark_dict()
    suites = {
        alias: names[name](task_order_index=protocol["task_order_index"])
        for alias, name in M02_SUITES.items()
    }
    setting = protocol["evaluation"]
    run_manifest = {
        "status": "RUNNING",
        "policy_repo": POLICY_ID,
        "policy_revision": POLICY_REVISION,
        "checkpoint_sha256": frozen_model_sha,
        "m02_manifest_sha256": file_hash(m02 / "manifest.json"),
        "data_metadata_sha256": file_hash(root / "data/libero/meta/info.json"),
        "policy_config": features,
        "gpu": gpu,
        "tolerances": {"qpos_qvel_linf": 1e-8, "sim_time_abs": 1e-10},
        "prefix_rule": "first floor((horizon*0.4)/50)*50 env actions from source M02 action_trace; fixed before treatment",
        "conditions": {
            "baseline": "no observation modification",
            "sham": "deep-copy no-op observation",
            "brightness": "1.15 RGB gain on both cameras for first 50 post-checkpoint observations",
            "occlusion": "zero central 12.5% square in image2 for first 50 post-checkpoint observations",
        },
        "candidate_rule": "M02 failures with m03_candidate=true and hash-addressable action+trajectory files; unique suite/task/init-state; priority spatial task5, long task0/4, reviewed placement failures, then canonical order; timeout not interpreted as cause",
        "groups": [],
        "development_source_groups": [
            f"{r['suite']}:{r['task_id']}:{r['init_state_id']}" for r in candidates
        ],
        "m06_reserved_source_groups": [
            f"{r['suite']}:{r['task_id']}:{r['init_state_id']}"
            for r in eligible_candidates
            if f"{r['suite']}:{r['task_id']}:{r['init_state_id']}"
            not in {f"{x['suite']}:{x['task_id']}:{x['init_state_id']}" for x in candidates}
        ],
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
        },
    }
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        prior_manifest = json.loads(manifest_path.read_text())
        if (
            prior_manifest.get("checkpoint_sha256") != frozen_model_sha
            or prior_manifest.get("m02_manifest_sha256") != run_manifest["m02_manifest_sha256"]
        ):
            raise RuntimeError("existing M03 output belongs to a different frozen baseline")
        run_manifest["groups"] = prior_manifest.get("groups", [])
    _write_json(manifest_path, run_manifest)
    fields = [
        "group_id",
        "suite",
        "task_id",
        "task_name",
        "instruction",
        "source_episode_id",
        "init_state_id",
        "seed",
        "prefix_steps",
        "condition",
        "status",
        "success",
        "termination_reason",
        "followup_steps",
        "state_match_qpos_linf",
        "state_match_qvel_linf",
        "state_match_time_abs",
        "action_delta_mean_abs_vs_baseline",
        "trajectory_delta_mean_l2_vs_baseline",
        "inference_mean_ms",
        "wall_seconds",
        "video_path",
        "video_sha256",
        "trajectory_path",
        "trajectory_sha256",
        "source_action_trace_sha256",
        "source_trajectory_sha256",
    ]
    csvpath = out / "paired_interventions.csv"
    rows_done = []
    if csvpath.exists():
        with csvpath.open() as f:
            rows_done = list(csv.DictReader(f))
    completed = set()
    for row in rows_done:
        result_path = out / "episodes" / f"{row.get('group_id')}_{row.get('condition')}.json"
        if (
            row.get("status") == "complete"
            and row.get("video_path")
            and Path(row["video_path"]).is_file()
            and file_hash(row["video_path"]) == row.get("video_sha256")
            and Path(row.get("trajectory_path", "")).is_file()
            and file_hash(row["trajectory_path"]) == row.get("trajectory_sha256")
            and result_path.is_file()
        ):
            completed.add((row["group_id"], row["condition"]))
    with csvpath.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_done)
        f.flush()
        for gi, source in enumerate(candidates):
            suite, task_id, init_id = (
                source["suite"],
                int(source["task_id"]),
                int(source["init_state_id"]),
            )
            task = suites[suite].get_task(task_id)
            horizon = int(source["horizon"])
            prefix = max(50, int((horizon * 0.4) // 50) * 50)
            budget = min(100, horizon - prefix)
            evidence = source["evidence"]
            trace = json.loads(Path(evidence["action_trace_path"]).read_text())["steps"]
            source_actions = [
                np.asarray(x["env_step_argument"], dtype=np.float32) for x in trace[:prefix]
            ]
            if len(source_actions) != prefix:
                raise RuntimeError("source action prefix is incomplete")
            source_state = np.load(evidence["trajectory_path"], allow_pickle=False)["states"][
                prefix
            ]
            group = f"m03-g{gi + 1:02d}-{suite}-t{task_id:02d}-init{init_id:02d}"
            cp_states = []
            outcomes = {}
            seed = int(source["seed"])
            for condition in ("baseline", "sham", "brightness" if gi % 2 == 0 else "occlusion"):
                if (group, condition) in completed:
                    found = json.loads((out / "episodes" / f"{group}_{condition}.json").read_text())
                    found["checkpoint_state"]["qpos"] = np.asarray(
                        found["checkpoint_state"]["qpos"]
                    )
                    found["checkpoint_state"]["qvel"] = np.asarray(
                        found["checkpoint_state"]["qvel"]
                    )
                    cp_states.append(found["checkpoint_state"])
                    outcomes[condition] = found
                    continue
                env = create_libero_env(
                    LiberoEnv,
                    suites[suite],
                    task_id,
                    M02_SUITES[suite],
                    protocol,
                    init_id,
                    256,
                    horizon,
                )
                torch.manual_seed(seed)
                np.random.seed(seed)
                obs, _ = env.reset(seed=seed)
                for act in source_actions:
                    obs, *_ = env.step(act)
                obs_source = libero_batch(obs)["observation.state"][0].numpy()
                error = float(np.max(np.abs(obs_source - source_state)))
                physical = _physical(env)
                cp_states.append(physical)
                policy.reset()
                actions = []
                states = [obs_source.copy()]
                sim_states = [physical]
                infer = []
                success = False
                reason = "budget_exhausted"
                video = out / "videos" / f"{group}_{condition}.mp4"
                traj = out / "episodes" / f"{group}_{condition}.npz"
                if video.exists() or traj.exists():
                    orphan_dir = out / "episodes" / "unregistered_attempts"
                    orphan_dir.mkdir(parents=True, exist_ok=True)
                    for artifact in (video, traj):
                        if artifact.exists():
                            artifact.replace(orphan_dir / f"{int(time.time())}_{artifact.name}")
                container = av.open(str(video), "w")
                stream = None
                started = time.perf_counter()
                treatment_frames = 0
                for step in range(budget):
                    act_obs = (
                        transform_observation(obs, condition, camera="image2")
                        if condition in ("brightness", "occlusion") and step < 50
                        else transform_observation(obs, "sham")
                    )
                    if condition in ("brightness", "occlusion") and step < 50:
                        treatment_frames += 1
                    pix = obs["pixels"]
                    frame = np.concatenate(
                        (np.flip(pix["image"], axis=(0, 1)), np.flip(pix["image2"], axis=(0, 1))),
                        axis=1,
                    ).copy()
                    if stream is None:
                        stream = container.add_stream("libx264", rate=int(setting["control_freq"]))
                        stream.width = frame.shape[1]
                        stream.height = frame.shape[0]
                        stream.pix_fmt = "yuv420p"
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                        container.mux(packet)
                    batch = libero_batch(act_obs)
                    batch["observation.state"], _ = fit_libero_state_to_checkpoint(
                        batch["observation.state"],
                        expected_dim=state_dim,
                        normalizer_dim=normalizer_dim,
                    )
                    batch["task"] = [task.language]
                    t = time.perf_counter()
                    raw = policy.select_action(pre(batch))
                    torch.cuda.synchronize()
                    infer.append((time.perf_counter() - t) * 1000)
                    command = post(raw).detach().float().cpu().numpy()[0]
                    command = np.clip(command, -1, 1).astype(np.float32)
                    obs, _, terminated, truncated, info = env.step(command)
                    actions.append(command)
                    states.append(libero_batch(obs)["observation.state"][0].numpy())
                    sim_states.append(_physical(env))
                    if "is_success" not in info:
                        raise RuntimeError("LIBERO step omitted measured is_success")
                    success = bool(info["is_success"])
                    if success or terminated or truncated:
                        reason = (
                            "success" if success else ("truncated" if truncated else "terminated")
                        )
                        break
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                env.close()
                elapsed = time.perf_counter() - started
                np.savez_compressed(
                    traj,
                    actions=np.asarray(actions),
                    states=np.asarray(states),
                    sim_qpos=np.stack([x["qpos"] for x in sim_states]),
                    sim_qvel=np.stack([x["qvel"] for x in sim_states]),
                    sim_time=np.asarray([x["time"] for x in sim_states]),
                    sim_ncon=np.asarray([x["ncon"] for x in sim_states]),
                )
                result = {
                    "group_id": group,
                    "suite": suite,
                    "task_id": task_id,
                    "task_name": task.name,
                    "instruction": task.language,
                    "source_episode_id": source["episode_id"],
                    "init_state_id": init_id,
                    "seed": seed,
                    "prefix_steps": prefix,
                    "condition": condition,
                    "status": "complete",
                    "success": success,
                    "termination_reason": reason,
                    "followup_steps": len(actions),
                    "source_state_linf": error,
                    "checkpoint_state": {
                        "qpos": physical["qpos"].tolist(),
                        "qvel": physical["qvel"].tolist(),
                        "time": physical["time"],
                        "ncon": physical["ncon"],
                    },
                    "treatment_observation_count": treatment_frames,
                    "mean_action": np.asarray(actions).mean(axis=0).tolist(),
                    "trajectory_path": str(traj),
                    "trajectory_sha256": file_hash(traj),
                    "video_path": str(video),
                    "video_sha256": file_hash(video),
                    "source_action_trace_sha256": evidence["action_trace_sha256"],
                    "source_trajectory_sha256": evidence["trajectory_sha256"],
                    "inference_mean_ms": float(np.mean(infer)),
                    "wall_seconds": elapsed,
                    "mujoco_snapshots": {
                        "qpos": physical["qpos"].tolist(),
                        "qvel": physical["qvel"].tolist(),
                        "time": physical["time"],
                        "ncon": physical["ncon"],
                    },
                }
                if condition != "baseline":
                    baseline_path = out / "episodes" / f"{group}_baseline.npz"
                    if baseline_path.is_file():
                        with np.load(baseline_path, allow_pickle=False) as base_npz:
                            base_actions = base_npz["actions"]
                            action_n = min(len(base_actions), len(actions))
                            n = min(len(base_npz["states"]), len(states))
                            result["action_delta_vs_baseline"] = action_delta(
                                base_actions[:action_n], np.asarray(actions)[:action_n]
                            )
                            result["action_comparison_steps"] = action_n
                            result["action_delta_mean_abs_vs_baseline"] = result[
                                "action_delta_vs_baseline"
                            ]["mean_abs"]
                            result["trajectory_delta_mean_l2_vs_baseline"] = float(
                                np.linalg.norm(
                                    base_npz["states"][:n] - np.asarray(states)[:n], axis=1
                                ).mean()
                            )
                            result["trajectory_comparison_steps"] = n - 1
                resultpath = out / "episodes" / f"{group}_{condition}.json"
                _write_json(resultpath, result)
                # Persist each condition immediately for crash-safe resume.
                row = {k: result.get(k, "") for k in fields}
                outcomes[condition] = result
                rows_done = [
                    r
                    for r in rows_done
                    if not (r.get("group_id") == group and r.get("condition") == condition)
                ]
                rows_done.append(row)
                f.seek(0)
                f.truncate()
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows_done)
                f.flush()
                os.fsync(f.fileno())
                print(
                    json.dumps(
                        {
                            "group": group,
                            "condition": condition,
                            "success": success,
                            "steps": len(actions),
                            "wall_s": round(elapsed, 2),
                            "source_state_linf": error,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if len(cp_states) == 3:
                ref = {k: cp_states[0][k] for k in ("qpos", "qvel", "time")}
                group_match = [
                    compare_sim_state(
                        ref, {k: st[k] for k in ("qpos", "qvel", "time")}, atol=1e-8
                    ).__dict__
                    for st in cp_states[1:]
                ]
            else:
                group_match = []
            group_row = {
                "group_id": group,
                "source_suite": suite,
                "source_task_id": task_id,
                "source_episode_id": source["episode_id"],
                "init_state_id": init_id,
                "seed": seed,
                "prefix_steps": prefix,
                "followup_budget": budget,
                "recovery_observation_state_linf_vs_m02": max(
                    float(item["source_state_linf"]) for item in outcomes.values()
                ),
                "physical_state_pairwise_match": group_match,
                "condition_results": _jsonable(outcomes),
                "eligible_for_matched_comparison": bool(
                    group_match and all(x["within_tolerance"] for x in group_match)
                ),
            }
            if group_match:
                for row in rows_done:
                    if row.get("group_id") == group:
                        row["state_match_qpos_linf"] = max(x["qpos_linf"] for x in group_match)
                        row["state_match_qvel_linf"] = max(x["qvel_linf"] for x in group_match)
                        row["state_match_time_abs"] = max(x["time_abs"] for x in group_match)
                with csvpath.open("w", newline="", encoding="utf-8") as metrics_file:
                    writer = csv.DictWriter(metrics_file, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows_done)
            run_manifest["groups"] = [
                item for item in run_manifest["groups"] if item.get("group_id") != group
            ]
            run_manifest["groups"].append(group_row)
            _write_json(out / "manifest.json", run_manifest)
    run_manifest["status"] = "COMPLETED"
    _write_json(out / "manifest.json", run_manifest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/models/smolvla_libero/31d453f7edd78c839a8bbc39744a292686daf0de"),
    )
    p.add_argument(
        "--vlm-root",
        type=Path,
        default=Path("/data/models/smolvlm_full/7b375e1b73b11138ff12fe22c8f2822d8fe03467"),
    )
    p.add_argument("--limit", type=int, default=10)
    run(p.parse_args())


if __name__ == "__main__":
    main()
