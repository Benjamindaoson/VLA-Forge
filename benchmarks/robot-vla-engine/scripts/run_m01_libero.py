"""Run the independently pinned official SmolVLA checkpoint on LIBERO/MuJoCo."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from robot_vla.m01 import (
    M01_SUITES,
    episode_success_metrics,
    fit_libero_state_to_checkpoint,
    validate_m01_coverage,
    validate_smoke_coverage,
)
from robot_vla.schema import file_hash

POLICY_ID = "lerobot/smolvla_libero"
POLICY_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
CSV_FIELDS = [
    "phase",
    "suite",
    "benchmark_suite",
    "task_id",
    "task_name",
    "instruction",
    "episode_id",
    "seed",
    "attempt",
    "status",
    "success",
    "total_action_steps",
    "action_chunk_size",
    "action_execution_steps",
    "control_mode",
    "source_state_dim",
    "model_state_dim",
    "state_projection",
    "input_shapes",
    "policy_inference_mean_ms",
    "policy_inference_p95_ms",
    "chunk_inference_mean_ms",
    "episode_wall_seconds",
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "video_path",
    "video_sha256",
    "trajectory_path",
    "metadata_path",
    "error",
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _json_write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def _load_rows(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["success"] == "True":
            row["success"] = True
        elif row["success"] == "False":
            row["success"] = False
        else:
            row["success"] = None
        for key in ("task_id", "episode_id", "seed", "attempt", "total_action_steps", "action_chunk_size", "action_execution_steps", "source_state_dim", "model_state_dim", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
            if row.get(key) not in (None, ""):
                row[key] = int(row[key])
        for key in ("policy_inference_mean_ms", "policy_inference_p95_ms", "chunk_inference_mean_ms", "episode_wall_seconds"):
            if row.get(key) not in (None, ""):
                row[key] = float(row[key])
    return rows


def _write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _key(row):
    return (row["phase"], row["suite"], str(row["task_id"]), str(row["episode_id"]))


def _upsert(rows, row):
    identity = _key(row)
    return [existing for existing in rows if _key(existing) != identity] + [row]


def _git(root):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        revision, status = None, ["git metadata unavailable"]
    return {"execution_checkout_revision": revision, "execution_worktree_status": status}


def _package_versions():
    names = ["lerobot", "torch", "transformers", "huggingface-hub", "hf-libero", "mujoco", "robosuite", "av", "safetensors"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _checkpoint_files(path: Path):
    records = []
    for file in sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts):
        records.append({"path": str(file), "size_bytes": file.stat().st_size, "sha256": file_hash(file)})
    if not records or not any(Path(row["path"]).name == "model.safetensors" for row in records):
        raise FileNotFoundError(f"model.safetensors missing from {path}")
    return records


def _episode_metrics(rows):
    groups = {}
    task_timing = {}
    for phase in ("smoke", "formal"):
        rows_phase = [row for row in rows if row.get("phase") == phase]
        if phase == "formal":
            groups[phase] = {
                suite: episode_success_metrics([r for r in rows_phase if r["suite"] == suite])
                for suite in M01_SUITES
            }
            for suite in M01_SUITES:
                selected = [r for r in rows_phase if r["suite"] == suite and r.get("status") == "complete"]
                latencies = [float(r["policy_inference_mean_ms"]) for r in selected if r.get("policy_inference_mean_ms") not in (None, "")]
                p95_values = [float(r["policy_inference_p95_ms"]) for r in selected if r.get("policy_inference_p95_ms") not in (None, "")]
                task_timing[suite] = {
                    "episode_wall_seconds_mean": sum(float(r.get("episode_wall_seconds") or 0) for r in selected) / len(selected) if selected else None,
                    "policy_inference_mean_ms_across_episodes": sum(latencies) / len(latencies) if latencies else None,
                    "policy_inference_p95_ms_mean_across_episodes": sum(p95_values) / len(p95_values) if p95_values else None,
                }
        else:
            groups[phase] = {"spatial": episode_success_metrics(rows_phase)}
    complete_formal = False
    try:
        validate_m01_coverage(rows)
        complete_formal = True
    except ValueError:
        pass
    smoke_ready = False
    try:
        validate_smoke_coverage(rows)
        smoke_ready = True
    except ValueError:
        pass
    video_rows = [row for row in rows if row.get("status") == "complete" and row.get("video_path")]
    verified_video_rows = [
        row for row in video_rows
        if Path(row["video_path"]).is_file()
        and row.get("video_sha256")
        and file_hash(row["video_path"]) == row["video_sha256"]
    ]
    video_verified = bool(verified_video_rows)
    return {
        "milestone": "M01_real_libero",
        "status": "PASS" if complete_formal and smoke_ready and video_verified else ("SMOKE_PASS" if smoke_ready and video_verified else "INCOMPLETE"),
        "smoke_gate_passed": smoke_ready,
        "formal_coverage_complete": complete_formal,
        "at_least_one_video_sha256_verified": video_verified,
        "verified_video_count": len(verified_video_rows),
        "formal": groups["formal"],
        "smoke": groups["smoke"],
        "formal_task_timing": task_timing,
        "formal_completed_episodes": sum(v["completed_episodes"] for v in groups["formal"].values()),
        "formal_expected_episodes": 40,
        "smoke_completed_episodes": groups["smoke"]["spatial"]["completed_episodes"],
        "smoke_expected_episodes": 3,
        "runtime_error_rows": sum(row.get("status") == "error" for row in rows),
        "episode_wall_seconds_sum": sum(float(row.get("episode_wall_seconds") or 0) for row in rows),
        "gpu_peak_allocated_bytes": max((int(row.get("gpu_peak_allocated_bytes") or 0) for row in rows), default=0),
        "gpu_peak_reserved_bytes": max((int(row.get("gpu_peak_reserved_bytes") or 0) for row in rows), default=0),
    }


def _gpu_peak_system_memory_mib(registry_root: Path):
    import sqlite3

    database = registry_root / "registry.sqlite"
    if not database.is_file():
        return None
    values = []
    with sqlite3.connect(database) as connection:
        records = connection.execute("SELECT payload_json FROM events WHERE kind='resources'").fetchall()
    for (encoded,) in records:
        payload = json.loads(encoded)
        gpu = payload.get("gpu")
        stdout = gpu.get("stdout") if isinstance(gpu, dict) else None
        if not stdout:
            continue
        first = stdout.splitlines()[0].split(",")
        if len(first) > 1:
            try:
                values.append(float(first[1].strip()))
            except ValueError:
                continue
    return max(values) if values else None


def _manifest_base(args, root, policy_cfg, policy_files, vlm_files, effective_features, git_info, versions, gpu):
    return {
        "milestone": "M01_real_libero",
        "model": {"repo_id": POLICY_ID, "revision": POLICY_REVISION, "snapshot_path": str(args.checkpoint.resolve()), "files": policy_files},
        "vision_language_backbone": {"repo_id": VLM_ID, "revision": VLM_REVISION, "snapshot_path": str(args.vlm_root.resolve()), "files": vlm_files},
        "dataset": {"repo_id": "lerobot/libero", "revision": policy_cfg.get("dataset_revision"), "source": "official checkpoint train_config.json"},
        "runtime": {"packages": versions, "gpu": gpu, "mujoco_gl": os.environ.get("MUJOCO_GL"), "libero_config": os.environ.get("LIBERO_CONFIG_PATH")},
        "policy_contract": effective_features,
        "experiment_contract": {
            "suite_task_selection": {alias: {"benchmark_suite": suite, "task_id": task_id} for alias, (suite, task_id) in M01_SUITES.items()},
            "long_suite_compatibility": "LIBERO's pinned official API names this suite libero_10",
            "smoke": {"suite": "spatial", "task_id": 0, "episodes": 3, "initial_state_indices": [0, 1, 2]},
            "formal_episodes_per_suite": 10,
            "formal_initial_state_indices": list(range(10)),
            "seed_rule": "seed_base + suite_ordinal*100 + episode_id",
            "control_mode": "relative (pinned LiberoEnv default; verified from installed LeRobot source)",
            "action_execution": "policy.select_action dequeues one action from the configured predicted chunk; postprocessor unnormalizes; command is clipped to [-1,1] and passed to env.step",
            "executed_action_evidence": "trajectory actions are the exact float32 vectors passed as env.step arguments; simulator internal actuator forces are not instrumented",
            "camera_mapping": {"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2", "camera3": "not fabricated; pinned config empty_cameras=0 and policy accepts present images"},
            "state_mapping": "official LiberoProcessorStep order is eef position xyz + axis-angle xyz + gripper qpos[2]; the pinned checkpoint processor's fitted state statistics are width 8 despite its declared model feature width 6; all 8 normalized values are passed to SmolVLAPolicy.prepare_state, which pads to max_state_dim",
        },
        "source": {**git_info, "requested_source_revision": args.source_revision, "tracked_source_hashes": {str(path.relative_to(root)): file_hash(path) for path in [root / "scripts/run_m01_libero.py", root / "src/robot_vla/m01.py", root / "scripts/evaluate_policy.py", root / "src/robot_vla/rollout.py", root / "src/robot_vla/observations.py", root / "src/robot_vla/instrumentation.py", root / "configs/data_protocol_v1.json"]}},
        "started_at": _now(),
        "runs": [],
    }


def _feature_shape(feature):
    return tuple(feature.shape)


def _main(args):
    root = args.root.resolve()
    output = args.output_dir.resolve()
    rows_path = output / "episodes.csv"
    manifest_path = output / "manifest.json"
    metrics_path = output / "metrics.json"
    output.mkdir(parents=True, exist_ok=True)
    if args.phase == "formal":
        try:
            validate_smoke_coverage(_load_rows(rows_path))
        except ValueError as exc:
            raise SystemExit(f"formal phase blocked by smoke gate: {exc}") from exc

    import av
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.libero import LiberoEnv
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from libero.libero import benchmark

    from robot_vla.gpu_budget import configure_gpu_budget
    from robot_vla.instrumentation import ChunkProfiler, ResourceMonitor
    from robot_vla.observations import libero_batch
    from robot_vla.preflight import create_libero_env, evaluation_settings
    from robot_vla.protocol import load_protocol
    from robot_vla.registry import Registry
    from robot_vla.rollout import record_episode
    from robot_vla.telemetry import snapshot

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M01 requires the verified CUDA GPU path")
    if not args.checkpoint.is_dir() or not args.vlm_root.is_dir():
        raise FileNotFoundError("both pinned policy and VLM snapshots must exist on the data disk")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed_base)
    np.random.seed(args.seed_base)
    gpu_props = torch.cuda.get_device_properties(0)
    if gpu_props.total_memory < 20 * 2**30:
        raise RuntimeError(f"M01 requires a 24GB-class GPU, found {gpu_props.total_memory} bytes")
    gpu = {"name": gpu_props.name, "total_memory_bytes": gpu_props.total_memory, "cuda": torch.version.cuda, "device_index": 0}
    gpu["pytorch_allocator_budget"] = configure_gpu_budget(22.0)

    # Importing the concrete class registers the pinned policy type with LeRobot.
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    if cfg.type != "smolvla":
        raise ValueError(f"official snapshot policy type is {cfg.type}, expected smolvla")
    if cfg.output_features["action"].shape != (7,):
        raise ValueError(f"checkpoint action contract is {cfg.output_features['action'].shape}, expected (7,)")
    state_dim = cfg.input_features["observation.state"].shape[0]
    if state_dim != 6:
        raise ValueError(f"unreviewed official checkpoint state contract {state_dim}")
    if cfg.n_action_steps < 1 or cfg.chunk_size < cfg.n_action_steps:
        raise ValueError("invalid official action-chunk configuration")
    cfg.device = "cuda"
    cfg.vlm_model_name = str(args.vlm_root.resolve())
    policy_files = _checkpoint_files(args.checkpoint)
    vlm_files = _checkpoint_files(args.vlm_root)
    policy_cfg_raw = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    train_cfg = json.loads((args.checkpoint / "train_config.json").read_text(encoding="utf-8"))
    preproc_config = json.loads((args.checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8"))
    rename_map = next(step["config"].get("rename_map", {}) for step in preproc_config["steps"] if step["registry_name"] == "rename_observations_processor")
    if rename_map != {"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}:
        raise ValueError(f"checkpoint camera rename contract changed: {rename_map}")

    load_start = time.perf_counter()
    # Checkpoint weights are loaded strictly; its declared VLM backbone is pinned separately.
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, config=cfg, strict=True, local_files_only=True)
    policy.eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": str(args.vlm_root.resolve())},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    # Read the fitted statistic width from the loaded official checkpoint
    # processor itself. Its JSON feature schema declares six state values, but
    # its serialized training statistics are eight-dimensional for LIBERO.
    normalizer_state_stats = pre.steps[-1].stats.get("observation.state", {})
    normalizer_state_mean = normalizer_state_stats.get("mean")
    normalizer_state_dim = int(np.asarray(normalizer_state_mean).shape[-1]) if normalizer_state_mean is not None else 0
    if state_dim != 6 or normalizer_state_dim != 8:
        raise ValueError(
            f"unreviewed pinned state contract: feature schema={state_dim}, fitted normalizer={normalizer_state_dim}"
        )
    model_load_seconds = time.perf_counter() - load_start
    torch.cuda.synchronize()
    model_load_peak_allocated = torch.cuda.max_memory_allocated()
    model_load_peak_reserved = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    measured = ChunkProfiler(policy, torch.cuda.synchronize)

    protocol = load_protocol(root / "configs/data_protocol_v1.json")
    settings = evaluation_settings(protocol)
    horizon_by_suite = settings["horizons"]
    protocol_suite_names = set(horizon_by_suite)
    if any(suite not in protocol_suite_names for suite, _ in M01_SUITES.values()):
        raise ValueError(f"frozen protocol lacks M01 suite horizons: {protocol_suite_names}")
    suites = {
        name: benchmark.get_benchmark_dict()[name](task_order_index=protocol["task_order_index"])
        for name, _ in M01_SUITES.values()
    }
    tasks = {
        alias: suites[benchmark_name].get_task(task_id)
        for alias, (benchmark_name, task_id) in M01_SUITES.items()
    }
    effective_features = {
        "policy_type": cfg.type,
        "input_features": {key: list(_feature_shape(value)) for key, value in cfg.input_features.items()},
        "fitted_state_normalizer_width": normalizer_state_dim,
        "output_features": {key: list(_feature_shape(value)) for key, value in cfg.output_features.items()},
        "n_action_steps": cfg.n_action_steps,
        "chunk_size": cfg.chunk_size,
        "image_resolution": list(cfg.input_features["observation.images.camera1"].shape[-2:]),
        "resize_imgs_with_padding": list(cfg.resize_imgs_with_padding),
        "checkpoint_empty_cameras": cfg.empty_cameras,
        "dataset_repo_id": train_cfg.get("dataset", {}).get("repo_id"),
        "dataset_revision": train_cfg.get("dataset", {}).get("revision"),
    }
    git_info = _git(root)
    versions = _package_versions()
    manifest = _manifest_base(args, root, policy_cfg_raw, policy_files, vlm_files, effective_features, git_info, versions, gpu)
    dataset_info = root / "data/libero/meta/info.json"
    manifest["dataset"] = {"repo_id": train_cfg.get("dataset", {}).get("repo_id"), "revision": protocol["revision"], "metadata_sha256": file_hash(dataset_info), "metadata_path": str(dataset_info.resolve()), "episode_count": json.loads(dataset_info.read_text(encoding="utf-8")).get("total_episodes"), "task_count": json.loads(dataset_info.read_text(encoding="utf-8")).get("total_tasks")}
    manifest["checkpoint_training_config"] = {"pretrained_path": train_cfg.get("pretrained_path"), "empty_cameras": cfg.empty_cameras, "load_vlm_weights": cfg.load_vlm_weights, "processor_rename_map": rename_map}
    manifest["model_load_seconds"] = model_load_seconds
    manifest["model_load_peak_allocated_bytes"] = model_load_peak_allocated
    manifest["model_load_peak_reserved_bytes"] = model_load_peak_reserved
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("model", {}).get("revision") != POLICY_REVISION or previous.get("vision_language_backbone", {}).get("revision") != VLM_REVISION:
            raise ValueError("existing M01 evidence belongs to a different model revision")
        manifest = previous
        manifest["runs"].append({"phase": args.phase, "started_at": _now(), "model_load_seconds": model_load_seconds})
    else:
        manifest["runs"] = [{"phase": args.phase, "started_at": _now(), "model_load_seconds": model_load_seconds}]
    manifest["source"] = {**manifest.get("source", {}), **git_info, "requested_source_revision": args.source_revision, "tracked_source_hashes": {str(path.relative_to(root)): file_hash(path) for path in [root / "scripts/run_m01_libero.py", root / "src/robot_vla/m01.py", root / "scripts/evaluate_policy.py", root / "src/robot_vla/rollout.py", root / "src/robot_vla/observations.py", root / "src/robot_vla/instrumentation.py", root / "configs/data_protocol_v1.json"]}}
    manifest["started_at"] = manifest.get("started_at", _now())
    rows = _load_rows(rows_path)
    metrics = _episode_metrics(rows)
    _json_write(manifest_path, manifest)
    _json_write(metrics_path, metrics)

    if args.phase == "smoke":
        plan = [("spatial", "libero_spatial", 0, i) for i in range(3)]
    else:
        plan = [
            (alias, benchmark_name, task_id, episode)
            for alias, (benchmark_name, task_id) in M01_SUITES.items()
            for episode in range(10)
        ]
    pending = []
    attempts_by_key = {}
    for row in rows:
        attempts_by_key[_key(row)] = max(attempts_by_key.get(_key(row), 0), int(row.get("attempt") or 1))
    for suite_alias, benchmark_name, task_id, episode_id in plan:
        prior = next((r for r in rows if _key(r) == (args.phase, suite_alias, str(task_id), str(episode_id))), None)
        if prior and prior["status"] == "complete":
            continue
        identity = (args.phase, suite_alias, str(task_id), str(episode_id))
        pending.append((suite_alias, benchmark_name, task_id, episode_id, attempts_by_key.get(identity, 0) + 1))

    session_start = time.perf_counter()
    registry = Registry(output / "registry")
    run_config = {
        "experiment_type": "m01_official_checkpoint_libero_rollout",
        "phase": args.phase,
        "policy_repo_id": POLICY_ID,
        "policy_revision": POLICY_REVISION,
        "policy_sha256": next(row["sha256"] for row in policy_files if Path(row["path"]).name == "model.safetensors"),
        "vlm_repo_id": VLM_ID,
        "vlm_revision": VLM_REVISION,
        "device": "cuda",
        "gpu": gpu,
        "seed_base": args.seed_base,
        "formal_suite_task_ids": {alias: task_id for alias, (_, task_id) in M01_SUITES.items()},
        "environment": snapshot(root),
        "model_contract": effective_features,
        "source_revision": args.source_revision,
    }
    suite_ordinals = {alias: i for i, alias in enumerate(M01_SUITES)}
    errors = 0
    with registry.run(run_config) as run, ResourceMonitor(run, interval=1.0, gpu=True), torch.inference_mode():
        for suite_alias, benchmark_name, task_id, episode_id, attempt in pending:
            suite = suites[benchmark_name]
            task = tasks.get(suite_alias) or suite.get_task(task_id)
            language = task.language
            seed = args.seed_base + suite_ordinals.get(suite_alias, 0) * 100 + episode_id
            horizon = horizon_by_suite[benchmark_name]
            size = cfg.input_features["observation.images.camera1"].shape[-1]
            stem = f"{args.phase}_{suite_alias}_task{task_id}_episode{episode_id:02d}_attempt{attempt}"
            video_path = output / "videos" / f"{stem}.mp4"
            episode_dir = run.directory / "episodes" / stem
            episode_dir.mkdir(parents=True, exist_ok=False)
            env = None
            container = None
            stream = None
            trace = []
            input_shapes = {}
            state_semantics = "full LIBERO pose and gripper state; official normalizer width 8; policy feature schema width 6"
            clip_counts = {"clipped_components": 0, "total_components": 0}
            torch.cuda.reset_peak_memory_stats()
            episode_start = time.perf_counter()
            try:
                env = create_libero_env(LiberoEnv, suite, task_id, benchmark_name, protocol, episode_id, size, horizon)
                container = av.open(str(video_path), "w")
                stream = container.add_stream("libx264", rate=settings["control_freq"])
                stream.width, stream.height = size, size
                stream.pix_fmt = "yuv420p"

                def preprocess(obs):
                    nonlocal input_shapes, state_semantics
                    batch = libero_batch(obs)
                    batch["observation.state"], state_semantics = fit_libero_state_to_checkpoint(
                        batch["observation.state"], expected_dim=state_dim, normalizer_dim=normalizer_state_dim
                    )
                    batch["task"] = [language]
                    processed = pre(batch)
                    if not input_shapes:
                        camera_inputs = {key: list(value.shape) for key, value in processed.items() if key.startswith("observation.images.") and hasattr(value, "shape")}
                        if set(camera_inputs) != {"observation.images.camera1", "observation.images.camera2"}:
                            raise ValueError(f"observed camera inputs do not match official checkpoint processor: {camera_inputs}")
                        state_value = processed["observation.state"]
                        if tuple(state_value.shape) != (1, normalizer_state_dim):
                            raise ValueError(f"processed state shape {tuple(state_value.shape)} != (1,{normalizer_state_dim})")
                        input_shapes = {key: list(value.shape) for key, value in processed.items() if hasattr(value, "shape")}
                    return processed

                def postprocess(action):
                    raw = action.detach().float().cpu().numpy()[0]
                    result = post(action)
                    command = result.detach().float().cpu().numpy()[0]
                    clipped = np.clip(command, -1.0, 1.0).astype(np.float32, copy=False)
                    clip_counts["clipped_components"] += int(np.count_nonzero(clipped != command))
                    clip_counts["total_components"] += int(command.size)
                    trace.append({"policy_select_action_output": raw.tolist(), "env_step_argument": clipped.tolist()})
                    return clipped

                def video_sink(obs, _step):
                    image = np.flip(obs["pixels"]["image"], axis=(0, 1)).copy()
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)

                measured.chunk_calls.clear()
                result = record_episode(
                    env,
                    measured,
                    run,
                    suite=suite_alias,
                    task_id=task_id,
                    episode_id=episode_id,
                    seed=seed,
                    instruction=language,
                    max_steps=horizon,
                    preprocess=preprocess,
                    postprocess=postprocess,
                    synchronize=torch.cuda.synchronize,
                    state_extract=lambda obs: libero_batch(obs)["observation.state"][0].numpy(),
                    observation_sink=video_sink,
                    environment_config={"task_name": task.name, "control_freq": settings["control_freq"], "num_steps_wait": settings["num_steps_wait"], "hard_reset": settings["hard_reset"], "init_state_index": episode_id, "camera_resolution": [size, size], "render_backend": os.environ["MUJOCO_GL"], "control_mode": "relative"},
                    policy_metadata={"repo_id": POLICY_ID, "revision": POLICY_REVISION, "action_chunk_size": cfg.chunk_size, "action_execution_steps": cfg.n_action_steps, "source_state_dim": 8, "model_state_dim": state_dim, "state_projection": state_semantics},
                )
                if not result.get("chunk_calls"):
                    raise ValueError("no real action-chunk inference was observed")
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                container = None
                with np.load(result["trajectory_path"], allow_pickle=False) as trajectory:
                    recorded_actions = trajectory["actions"]
                traced_commands = np.asarray([entry["env_step_argument"] for entry in trace], dtype=np.float32)
                if traced_commands.shape != recorded_actions.shape or not np.array_equal(traced_commands, recorded_actions):
                    raise ValueError("env.step command trace differs from episode trajectory actions")
                action_trace_path = episode_dir / "action_trace.json"
                _json_write(action_trace_path, {"scope": "normalized policy-selected action and final clipped vector passed into env.step", "steps": trace, "trajectory_match": True, "clip_counts": clip_counts})
                episode_result_path = episode_dir / "rollout.json"
                _json_write(episode_result_path, result)
                run.artifact(Path(result["trajectory_path"]), "rollout_trajectory")
                run.artifact(episode_result_path, "rollout_metadata")
                run.artifact(action_trace_path, "action_trace")
                run.artifact(video_path, "rollout_video")
                policy_latency = result["latency"]["forward"]
                chunk_latency = result.get("chunk_forward_latency", {})
                row = {
                    "phase": args.phase,
                    "suite": suite_alias,
                    "benchmark_suite": benchmark_name,
                    "task_id": task_id,
                    "task_name": task.name,
                    "instruction": language,
                    "episode_id": episode_id,
                    "seed": seed,
                    "attempt": attempt,
                    "status": "complete",
                    "success": bool(result["success"]),
                    "total_action_steps": result["episode_length"],
                    "action_chunk_size": cfg.chunk_size,
                    "action_execution_steps": cfg.n_action_steps,
                    "control_mode": "relative",
                    "source_state_dim": 8,
                    "model_state_dim": state_dim,
                    "state_projection": state_semantics,
                    "input_shapes": json.dumps(input_shapes, sort_keys=True),
                    "policy_inference_mean_ms": policy_latency["mean_ms"],
                    "policy_inference_p95_ms": policy_latency["p95_ms"],
                    "chunk_inference_mean_ms": chunk_latency.get("mean_ms"),
                    "episode_wall_seconds": result["wall_seconds"],
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "video_path": str(video_path),
                    "video_sha256": file_hash(video_path),
                    "trajectory_path": result["trajectory_path"],
                    "metadata_path": str(episode_result_path),
                    "error": "",
                }
                run.event("m01_episode", row)
            except Exception as exc:
                errors += 1
                if container is not None:
                    try:
                        if stream is not None:
                            for packet in stream.encode():
                                container.mux(packet)
                        container.close()
                    except Exception:
                        pass
                partial_path = episode_dir / "attempt_error.json"
                _json_write(partial_path, {"phase": args.phase, "suite": suite_alias, "task_id": task_id, "episode_id": episode_id, "seed": seed, "attempt": attempt, "error": f"{type(exc).__name__}: {exc}", "trace": trace, "input_shapes": input_shapes, "timestamp": _now()})
                run.artifact(partial_path, "rollout_error")
                row = {"phase": args.phase, "suite": suite_alias, "benchmark_suite": benchmark_name, "task_id": task_id, "task_name": task.name, "instruction": language, "episode_id": episode_id, "seed": seed, "attempt": attempt, "status": "error", "success": None, "total_action_steps": len(trace), "action_chunk_size": cfg.chunk_size, "action_execution_steps": cfg.n_action_steps, "control_mode": "relative", "source_state_dim": 8, "model_state_dim": state_dim, "state_projection": state_semantics or "not_reached", "input_shapes": json.dumps(input_shapes, sort_keys=True), "policy_inference_mean_ms": None, "policy_inference_p95_ms": None, "chunk_inference_mean_ms": None, "episode_wall_seconds": time.perf_counter() - episode_start, "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(), "video_path": str(video_path) if video_path.is_file() else "", "video_sha256": file_hash(video_path) if video_path.is_file() else "", "trajectory_path": "", "metadata_path": str(partial_path), "error": f"{type(exc).__name__}: {exc}"}
                run.event("m01_episode_error", row)
            finally:
                if env is not None:
                    env.close()

            rows = _upsert(rows, row)
            _write_rows(rows_path, rows)
            with (output / "episode_attempts.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            metrics = _episode_metrics(rows)
            _json_write(metrics_path, metrics)
            run.event("m01_progress", {"phase": args.phase, "completed_rows": sum(r["status"] == "complete" and r["phase"] == args.phase for r in rows), "expected": 3 if args.phase == "smoke" else 40, "metrics_status": metrics["status"]})
            print(json.dumps({"phase": args.phase, "suite": suite_alias, "task_id": task_id, "episode_id": episode_id, "attempt": attempt, "status": row["status"], "success": row["success"], "action_steps": row["total_action_steps"], "seconds": row["episode_wall_seconds"], "error": row["error"]}, ensure_ascii=False), flush=True)
    if args.phase == "smoke":
        try:
            validate_smoke_coverage(rows)
        except ValueError as exc:
            stage_status = "INCOMPLETE"
            reason = str(exc)
        else:
            stage_status = "SMOKE_PASS"
            reason = None
    else:
        try:
            validate_m01_coverage(rows)
            validate_smoke_coverage(rows)
        except ValueError as exc:
            stage_status = "INCOMPLETE"
            reason = str(exc)
        else:
            stage_status = "PASS"
            reason = None
    metrics = _episode_metrics(rows)
    if stage_status in {"SMOKE_PASS", "PASS"} and not metrics["at_least_one_video_sha256_verified"]:
        stage_status = "INCOMPLETE"
        reason = "no completed rollout video passed SHA-256 verification"
    if args.phase == "formal" and stage_status == "PASS" and metrics["status"] != "PASS":
        stage_status = "INCOMPLETE"
        reason = "formal data, smoke gate, or video verification did not pass"
    metrics["phase_status"] = stage_status
    metrics["phase_reason"] = reason
    metrics["phase_elapsed_seconds"] = time.perf_counter() - session_start
    metrics["model_load_seconds"] = model_load_seconds
    metrics["model_load_peak_allocated_bytes"] = model_load_peak_allocated
    metrics["model_load_peak_reserved_bytes"] = model_load_peak_reserved
    metrics["gpu_peak_system_memory_mib"] = _gpu_peak_system_memory_mib(output / "registry")
    _json_write(metrics_path, metrics)
    manifest["runs"][-1].update({"ended_at": _now(), "elapsed_seconds": time.perf_counter() - session_start, "phase_status": stage_status, "runtime_errors": errors})
    manifest["latest_status"] = stage_status
    manifest["updated_at"] = _now()
    _json_write(manifest_path, manifest)
    _write_report(output, manifest, metrics)
    print(json.dumps({"phase_status": stage_status, "metrics": str(metrics_path), "report": str(output / "REPORT.md"), "episodes_csv": str(rows_path)}, ensure_ascii=False), flush=True)
    return 0 if stage_status in {"SMOKE_PASS", "PASS"} else 2


def _write_report(output, manifest, metrics):
    lines = [
        "# M01: Real SmolVLA–LIBERO Simulation Loop",
        "",
        f"**Acceptance result:** {metrics.get('status')} (phase: {metrics.get('phase_status', 'not_run')})",
        "",
        "## Executed configuration",
        "",
        f"- Policy: `{POLICY_ID}@{POLICY_REVISION}`",
        f"- VLM backbone: `{VLM_ID}@{VLM_REVISION}`",
        f"- GPU: `{manifest.get('runtime', {}).get('gpu', {}).get('name', 'not recorded')}`",
        f"- Suite/task mapping: `{json.dumps(manifest.get('experiment_contract', {}).get('suite_task_selection', {}), ensure_ascii=False)}`",
        f"- Action chunk: {manifest.get('policy_contract', {}).get('chunk_size')} predicted actions; {manifest.get('policy_contract', {}).get('n_action_steps')} actions executed per chunk",
        f"- Formal coverage: {metrics.get('formal_completed_episodes', 0)}/40; smoke: {metrics.get('smoke_completed_episodes', 0)}/3",
        f"- GPU peak allocated: {metrics.get('gpu_peak_allocated_bytes', 0)} bytes",
        f"- GPU peak reserved: {metrics.get('gpu_peak_reserved_bytes', 0)} bytes",
        f"- GPU peak system memory used: {metrics.get('gpu_peak_system_memory_mib')} MiB (sampled by nvidia-smi)",
        f"- Episode wall-time sum: {metrics.get('episode_wall_seconds_sum', 0):.3f}s",
        "",
        "## Measured success rates",
        "",
    ]
    for suite, values in metrics.get("formal", {}).items():
        rate = values.get("success_rate")
        lines.append(f"- {suite}: {values.get('successes', 0)}/{values.get('completed_episodes', 0)} ({'N/A' if rate is None else f'{rate:.1%}'})")
    lines.extend([
        "",
        "## Evidence",
        "",
        "- Episode ledger: `episodes.csv`",
        "- Metrics: `metrics.json`",
        "- Registry and per-episode trajectories: `registry/`",
        "- Simulator videos and action traces: `videos/` and `registry/`",
        "- Complete process output: `run.log` (when invoked through the reproduction entry point)",
        "- Reproduction command: `reproduce.sh`",
        "",
        "## Compatibility decisions and evidence boundary",
        "",
            "The checkpoint processor renames the LIBERO two-camera inputs to `camera1` and `camera2`; the absent `camera3` is not fabricated. The pinned LIBERO processor emits state `[eef_pos(3), axis_angle(3), gripper_qpos(2)]`. Its official fitted normalizer has measured state statistics width 8 although the serialized feature schema declares 6; all eight observed and normalized values are retained. Pinned LeRobot `SmolVLAPolicy.prepare_state` accepts this state and pads it to `max_state_dim`. The checkpoint outputs 7 actions; its own LeRobot postprocessor unnormalizes them, and the final clipped vector is recorded as the exact argument passed to MuJoCo `env.step`. The MuJoCo internal actuator-force signal is outside this evidence boundary.",
        "",
        "Low task success is an observed result, not grounds to alter the evaluation or discard episodes. Infrastructure errors remain `error` rows and prevent PASS.",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "formal"), required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/vla_forge_2_transition_20261010/milestones/M01_real_libero"))
    parser.add_argument("--checkpoint", type=Path, default=Path(f"/data/models/smolvla_libero/{POLICY_REVISION}"))
    parser.add_argument("--vlm-root", type=Path, default=Path(f"/data/models/smolvlm_full/{VLM_REVISION}"))
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--seed-base", type=int, default=81000)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    raise SystemExit(_main(args))


if __name__ == "__main__":
    main()
