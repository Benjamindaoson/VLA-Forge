"""Run or resume all 400 frozen SmolVLA LIBERO 40-task Episodes."""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.metadata
import json
import os
import random
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from robot_vla.m01 import fit_libero_state_to_checkpoint
from robot_vla.m02 import (
    M02_SUITES,
    aggregate_binary,
    artifact_is_valid,
    episode_seed,
    expected_episode_identities,
)
from robot_vla.observations import libero_batch
from robot_vla.preflight import create_libero_env, evaluation_settings
from robot_vla.protocol import load_protocol
from robot_vla.registry import Registry
from robot_vla.rollout import record_episode
from robot_vla.schema import file_hash
from robot_vla.telemetry import snapshot

POLICY_ID = "lerobot/smolvla_libero"
POLICY_REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
VLM_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
CSV_FIELDS = [
    "suite", "benchmark_suite", "task_id", "task_name", "instruction", "episode_id", "registry_run_id",
    "init_state_id", "seed", "attempt", "status", "success", "termination_reason",
    "total_action_steps", "horizon", "action_chunk_size", "action_execution_steps",
    "control_mode", "checkpoint_revision", "checkpoint_sha256", "dataset_revision",
    "environment_config_hash", "input_shapes", "policy_inference_mean_ms",
    "policy_inference_p95_ms", "chunk_inference_mean_ms", "episode_wall_seconds",
    "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes", "video_path", "video_sha256",
    "camera_mapping", "trajectory_path", "trajectory_sha256", "action_trace_path",
    "action_trace_sha256", "metadata_path", "metadata_sha256", "error",
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_rows(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("task_id", "episode_id", "init_state_id", "seed", "attempt", "total_action_steps", "horizon", "action_chunk_size", "action_execution_steps", "gpu_peak_allocated_bytes", "gpu_peak_reserved_bytes"):
            if row.get(field) not in (None, ""):
                row[field] = int(row[field])
        for field in ("policy_inference_mean_ms", "policy_inference_p95_ms", "chunk_inference_mean_ms", "episode_wall_seconds"):
            if row.get(field) not in (None, ""):
                row[field] = float(row[field])
        row["success"] = True if row.get("success") == "True" else False if row.get("success") == "False" else None
    return rows


def _write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["suite"], int(row["task_id"]), int(row["episode_id"]))))
    temporary.replace(path)


def _identity(row):
    return (row["suite"], int(row["task_id"]), int(row["episode_id"]))


def _valid_complete(row):
    if row.get("status") != "complete" or type(row.get("success")) is not bool:
        return False
    return all(
        artifact_is_valid(row.get(path_key), row.get(hash_key))
        for path_key, hash_key in (
            ("video_path", "video_sha256"),
            ("trajectory_path", "trajectory_sha256"),
            ("action_trace_path", "action_trace_sha256"),
            ("metadata_path", "metadata_sha256"),
        )
    )


def _versions():
    result = {}
    for name in ("lerobot", "torch", "transformers", "huggingface-hub", "hf-libero", "mujoco", "robosuite", "av", "safetensors"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _source_identity(root):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        revision, status = None, ["git metadata unavailable"]
    files = [root / "scripts/run_m02_libero40.py", root / "src/robot_vla/m02.py", root / "src/robot_vla/rollout.py", root / "src/robot_vla/observations.py", root / "configs/data_protocol_v1.json"]
    return {
        "revision": revision,
        "worktree_status_at_start": status,
        "files": {str(path.relative_to(root)): file_hash(path) for path in files},
    }


def _checkpoint_manifest(checkpoint: Path, vlm_root: Path, baseline_manifest: dict):
    snapshots = {
        "policy": (checkpoint, baseline_manifest["model"]["files"]),
        "vlm": (vlm_root, baseline_manifest["vision_language_backbone"]["files"]),
    }
    output = {}
    for label, (root, expected_records) in snapshots.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{label} snapshot is missing: {root}")
        expected = {
            Path(record["path"]).name: (int(record["size_bytes"]), record["sha256"])
            for record in expected_records
        }
        actual_files = [path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts]
        actual = {path.name: (path.stat().st_size, file_hash(path)) for path in actual_files}
        if actual != expected:
            raise ValueError(f"{label} checkpoint files differ from the frozen M01 manifest")
        output[label] = [
            {"path": str(root / name), "size_bytes": size, "sha256": digest}
            for name, (size, digest) in sorted(actual.items())
        ]
    return output


def _load_policy(args, baseline):
    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not torch.cuda.is_available():
        raise RuntimeError("M02 requires the verified CUDA path")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed_base)
    np.random.seed(args.seed_base)
    gpu_props = torch.cuda.get_device_properties(0)
    if gpu_props.name != baseline["runtime"]["gpu"]["name"]:
        raise ValueError("GPU model differs from M01's frozen runtime")
    from robot_vla.gpu_budget import configure_gpu_budget

    budget = configure_gpu_budget(22.0)
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    if config.type != "smolvla" or tuple(config.output_features["action"].shape) != (7,):
        raise ValueError("official checkpoint policy/action contract changed")
    state_dim = config.input_features["observation.state"].shape[0]
    config.device = "cuda"
    config.vlm_model_name = str(args.vlm_root.resolve())
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint, config=config, strict=True, local_files_only=True)
    policy.eval()
    pre, post = make_pre_post_processors(
        config,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": str(args.vlm_root.resolve())},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    stats = pre.steps[-1].stats.get("observation.state", {})
    normalizer_mean = stats.get("mean")
    normalizer_dim = int(np.asarray(normalizer_mean).shape[-1]) if normalizer_mean is not None else 0
    if state_dim != 6 or normalizer_dim != 8:
        raise ValueError(f"unreviewed checkpoint state contract schema={state_dim}, normalizer={normalizer_dim}")
    preprocessor = json.loads((args.checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8"))
    rename_map = next(
        step["config"].get("rename_map", {})
        for step in preprocessor["steps"]
        if step["registry_name"] == "rename_observations_processor"
    )
    expected_rename = {
        "observation.images.image": "observation.images.camera1",
        "observation.images.image2": "observation.images.camera2",
    }
    if rename_map != expected_rename:
        raise ValueError(f"checkpoint camera rename contract changed: {rename_map}")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    from robot_vla.instrumentation import ChunkProfiler

    measured = ChunkProfiler(policy, torch.cuda.synchronize)
    gpu = {
        "name": gpu_props.name,
        "total_memory_bytes": int(gpu_props.total_memory),
        "cuda": torch.version.cuda,
        "device_index": 0,
        "allocator_budget": budget,
    }
    features = {
        "policy_type": config.type,
        "input_features": {key: list(value.shape) for key, value in config.input_features.items()},
        "normalizer_state_width": normalizer_dim,
        "output_features": {key: list(value.shape) for key, value in config.output_features.items()},
        "action_chunk_size": int(config.chunk_size),
        "action_execution_steps": int(config.n_action_steps),
        "resize_imgs_with_padding": list(config.resize_imgs_with_padding),
    }
    return torch, policy, measured, pre, post, config, state_dim, normalizer_dim, rename_map, gpu, features


def _preflight(args):
    from robot_vla.m02 import M02_SUITES

    baseline_path = args.root / "reports/vla_forge_2_transition_20261010/milestones/M01_real_libero/manifest.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline["model"]["repo_id"] != POLICY_ID or baseline["model"]["revision"] != POLICY_REVISION:
        raise ValueError("M01 checkpoint identity is not the expected official policy")
    checkpoint = _checkpoint_manifest(args.checkpoint, args.vlm_root, baseline)
    protocol = load_protocol(args.root / "configs/data_protocol_v1.json")
    settings = evaluation_settings(protocol)
    if settings["initial_state_indices"] != list(range(10)):
        raise ValueError("frozen M02 requires initial state IDs 0 through 9")
    if set(M02_SUITES.values()) != set(settings["horizons"]):
        raise ValueError("M02 suite set does not match the frozen protocol")
    data_info = args.root / "data/libero/meta/info.json"
    data = json.loads(data_info.read_text(encoding="utf-8"))
    if int(data.get("total_tasks", -1)) != 40:
        raise ValueError("pinned LIBERO metadata does not contain 40 tasks")
    expected_data_hash = baseline["dataset"]["metadata_sha256"]
    if file_hash(data_info) != expected_data_hash:
        raise ValueError("LIBERO dataset metadata differs from M01")
    return baseline, checkpoint, protocol, settings, file_hash(data_info), data


def _append_jsonl(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_live_metrics(output: Path, rows):
    complete = [row for row in rows if row.get("status") == "complete" and type(row.get("success")) is bool]
    groups = {suite: aggregate_binary([row for row in complete if row["suite"] == suite]) for suite in M02_SUITES}
    metrics = {
        "status": "RUNNING",
        "expected_episodes": 400,
        "completed_episodes": len(complete),
        "incomplete_or_error_rows": sum(row.get("status") != "complete" for row in rows),
        "overall": aggregate_binary(complete),
        "suites": groups,
        "updated_at": _now(),
    }
    _write_json(output / "metrics.json", metrics)


def _main(args):
    import av
    import numpy as np
    from lerobot.envs.libero import LiberoEnv
    from libero.libero import benchmark

    from robot_vla.instrumentation import ResourceMonitor
    from robot_vla.m02 import EPISODES_PER_TASK, M02_SUITES, TASKS_PER_SUITE

    root, output, artifact_root = args.root.resolve(), args.output_dir.resolve(), args.artifact_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".m02.lock"
    lock_handle = lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"another M02 process holds {lock_path}") from exc
    lock_handle.seek(0)
    lock_handle.truncate()
    lock_handle.write(f"pid={os.getpid()} started_at={_now()}\n")
    lock_handle.flush()

    baseline, checkpoint_files, protocol, settings, dataset_hash, dataset_info = _preflight(args)
    model_start = time.perf_counter()
    (
        torch, policy, measured, pre, post, config, state_dim, normalizer_dim,
        rename_map, gpu, features,
    ) = _load_policy(args, baseline)
    model_load_seconds = time.perf_counter() - model_start

    suite_classes = benchmark.get_benchmark_dict()
    suites = {
        alias: suite_classes[benchmark_name](task_order_index=protocol["task_order_index"])
        for alias, benchmark_name in M02_SUITES.items()
    }
    if any(int(suite.n_tasks) != TASKS_PER_SUITE for suite in suites.values()):
        raise ValueError("pinned official LIBERO API must expose exactly ten tasks per suite")
    task_map = {
        (alias, task_id): suites[alias].get_task(task_id)
        for alias in M02_SUITES
        for task_id in range(TASKS_PER_SUITE)
    }
    task_names = {
        alias: [
            {"task_id": task_id, "name": task_map[(alias, task_id)].name, "instruction": task_map[(alias, task_id)].language}
            for task_id in range(TASKS_PER_SUITE)
        ]
        for alias in M02_SUITES
    }
    checkpoint_sha = next(row["sha256"] for row in checkpoint_files["policy"] if Path(row["path"]).name == "model.safetensors")
    source = _source_identity(root)
    manifest_path = output / "manifest.json"
    manifest = {
        "milestone": "M02_libero40_failure_mining",
        "status": "RUNNING",
        "started_at": _now(),
        "model": {"repo_id": POLICY_ID, "revision": POLICY_REVISION, "sha256": checkpoint_sha, "files": checkpoint_files["policy"], "snapshot_path": str(args.checkpoint)},
        "vision_language_backbone": {"repo_id": VLM_ID, "revision": VLM_REVISION, "files": checkpoint_files["vlm"], "snapshot_path": str(args.vlm_root)},
        "model_config_sha256": file_hash(args.checkpoint / "config.json"),
        "processor": {"camera_rename_map": rename_map, "fitted_state_normalizer_width": normalizer_dim},
        "runtime": {"packages": _versions(), "gpu": gpu, "mujoco_gl": os.environ.get("MUJOCO_GL"), "libero_config": os.environ.get("LIBERO_CONFIG_PATH")},
        "dataset": {"repo_id": "lerobot/libero", "revision": protocol["revision"], "metadata_path": str((root / "data/libero/meta/info.json").resolve()), "metadata_sha256": dataset_hash, "episode_count": dataset_info["total_episodes"], "task_count": dataset_info["total_tasks"]},
        "protocol": {"protocol_sha256": file_hash(root / "configs/data_protocol_v1.json"), "suite_task_order_index": protocol["task_order_index"], "suite_task_names": task_names, "episodes_per_task": EPISODES_PER_TASK, "initial_state_ids": settings["initial_state_indices"], "seed_rule": "seed_base + suite_ordinal*1000 + task_id*10 + init_state_id", "seed_base": args.seed_base, "horizons": settings["horizons"], "control_frequency_hz": settings["control_freq"], "num_steps_wait": settings["num_steps_wait"], "hard_reset": settings["hard_reset"], "control_mode": "relative", "action_chunk_size": config.chunk_size, "action_execution_steps": config.n_action_steps, "camera_views": ["image", "image2"], "paired_video_layout": "image left, image2 right"},
        "policy_contract": features,
        "source": source,
        "baseline_m01_manifest_sha256": file_hash(root / "reports/vla_forge_2_transition_20261010/milestones/M01_real_libero/manifest.json"),
        "artifact_root": str(artifact_root),
        "runs": [],
    }
    if manifest_path.exists():
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior_manifest.get("model", {}).get("sha256") != checkpoint_sha or prior_manifest.get("protocol", {}).get("protocol_sha256") != manifest["protocol"]["protocol_sha256"]:
            raise ValueError("existing M02 manifest belongs to another frozen model/protocol")
        manifest = prior_manifest
    rows_path = output / "episodes.csv"
    attempts_path = output / "episode_attempts.jsonl"
    rows = _read_rows(rows_path)
    valid_by_identity = {_identity(row): _valid_complete(row) for row in rows}
    attempts_by_identity = {}
    if attempts_path.is_file():
        for line in attempts_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            key = (entry["suite"], int(entry["task_id"]), int(entry["episode_id"]))
            attempts_by_identity[key] = max(attempts_by_identity.get(key, 0), int(entry["attempt"]))

    registry = Registry(artifact_root / "registry")
    run_config = {
        "experiment_type": "m02_frozen_smolvla_libero40",
        "policy_repo_id": POLICY_ID,
        "policy_revision": POLICY_REVISION,
        "policy_sha256": checkpoint_sha,
        "vlm_repo_id": VLM_ID,
        "vlm_revision": VLM_REVISION,
        "seed_base": args.seed_base,
        "protocol_sha256": manifest["protocol"]["protocol_sha256"],
        "metadata_sha256": dataset_hash,
        "source_revision": source["revision"],
        "gpu": gpu,
        "environment": snapshot(root),
        "model_contract": features,
    }
    with registry.run(run_config) as run, ResourceMonitor(run, interval=1.0, gpu=True), torch.inference_mode():
        run_started = _now()
        manifest["runs"].append({"run_id": run.run_id, "started_at": run_started, "model_load_seconds": model_load_seconds})
        _write_json(manifest_path, manifest)
        _write_rows(rows_path, rows)
        _write_live_metrics(output, rows)
        plan = [
            (suite_alias, task_id, episode_id)
            for suite_alias in M02_SUITES
            for task_id in range(TASKS_PER_SUITE)
            for episode_id in range(EPISODES_PER_TASK)
        ]
        attempted_this_invocation = 0
        for suite_alias, task_id, episode_id in plan:
            identity = (suite_alias, task_id, episode_id)
            if valid_by_identity.get(identity, False):
                continue
            attempt = attempts_by_identity.get(identity, 0) + 1
            if attempt > args.max_attempts:
                continue
            if args.limit_episodes is not None and attempted_this_invocation >= args.limit_episodes:
                break
            attempted_this_invocation += 1
            benchmark_name = M02_SUITES[suite_alias]
            task = task_map[(suite_alias, task_id)]
            language = task.language
            seed = episode_seed(args.seed_base, suite_alias, task_id, episode_id)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            horizon = int(settings["horizons"][benchmark_name])
            size = int(config.input_features["observation.images.camera1"].shape[-1])
            stem = f"{suite_alias}_task{task_id:02d}_episode{episode_id:02d}_attempt{attempt}"
            episode_dir = run.directory / "episodes" / stem
            episode_dir.mkdir(parents=True, exist_ok=False)
            video_path = run.directory / "videos" / f"{stem}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            env = None
            container = None
            stream = None
            action_trace = []
            input_shapes = {}
            clip_counts = {"clipped_components": 0, "total_components": 0}
            episode_start = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            metadata = {
                "registry_run_id": run.run_id,
                "suite": suite_alias,
                "benchmark_suite": benchmark_name,
                "task_id": task_id,
                "task_name": task.name,
                "instruction": language,
                "episode_id": episode_id,
                "init_state_id": episode_id,
                "seed": seed,
                "checkpoint_revision": POLICY_REVISION,
                "checkpoint_sha256": checkpoint_sha,
                "environment": {"control_mode": "relative", "control_frequency_hz": settings["control_freq"], "num_steps_wait": settings["num_steps_wait"], "hard_reset": settings["hard_reset"], "horizon": horizon, "camera_resolution": [size, size], "render_backend": os.environ["MUJOCO_GL"]},
            }
            try:
                env = create_libero_env(LiberoEnv, suites[suite_alias], task_id, benchmark_name, protocol, episode_id, size, horizon)
                container = av.open(str(video_path), "w")
                stream = None

                def preprocess(obs):
                    nonlocal input_shapes
                    batch = libero_batch(obs)
                    batch["observation.state"], _ = fit_libero_state_to_checkpoint(batch["observation.state"], expected_dim=state_dim, normalizer_dim=normalizer_dim)
                    batch["task"] = [language]
                    processed = pre(batch)
                    camera_shapes = {key: list(value.shape) for key, value in processed.items() if key in {"observation.images.camera1", "observation.images.camera2"}}
                    if set(camera_shapes) != {"observation.images.camera1", "observation.images.camera2"}:
                        raise ValueError(f"expected two actual checkpoint camera inputs, got {camera_shapes}")
                    if tuple(processed["observation.state"].shape) != (1, normalizer_dim):
                        raise ValueError("processed state differs from frozen M01 normalizer width")
                    input_shapes = {key: list(value.shape) for key, value in processed.items() if hasattr(value, "shape")}
                    return processed

                def postprocess(action):
                    raw = action.detach().float().cpu().numpy()[0]
                    command = post(action).detach().float().cpu().numpy()[0]
                    clipped = np.clip(command, -1.0, 1.0).astype(np.float32, copy=False)
                    clip_counts["clipped_components"] += int(np.count_nonzero(clipped != command))
                    clip_counts["total_components"] += int(command.size)
                    action_trace.append({"policy_select_action_output": raw.tolist(), "env_step_argument": clipped.tolist()})
                    return clipped

                def video_sink(obs, _step):
                    nonlocal stream
                    pixels = obs["pixels"]
                    if "image" not in pixels or "image2" not in pixels:
                        raise ValueError(f"two-camera observation missing: {tuple(pixels)}")
                    left = np.flip(pixels["image"], axis=(0, 1)).copy()
                    right = np.flip(pixels["image2"], axis=(0, 1)).copy()
                    frame = np.concatenate((left, right), axis=1)
                    if stream is None:
                        stream = container.add_stream("libx264", rate=settings["control_freq"])
                        stream.width, stream.height = frame.shape[1], frame.shape[0]
                        stream.pix_fmt = "yuv420p"
                    video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                    for packet in stream.encode(video_frame):
                        container.mux(packet)

                measured.chunk_calls.clear()
                measured.control_step = 0
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
                    environment_config=metadata["environment"],
                    policy_metadata={"repo_id": POLICY_ID, "revision": POLICY_REVISION, "sha256": checkpoint_sha, "action_chunk_size": config.chunk_size, "action_execution_steps": config.n_action_steps, "control_mode": "relative", "camera_mapping": rename_map},
                )
                if not result.get("chunk_calls"):
                    raise ValueError("Episode completed without a measured action-chunk inference")
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                container = None
                trajectory_path = Path(result["trajectory_path"])
                with np.load(trajectory_path, allow_pickle=False) as trajectory:
                    recorded_actions = trajectory["actions"]
                    states = trajectory["states"]
                traced = np.asarray([entry["env_step_argument"] for entry in action_trace], dtype=np.float32)
                if traced.shape != recorded_actions.shape or not np.array_equal(traced, recorded_actions):
                    raise ValueError("stored action trace differs from the exact trajectory passed to env.step")
                if states.ndim != 2 or states.shape[1] != 8:
                    raise ValueError(f"expected full eight-dimensional LIBERO robot state trace, got {states.shape}")
                trace_path = episode_dir / "action_trace.json"
                _write_json(trace_path, {"scope": "normalized policy-selected action and exact final vector passed to env.step", "steps": action_trace, "trajectory_exact_match": True, "clipping": clip_counts})
                metadata.update({"success": bool(result["success"]), "termination_reason": result["termination_reason"], "total_action_steps": result["episode_length"], "horizon": horizon, "wall_seconds": result["wall_seconds"], "latency": result["latency"], "chunk_forward_latency": result.get("chunk_forward_latency"), "video_layout": {"left": "pixels.image", "right": "pixels.image2"}, "trajectory_state_shape": list(states.shape), "trajectory_action_shape": list(recorded_actions.shape), "trajectory_state_semantics": "EEF position xyz, axis-angle xyz, gripper qpos[2]"})
                metadata_path = episode_dir / "episode.json"
                _write_json(metadata_path, metadata)
                run.artifact(trajectory_path, "rollout_trajectory")
                run.artifact(trace_path, "action_trace")
                run.artifact(metadata_path, "rollout_metadata")
                run.artifact(video_path, "paired_camera_rollout_video")
                policy_latency = result["latency"]["forward"]
                chunk_latency = result.get("chunk_forward_latency", {})
                row = {
                    **metadata,
                    "attempt": attempt,
                    "status": "complete",
                    "success": bool(result["success"]),
                    "total_action_steps": int(result["episode_length"]),
                    "action_chunk_size": int(config.chunk_size),
                    "action_execution_steps": int(config.n_action_steps),
                    "control_mode": "relative",
                    "dataset_revision": protocol["revision"],
                    "environment_config_hash": __import__("robot_vla.schema", fromlist=["content_hash"]).content_hash(metadata["environment"]),
                    "input_shapes": json.dumps(input_shapes, sort_keys=True),
                    "policy_inference_mean_ms": policy_latency["mean_ms"],
                    "policy_inference_p95_ms": policy_latency["p95_ms"],
                    "chunk_inference_mean_ms": chunk_latency.get("mean_ms"),
                    "episode_wall_seconds": float(result["wall_seconds"]),
                    "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "video_path": str(video_path),
                    "video_sha256": file_hash(video_path),
                    "camera_mapping": "image:left,image2:right",
                    "trajectory_path": str(trajectory_path),
                    "trajectory_sha256": file_hash(trajectory_path),
                    "action_trace_path": str(trace_path),
                    "action_trace_sha256": file_hash(trace_path),
                    "metadata_path": str(metadata_path),
                    "metadata_sha256": file_hash(metadata_path),
                    "checkpoint_revision": POLICY_REVISION,
                    "checkpoint_sha256": checkpoint_sha,
                    "horizon": horizon,
                    "error": "",
                }
            except Exception as exc:
                if container is not None:
                    try:
                        if stream is not None:
                            for packet in stream.encode():
                                container.mux(packet)
                        container.close()
                    except Exception:
                        pass
                error_path = episode_dir / "attempt_error.json"
                _write_json(error_path, {**metadata, "attempt": attempt, "status": "error", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "partial_action_trace": action_trace, "timestamp": _now()})
                if error_path.is_file():
                    run.artifact(error_path, "rollout_error")
                row = {
                    "suite": suite_alias, "benchmark_suite": benchmark_name, "task_id": task_id,
                    "task_name": task.name, "instruction": language, "episode_id": episode_id,
                    "init_state_id": episode_id, "seed": seed, "attempt": attempt,
                    "status": "error", "success": None,
                    "termination_reason": "infrastructure_or_runtime_error",
                    "total_action_steps": len(action_trace), "horizon": horizon,
                    "action_chunk_size": int(config.chunk_size), "action_execution_steps": int(config.n_action_steps),
                    "control_mode": "relative", "checkpoint_revision": POLICY_REVISION,
                    "checkpoint_sha256": checkpoint_sha, "dataset_revision": protocol["revision"],
                    "environment_config_hash": "", "input_shapes": json.dumps(input_shapes, sort_keys=True),
                    "policy_inference_mean_ms": "", "policy_inference_p95_ms": "", "chunk_inference_mean_ms": "",
                    "episode_wall_seconds": time.perf_counter() - episode_start,
                    "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "video_path": str(video_path) if video_path.is_file() else "",
                    "video_sha256": file_hash(video_path) if video_path.is_file() else "",
                    "camera_mapping": "image:left,image2:right", "trajectory_path": "", "trajectory_sha256": "",
                    "action_trace_path": "", "action_trace_sha256": "", "metadata_path": str(error_path),
                    "metadata_sha256": file_hash(error_path), "error": f"{type(exc).__name__}: {exc}",
                }
                run.event("m02_episode_error", row)
            finally:
                if env is not None:
                    env.close()

            rows = [old for old in rows if _identity(old) != identity]
            rows.append(row)
            _write_rows(rows_path, rows)
            _append_jsonl(attempts_path, row)
            _write_live_metrics(output, rows)
            attempts_by_identity[identity] = attempt
            valid_by_identity[identity] = _valid_complete(row)
            run.event("m02_progress", {"identity": identity, "attempt": attempt, "status": row["status"], "success": row["success"], "completed": sum(r.get("status") == "complete" for r in rows), "expected": 400})
            print(json.dumps({"suite": suite_alias, "task_id": task_id, "episode_id": episode_id, "attempt": attempt, "status": row["status"], "success": row["success"], "steps": row["total_action_steps"], "seconds": row["episode_wall_seconds"], "error": row["error"]}, ensure_ascii=False), flush=True)
        run.event("m02_run_finished", {"completed": sum(r.get("status") == "complete" for r in rows), "expected": 400, "ended_at": _now()})

    manifest["runs"][-1].update({"ended_at": _now(), "elapsed_seconds": time.perf_counter() - model_start, "completed_episodes": sum(row.get("status") == "complete" for row in rows)})
    manifest["latest_completed_episodes"] = sum(row.get("status") == "complete" for row in rows)
    valid_rows = [row for row in rows if _valid_complete(row)]
    actual_identities = {_identity(row) for row in valid_rows}
    manifest["status"] = (
        "COMPLETE"
        if len(valid_rows) == 400 and actual_identities == expected_episode_identities()
        else "INCOMPLETE"
    )
    manifest["updated_at"] = _now()
    _write_json(manifest_path, manifest)
    _write_live_metrics(output, rows)
    lock_handle.close()
    return 0 if manifest["status"] == "COMPLETE" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/vla_forge_2_transition_20261010/milestones/M02_libero40_failure_mining"))
    parser.add_argument("--artifact-root", type=Path, default=Path("/data/vla-forge-artifacts/M02_libero40"))
    parser.add_argument("--checkpoint", type=Path, default=Path(f"/data/models/smolvla_libero/{POLICY_REVISION}"))
    parser.add_argument("--vlm-root", type=Path, default=Path(f"/data/models/smolvlm_full/{VLM_REVISION}"))
    parser.add_argument("--seed-base", type=int, default=82000)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--limit-episodes", type=int, default=None, help="run at most this many pending Episodes; used for the first-episode acceptance gate")
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be positive")
    if args.limit_episodes is not None and args.limit_episodes < 1:
        parser.error("--limit-episodes must be positive")
    return _main(args)


if __name__ == "__main__":
    raise SystemExit(main())
