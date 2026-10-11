"""Build a fresh LeRobot v3 set from unique, independently replay-verified M04 corrections."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.env_processor import LiberoProcessorStep

ROOT = Path("/data/VLA-Forge")
EVID = ROOT / "Evidence/M04"
DEST = Path(os.environ.get("M04_DATASET_PATH", EVID / "dataset/lerobot_v3_verified_r2"))
REPO_ID = "vla_forge/m04_verified_corrections"
FPS = 20
SOURCE_SPECS = [
    {
        "episode_id": "m04-m03-treatment-replay-01",
        "npz": EVID / "artifacts/first_correction/m04-first-correction-replay-01.npz",
        "json": EVID / "artifacts/first_correction/m04-first-correction-replay-01.json",
        "start": 100,
        "unique_action_source": "M03 frozen SmolVLA brightness-treatment actions, independently physically replayed by M04",
    },
    {
        "episode_id": "m04-policy-seed84075-replay-01",
        "npz": EVID
        / "artifacts/verified_corrections/m04-goal-t06-e07-brightness115-first50-seed84075/m04-goal-t06-e07-brightness115-first50-seed84075-independent-replay-01.npz",
        "json": EVID
        / "artifacts/verified_corrections/m04-goal-t06-e07-brightness115-first50-seed84075/m04-goal-t06-e07-brightness115-first50-seed84075-independent-replay-01.json",
        "start": 100,
        "unique_action_source": "fresh M04 SmolVLA inference at policy seed 84075 with 1.15 RGB gain for the first 50 post-checkpoint observations, independently replayed twice",
    },
    {
        "episode_id": "m04-policy-wrist-camera115-replay-01",
        "npz": EVID
        / "artifacts/verified_corrections/m04-goal-t06-e07-wrist115-first50-seed84067/m04-goal-t06-e07-wrist115-first50-seed84067-independent-replay-01.npz",
        "json": EVID
        / "artifacts/verified_corrections/m04-goal-t06-e07-wrist115-first50-seed84067/m04-goal-t06-e07-wrist115-first50-seed84067-independent-replay-01.json",
        "start": 100,
        "unique_action_source": "fresh M04 SmolVLA inference at policy seed 84067 with 1.15 gain applied only to wrist-camera observations for the first 50 post-checkpoint policy calls, independently replayed twice",
    },
    {
        "episode_id": "m04-policy-wrist110-seed84068-replay-01",
        "npz": EVID / "artifacts/verified_corrections/m04-goal-t06-e07-wrist110-first50-seed84068/m04-goal-t06-e07-wrist110-first50-seed84068-independent-replay-01.npz",
        "json": EVID / "artifacts/verified_corrections/m04-goal-t06-e07-wrist110-first50-seed84068/m04-goal-t06-e07-wrist110-first50-seed84068-independent-replay-01.json",
        "start": 100,
        "unique_action_source": "fresh M04 SmolVLA inference at seed 84068 with 1.10 wrist-camera gain for 50 policy calls; independently replayed twice",
    },
    {
        "episode_id": "m04-policy-wrist110-seed84071-replay-01",
        "npz": EVID / "artifacts/verified_corrections/m04-goal-t06-e07-wrist110-first50-seed84071/m04-goal-t06-e07-wrist110-first50-seed84071-independent-replay-01.npz",
        "json": EVID / "artifacts/verified_corrections/m04-goal-t06-e07-wrist110-first50-seed84071/m04-goal-t06-e07-wrist110-first50-seed84071-independent-replay-01.json",
        "start": 100,
        "unique_action_source": "fresh M04 SmolVLA inference at seed 84071 with 1.10 wrist-camera gain for 50 policy calls; independently replayed twice",
    },
    {
        "episode_id": "m04-policy-wrist110-seed84072-replay-01",
        "npz": EVID / "artifacts/verified_corrections/m04-goal-t06-e07-wrist110-first50-seed84072/m04-goal-t06-e07-wrist110-first50-seed84072-independent-replay-01.npz",
        "json": EVID / "artifacts/verified_corrections/m04-goal-t06-e07-wrist110-first50-seed84072/m04-goal-t06-e07-wrist110-first50-seed84072-independent-replay-01.json",
        "start": 100,
        "unique_action_source": "fresh M04 SmolVLA inference at seed 84072 with 1.10 wrist-camera gain for 50 policy calls; independently replayed twice",
    },
]


def all_source_specs() -> list[dict]:
    specs = list(SOURCE_SPECS)
    sweep_path = EVID / "logs/seed-sweep-summary.json"
    if not sweep_path.is_file():
        return specs
    sweep = json.loads(sweep_path.read_text())
    for item in sweep.get("attempts", []):
        if item.get("status") != "NEW_VERIFIED_UNIQUE_SEQUENCE":
            continue
        run_id = item["run_id"]
        replay_dir = EVID / "artifacts/verified_corrections" / run_id
        specs.append(
            {
                "episode_id": f"{run_id}-verified-replay-01",
                "npz": replay_dir / f"{run_id}-independent-replay-01.npz",
                "json": replay_dir / f"{run_id}-independent-replay-01.json",
                "start": 100,
                "unique_action_source": (
                    "fresh M04 SmolVLA policy seed sweep success, independently physically replayed twice"
                ),
            }
        )
    return specs


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if DEST.exists():
        raise FileExistsError(f"refusing to overwrite {DEST}")
    processor = LiberoProcessorStep()
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "observation.images.agentview_image": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.robot0_eye_in_hand_image": {
            "dtype": "video",
            "shape": (256, 256, 3),
            "names": ["height", "width", "channel"],
        },
        "action": {"dtype": "float32", "shape": (7,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    source_rows = []
    prepared = []
    seen_action_hashes = set()
    for spec in all_source_specs():
        meta = json.loads(spec["json"].read_text())
        source_m02 = json.loads(
            (
                ROOT
                / "Evidence/M02/source/M02_libero40/registry/runs/20261010T030315-982cacc8515f/episodes/goal_task06_episode07_attempt1/episode.json"
            ).read_text()
        )
        instruction = meta.get("instruction", source_m02["instruction"])
        if meta.get("success") is not True:
            raise RuntimeError(f"source not Success=True: {spec['episode_id']}")
        if meta.get("prefix_recovery", meta.get("recovery", {})).get("qpos_linf") != 0.0:
            raise RuntimeError(f"source checkpoint does not exactly recover: {spec['episode_id']}")
        with np.load(spec["npz"]) as z:
            frames = z["dual_camera_frames"].copy()
            actions = z["actions"].astype(np.float32, copy=True)
            pos = z["eef_pos"].astype(np.float32, copy=True)
            quat = z["eef_quat"].astype(np.float32, copy=True)
            grip = z["gripper_qpos"].astype(np.float32, copy=True)
        start = int(spec["start"])
        n = int(meta["correction_actions_executed"])
        if frames.shape[0] != actions.shape[0] + 1 or start + n != actions.shape[0]:
            raise RuntimeError(f"T/T+1 source alignment fails: {spec['episode_id']}")
        if frames.shape[1:] != (256, 512, 3) or pos.shape[0] != frames.shape[0]:
            raise RuntimeError(f"camera/state shape mismatch: {spec['episode_id']}")
        correction = actions[start : start + n]
        action_hash = hashlib.sha256(correction.tobytes()).hexdigest()
        if action_hash in seen_action_hashes:
            raise RuntimeError("duplicate corrected action sequence; do not count twice")
        seen_action_hashes.add(action_hash)
        aa = processor._quat2axisangle(torch.from_numpy(quat[start : start + n])).cpu().numpy()
        state = np.concatenate(
            [pos[start : start + n], aa, grip[start : start + n]], axis=-1
        ).astype(np.float32)
        if (
            state.shape != (n, 8)
            or correction.shape != (n, 7)
            or not np.isfinite(state).all()
            or not np.isfinite(correction).all()
        ):
            raise RuntimeError(f"invalid feature rows: {spec['episode_id']}")
        prepared.append(
            {
                "spec": spec,
                "meta": meta,
                "instruction": instruction,
                "frames": frames,
                "actions": correction,
                "state": state,
                "count": n,
                "action_sha256": action_hash,
            }
        )
        source_rows.append(
            {
                "episode_id": spec["episode_id"],
                "replay_episode_id": meta["episode_id"],
                "source_m02_episode": meta.get("source_m02_episode", source_m02["episode_id"]),
                "source_m02_trace_sha256": meta.get(
                    "source_m02_trace_sha256", meta.get("source_m02_trace_sha256")
                ),
                "suite": source_m02["suite"],
                "task_id": source_m02["task_id"],
                "task_name": source_m02["task_name"],
                "instruction": instruction,
                "init_state_id": source_m02["init_state_id"],
                "environment_seed": source_m02["seed"],
                "source_trajectory": str(spec["npz"]),
                "source_trajectory_sha256": sha(spec["npz"]),
                "source_metadata_sha256": sha(spec["json"]),
                "correction_action_sha256": action_hash,
                "correction_actions": n,
                "source_policy_attempt": meta.get("policy_attempt_id"),
                "policy_seed": meta.get("policy_seed"),
                "condition": meta.get("condition"),
                "brightness_gain": meta.get("brightness_gain"),
                "independent_replay_success": meta["success"],
                "replay_checkpoint_error": meta.get("recovery", meta.get("prefix_recovery")),
                "policy_inference_gpu_hours": meta.get(
                    "policy_inference_gpu_hours", meta.get("gpu_hours_policy_inference_in_this_run")
                ),
                "unique_action_source": spec["unique_action_source"],
            }
        )
    DEST.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=DEST,
        fps=FPS,
        features=features,
        use_videos=True,
        image_writer_threads=2,
        video_backend="pyav",
    )
    ep_ranges = []
    cursor = 0
    for item in prepared:
        spec, meta, frames, actions, states, n = (
            item["spec"],
            item["meta"],
            item["frames"],
            item["actions"],
            item["state"],
            item["count"],
        )
        start = int(spec["start"])
        first = cursor
        for i in range(n):
            obs_frame = frames[start + i]
            last = i == n - 1
            ds.add_frame(
                {
                    "observation.state": states[i],
                    "observation.images.agentview_image": obs_frame[:, :256, :],
                    "observation.images.robot0_eye_in_hand_image": obs_frame[:, 256:, :],
                    "action": actions[i],
                    "next.success": np.asarray([last], dtype=np.bool_),
                    "next.done": np.asarray([last], dtype=np.bool_),
                    "task": item["instruction"],
                }
            )
            cursor += 1
        ds.save_episode(parallel_encoding=True)
        ep_ranges.append(
            {
                "episode_id": spec["episode_id"],
                "global_start": first,
                "global_end_inclusive": cursor - 1,
                "transitions": n,
            }
        )
    if hasattr(ds, "finalize"):
        ds.finalize()
    del ds
    encode_seconds = time.perf_counter() - started
    loaded = LeRobotDataset(
        REPO_ID, root=DEST, download_videos=False, video_backend="pyav", return_uint8=True
    )
    if loaded.num_episodes != len(prepared) or len(loaded) != sum(x["count"] for x in prepared):
        raise RuntimeError("reloaded LeRobot dataset count mismatch")
    checks = []
    for item, ep_range in zip(prepared, ep_ranges):
        start = int(item["spec"]["start"])
        errors = {"action_linf": 0.0, "state_linf": 0.0}
        success_indices = []
        camera_mae = {"agentview": [], "wrist": []}
        for local in range(item["count"]):
            row = loaded[ep_range["global_start"] + local]
            act = np.asarray(row["action"], dtype=np.float32)
            state = np.asarray(row["observation.state"], dtype=np.float32)
            ae = float(np.max(np.abs(act - item["actions"][local])))
            se = float(np.max(np.abs(state - item["state"][local])))
            errors["action_linf"] = max(errors["action_linf"], ae)
            errors["state_linf"] = max(errors["state_linf"], se)
            if ae > 1e-6 or se > 1e-6:
                raise RuntimeError(
                    f"v3 numeric roundtrip mismatch {item['spec']['episode_id']} row {local}: {ae}/{se}"
                )
            if bool(np.asarray(row["next.success"]).item()):
                success_indices.append(local)
            for key, half, name in [
                ("observation.images.agentview_image", 0, "agentview"),
                ("observation.images.robot0_eye_in_hand_image", 1, "wrist"),
            ]:
                got = np.asarray(row[key])
                if got.shape != (3, 256, 256) or got.dtype != np.uint8:
                    raise RuntimeError(f"bad decoded image {key}: {got.shape}/{got.dtype}")
                source = item["frames"][
                    start + local, :, half * 256 : (half + 1) * 256, :
                ].transpose(2, 0, 1)
                camera_mae[name].append(
                    float(np.abs(got.astype(np.float32) - source.astype(np.float32)).mean())
                )
        if success_indices != [item["count"] - 1]:
            raise RuntimeError(
                f"bad v3 terminal success markers for {item['spec']['episode_id']}: {success_indices}"
            )
        checks.append(
            {
                "episode_id": item["spec"]["episode_id"],
                "transitions": item["count"],
                **errors,
                "success_local_indices": success_indices,
                "image_mean_abs_error_vs_source": {
                    k: float(np.mean(v)) for k, v in camera_mae.items()
                },
            }
        )
    receipt = {
        "schema": "vla-forge.m04.lerobot-v3-verified-corrections.v1",
        "repo_id": REPO_ID,
        "root": str(DEST),
        "codebase_version": loaded.meta.info.codebase_version,
        "fps": FPS,
        "episodes": loaded.num_episodes,
        "transitions": len(loaded),
        "unique_action_sequences": len(seen_action_hashes),
        "episodes_index": ep_ranges,
        "sources": source_rows,
        "roundtrip_validation": checks,
        "write_and_encode_seconds": encode_seconds,
        "transitions_per_second": len(loaded) / encode_seconds,
        "release_scope": f"engineering preview: {len(seen_action_hashes)} unique successful corrected trajectories originate from the same LIBERO Goal task 6, M02 init state 7, M02 failed source episode. Not evidence of task/init-state generalization; public release remains gated by data quality and asset-license audit.",
    }
    (DEST / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
