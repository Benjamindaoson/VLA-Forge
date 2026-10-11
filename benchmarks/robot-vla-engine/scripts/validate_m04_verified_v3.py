"""Independently validate the frozen two-episode M04 LeRobot v3 dataset."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.env_processor import LiberoProcessorStep

DATASET = Path(
    os.environ.get(
        "M04_DATASET_PATH", "/data/VLA-Forge/Evidence/M04/dataset/lerobot_v3_verified_r2"
    )
)
OUT = DATASET / "independent_validation.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    provenance = json.loads((DATASET / "provenance.json").read_text())
    dataset = LeRobotDataset(
        provenance["repo_id"],
        root=DATASET,
        download_videos=False,
        video_backend="pyav",
        return_uint8=True,
    )
    if dataset.num_episodes != provenance["episodes"] or len(dataset) != provenance["transitions"]:
        raise AssertionError(
            f"dataset coverage mismatch: {dataset.num_episodes} episodes / {len(dataset)} frames"
        )
    state_max = action_max = 0.0
    image_means: list[float] = []
    episode_receipts = []
    processor = LiberoProcessorStep()
    for index, source in zip(provenance["episodes_index"], provenance["sources"], strict=True):
        if source["independent_replay_success"] is not True:
            raise AssertionError(f"source replay is not verified: {source['episode_id']}")
        source_path = Path(source["source_trajectory"])
        if sha256(source_path) != source["source_trajectory_sha256"]:
            raise AssertionError(f"source trajectory hash mismatch: {source_path}")
        count = int(source["correction_actions"])
        with np.load(source_path) as z:
            start = len(z["qpos"]) - 1 - count
            actions = z["actions"][start : start + count]
            quat = torch.from_numpy(z["eef_quat"][start : start + count].astype(np.float32))
            axis_angle = processor._quat2axisangle(quat).cpu().numpy()
            states = np.concatenate(
                [
                    z["eef_pos"][start : start + count],
                    axis_angle,
                    z["gripper_qpos"][start : start + count],
                ],
                axis=-1,
            ).astype(np.float32)
            camera_frames = z["dual_camera_frames"][start : start + count]
        if len(actions) != count or states.shape != (count, 8):
            raise AssertionError(f"source alignment mismatch: {source['episode_id']}")
        begin, end = int(index["global_start"]), int(index["global_end_inclusive"])
        if end - begin + 1 != count:
            raise AssertionError(f"episode range mismatch: {source['episode_id']}")
        episode_state_max = episode_action_max = 0.0
        episode_image_mean = []
        success_indices, done_indices = [], []
        for offset in range(count):
            row = dataset[begin + offset]
            action_err = float(
                np.max(np.abs(np.asarray(row["action"], dtype=np.float32) - actions[offset]))
            )
            state_err = float(
                np.max(
                    np.abs(np.asarray(row["observation.state"], dtype=np.float32) - states[offset])
                )
            )
            episode_action_max = max(episode_action_max, action_err)
            episode_state_max = max(episode_state_max, state_err)
            if action_err > 1e-6 or state_err > 1e-6:
                raise AssertionError(
                    f"numeric roundtrip mismatch at {source['episode_id']}:{offset}"
                )
            for key, camera_index in (
                ("observation.images.agentview_image", 0),
                ("observation.images.robot0_eye_in_hand_image", 1),
            ):
                actual = np.asarray(row[key], dtype=np.uint8)
                expected = camera_frames[
                    offset, :, camera_index * 256 : (camera_index + 1) * 256
                ].transpose(2, 0, 1)
                if actual.shape != (3, 256, 256) or not np.any(actual):
                    raise AssertionError(
                        f"invalid image frame for {source['episode_id']}:{offset}:{key}"
                    )
                image_error = float(
                    np.abs(actual.astype(np.int16) - expected.astype(np.int16)).mean()
                )
                episode_image_mean.append(image_error)
            if bool(np.asarray(row["next.success"]).item()):
                success_indices.append(offset)
            if bool(np.asarray(row["next.done"]).item()):
                done_indices.append(offset)
        if success_indices != [count - 1] or done_indices != [count - 1]:
            raise AssertionError(f"terminal success/done mismatch for {source['episode_id']}")
        state_max = max(state_max, episode_state_max)
        action_max = max(action_max, episode_action_max)
        image_means.extend(episode_image_mean)
        episode_receipts.append(
            {
                "episode_id": source["episode_id"],
                "source_trajectory_sha256": source["source_trajectory_sha256"],
                "transitions": count,
                "action_linf": episode_action_max,
                "state_linf": episode_state_max,
                "mean_abs_image_error": float(np.mean(episode_image_mean)),
                "success_index": count - 1,
                "done_index": count - 1,
                "physical_replay_success": True,
            }
        )
    files = [p for p in DATASET.rglob("*") if p.is_file() and p != OUT]
    receipt = {
        "schema": "vla-forge.m04.lerobot-v3-independent-validation.v1",
        "status": "PASS",
        "dataset_root": str(DATASET),
        "repo_id": provenance["repo_id"],
        "episodes": dataset.num_episodes,
        "transitions": len(dataset),
        "action_max_linf": action_max,
        "state_max_linf": state_max,
        "mean_abs_image_error_across_camera_frames": float(np.mean(image_means)),
        "episode_receipts": episode_receipts,
        "file_manifest": [
            {
                "path": str(path.relative_to(DATASET)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(files)
        ],
    }
    OUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "file_manifest"}, indent=2))


if __name__ == "__main__":
    main()
