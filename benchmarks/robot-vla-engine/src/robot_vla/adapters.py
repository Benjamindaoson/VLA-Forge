"""Source adapters preserve source semantics and unavailable fields."""

import json
from collections import defaultdict
from pathlib import Path

import h5py
import pyarrow.parquet as pq

from .schema import Episode


def hdf5_episodes(
    path, *, dataset_id, revision, embodiment, action_semantics, action_key="actions"
):
    with h5py.File(path, "r") as handle:
        for name, group in sorted(handle["data"].items()):
            action = group.get(action_key)
            obs = group.get("obs", {})
            shapes = {
                key: list(value.shape) for key, value in obs.items() if hasattr(value, "shape")
            }
            frames = len(action) if action is not None else int(group.attrs.get("num_samples", 0))
            yield Episode(
                dataset_id=dataset_id,
                revision=revision,
                episode_id=name,
                task=Path(path).stem,
                embodiment=embodiment,
                frames=frames,
                source=f"{Path(path).resolve()}::/data/{name}",
                action_dim=action.shape[-1] if action is not None else None,
                action_semantics=action_semantics if action is not None else None,
                timestamp_basis="source" if "timestamps" in group else "unavailable",
                observations=shapes,
                metadata={
                    "source_format": "robomimic_hdf5",
                    "action_key": action_key,
                    "action_shape": list(action.shape) if action is not None else None,
                },
            )


def lerobot_episodes(root, *, dataset_id, revision):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    task_names = {}
    for row in pq.read_table(root / "meta/tasks.parquet").to_pylist():
        task_names[row["task_index"]] = row.get("task", row.get("__index_level_0__"))
    episode_tasks = defaultdict(set)
    for file in sorted((root / "data").rglob("*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(columns=["episode_index", "task_index"]):
            for item in batch.to_pylist():
                episode_tasks[item["episode_index"]].add(item["task_index"])
    for file in sorted((root / "meta/episodes").rglob("*.parquet")):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                indices = episode_tasks[row["episode_index"]]
                if len(indices) != 1:
                    raise ValueError(
                        f"episode {row['episode_index']} has missing or ambiguous task mapping"
                    )
                task_index = next(iter(indices))
                if task_index not in task_names or not task_names[task_index]:
                    raise ValueError(f"unknown task index {task_index}")
                tasks = [task_names[task_index]]
                yield Episode(
                    dataset_id=dataset_id,
                    revision=revision,
                    episode_id=str(row["episode_index"]),
                    task=tasks[0] if tasks else "unavailable",
                    instruction=tasks[0] if tasks else None,
                    embodiment=info["robot_type"],
                    frames=int(row["length"]),
                    source=str(root.resolve()),
                    timestamp_basis="index/fps",
                    fps=info["fps"],
                    action_dim=info["features"]["action"]["shape"][0],
                    action_semantics="LIBERO relative OSC_POSE: xyz+axis_angle delta, gripper",
                    state_dim=info["features"]["observation.state"]["shape"][0],
                    observations={k: v for k, v in info["features"].items() if ".images." in k},
                    metadata={
                        k: v
                        for k, v in row.items()
                        if k.startswith(("data/", "videos/", "dataset_"))
                    },
                )
