"""Bounded-memory audits with explicit coverage and unmeasured semantics."""

import hashlib
from collections import defaultdict

import h5py
import numpy as np


def numeric_stats(dataset, block_size=256):
    minimum, maximum, nonfinite, count = None, None, 0, 0
    for start in range(0, len(dataset), block_size):
        values = np.asarray(dataset[start : start + block_size])
        count += values.size
        mask = np.isfinite(values)
        nonfinite += int((~mask).sum())
        finite = values[mask]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            minimum = lo if minimum is None else min(minimum, lo)
            maximum = hi if maximum is None else max(maximum, hi)
    return dict(
        shape=list(dataset.shape), min=minimum, max=maximum, nonfinite=nonfinite, count=count
    )


def audit_hdf5(path, max_episodes=None, action_key="actions"):
    episodes, content = [], defaultdict(list)
    problems = []
    with h5py.File(path, "r") as f:
        names = sorted(f["data"])
        masks = {
            name: {x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]}
            for name, ds in f.get("mask", {}).items()
        }
        train = masks.get("train", set())
        heldout = set().union(*(v for k, v in masks.items() if k in {"valid", "val", "test"}))
        overlap = sorted(train & heldout)
        for name in names[:max_episodes]:
            group = f["data"][name]
            actions = group.get(action_key)
            frames = len(actions) if actions is not None else int(group.attrs.get("num_samples", 0))
            record = dict(
                episode_id=name,
                frames=frames,
                actions=None,
                observations={},
                timestamps=None,
                success=None,
                termination=None,
                pair_id=None,
            )
            digest = hashlib.sha256()
            if actions is not None:
                record["actions"] = numeric_stats(actions)
                for start in range(0, frames, 256):
                    digest.update(np.asarray(actions[start : start + 256]).tobytes())
                if record["actions"]["nonfinite"]:
                    problems.append(f"{name}: nonfinite action")
            else:
                problems.append(f"{name}: missing action")
            for key, ds in group.get("obs", {}).items():
                if not isinstance(ds, h5py.Dataset):
                    continue
                stats = (
                    numeric_stats(ds)
                    if np.issubdtype(ds.dtype, np.number)
                    else {"shape": list(ds.shape), "numeric": False}
                )
                record["observations"][key] = stats
                digest.update(key.encode())
                for start in range(0, len(ds), 64):
                    digest.update(np.asarray(ds[start : start + 64]).tobytes())
                if len(ds) != frames:
                    problems.append(f"{name}: stream length mismatch {key}")
                if stats.get("nonfinite", 0):
                    problems.append(f"{name}: nonfinite observation {key}")
            if "timestamps" in group:
                times = np.asarray(group["timestamps"][:]).reshape(-1)
                monotonic = bool(np.isfinite(times).all() and (np.diff(times) > 0).all())
                record["timestamps"] = dict(
                    strictly_increasing=monotonic,
                    aligned=len(times) == frames,
                    duration=float(times[-1] - times[0]) if monotonic and len(times) else None,
                )
                if not monotonic or len(times) != frames:
                    problems.append(f"{name}: invalid timestamps")
            record["content_hash"] = digest.hexdigest()
            content[digest.hexdigest()].append(name)
            episodes.append(record)
        duplicates = [v for v in content.values() if len(v) > 1]
        return dict(
            source=str(path),
            source_episode_count=len(names),
            episode_count=len(episodes),
            frame_count=sum(r["frames"] for r in episodes),
            episodes=episodes,
            coverage="full" if len(episodes) == len(names) else "sample",
            split_overlap=overlap,
            split_counts={k: len(v) for k, v in masks.items()},
            duplicate_groups=duplicates,
            problems=problems,
            near_duplicates="N/A: not measured",
            human_robot_pair_integrity="N/A: no pairing manifest",
            formal_training_ready=False,
            readiness_reason="requires explicit source semantics and validated split; audit alone cannot authorize training",
        )
