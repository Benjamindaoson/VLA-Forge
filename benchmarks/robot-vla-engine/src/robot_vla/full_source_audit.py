"""Complete selected-source HDF5 audit; no invented timing or pairing semantics."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from .schema import content_hash, file_hash

NAMES = ("groceries_human.hdf5", "groceries_robot.hdf5")


def verify_full_sources(folder, metadata):
    folder = Path(folder)
    receipt = json.loads((folder / "download_receipt.json").read_text(encoding="utf-8"))
    records = receipt.get("files", [])
    if (
        not receipt.get("completed")
        or receipt.get("revision") != metadata["sha"]
        or receipt.get("count") != 2
        or len(records) != 2
        or {r["path"] for r in records} != set(NAMES)
    ):
        raise ValueError("incomplete or mismatched source receipt")
    expected = {r["rfilename"]: r for r in metadata["siblings"]}
    for row in records:
        ref = expected[row["path"]]
        p = folder / row["path"]
        if (
            not row.get("upstream_lfs_verified")
            or row["bytes"] != ref["size"]
            or row["sha256"] != ref["lfs"]["sha256"]
            or p.stat().st_size != row["bytes"]
            or file_hash(p) != row["sha256"]
        ):
            raise ValueError(f"source checksum/size mismatch: {row['path']}")
    return records


def _scan(ds, block_bytes=32 * 1024**2):
    shape = list(ds.shape)
    if not shape:
        raise ValueError(f"scalar episode stream is unsupported: {ds.name}")
    stride = max(1, block_bytes // max(1, int(np.prod(shape[1:])) * ds.dtype.itemsize))
    digest = hashlib.sha256()
    minimum = maximum = None
    count = nonfinite = frames = 0
    numeric = np.issubdtype(ds.dtype, np.number) or ds.dtype == np.dtype(bool)
    if not numeric:
        raise ValueError(f"non-numeric episode stream is unsupported: {ds.name}")
    for start in range(0, len(ds), stride):
        values = np.asarray(ds[start : start + stride])
        frames += len(values)
        count += values.size
        digest.update(values.tobytes())
        if np.issubdtype(ds.dtype, np.inexact):
            finite = np.isfinite(values)
            nonfinite += int(values.size - finite.sum())
            values = values[finite]
        if values.size:
            lo, hi = float(values.min()), float(values.max())
            minimum = lo if minimum is None else min(minimum, lo)
            maximum = hi if maximum is None else max(maximum, hi)
    return dict(
        shape=shape,
        dtype=str(ds.dtype),
        frames_read=frames,
        elements=count,
        min=minimum,
        max=maximum,
        nonfinite=nonfinite,
        sha256=digest.hexdigest(),
        logical_bytes=count * ds.dtype.itemsize,
    )


def _signature(action, state):
    if len(action) != len(state):
        return None
    points = np.linspace(0, len(action) - 1, 32)
    lower, upper = np.floor(points).astype(int), np.ceil(points).astype(int)
    indices = np.unique(np.concatenate([lower, upper]))
    fraction = (points - lower)[:, None]
    result = []
    for ds in (action, state):
        values = np.asarray(ds[indices]).reshape(len(indices), -1)
        a, b = values[np.searchsorted(indices, lower)], values[np.searchsorted(indices, upper)]
        result.append(a * (1 - fraction) + b * fraction)
    return tuple(result)


def _timestamps(ds, frames, block_rows=4096):
    first = last = None
    count, monotonic = 0, True
    for start in range(0, len(ds), block_rows):
        values = np.asarray(ds[start : start + block_rows]).reshape(-1)
        count += len(values)
        if len(values):
            monotonic &= bool(
                np.isfinite(values).all()
                and (np.diff(values) > 0).all()
                and (last is None or values[0] > last)
            )
            first = float(values[0]) if first is None else first
            last = float(values[-1])
    return dict(
        strictly_increasing=monotonic,
        length_aligned=count == frames,
        duration=last - first if monotonic and count else None,
        physical_stream_alignment=None,
    )


def audit_complete_hdf5(path, *, action_key, embodiment):
    episodes, problems, signatures = [], [], {}
    duplicate_groups = defaultdict(list)
    with h5py.File(path, "r") as f:
        names = sorted(f["data"])
        masks = {
            k: [x.decode() if isinstance(x, bytes) else str(x) for x in v[:]]
            for k, v in f.get("mask", {}).items()
        }
        heldout = set().union(*(set(v) for k, v in masks.items() if k in {"valid", "val", "test"}))
        overlap = sorted(set(masks.get("train", [])) & heldout)
        unknown = sorted(set().union(*map(set, masks.values())) - set(names)) if masks else []
        for name in names:
            group = f["data"][name]
            if action_key not in group:
                raise ValueError(f"missing required action stream: {name}/{action_key}")
            frames = len(group[action_key])
            if frames == 0:
                raise ValueError(f"empty episode: {name}")
            streams = {}

            def visit(key, obj):
                if isinstance(obj, h5py.Dataset):
                    streams[key] = _scan(obj)

            group.visititems(visit)
            rgb = []
            for key, stats in streams.items():
                if stats["frames_read"] != frames:
                    problems.append(f"{name}: length mismatch {key}")
                if stats["nonfinite"]:
                    problems.append(f"{name}: nonfinite {key}")
                if key.startswith("obs/") and "img" in key and not key.endswith("_mask"):
                    valid_image = (
                        len(stats["shape"]) == 4
                        and all(dim > 0 for dim in stats["shape"])
                        and stats["shape"][-1] == 3
                        and stats["dtype"] == "uint8"
                    )
                    stats["raw_rgb_shape_valid"] = valid_image
                    if not valid_image:
                        problems.append(f"{name}: invalid image shape/dtype {key}")
                    else:
                        rgb.append(key)
            if not rgb:
                problems.append(f"{name}: missing RGB observation")
            if "obs/ee_pose" not in streams:
                problems.append(f"{name}: missing state obs/ee_pose")
            times = None
            if "timestamps" in group:
                times = _timestamps(group["timestamps"], frames)
                if not times["strictly_increasing"] or not times["length_aligned"]:
                    problems.append(f"{name}: invalid timestamps")
            signature = (
                _signature(group[action_key], group["obs/ee_pose"])
                if "obs/ee_pose" in group
                and not streams[action_key]["nonfinite"]
                and not streams["obs/ee_pose"]["nonfinite"]
                else None
            )
            signatures[name] = signature
            digest = content_hash(
                {k: {f: v[f] for f in ("shape", "dtype", "sha256")} for k, v in streams.items()}
            )
            duplicate_groups[digest].append(name)
            episodes.append(
                dict(
                    episode_id=name,
                    frames=frames,
                    streams=streams,
                    source_masks=[k for k, v in masks.items() if name in v],
                    timestamps=times,
                    success=None,
                    termination=None,
                    pair_id=None,
                    content_hash=digest,
                    rgb_streams=rgb,
                )
            )
    candidates, compared = [], 0
    for i, left in enumerate(episodes):
        for right in episodes[i + 1 :]:
            a, b = signatures[left["episode_id"]], signatures[right["episode_id"]]
            if a is None or b is None or any(x.shape != y.shape for x, y in zip(a, b)):
                continue
            compared += 1
            ratio = max(left["frames"], right["frames"]) / min(left["frames"], right["frames"])
            errors = [float(np.sqrt(np.mean((x - y) ** 2))) for x, y in zip(a, b)]
            if ratio <= 1.1 and errors[0] <= 0.02 and errors[1] <= 0.01:
                candidates.append(
                    dict(
                        left=left["episode_id"],
                        right=right["episode_id"],
                        length_ratio=ratio,
                        action_rmse=errors[0],
                        state_rmse=errors[1],
                    )
                )
    return dict(
        source=str(path),
        embodiment=embodiment,
        action_key=action_key,
        coverage="all episodes and all numeric datasets in this selected source file",
        episode_count=len(episodes),
        frame_count=sum(e["frames"] for e in episodes),
        episodes=episodes,
        split_counts={k: len(v) for k, v in masks.items()},
        source_masks=masks,
        split_overlap=overlap,
        unknown_mask_episodes=unknown,
        duplicate_groups=[v for v in duplicate_groups.values() if len(v) > 1],
        problems=problems,
        near_duplicates=dict(
            method="within-file 32-point normalized-time interpolation of full action horizon and ee_pose",
            thresholds=dict(length_ratio_max=1.1, action_rmse_max=0.02, state_rmse_max=0.01),
            interpretation="uncalibrated numeric screening in source units; candidates are not duplicate proof; no image comparison",
            pairs_compared=compared,
            candidates=candidates,
            skipped_episodes=[k for k, v in signatures.items() if v is None],
        ),
        human_robot_pair_integrity=None,
        physical_timestamp_alignment=None,
        success_and_termination="N/A: not supplied by this source contract",
        formal_training_ready=False,
        readiness_reason="Data audit does not validate temporal control semantics or authorize cross-embodiment policy training",
    )
