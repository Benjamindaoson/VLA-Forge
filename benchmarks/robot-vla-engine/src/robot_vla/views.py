"""Traceable source-frame selection without inventing temporal alignment."""

import h5py
import numpy as np

from .schema import file_hash


def extract_view(source, destination, selections, keys):
    records = []
    with h5py.File(destination, "x") as out:
        for name, indices in selections.items():
            parent = source["data"][name]
            indices = np.asarray(indices, dtype=np.int64)
            length = parent[keys[0]].shape[0]
            if (
                indices.ndim != 1
                or not len(indices)
                or (indices < 0).any()
                or (indices >= length).any()
                or (np.diff(indices) <= 0).any()
            ):
                raise ValueError("frame selections must be sorted, unique and within source bounds")
            group = out.require_group(f"data/{name}")
            group.attrs["num_samples"] = len(indices)
            group.attrs["parent_num_samples"] = length
            group.attrs["is_sparse_view"] = True
            group.create_dataset("source_frame_index", data=indices)
            for key in keys:
                dataset = parent[key]
                if len(dataset) != length:
                    raise ValueError(f"unaligned source stream: {key}")
                target = group.create_dataset(
                    key,
                    shape=(len(indices), *dataset.shape[1:]),
                    dtype=dataset.dtype,
                    compression="gzip",
                    compression_opts=1,
                )
                for i, frame in enumerate(indices):
                    target[i] = dataset[int(frame)]
            records.append(
                dict(
                    episode_id=name,
                    parent_frames=length,
                    selected_frames=indices.tolist(),
                    keys=keys,
                )
            )
    return dict(
        coverage="sparse source-frame view, not full episodes",
        copied_frames=sum(len(x["selected_frames"]) for x in records),
        episodes=records,
        sha256=file_hash(destination),
        pair_relationship=None,
    )
