"""Strict sampled input equivalence for alternative official video decoders."""

import hashlib

import numpy as np


def verify_bindings(protocol, compute, source, corpus):
    if (
        compute["data_protocol_identity"] != protocol["identity"]
        or corpus["data_protocol_identity"] != protocol["identity"]
        or source["revision"] != protocol["revision"]
        or source != corpus["source"]
    ):
        raise ValueError("compute/data/source/corpus identities differ")


def choose_probe_indices(rows, episode_ids):
    if any(not (type(e) is int or isinstance(e, str) and e.isdecimal()) for e in episode_ids):
        raise ValueError("episode IDs must be integers or decimal strings")
    wanted = {int(e) for e in episode_ids}
    if not wanted or len(wanted) != len(episode_ids):
        raise ValueError("invalid episode selection")
    groups = {e: [] for e in wanted}
    seen = set()
    for row in rows:
        if row["episode_index"] not in wanted:
            continue
        coordinate = row["episode_index"], row["frame_index"]
        if coordinate in seen:
            raise ValueError("duplicate source coordinate")
        seen.add(coordinate)
        groups[row["episode_index"]].append(row)
    result = []
    for episode in sorted(wanted):
        group = sorted(groups[episode], key=lambda r: r["frame_index"])
        if len(group) < 3 or [r["frame_index"] for r in group] != list(range(len(group))):
            raise ValueError("missing or discontinuous source episode")
        if len({r["task_index"] for r in group}) != 1:
            raise ValueError("source episode has multiple tasks")
        result.extend(group[i]["index"] for i in [0, len(group) // 2, len(group) - 1])
    if len(set(result)) != len(result):
        raise ValueError("duplicate global source index")
    return result


def arrays(item):
    result = {}
    for key, value in item.items():
        if isinstance(value, str):
            result[key] = value
            continue
        if hasattr(value, "detach"):
            if value.device.type != "cpu":
                raise ValueError("backend audit requires CPU tensors")
            value = value.detach().numpy()
        value = np.asarray(value)
        if value.dtype.kind not in "buif" or not np.isfinite(value).all():
            raise ValueError("unsupported/nonfinite dataset field")
        result[key] = value.copy()
    return result


def item_hashes(item):
    return {
        k: dict(
            dtype=str(v.dtype), shape=list(v.shape), sha256=hashlib.sha256(v.tobytes()).hexdigest()
        )
        if isinstance(v, np.ndarray)
        else dict(text=v)
        for k, v in arrays(item).items()
    }


def compare_items(reference, candidate):
    reference, candidate = arrays(reference), arrays(candidate)
    if reference.keys() != candidate.keys():
        raise ValueError("dataset field set differs")
    fields = {}
    for key, left in reference.items():
        right = candidate[key]
        if isinstance(left, str) or isinstance(right, str):
            if type(left) is not type(right):
                raise ValueError("dataset text type differs")
            fields[key] = dict(exact=left == right, max_abs=None, rmse=None)
            continue
        if left.dtype != right.dtype or left.shape != right.shape:
            raise ValueError("dataset dtype/shape differs")
        delta = left.astype(np.float64) - right.astype(np.float64)
        fields[key] = dict(
            exact=bool(np.array_equal(left, right)),
            max_abs=float(np.abs(delta).max()) if delta.size else 0.0,
            rmse=float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
        )
    return dict(exact=all(v["exact"] for v in fields.values()), fields=fields)
