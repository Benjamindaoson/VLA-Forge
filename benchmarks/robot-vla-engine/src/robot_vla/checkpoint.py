"""Atomic model/optimizer/RNG snapshots with immutable experiment identity."""

import random
from pathlib import Path

import numpy as np
import torch

from .schema import content_hash


def save(path, model, optimizer, step, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np_state = np.random.get_state()
    state = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        step=step,
        identity_hash=content_hash(identity),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        python_rng=random.getstate(),
        numpy_rng=(
            np_state[0],
            torch.tensor(np_state[1].astype("int64")),
            np_state[2],
            np_state[3],
            np_state[4],
        ),
    )
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    temp.replace(path)


def restore(path, model, optimizer, identity):
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if state["identity_hash"] != content_hash(identity):
        raise ValueError("checkpoint experiment identity mismatch")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint RNG cannot be restored on CPU")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    n = state["numpy_rng"]
    np.random.set_state((n[0], n[1].numpy().astype("uint32"), n[2], n[3], n[4]))
    return state["step"]
