"""Bounded, actual M05 video-reader comparison; no GPU inference overlaps training."""

import json
import time

import numpy as np
import psutil
from torch.utils.data import DataLoader, Subset

from robot_vla.m05_training import (
    EVIDENCE,
    atomic_json,
    corrective_dataset,
    dataset_for,
    training_batch,
)

out = EVIDENCE / "training_perf/data_pipeline"
out.mkdir(exist_ok=True)
ids = np.random.default_rng(5101).integers(0, 857, size=32).tolist()
rows = []
images = {}
for backend in ["pyav", "torchcodec"]:
    try:
        t = time.perf_counter()
        ds = corrective_dataset(backend)
        load = time.perf_counter() - t
        for workers in [0, 2]:
            loader = DataLoader(
                Subset(ds, ids), batch_size=2, num_workers=workers, pin_memory=False
            )
            t = time.perf_counter()
            last = t
            lat = []
            for batch in loader:
                now = time.perf_counter()
                lat.append(now - last)
                last = now
            rows.append(
                {
                    "backend": backend,
                    "workers": workers,
                    "samples": 32,
                    "wall_s": time.perf_counter() - t,
                    "batch_wait_s": lat,
                    "rss": psutil.Process().memory_info().rss,
                    "load_s": load,
                }
            )
        images[backend] = [ds[i] for i in ids[:4]]
    except Exception as exc:
        rows.append({"backend": backend, "error": repr(exc)})
comparisons = []
if len(images) == 2:
    for i, (a, b) in enumerate(zip(images["pyav"], images["torchcodec"])):
        comparisons.append(
            {
                "index": ids[i],
                "state_linf": float((a["observation.state"] - b["observation.state"]).abs().max()),
                "action_linf": float((a["action"] - b["action"]).abs().max()),
                "image_mae": float(
                    (
                        a["observation.images.agentview_image"]
                        - b["observation.images.agentview_image"]
                    )
                    .abs()
                    .mean()
                ),
            }
        )
# Frozen genuine observations reused by every inference backend.
batches = []
for kind, positions in [("corrective", [0, 300, 856]), ("standard", [0, 300, 856])]:
    ds = dataset_for(kind)
    for pos in positions:
        b = training_batch(ds, [pos])
        record = {
            "image1": (b["observation.images.image"][0].permute(1, 2, 0).numpy() * 255)
            .round()
            .astype(np.uint8),
            "image2": (b["observation.images.image2"][0].permute(1, 2, 0).numpy() * 255)
            .round()
            .astype(np.uint8),
            "state": b["observation.state"][0].numpy(),
            "task": np.asarray(b["task"][0]),
        }
        p = out / f"observation-{kind}-{pos}.npz"
        np.savez_compressed(p, **record)
        batches.append({"path": str(p), "dataset": kind, "local_index": pos})
atomic_json(
    out / "results.json",
    {
        "rows": rows,
        "comparisons": comparisons,
        "observations": batches,
        "ordering": "PyAV 0/2 workers then TorchCodec 0/2; descriptive cold process measurement, not randomized causal comparison",
        "h2d": "Measured in GPU training processor phase; this CPU-only comparison intentionally excludes CUDA.",
    },
)
print(json.dumps({"rows": rows, "comparisons": comparisons}))
