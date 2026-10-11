"""Full-action-chunk eager/CUDA-graph benchmark on real recorded observations."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from robot_vla.m01 import fit_libero_state_to_checkpoint
from robot_vla.m04_acceleration import CudaGraphSampler
from robot_vla.observations import libero_batch

ROOT = Path("/data/VLA-Forge")
CKPT = ROOT / "Assets/models/smolvla_libero"
VLM = ROOT / "Assets/models/smolvlm2/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
NPZ = ROOT / "Evidence/M04/artifacts/first_correction/m04-first-correction-replay-01.npz"
OUT = ROOT / "Evidence/M04/performance/cuda_graph_benchmark.json"


def sha(p):

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    cfg = PreTrainedConfig.from_pretrained(CKPT)
    cfg.device = "cuda"
    cfg.vlm_model_name = str(VLM)
    policy = SmolVLAPolicy.from_pretrained(
        CKPT, config=cfg, strict=True, local_files_only=True
    ).eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=CKPT,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": str(VLM)},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    with np.load(NPZ) as z:
        i = 110
        frame = z["dual_camera_frames"][i]
        quat = z["eef_quat"][i]
        pos = z["eef_pos"][i]
        grip = z["gripper_qpos"][i]
    obs = {
        "pixels": {"image": frame[:, :256].copy(), "image2": frame[:, 256:].copy()},
        "robot_state": {
            "eef": {"pos": pos.copy(), "quat": quat.copy()},
            "gripper": {"qpos": grip.copy()},
        },
    }
    batch = libero_batch(obs)
    batch["observation.state"], sem = fit_libero_state_to_checkpoint(
        batch["observation.state"], expected_dim=6, normalizer_dim=8
    )
    batch["task"] = ["put the cream cheese in the bowl"]
    inp = pre(batch)

    original = policy.model.sample_actions
    graph = CudaGraphSampler(policy.model)

    def one(mode, seed, value):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        policy.model.sample_actions = original if mode == "eager" else graph
        policy.reset()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            raw = policy.predict_action_chunk(value)
        torch.cuda.synchronize()
        return (
            raw.detach().float().cpu().numpy().copy(),
            (time.perf_counter() - t) * 1000,
            torch.cuda.get_rng_state().cpu().numpy(),
        )

    one("eager", 100, inp)
    one("graph", 100, inp)
    times = {"eager": [], "graph": []}
    pairs = []
    with np.load(NPZ) as z:
        for index in [100, 110, 120]:
            frame = z["dual_camera_frames"][index]
            obs = {
                "pixels": {"image": frame[:, :256].copy(), "image2": frame[:, 256:].copy()},
                "robot_state": {
                    "eef": {"pos": z["eef_pos"][index].copy(), "quat": z["eef_quat"][index].copy()},
                    "gripper": {"qpos": z["gripper_qpos"][index].copy()},
                },
            }
            batch = libero_batch(obs)
            batch["observation.state"], _ = fit_libero_state_to_checkpoint(
                batch["observation.state"], expected_dim=6, normalizer_dim=8
            )
            batch["task"] = ["put the cream cheese in the bowl"]
            inp = pre(batch)
            for seed in range(200, 206):
                results = {}
                for mode in ("eager", "graph") if seed % 2 else ("graph", "eager"):
                    results[mode] = one(mode, seed, inp)
                    times[mode].append(results[mode][1])
                a, _, r1 = results["eager"]
                b, _, r2 = results["graph"]
                pairs.append(
                    {
                        "observation_index": index,
                        "seed": seed,
                        "shape": list(a.shape),
                        "max_abs_error": float(np.abs(a - b).max()),
                        "rng_equal": bool(np.array_equal(r1, r2)),
                    }
                )

    def pct(xs):
        return {f"p{n}_ms": float(np.percentile(xs, n)) for n in [50, 95, 99]}

    speed = float(np.median(times["eager"]) / np.median(times["graph"]))
    valid = all(x["max_abs_error"] <= 1e-5 and x["rng_equal"] for x in pairs)
    result = {
        "checkpoint_sha256": sha(CKPT / "model.safetensors"),
        "source_trajectory_sha256": sha(NPZ),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "samples": len(pairs),
        "pairs": pairs,
        "eager": pct(times["eager"]),
        "cuda_graph": pct(times["graph"]),
        "speedup": speed,
        "cold_capture_seconds": graph.cold_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "correctness_pass": valid,
        "performance_pass": speed > 1.05,
        "decision": "ACCEPT_FOR_ROLLOUT" if valid and speed > 1.05 else "REJECT",
        "timing_scope": "full raw action chunk, GPU synchronized; preprocessing and simulator excluded",
        "times_ms": times,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
