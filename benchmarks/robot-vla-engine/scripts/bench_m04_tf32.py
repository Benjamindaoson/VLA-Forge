"""Paired FP32/TF32 SmolVLA forward benchmark on a real M04 observation."""

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
from robot_vla.observations import libero_batch

ROOT = Path("/data/VLA-Forge")
CKPT = ROOT / "Assets/models/smolvla_libero"
VLM = ROOT / "Assets/models/smolvlm2/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
NPZ = ROOT / "Evidence/M04/artifacts/first_correction/m04-first-correction-replay-01.npz"
OUT = ROOT / "Evidence/M04/performance/tf32_benchmark.json"


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

    def one(mode, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        allow_tf32 = mode == "tf32"
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        policy.reset()
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode():
            raw = policy.select_action(inp)
        torch.cuda.synchronize()
        return raw.detach().float().cpu().numpy()[0].copy(), (time.perf_counter() - t) * 1000

    for mode in ("fp32", "tf32"):
        for seed in (100, 101):
            one(mode, seed)
    pairs = []
    times = {"fp32": [], "tf32": []}
    for seed in range(200, 212):
        a, t1 = one("fp32", seed)
        b, t2 = one("tf32", seed)
        times["fp32"].append(t1)
        times["tf32"].append(t2)
        pairs.append(
            {"seed": seed, "mae": float(np.abs(a - b).mean()), "linf": float(np.abs(a - b).max())}
        )

    def pct(xs):
        return {
            "p50_ms": float(np.percentile(xs, 50)),
            "p95_ms": float(np.percentile(xs, 95)),
            "p99_ms": float(np.percentile(xs, 99)),
        }

    mae = max(x["mae"] for x in pairs)
    linf = max(x["linf"] for x in pairs)
    speed = float(np.median(times["fp32"]) / np.median(times["tf32"]))
    result = {
        "checkpoint_sha256": sha(CKPT / "model.safetensors"),
        "source_m04_trajectory_sha256": sha(NPZ),
        "observation_index": i,
        "state_semantics": sem,
        "gpu": torch.cuda.get_device_name(0),
        "samples": len(pairs),
        "baseline_fp32": pct(times["fp32"]),
        "candidate_tf32": pct(times["tf32"]),
        "speedup_median": speed,
        "paired_output_max_mae": mae,
        "paired_output_max_linf": linf,
        "tolerance": {"max_mae": 0.002, "max_linf": 0.02},
        "output_correctness_pass": bool(mae <= 0.002 and linf <= 0.02),
        "performance_pass": bool(speed > 1.05),
        "decision": "candidate accepted for rollout test"
        if mae <= 0.002 and linf <= 0.02 and speed > 1.05
        else "reject; keep FP32",
        "paired_samples": pairs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
