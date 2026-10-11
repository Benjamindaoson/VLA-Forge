"""Canonical M05 supervision and resumable sample ordering."""

import contextlib
import hashlib
import json
import os
import random
import signal
import threading
import time
from pathlib import Path

import numpy as np


def action_window(start, end, chunk):
    if not 0 <= start < end or chunk < 1:
        raise ValueError("Invalid action window")
    raw = np.arange(start, start + chunk)
    return np.minimum(raw, end - 1), raw >= end


def validate_sample(state, actions):
    state, actions = np.asarray(state), np.asarray(actions)
    if state.shape != (8,) or actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError("Expected canonical 8D state and 7D actions")
    if (
        not np.isfinite(state).all()
        or not np.isfinite(actions).all()
        or np.abs(actions).max() > 1.00001
    ):
        raise ValueError("Nonfinite or illegal supervision")


class IndexSampler:
    def __init__(self, size, seed):
        if size < 1:
            raise ValueError("Empty data")
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.exposures = 0

    def take(self, count):
        result = self.rng.integers(0, self.size, size=count)
        self.exposures += count
        return result

    def state_dict(self):
        return {"size": self.size, "rng": self.rng.bit_generator.state, "exposures": self.exposures}

    def load_state_dict(self, state):
        if state["size"] != self.size:
            raise ValueError("Dataset changed on resume")
        self.rng.bit_generator.state = state["rng"]
        self.exposures = state["exposures"]


# The implementation below delegates the objective and preprocessing to pinned LeRobot.
ROOT = Path("/data/VLA-Forge")
BASE = ROOT / "Assets/models/smolvla_libero"
VLM = ROOT / "Assets/models/smolvlm2/7b375e1b73b11138ff12fe22c8f2822d8fe03467"
DATA = ROOT / "Evidence/M04/dataset/lerobot_v3_verified_r7"
EVIDENCE = ROOT / "Evidence/M05"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def append_event(path, obj):
    with Path(path).open("a") as f:
        f.write(json.dumps(obj, default=str) + "\n")
        f.flush()


class Telemetry:
    def __init__(self, path):
        self.path = Path(path)
        self.stop = threading.Event()
        self.phase = "startup"

    def __enter__(self):
        import psutil
        import pynvml as nv

        nv.nvmlInit()
        h = nv.nvmlDeviceGetHandleByIndex(0)
        proc = psutil.Process()

        def sample():
            while not self.stop.is_set():
                try:
                    u = nv.nvmlDeviceGetUtilizationRates(h)
                    m = nv.nvmlDeviceGetMemoryInfo(h)
                    append_event(
                        self.path,
                        {
                            "utc_s": time.time(),
                            "monotonic_s": time.perf_counter(),
                            "phase": self.phase,
                            "pid": os.getpid(),
                            "rss": proc.memory_info().rss,
                            "cpu_percent": proc.cpu_percent(),
                            "gpu_util": u.gpu,
                            "gpu_memory": m.used,
                            "gpu_power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000,
                            "io": proc.io_counters()._asdict(),
                            "gpu_uuid": nv.nvmlDeviceGetUUID(h),
                        },
                    )
                except Exception as exc:
                    append_event(self.path, {"error": repr(exc)})
                self.stop.wait(0.5)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def load_policy(adapter=None, train=False, lora=True):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = PreTrainedConfig.from_pretrained(BASE)
    cfg.device = "cuda"
    cfg.vlm_model_name = str(VLM)
    p = SmolVLAPolicy.from_pretrained(BASE, config=cfg, strict=True, local_files_only=True)
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=BASE,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": str(VLM)},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    if adapter:
        from peft import PeftModel

        p = PeftModel.from_pretrained(p, str(adapter), is_trainable=train)
    elif lora:
        p = p.wrap_with_peft(peft_cli_overrides={"r": 8, "lora_alpha": 16, "lora_dropout": 0.0})
    p.train(train)
    return p, pre, post


def corrective_dataset(backend="pyav"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        "vla_forge/m04_verified_corrections",
        root=DATA,
        episodes=list(range(1, 12)),
        delta_timestamps={"action": [i / 20 for i in range(50)]},
        video_backend=backend,
    )


def training_batch(dataset, indices):
    import torch
    from torch.utils.data import default_collate

    samples = []
    for i in indices:
        item = dict(dataset[int(i)])
        for old, new in [
            ("observation.images.agentview_image", "observation.images.image"),
            ("observation.images.robot0_eye_in_hand_image", "observation.images.image2"),
        ]:
            if old in item:
                item[new] = item.pop(old)
        required = [
            "observation.images.image",
            "observation.images.image2",
            "observation.state",
            "action",
            "action_is_pad",
            "task",
        ]
        samples.append({k: item[k] for k in required})
    batch = default_collate(samples)
    for s, a in zip(batch["observation.state"], batch["action"]):
        validate_sample(s.numpy(), a.numpy())
    assert batch["action_is_pad"].dtype == torch.bool
    return batch


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(s):
    import torch

    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def train_job(config):
    import subprocess

    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file

    start = time.perf_counter()
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    steps = int(config["steps"])
    micro = int(config.get("microbatch", 1))
    accum = int(config.get("accumulation", 1))
    if not 1 <= steps <= 1000 or micro < 1 or accum < 1:
        raise ValueError("Unbounded/invalid run")
    precision = config.get("precision", "fp32")
    if precision not in ("fp32", "bf16", "fp16", "tf32"):
        raise ValueError(precision)
    torch.set_num_threads(1)
    seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
    torch.backends.cudnn.allow_tf32 = precision == "tf32"
    resume = config.get("resume")
    if (out / "result.json").exists() and not resume:
        raise FileExistsError("Run already completed")
    atomic_json(out / "config.json", config)
    config_hash = sha256(out / "config.json")
    stop = {"value": False}
    old_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda *_: stop.update(value=True))
    with Telemetry(out / "resources.jsonl") as telemetry:
        telemetry.phase = "model_load"
        t = time.perf_counter()
        policy, pre, post = load_policy(train=True)
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t
        if config.get("gradient_checkpointing"):
            from torch.utils.checkpoint import checkpoint

            expert = policy.get_base_model().model.vlm_with_expert
            original = expert.forward
            expert.forward = lambda *a, **kw: checkpoint(original, *a, use_reentrant=False, **kw)
        parameters = [p for p in policy.parameters() if p.requires_grad]
        names = [n for n, p in policy.named_parameters() if p.requires_grad]
        initial = {
            n: p.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad
        }
        total = sum(p.numel() for p in policy.parameters())
        trainable = sum(p.numel() for p in parameters)
        atomic_json(
            out / "identity.json",
            {
                "schema": "vla.m05.train.v1",
                "run_id": config["run_id"],
                "config_sha256": config_hash,
                "base_sha256": sha256(BASE / "model.safetensors"),
                "processor_sha256": sha256(BASE / "policy_preprocessor.json"),
                "normalizer_sha256": sha256(
                    BASE / "policy_preprocessor_step_5_normalizer_processor.safetensors"
                ),
                "dataset_sha256": sha256(DATA / "provenance.json")
                if config["dataset"] == "corrective"
                else sha256(EVIDENCE / "data/selection.json"),
                "code_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "trainable_parameters": trainable,
                "total_parameters": total,
                "lora_trainable_ratio": trainable / total,
                "trainable_names": names,
                "precision": precision,
                "microbatch": micro,
                "accumulation": accum,
                "effective_batch": micro * accum,
                "model_load_s": load_s,
                "hardware": torch.cuda.get_device_name(),
                "off_policy_supervision": True,
            },
        )
        telemetry.phase = "dataset_load"
        t = time.perf_counter()
        ds = dataset_for(config["dataset"], config.get("video_backend", "pyav"))
        sampler = IndexSampler(len(ds), config["seed"] + 1)
        dataset_load_s = time.perf_counter() - t
        opt = torch.optim.AdamW(
            parameters, lr=float(config["lr"]), weight_decay=0.01, fused=config.get("fused", False)
        )
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda n: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
        step0 = 0
        if resume:
            ck = torch.load(resume, map_location="cpu", weights_only=False)
            if ck["identity"] != {
                k: config.get(k)
                for k in [
                    "dataset",
                    "seed",
                    "lr",
                    "precision",
                    "microbatch",
                    "accumulation",
                    "steps",
                ]
            }:
                raise ValueError("Resume recipe mismatch")
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(policy, ck["adapter"])
            opt.load_state_dict(ck["optimizer"])
            sched.load_state_dict(ck["scheduler"])
            scaler.load_state_dict(ck["scaler"])
            sampler.load_state_dict(ck["sampler"])
            restore_rng(ck["rng"])
            step0 = ck["step"]
        grad_finite = True
        rows = []
        torch.cuda.reset_peak_memory_stats()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
        profile = bool(config.get("profile", False))
        prof = (
            torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            )
            if profile
            else contextlib.nullcontext()
        )
        telemetry.phase = "training"
        interrupted = False
        with prof as recorder:
            for step in range(step0, steps):
                t0 = time.perf_counter()
                opt.zero_grad(set_to_none=True)
                loss_values = []
                decode = prep = forward = backward = 0.0
                sampled = []
                for _ in range(accum):
                    ids = sampler.take(micro)
                    sampled.extend(ids.tolist())
                    t = time.perf_counter()
                    with torch.profiler.record_function("m05/data_decode"):
                        batch = training_batch(ds, ids)
                    decode += time.perf_counter() - t
                    t = time.perf_counter()
                    with torch.profiler.record_function("m05/processor_and_h2d"):
                        batch = pre(batch)
                    torch.cuda.synchronize()
                    prep += time.perf_counter() - t
                    t = time.perf_counter()
                    with torch.profiler.record_function("m05/official_flow_forward"):
                        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
                            loss, _ = policy(batch)
                    torch.cuda.synchronize()
                    forward += time.perf_counter() - t
                    t = time.perf_counter()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss")
                    with torch.profiler.record_function("m05/backward"):
                        scaler.scale(loss / accum).backward()
                    torch.cuda.synchronize()
                    backward += time.perf_counter() - t
                    loss_values.append(float(loss.detach()))
                scaler.unscale_(opt)
                if step == 0:
                    torch.save(
                        {
                            n: p.grad.detach().cpu().clone()
                            for n, p in policy.named_parameters()
                            if p.requires_grad and p.grad is not None
                        },
                        out / "first_gradients.pt",
                    )
                grad = torch.nn.utils.clip_grad_norm_(parameters, 10.0)
                grad_finite = grad_finite and bool(torch.isfinite(grad))
                if not grad_finite:
                    raise FloatingPointError("Nonfinite gradient")
                t = time.perf_counter()
                with torch.profiler.record_function("m05/optimizer"):
                    scaler.step(opt)
                    scaler.update()
                    sched.step()
                torch.cuda.synchronize()
                optimizer_s = time.perf_counter() - t
                row = {
                    "run_id": config["run_id"],
                    "step": step + 1,
                    "loss": float(np.mean(loss_values)),
                    "gradient_norm": float(grad),
                    "lr": opt.param_groups[0]["lr"],
                    "indices": sampled,
                    "sample_exposures": sampler.exposures,
                    "decode_s": decode,
                    "processor_h2d_s": prep,
                    "forward_s": forward,
                    "backward_s": backward,
                    "optimizer_s": optimizer_s,
                    "update_wall_s": time.perf_counter() - t0,
                    "allocated_peak": torch.cuda.max_memory_allocated(),
                    "reserved_peak": torch.cuda.max_memory_reserved(),
                    "grad_scaler": scaler.get_scale(),
                    "precision": precision,
                }
                append_event(out / "steps.jsonl", row)
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "step": step + 1,
                            "loss": row["loss"],
                            "grad": float(grad),
                            "seconds": row["update_wall_s"],
                        }
                    ),
                    flush=True,
                )
                if profile:
                    recorder.step()
                if stop["value"] or (config.get("stop_after") and step + 1 >= config["stop_after"]):
                    interrupted = True
                    break
        if profile:
            recorder.export_chrome_trace(str(out / "profile.json"))
        telemetry.phase = "checkpoint_save"
        t = time.perf_counter()
        completed_step = step + 1
        adapter = out / f"adapter-{completed_step:06d}"
        policy.save_pretrained(adapter)
        ck = {
            "step": completed_step,
            "adapter": {k: v.detach().cpu() for k, v in get_peft_model_state_dict(policy).items()},
            "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "sampler": sampler.state_dict(),
            "rng": rng_state(),
            "identity": {
                k: config.get(k)
                for k in [
                    "dataset",
                    "seed",
                    "lr",
                    "precision",
                    "microbatch",
                    "accumulation",
                    "steps",
                ]
            },
        }
        tmp = out / "resume.tmp"
        torch.save(ck, tmp)
        os.replace(tmp, out / "resume.pt")
        save_s = time.perf_counter() - t
        delta = max(
            float((p.detach().cpu() - initial[n]).abs().max())
            for n, p in policy.named_parameters()
            if p.requires_grad
        )
        saved = load_file(str(adapter / "adapter_model.safetensors"))
        current = get_peft_model_state_dict(policy)
        reload_linf = max(
            float((saved[k] - v.detach().cpu()).abs().max()) for k, v in current.items()
        )
        result = {
            "run_id": config["run_id"],
            "status": "PAUSED" if interrupted else "COMPLETED",
            "optimizer_updates": completed_step,
            "parameter_delta_linf": delta,
            "reload_linf": reload_linf,
            "finite_gradients": grad_finite,
            "adapter": str(adapter),
            "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
            "model_load_s": load_s,
            "dataset_load_s": dataset_load_s,
            "checkpoint_save_s": save_s,
            "wall_s": time.perf_counter() - start,
            "sample_exposures": sampler.exposures,
            "rows_this_process": len(rows),
            "allocated_peak": torch.cuda.max_memory_allocated(),
            "reserved_peak": torch.cuda.max_memory_reserved(),
            "last_loss": rows[-1]["loss"],
            "resume_checkpoint": str(out / "resume.pt"),
        }
        atomic_json(out / "result.json", result)
    signal.signal(signal.SIGTERM, old_handler)
    return result


def dataset_for(kind, backend="pyav"):
    if kind == "corrective":
        return corrective_dataset(backend)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torch.utils.data import ConcatDataset, Subset

    spec = json.loads((EVIDENCE / "data/selection.json").read_text())
    root = EVIDENCE / "data/expert_metadata"
    group = "standard" if kind == "standard" else "replay"
    ids = spec[group]["episodes"]
    ds = LeRobotDataset(
        "lerobot/libero",
        root=root,
        episodes=ids,
        delta_timestamps={"action": [i / 10 for i in range(50)]},
        video_backend=backend,
    )
    ds = Subset(ds, spec[group]["selected_local_indices"])
    if kind == "replay":
        return ConcatDataset([corrective_dataset(backend), ds])
    if kind != "standard":
        raise ValueError("Unknown training condition")
    return ds
