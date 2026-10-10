"""Pinned LIBERO and FP32 SmolVLA adapters for correction acquisition."""

import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .correction_assets import BASELINE, ORIGIN, TEACHER, relocate, verify_asset
from .preflight import bind_processor


@contextmanager
def gpu_exclusion(path):
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def snapshot_batch(observation, instruction):
    import torch
    from lerobot.processor.env_processor import LiberoProcessorStep

    batch = {"observation.state": torch.from_numpy(observation["observation"].copy())[None]}
    for name in ("image", "image2"):
        batch["observation.images." + name] = (
            torch.from_numpy(observation["pixels_" + name].copy()).permute(2, 0, 1).float()[None]
            / 255
        )
    batch = LiberoProcessorStep()._process_observation(batch)
    batch["task"] = [instruction]
    return batch


def load_archived_policy(root, asset_root, preflight, *, role="teacher"):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    from .policies import load_smol_checkpoint

    if preflight["status"] != "ASSETS_VERIFIED_RUNTIME_PENDING":
        raise ValueError("preflight required")
    run_id = {"teacher": TEACHER, "baseline": BASELINE}[role]
    checkpoint = relocate(
        asset_root, str(ORIGIN / f"artifacts/experiments/runs/{run_id}/pretrained_model")
    )
    for row in preflight["verified"]:
        if row["role"] == role + "_checkpoint":
            verify_asset(asset_root, row["original"], row["sha256"])
    bind_processor(
        Path(root) / "data/models/smolvlm",
        "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        preflight[role]["processor"],
    )
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if (
        cfg.type != "smolvla"
        or cfg.n_action_steps != 10
        or cfg.input_features["observation.state"].shape != (8,)
        or cfg.output_features["action"].shape != (7,)
    ):
        raise ValueError("checkpoint policy/action interface differs")
    cfg.device, cfg.load_vlm_weights = "cuda", False
    cfg.vlm_model_name = str((Path(root) / "data/models/smolvlm").resolve())
    model = load_smol_checkpoint(cfg, checkpoint).eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": cfg.vlm_model_name},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return model, pre, post


class PolicyProducer:
    def __init__(self, model, pre, post, instruction):
        self.model, self.pre, self.post, self.instruction = model, pre, post, instruction

    def reset(self, seed):
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.model.reset()
        for processor in (self.pre, self.post):
            if hasattr(processor, "reset"):
                processor.reset()

    def predict(self, observation):
        import torch

        with torch.inference_mode():
            raw = self.model.select_action(self.pre(snapshot_batch(observation, self.instruction)))
            prediction = raw.detach().cpu().numpy()[0].copy()
            # Processor receives a clone because some steps mutate their input.
            command = self.post(raw.clone()).detach().cpu().numpy()[0].copy()
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError("postprocessed command must be a finite seven-vector")
        # Match the frozen deployment's bounded command adapter, while preserving raw output.
        return prediction, np.clip(command, -1, 1)


class LiberoScene:
    def __init__(self, protocol, case, size=256):
        from lerobot.envs.libero import LiberoEnv
        from libero.libero import benchmark

        from .preflight import create_libero_env

        suite = benchmark.get_benchmark_dict()[case["suite"]](
            task_order_index=protocol["task_order_index"]
        )
        self.env = create_libero_env(
            LiberoEnv, suite, case["task_id"], case["suite"], protocol, 0, size, case["horizon"]
        )

    def _snapshot(self, observation, success):
        from .observations import libero_batch

        return dict(
            sim=self.env._env.get_sim_state().copy(),
            observation=libero_batch(observation)["observation.state"][0].numpy().copy(),
            success=success,
            **{"pixels_" + k: observation["pixels"][k].copy() for k in ("image", "image2")},
        )

    def reset(self, seed):
        observation, _ = self.env.reset(seed=seed)
        return self._snapshot(observation, bool(self.env._env.check_success()))

    def step(self, command):
        observation, _, terminal, truncated, info = self.env.step(command)
        success = info.get("is_success")
        if type(success) not in (bool, np.bool_):
            raise ValueError("missing original task predicate")
        return self._snapshot(observation, bool(success)), terminal, truncated

    def close(self):
        self.env.close()


def verify_runtime_sources(root):
    import inspect

    from lerobot.envs.libero import LiberoEnv

    from .protocol import load_protocol
    from .schema import file_hash
    from .training_contracts import verify_implementation

    root = Path(root)
    verify_implementation(root, load_protocol(root / "configs/compute_protocol.json"))
    config = json.loads((root / "reports/repair_gate0/run_config_v2.json").read_text())
    expected = config["source_hashes"][str(ORIGIN / "external/lerobot/src/lerobot/envs/libero.py")]
    if file_hash(inspect.getsourcefile(LiberoEnv)) != expected:
        raise ValueError("LIBERO adapter differs from archived Gate 0")
    return dict(libero_adapter_sha256=expected, frozen_implementation="PASS")
