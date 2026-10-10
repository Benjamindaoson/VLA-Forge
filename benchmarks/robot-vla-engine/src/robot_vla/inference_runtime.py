"""Owned SmolVLA checkpoints and lossless, registry-bound offline observations."""

import json
from pathlib import Path

import numpy as np

from .preflight import bind_processor
from .protocol import load_protocol
from .registry import Registry
from .schema import content_hash, file_hash


def gpu_identity():
    import subprocess

    import torch

    # This experiment is explicitly single-device; fail closed on an ambiguous host.
    query = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,driver_version,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        .stdout.strip()
        .splitlines()
    )
    if len(query) != 1 or torch.cuda.device_count() != 1:
        raise ValueError("inference protocol requires one unambiguous physical GPU")
    uuid, driver, name = [s.strip() for s in query[0].split(",")]
    return dict(
        uuid=uuid,
        driver=driver,
        name=name,
        cuda_runtime=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
    )


def verify_inference_protocol(root, corpus_identity):
    root = Path(root)
    protocol = load_protocol(root / "configs/inference_protocol_v1.json")
    if protocol["status"] != "FROZEN" or protocol["corpus_identity"] != corpus_identity:
        raise ValueError("inference protocol status or corpus differs")
    for name, expected in protocol["source_hashes"].items():
        if content_hash((root / name).read_text(encoding="utf-8")) != expected:
            raise ValueError(f"inference source differs: {name}")
    return protocol


def load_corpus(root):
    root = Path(root)
    folder = root / "data/inference_corpus_v1"
    manifest = load_protocol(folder / "manifest.json")
    protocol = load_protocol(root / "configs/data_protocol_v1.json")
    if manifest["data_protocol_identity"] != protocol["identity"] or len(manifest["cases"]) != 40:
        raise ValueError("corpus protocol or task count mismatch")
    registry = Registry(root / "artifacts/experiments")
    with registry.connect() as db:
        matches = db.execute(
            "SELECT r.config_json,r.run_id FROM artifacts a JOIN runs r ON r.run_id=a.run_id WHERE a.role='inference_corpus_manifest' AND a.sha256=? AND r.status='COMPLETE'",
            (file_hash(folder / "manifest.json"),),
        ).fetchall()
    if not any(
        json.loads(r[0]).get("experiment_type") == "engineering_corpus"
        and not registry.verify_artifacts(r[1])
        for r in matches
    ):
        raise ValueError("corpus has no verified complete creation record")
    cases = []
    for case in manifest["cases"]:
        path = folder / case["path"]
        if path.parent.resolve() != folder.resolve() or file_hash(path) != case["sha256"]:
            raise ValueError("corpus case integrity mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            values = {key: arrays[key].copy() for key in case["shapes"]}
        if any(
            list(value.shape) != case["shapes"][key] or not np.isfinite(value).all()
            for key, value in values.items()
        ):
            raise ValueError("invalid corpus tensor")
        cases.append((case, values))
    return manifest, cases


def verify_parent_protocol(parent, compute, data_identity):
    if (
        parent.get("protocol_identity") != data_identity
        or compute.get("data_protocol_identity") != data_identity
    ):
        raise ValueError("training/data/corpus protocol identity mismatch")


def load_owned_smol(root, training_run_id, device, development=False):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    from .policies import load_smol_checkpoint

    root = Path(root)
    registry = Registry(root / "artifacts/experiments")
    record = registry.get(training_run_id)
    config = json.loads(record["config_json"])
    stored = json.loads(
        (registry.root / "runs" / training_run_id / "config.json").read_text(encoding="utf-8")
    )
    if config != stored or content_hash(config) != record["config_hash"]:
        raise ValueError("training configuration identity mismatch")
    if record["status"] != "COMPLETE" or registry.verify_artifacts(training_run_id):
        raise ValueError("incomplete or changed training artifacts")
    if config.get("recipe", {}).get("policy") != "smolvla":
        raise ValueError("this benchmark requires an owned SmolVLA checkpoint")
    if not development:
        from .training_contracts import verify_implementation

        compute = load_protocol(root / "configs/compute_protocol.json")
        verify_implementation(root, compute)
        verify_parent_protocol(
            config, compute, load_protocol(root / "configs/data_protocol_v1.json")["identity"]
        )
        if (
            config.get("experiment_type") != "formal_training"
            or config.get("compute_protocol_identity") != compute["identity"]
        ):
            raise ValueError("requires the frozen formal training checkpoint")
        if config["recipe"]["condition"] != "baseline_full":
            raise ValueError("inference optimization requires the baseline_full checkpoint")
    with registry.connect() as db:
        artifacts = [
            dict(r)
            for r in db.execute(
                "SELECT * FROM artifacts WHERE run_id=? AND role='policy_checkpoint'",
                (training_run_id,),
            )
        ]
    weights = [r for r in artifacts if Path(r["path"]).name == "model.safetensors"]
    if len(weights) != 1:
        raise ValueError("ambiguous owned model weights")
    checkpoint = Path(weights[0]["path"]).parent
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    if cfg.type != "smolvla":
        raise ValueError("checkpoint policy type mismatch")
    binding = bind_processor(
        root / "data/models/smolvlm",
        "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        config.get("verified_sources", {}).get("processor"),
        development=development,
    )
    cfg.device = device
    cfg.load_vlm_weights = False
    cfg.vlm_model_name = str((root / "data/models/smolvlm").resolve())
    model = load_smol_checkpoint(cfg, checkpoint)
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "tokenizer_processor": {"tokenizer_name": cfg.vlm_model_name},
        },
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    metadata = dict(
        training_run_id=training_run_id,
        checkpoint_sha256=weights[0]["sha256"],
        checkpoint_config_sha256=file_hash(checkpoint / "config.json"),
        processor_binding=binding,
        formal_training=config.get("experiment_type") == "formal_training",
        compute_protocol_identity=config.get("compute_protocol_identity"),
        training_data_protocol_identity=config.get("protocol_identity"),
        training_seed=config.get("recipe", {}).get("seed"),
    )
    return model, pre, post, metadata
