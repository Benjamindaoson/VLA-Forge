"""Verify current bytes and effective settings before constructing any policy."""

import json
from pathlib import Path

from .schema import content_hash, file_hash


def bind_processor(root, revision, training_identity, *, development=False):
    current = verify_receipt(root, revision)
    if training_identity is None:
        if not development:
            raise ValueError("missing training processor identity")
        binding = "missing_legacy_development"
    elif current != training_identity:
        raise ValueError("current source differs from training processor identity")
    else:
        binding = "verified"
    return dict(identity=current, training_binding=binding)


def verify_receipt(root, revision):
    root = Path(root).resolve()
    path = root / "download_receipt.json"
    receipt = json.loads(path.read_text())
    records = receipt["files"]
    if (
        not receipt.get("completed")
        or receipt.get("revision") != revision
        or receipt["count"] != len(records)
    ):
        raise ValueError("incomplete or wrong source receipt")
    expected = {r["path"] for r in records}
    if len(expected) != len(records) or not records:
        raise ValueError("empty/duplicate source receipt")
    actual = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file()
        and ".cache" not in p.relative_to(root).parts
        and p.name != "download_receipt.json"
    }
    if expected != actual:
        raise ValueError("source file set differs from receipt")
    for row in records:
        source = (root / row["path"]).resolve()
        if (
            root not in source.parents
            or source.stat().st_size != row["bytes"]
            or file_hash(source) != row["sha256"]
        ):
            raise ValueError(f"source checksum/size mismatch: {row['path']}")
    return dict(
        receipt_sha256=file_hash(path),
        revision=revision,
        verified_files=len(records),
        verified_bytes=sum(r["bytes"] for r in records),
    )


def validate_normalizer(raw, protocol, condition):
    if (
        raw.get("protocol_identity") != protocol["identity"]
        or raw.get("condition") != condition
        or raw.get("episode_ids_hash")
        != content_hash(protocol["conditions"][condition]["episode_ids"])
    ):
        raise ValueError("normalizer identity does not match training condition")
    return raw["stats"]


def verified_normalizer(root, protocol, condition):
    root = Path(root)
    raw = json.loads((root / f"configs/{condition}_stats.json").read_text())
    validate_normalizer(raw, protocol, condition)
    manifest = json.loads((root / "configs/normalizer_manifest.json").read_text())
    if (
        manifest["protocol_identity"] != protocol["identity"]
        or content_hash({k: v for k, v in manifest.items() if k != "identity"})
        != manifest["identity"]
        or file_hash(root / f"configs/{condition}_stats.json")
        != manifest["files"][f"{condition}_stats.json"]
    ):
        raise ValueError("normalizer frozen content identity mismatch")
    return raw


def evaluation_settings(protocol):
    value = protocol["evaluation"]
    indices = value["initial_state_indices"]
    if (
        len(indices) != value["episodes_per_task"]
        or len(set(indices)) != len(indices)
        or any(i < 0 for i in indices)
    ):
        raise ValueError("initial state identities do not match episode count")
    if (
        value["control_freq"] <= 0
        or value["num_steps_wait"] < 0
        or type(value["hard_reset"]) is not bool
    ):
        raise ValueError("invalid evaluation settings")
    return value


def create_libero_env(factory, suite, task_id, suite_name, protocol, initial_state, size, horizon):
    settings = evaluation_settings(protocol)
    if initial_state not in settings["initial_state_indices"]:
        raise ValueError("initial state not in protocol")
    return factory(
        suite,
        task_id,
        suite_name,
        episode_length=horizon,
        obs_type="pixels_agent_pos",
        observation_width=size,
        observation_height=size,
        episode_index=initial_state,
        init_states=True,
        hard_reset=settings["hard_reset"],
        control_freq=settings["control_freq"],
        num_steps_wait=settings["num_steps_wait"],
    )
