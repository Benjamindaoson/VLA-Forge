"""CPU-only M3 input admission; does not train, normalize or qualify a release."""

import argparse
import bisect
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

from .correction_records import FIELDS, verify_record
from .schema import canonical_json, content_hash, file_hash


def read_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    canonical_json(data)  # Reject NaN/Infinity, including nested fields.
    return data


def registered_run(root, run_id):
    root = Path(root).resolve()
    database = root / "registry.sqlite"
    if not database.is_file():
        raise FileNotFoundError("correction Registry is missing")
    directory = (root / "runs" / run_id).resolve()
    if directory.parent != root / "runs":
        raise ValueError("invalid run identity")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        artifacts = db.execute("SELECT * FROM artifacts WHERE run_id=?", (run_id,)).fetchall()
    if row is None or row["status"] != "COMPLETE":
        raise ValueError("acquisition run must be COMPLETE")
    config = json.loads(row["config_json"])
    if content_hash(config) != row["config_hash"] or read_json(directory / "config.json") != config:
        raise ValueError("run config identity differs")
    if not isinstance(config, dict) or not isinstance(config.get("provenance"), dict):
        raise ValueError("acquisition config and provenance must be objects")
    if (
        config.get("experiment_type") != "engineering_verified_correction_acquisition"
        or config.get("precision") != "fp32"
        or config.get("n_action_steps") != 10
        or config.get("control_freq") != 20
        or config.get("policy_training") is not False
        or config.get("case_count") != 4
        or config.get("provenance", {}).get("status") != "ASSETS_VERIFIED_RUNTIME_PENDING"
    ):
        raise ValueError("unsupported acquisition contract")
    registered = {}
    for item in artifacts:
        path = Path(item["path"]).resolve()
        if (
            not path.is_relative_to(directory)
            or path in registered
            or not path.is_file()
            or path.stat().st_size != item["size"]
            or file_hash(path) != item["sha256"]
        ):
            raise ValueError("registered artifact integrity differs")
        registered[path] = dict(item)
    return root, directory, config, registered


def checked_artifact(path, registered, role):
    entry = registered.get(Path(path).resolve())
    if entry is None or entry["role"] != role:
        raise ValueError("missing registered " + role)
    return entry


def partitions(document, groups):
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("invalid split schema")
    parts = document.get("partitions")
    if not isinstance(parts, dict) or set(parts) != {"train", "development", "test"}:
        raise ValueError("invalid partition names")
    result = {}
    for name, values in parts.items():
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            raise ValueError("invalid scene group list")
        for group in values:
            if group not in groups or group in result:
                raise ValueError("unknown or cross-partition scene group")
            result[group] = name
    if set(result) != groups:
        raise ValueError("every source scene group requires a partition")
    return result


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in ("commands", *FIELDS)}
    commands = data["commands"]
    n = len(commands)
    if (
        commands.shape != (n, 7)
        or n < 1
        or commands.dtype.kind not in "fiu"
        or not np.isfinite(commands).all()
        or np.any(np.abs(commands) > 1)
    ):
        raise ValueError("invalid dispatched commands")
    for key, value in data.items():
        if not value.size or value.dtype.kind not in "buif" or not np.isfinite(value).all():
            raise ValueError("invalid array: " + key)
        if key != "commands" and len(value) != n + 1:
            raise ValueError("observation/action alignment differs")
    if data["observation"].shape != (n + 1, 8) or data["sim"].ndim != 2:
        raise ValueError("state shape differs")
    if data["success"].dtype != np.bool_ or data["success"].shape != (n + 1,):
        raise ValueError("success must be a boolean sequence")
    for key in ("pixels_image", "pixels_image2"):
        image = data[key]
        if image.dtype != np.uint8 or image.ndim != 4 or image.shape[-1] != 3:
            raise ValueError("camera must be uint8 RGB")
    return data


def _prepare(root, run_id, split, split_sha256):
    root, directory, config, registered = registered_run(root, run_id)
    cases = config["provenance"]["cases"]
    if (
        len(cases) != 4
        or len({c["case_id"] for c in cases}) != 4
        or {(c["suite"], c["task_id"], c["prefix"]) for c in cases}
        != {(s, 0, p) for s in ("libero_10", "libero_goal") for p in (20, 60)}
    ):
        raise ValueError("unsupported source case identities")
    keys = (
        "suite",
        "task_id",
        "prefix",
        "baseline_run_id",
        "prefix_run_id",
        "reference_sha256",
        "reconstruction_sha256",
    )
    if any(c["case_id"] != content_hash({k: c[k] for k in keys}) for c in cases):
        raise ValueError("source case identity mismatch")
    group_by_id = {c["case_id"]: f"{c['suite']}/task{c['task_id']}/init0" for c in cases}
    assigned = partitions(split, set(group_by_id.values()))
    parent_path = directory / "summary.json"
    checked_artifact(parent_path, registered, "correction_summary")
    parent = read_json(parent_path)
    if (
        parent["run_id"] != run_id
        or parent["status"] != "EXECUTED"
        or len(parent["cases"]) != len(cases)
        or {c["case_id"] for c in parent["cases"]} != set(group_by_id)
    ):
        raise ValueError("parent summary case identity mismatch")
    episodes, excluded, seen_content = [], [], set()
    verified_count = 0
    for case in cases:
        folder = directory / case["case_id"]
        path = folder / "summary.json"
        checked_artifact(path, registered, "correction_evidence")
        summary = read_json(path)
        parent_case = next(r for r in parent["cases"] if r["case_id"] == case["case_id"])
        if summary != parent_case:
            raise ValueError("parent/case summary differs")
        group = group_by_id[case["case_id"]]
        if summary.get("verified_correction") is not True:
            excluded.append(dict(case_id=case["case_id"], reason="NOT_VERIFIED", group=group))
            continue
        if (
            summary.get("prefix_steps") != case["prefix"]
            or summary.get("environment_seed") != 42
            or summary.get("horizon") != (520 if case["suite"] == "libero_10" else 300)
        ):
            raise ValueError("case scenario differs")
        summary = verify_record(folder)
        for relative in summary["files"]:
            checked_artifact(folder / relative, registered, "correction_evidence")
        verified_count += 1
        attempt = next(row for row in summary["attempts"] if row["status"] == "SUCCESS")
        trajectory = folder / attempt["relative_directory"] / "trajectory.npz"
        entry = checked_artifact(trajectory, registered, "correction_evidence")
        data = arrays(trajectory)
        # Duplicate numerical content is not a new episode, regardless of zip metadata.
        digest = content_hash({k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in data.items()})
        if digest in seen_content:
            raise ValueError("duplicate correction content")
        seen_content.add(digest)
        if assigned[group] != "train":
            excluded.append(dict(case_id=case["case_id"], reason=assigned[group], group=group))
            continue
        episodes.append(
            dict(
                case_id=case["case_id"],
                source_scene_group=group,
                suite=case["suite"],
                task_id=case["task_id"],
                initial_state=0,
                prefix=case["prefix"],
                environment_seed=42,
                trajectory=trajectory.relative_to(root).as_posix(),
                trajectory_sha256=entry["sha256"],
                summary_sha256=registered[path.resolve()]["sha256"],
                commands=len(data["commands"]),
                action_semantics="LIBERO dispatched normalized command; not actuator feedback",
            )
        )
    if (
        type(parent["verified_corrections"]) is not int
        or parent["verified_corrections"] != verified_count
    ):
        raise ValueError("verified correction count differs")
    if not episodes:
        raise ValueError("no verified training corrections")
    manifest = dict(
        schema_version=1,
        status="INPUT_CONTRACT_VERIFIED",
        run_id=run_id,
        source_config_identity=content_hash(config),
        source_summary_sha256=file_hash(parent_path),
        split=split,
        split_sha256=split_sha256,
        episodes=episodes,
        excluded=excluded,
        source_verified_corrections=verified_count,
        source_scene_groups=len({e["source_scene_group"] for e in episodes}),
        training_commands=sum(e["commands"] for e in episodes),
        model_training="NOT_RUN",
        generalization_claim=False,
        actuator_feedback=None,
        scope="CPU data contract only; requires current model/runtime and trainer qualification",
    )
    manifest["identity"] = content_hash(manifest)
    return manifest


def prepare_inputs(registry_root, run_id, split_path, expected_split_sha256):
    if file_hash(split_path) != expected_split_sha256:
        raise ValueError("split file hash differs")
    return _prepare(registry_root, run_id, read_json(split_path), expected_split_sha256)


class CorrectionDataset:
    """Validated immutable in-memory NumPy snapshots, one sample per dispatched action."""

    def __init__(self, registry_root, manifest, expected_identity, chunk_size=50):
        if type(chunk_size) is not int or not 1 <= chunk_size <= 50:
            raise ValueError("chunk size must be 1..50")
        claimed = {k: v for k, v in manifest.items() if k != "identity"}
        if (
            manifest.get("identity") != expected_identity
            or content_hash(claimed) != expected_identity
        ):
            raise ValueError("manifest identity differs")
        actual = _prepare(
            registry_root, manifest["run_id"], manifest["split"], manifest["split_sha256"]
        )
        if actual != manifest:
            raise ValueError("source changed since admission")
        self.chunk_size = chunk_size
        self.episodes = []
        self.ends = []
        total = 0
        for episode in actual["episodes"]:
            path = Path(registry_root) / episode["trajectory"]
            data = arrays(path)
            if file_hash(path) != episode["trajectory_sha256"]:
                raise ValueError("source changed during load")
            self.episodes.append((episode, data))
            total += episode["commands"]
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        episode_index = bisect.bisect_right(self.ends, index)
        offset = index - (self.ends[episode_index - 1] if episode_index else 0)
        episode, data = self.episodes[episode_index]
        valid = min(self.chunk_size, episode["commands"] - offset)
        action = np.zeros((self.chunk_size, 7), dtype=np.float32)
        action[:valid] = data["commands"][offset : offset + valid]
        mask = np.arange(self.chunk_size) >= valid
        return {
            "observation.state": data["observation"][offset].copy(),
            "observation.images.image": data["pixels_image"][offset].copy(),
            "observation.images.image2": data["pixels_image2"][offset].copy(),
            "action": action,
            "action_is_pad": mask,
            "actuator_feedback": None,
            "source": dict(episode, correction_step=offset),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = prepare_inputs(args.registry, args.run_id, args.split, args.split_sha256)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.DatabaseError) as exc:
        result = dict(
            status="BLOCKED_INPUTS",
            reason=str(exc),
            model_training="NOT_RUN",
            verified_training_corrections=None,
        )
        (args.output / "blocked.json").write_text(canonical_json(result), encoding="utf-8")
        print(canonical_json(result))
        return 2
    (args.output / "manifest.json").write_text(canonical_json(result), encoding="utf-8")
    print(
        canonical_json(
            dict(
                status=result["status"],
                identity=result["identity"],
                episodes=len(result["episodes"]),
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
