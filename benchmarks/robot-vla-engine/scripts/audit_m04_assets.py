"""Independent provenance audit for M04's frozen M02/M03 inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mapped_artifact(path: str, source_root: Path) -> Path:
    marker = "/data/vla-forge-artifacts/M02_libero40/"
    if marker not in path:
        raise ValueError(f"unexpected artifact root: {path}")
    relative = path.split(marker, 1)[1]
    return source_root / relative


def audit(root: Path) -> dict:
    m02 = root / "Evidence/M02/source/M02_libero40"
    m03 = root / "Evidence/M03/source/M03_interventional_diagnosis"
    model_dir = root / "Assets/models/smolvla_libero"
    model = model_dir / "model.safetensors"
    if not model.is_file():
        raise FileNotFoundError(model)
    model_sha = digest(model)

    m03_manifest = json.loads((m03 / "manifest.json").read_text())
    if m03_manifest.get("status") != "COMPLETED":
        raise ValueError("M03 source manifest is not COMPLETED")
    if m03_manifest.get("checkpoint_sha256") != model_sha:
        raise ValueError("M03 checkpoint hash does not match project-local model")
    groups = m03_manifest.get("groups", [])
    if len(groups) != 10:
        raise ValueError(f"expected 10 M03 groups, found {len(groups)}")
    development = set(m03_manifest.get("development_source_groups", []))
    reserved = set(m03_manifest.get("m06_reserved_source_groups", []))
    if development & reserved:
        raise ValueError("M03 development and M06 reserved groups overlap")
    intervention_types = set()
    m03_successes = Counter()
    m03_files_checked = 0
    for group in groups:
        matches = group.get("physical_state_pairwise_match", [])
        if len(matches) != 2 or any(
            item.get("exact") is not True or item.get("within_tolerance") is not True
            for item in matches
        ):
            raise ValueError(f"unverified state match: {group['group_id']}")
        conditions = group.get("condition_results", {})
        if not {"baseline", "sham"} <= set(conditions):
            raise ValueError(f"missing baseline/sham: {group['group_id']}")
        for name, result in conditions.items():
            m03_successes[name] += int(result.get("success") is True)
            if name not in ("baseline", "sham"):
                intervention_types.add(name)
            for path_key, hash_key in (
                ("trajectory_path", "trajectory_sha256"),
                ("video_path", "video_sha256"),
            ):
                raw = result.get(path_key)
                expected = result.get(hash_key)
                if raw and expected:
                    source_marker = "/data/vla-forge-artifacts/M03_interventional_diagnosis/"
                    if source_marker not in raw:
                        raise ValueError(f"unexpected M03 artifact root: {raw}")
                    file = m03 / raw.split(source_marker, 1)[1]
                    if not file.is_file() or digest(file) != expected:
                        raise ValueError(f"M03 artifact hash mismatch: {file}")
                    m03_files_checked += 1
    if len(intervention_types) < 2:
        raise ValueError("M03 evidence has fewer than two intervention types")

    db = m02 / "registry/registry.sqlite"
    conn = sqlite3.connect(db)
    try:
        runs = conn.execute(
            "SELECT run_id,status,config_json,config_hash FROM runs ORDER BY run_id"
        ).fetchall()
        artifacts = conn.execute(
            "SELECT run_id,role,path,sha256,size FROM artifacts ORDER BY run_id,role,path"
        ).fetchall()
    finally:
        conn.close()
    if not runs or any(row[1] != "COMPLETE" for row in runs):
        raise ValueError("M02 registry has missing or incomplete runs")
    counts = Counter(row[1] for row in artifacts)
    required = {
        "action_trace": 400,
        "paired_camera_rollout_video": 400,
        "rollout_trajectory": 800,
        "rollout_metadata": 800,
    }
    for role, expected_count in required.items():
        if counts[role] != expected_count:
            raise ValueError(f"M02 {role}: expected {expected_count}, found {counts[role]}")
    for run_id, role, raw, expected, size in artifacts:
        file = mapped_artifact(raw, m02)
        if not file.is_file() or file.stat().st_size != size or digest(file) != expected:
            raise ValueError(f"M02 registry artifact failed hash: {run_id}/{role}/{file}")

    configs = [json.loads(row[2]) for row in runs]
    policy_ids = {
        (c.get("policy_repo_id"), c.get("policy_revision"), c.get("policy_sha256")) for c in configs
    }
    if len(policy_ids) != 1:
        raise ValueError("M02 registry mixes policy identities")
    return {
        "status": "PASS",
        "m02": {
            "run_count": len(runs),
            "episode_action_traces": counts["action_trace"],
            "episode_videos": counts["paired_camera_rollout_video"],
            "trajectory_records": counts["rollout_trajectory"],
            "metadata_records": counts["rollout_metadata"],
            "artifact_files_hash_verified": len(artifacts),
            "policy_identity": list(next(iter(policy_ids))),
        },
        "m03": {
            "groups": len(groups),
            "pairwise_state_matches": sum(len(g["physical_state_pairwise_match"]) for g in groups),
            "intervention_types": sorted(intervention_types),
            "condition_successes": dict(m03_successes),
            "source_files_hash_verified": m03_files_checked,
            "m06_group_overlap": 0,
        },
        "model": {
            "path": str(model),
            "sha256": model_sha,
            "bytes": model.stat().st_size,
        },
        "evidence_roots": {
            "m02": str(m02),
            "m03": str(m03),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/VLA-Forge"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
