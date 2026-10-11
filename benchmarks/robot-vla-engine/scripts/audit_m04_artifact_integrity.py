"""Audit all M04 episode artifact hashes without repairing source evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/data/VLA-Forge/Evidence/M04")
OUT = ROOT / "artifact_integrity.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / "artifacts/policy_attempts/attempts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    refs_checked = 0
    missing = []
    mismatches = []
    actions_reconciled = []
    for row in rows:
        for kind in ("trajectory", "video", "action_trace"):
            item = row.get(kind, {})
            if not item.get("path"):
                continue
            path = Path(item["path"])
            refs_checked += 1
            if not path.is_file():
                missing.append({"episode_id": row["episode_id"], "kind": kind, "path": str(path)})
                continue
            actual = sha256(path)
            if actual != item.get("sha256"):
                entry = {
                    "episode_id": row["episode_id"],
                    "kind": kind,
                    "path": str(path),
                    "registry_sha256": item.get("sha256"),
                    "current_sha256": actual,
                }
                if kind == "action_trace" and row.get("trajectory", {}).get("path"):
                    trace = json.loads(path.read_text())
                    import numpy as np

                    env_actions = np.asarray(
                        [step["env_step_argument"] for step in trace.get("steps", [])],
                        dtype=np.float32,
                    )
                    with np.load(row["trajectory"]["path"]) as data:
                        stored_actions = data["actions"].astype(np.float32)
                    if env_actions.shape == stored_actions.shape:
                        entry["action_array_max_abs_error_vs_npz"] = float(
                            np.max(np.abs(env_actions - stored_actions))
                        )
                        entry["action_array_matches_npz"] = bool(
                            np.array_equal(env_actions, stored_actions)
                        )
                        if entry["action_array_matches_npz"]:
                            actions_reconciled.append(row["episode_id"])
                mismatches.append(entry)
    source_doc = json.loads((ROOT / "demo/video_sources.json").read_text())
    source_checks = []
    for entry in source_doc["episode_sources"]:
        source = Path(entry["source_path"])
        copied = Path(entry["copied_path"])
        ok = (
            source.is_file()
            and copied.is_file()
            and sha256(source) == entry["source_sha256"]
            and sha256(copied) == entry["copied_sha256"]
            and entry["source_sha256"] == entry["copied_sha256"]
        )
        source_checks.append({"source_id": entry["source_id"], "verified": ok})
    video = ROOT / "demo/M04_demo_200s.mp4"
    video_receipt = json.loads((ROOT / "demo/video_validation.json").read_text())
    output_video_ok = video.is_file() and sha256(video) == video_receipt["sha256"]
    receipt = {
        "schema": "vla-forge.m04.artifact-integrity.v1",
        "status": "PASS"
        if not missing
        and not mismatches
        and all(x["verified"] for x in source_checks)
        and output_video_ok
        else "PARTIAL_WITH_DECLARED_EXCEPTION",
        "registered_attempt_rows": len(rows),
        "episode_artifact_references_checked": refs_checked,
        "missing_artifacts": missing,
        "registry_hash_mismatches": mismatches,
        "action_arrays_reconciled_against_npz": actions_reconciled,
        "demo_video_sha256_verified": output_video_ok,
        "demo_episode_sources_verified": sum(x["verified"] for x in source_checks),
        "demo_episode_source_count": len(source_checks),
        "evidence_boundary": "No file was repaired or silently rewritten. A registry hash mismatch remains a provenance exception even when stored action arrays match the NPZ exactly.",
    }
    OUT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
