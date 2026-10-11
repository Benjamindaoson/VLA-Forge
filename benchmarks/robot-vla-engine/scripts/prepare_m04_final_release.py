"""File-level publication allowlist; never relicense upstream visual assets."""

import csv
import hashlib
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/data/VLA-Forge")
E = ROOT / "Evidence/M04"
OUT = E / "finalization_20261011"
LIB = ROOT / "third_party/LIBERO-8f1084e3132a39270c3a13ebe37270a43ece2a01"
ASSETS = ROOT / "Assets/LIBERO-assets"
DS = E / "dataset/lerobot_v3_verified_r7"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    source_dir = OUT / "license_sources"
    source_dir.mkdir(exist_ok=True)
    sources = []
    for name, url in [
        (
            "libero_readme",
            "https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/8f1084e3132a39270c3a13ebe37270a43ece2a01/README.md",
        ),
        (
            "libero_license",
            "https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/8f1084e3132a39270c3a13ebe37270a43ece2a01/LICENSE",
        ),
        ("hope_readme", "https://raw.githubusercontent.com/swtyree/hope-dataset/master/README.md"),
        (
            "hope_robosuite_derivation",
            "https://raw.githubusercontent.com/NVlabs/RoboLab/main/assets/objects/hope/LICENSE",
        ),
    ]:
        p = source_dir / (name + ".txt")
        try:
            if not p.exists():
                p.write_bytes(urllib.request.urlopen(url, timeout=30).read())
            sources.append({"name": name, "url": url, "path": str(p), "sha256": sha(p)})
        except Exception as exc:
            sources.append({"name": name, "url": url, "error": str(exc)})
    inventory = []
    for p in sorted(ASSETS.rglob("*")):
        if not p.is_file() or ".cache" in p.parts:
            continue
        rel = p.relative_to(ASSETS)
        original = LIB / "libero/libero/assets" / rel
        h = sha(p)
        family = rel.parts[0]
        if family == "stable_hope_objects":
            declared = "LIBERO root MIT; upstream HOPE CC-BY-NC-SA-4.0"
            decision = "PRIVATE_PENDING_ASSET_SPECIFIC_ATTRIBUTION_AND_LICENSE_RECONCILIATION"
            basis = "HOPE upstream README includes textured meshes; retain NC-SA provenance and do not infer blanket CC-BY permission."
        elif family in ["turbosquid_objects", "textures", "scenes", "articulated_objects"]:
            declared = "LIBERO root MIT; original asset-specific license not supplied"
            decision = "PRIVATE_PENDING_PER_ASSET_SOURCE_AND_REDISTRIBUTION_TERMS"
            basis = "Exact LIBERO provenance verified where matching; no original item URL/purchase or attribution receipt establishing separate third-party rights."
        elif family == "stable_scanned_objects":
            declared = "LIBERO root MIT; Google Scanned Objects family identification, item-level mapping incomplete"
            decision = "PRIVATE_PENDING_ITEM_ATTRIBUTION"
            basis = "Do not infer exact original asset item from directory family alone."
        else:
            declared = "LIBERO root MIT where matching"
            decision = "NOT_IN_PUBLIC_ALLOWLIST"
            basis = "No need to redistribute simulator infrastructure in numeric-only release."
        inventory.append(
            {
                "path": str(p),
                "relative_path": str(rel),
                "sha256": h,
                "kind": "simulation_asset",
                "upstream_path": str(original),
                "official_pinned_byte_match": original.is_file() and sha(original) == h,
                "declared_license": declared,
                "publication_decision": decision,
                "basis": basis,
            }
        )
    media = []
    for base in [E / "artifacts", E / "demo", DS, OUT / "cross_task/policy_attempts"]:
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in [".mp4", ".png", ".jpg", ".jpeg", ".npz"]:
                continue
            media.append(
                {
                    "path": str(p),
                    "relative_path": str(p.relative_to(E)),
                    "sha256": sha(p),
                    "kind": "rendered_media_or_camera_containing_trajectory",
                    "upstream_path": "See simulation_asset inventory and M02/M03/M04 lineage",
                    "official_pinned_byte_match": "",
                    "declared_license": "Mixed visual asset provenance; no blanket relicensing",
                    "publication_decision": "KEEP_PRIVATE",
                    "basis": "Contains or may contain third-party asset depictions; full visual scene includes HOPE cream cheese, scanned bowl and other fixtures. Numeric-only extraction excludes camera frames, textures, meshes and simulation qpos.",
                }
            )
    inventory += media
    with (OUT / "license_inventory.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=inventory[0].keys())
        w.writeheader()
        w.writerows(inventory)
    decision = {
        "sources": sources,
        "files_reviewed": len(inventory),
        "simulation_assets": len(inventory) - len(media),
        "pinned_matches": sum(x["official_pinned_byte_match"] is True for x in inventory),
        "media_files_kept_private": len(media),
        "status": "PARTIAL_PUBLICATION_ALLOWED_NUMERIC_ONLY",
        "public_allowlist_scope": "Own newly generated 7D robot commands, 8D robot proprioception, numeric terminal labels and original engineering reports. No third-party mesh, texture, RGB frame, video or object geometry.",
        "full_dataset_and_demo": "KEEP_PRIVATE",
        "unblocking": "For depicted HOPE assets: preserve original attribution and resolve MIT / CC-BY versus NC-SA downstream treatment. For TurboSquid/scanned/scene items: obtain exact source and redistribution terms or replace with clearly licensed assets in a future separately authorized run. Existing recordings remain unchanged.",
        "actual_goal6_objects": [
            "cream_cheese",
            "akita_black_bowl",
            "wine_bottle",
            "plate",
            "wooden_cabinet",
            "flat_stove",
            "wine_rack",
            "table/robot/scene",
        ],
        "scope_boundary": "Engineering publication clearance based on inspected provenance, not a legal ruling that all rendered images are unlawful.",
    }
    (OUT / "license_review.json").write_text(json.dumps(decision, indent=2) + "\n")
    payload = OUT / "public_numeric_dataset"
    payload.mkdir(exist_ok=True)
    t = pq.read_table(sorted(DS.glob("data/**/*.parquet")))
    allowed = [
        "action",
        "observation.state",
        "next.success",
        "next.done",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    assert set(allowed) <= set(t.column_names)
    pq.write_table(t.select(allowed), payload / "data.parquet", compression="zstd")
    prov = json.loads((DS / "provenance.json").read_text())
    episode_records = []
    for i, (s, r) in enumerate(zip(prov["sources"], prov["episodes_index"])):
        episode_records.append(
            {
                "episode_index": i,
                "episode_id": s["episode_id"],
                "source_suite": s["suite"],
                "source_task_id": s["task_id"],
                "source_init_state_id": s["init_state_id"],
                "source_trace_sha256": s["source_m02_trace_sha256"],
                "source_trajectory_sha256": s["source_trajectory_sha256"],
                "action_sequence_sha256": s["correction_action_sha256"],
                "first_row": r["global_start"],
                "last_row": r["global_end_inclusive"],
                "transitions": s["correction_actions"],
                "origin": "fresh_M04" if s.get("source_policy_attempt") else "M03_derived",
                "success": s["independent_replay_success"],
            }
        )
    (payload / "episodes.json").write_text(json.dumps(episode_records, indent=2) + "\n")
    (payload / "README.md").write_text(
        """---
license: cc-by-4.0
task_categories:
- robotics
language:
- en
size_categories:
- n<1K
pretty_name: VLA-Forge M04 Numeric Correction Evidence
---
# VLA-Forge M04 numeric correction evidence

**12 successful correction segments, 956 transitions, one LIBERO Goal task and two source initial states.**
11 segments were newly inferred in M04; one was inherited from M03. The 11 new segments each have two successful independent physical replay receipts.

This is an openly readable **numeric-only evidence subset**, not the full LeRobot v3 visual dataset. It cannot replace the image observations needed to train SmolVLA. The complete visual r7 dataset and demo remain private pending asset-specific license reconciliation.

## Files and fields
- data.parquet: executed 7D relative robot action, pre-action 8D robot state (EEF position, axis-angle orientation, two gripper positions), terminal success/done labels, 20 Hz timestamps and integer row/episode IDs.
- episodes.json: source IDs, immutable SHA256 references and row ranges.
- release_manifest.json: exact files, hashes and release boundaries.

Rows stop at the successful terminal action. Success/done is true only for each segment's final row. Frame i and state i supervise action i. Segments start after a 100-action source prefix. Commands are finite, clipped to [-1,1] and verified against the executed action log and independent replays. Camera brightness interventions produced some action proposals; canonical replay observations were used in the private visual dataset.

## Limitations
One task (Goal task 6) and two initial states (6 and 7); no cross-task generalization claim. Three budget-limited Goal task 4 attempts failed and are not represented as corrections. This data is development-only. Source families, including failed attempts, must stay separate from evaluation. The numeric subset has no camera frames, meshes, textures, simulation object geometry or pretrained weights.

## License and attribution
Newly generated numerical robot telemetry, labels and the original text in this subset are released under CC BY 4.0 by the VLA-Forge project. Attribute VLA-Forge and link this dataset. This license does not grant rights in excluded third-party visual assets.
Experiments use [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) (code MIT; official datasets declared CC BY 4.0) and [SmolVLA](https://huggingface.co/lerobot/smolvla_libero). Visual scenes include assets with separate provenance, including HOPE (CC BY-NC-SA 4.0), which are excluded from this release.
No robot hardware data, customer data, human personal data or secret credentials are included.

## Reproduce and evidence
Source: https://github.com/Benjamindaoson/VLA-Forge/tree/codex/m04-final-admission
Physical source recordings remain in the private evidence archive; hash references enable authorized verification. Public users can verify numeric row alignment, action bounds, labels and unique action hashes.
""",
        encoding="utf-8",
    )
    release = {
        "schema": "vla-forge.numeric-only-public-release.v1",
        "source_dataset": "lerobot_v3_verified_r7",
        "source_provenance_sha256": sha(DS / "provenance.json"),
        "episodes": 12,
        "transitions": 956,
        "visual_observations_included": False,
        "public_license": "CC-BY-4.0",
        "excluded": [
            "videos",
            "RGB images",
            "mesh/texture assets",
            "simulator object state",
            "model weights",
        ],
        "files": {
            p.name: sha(p)
            for p in payload.iterdir()
            if p.is_file() and p.name != "release_manifest.json"
        },
    }
    (payload / "release_manifest.json").write_text(json.dumps(release, indent=2) + "\n")
    print(
        json.dumps(
            {
                "asset_count": decision["simulation_assets"],
                "media_private": len(media),
                "match": decision["pinned_matches"],
                "public_columns": allowed,
                "rows": t.num_rows,
            }
        )
    )


if __name__ == "__main__":
    main()
