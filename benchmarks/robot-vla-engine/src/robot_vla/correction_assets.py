"""Explicit, read-only relocation of archived evidence; never rewrite historical paths."""

import json
import sqlite3
from pathlib import Path, PurePosixPath

import numpy as np

from .correction_sources import bind_cases, load_gate0_evidence
from .preflight import bind_processor, verify_receipt
from .protocol import load_protocol
from .schema import file_hash

ORIGIN = PurePosixPath("/root/autodl-tmp/robot-vla")
TEACHER = "20261007T150730-84383440f521"
BASELINE = "20261007T121952-d03c6e7f64c8"


def relocate(asset_root, original):
    source = PurePosixPath(original)
    if ".." in source.parts or not source.is_absolute():
        raise ValueError("invalid archived path")
    relative = source.relative_to(ORIGIN)
    root = Path(asset_root).resolve()
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError("asset path escapes archive")
    return target


def verify_asset(asset_root, original, expected):
    path = relocate(asset_root, original)
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_hash(path) != expected:
        raise ValueError("asset hash mismatch: " + str(path))
    return path


def asset_preflight(root, asset_root):
    root, asset_root = Path(root).resolve(), Path(asset_root).resolve()
    baseline, prefix = load_gate0_evidence(root / "reports/repair_gate0")
    cases = bind_cases(baseline, prefix)
    data = load_protocol(root / "configs/data_protocol_v1.json")
    compute = load_protocol(root / "configs/compute_protocol.json")
    report = dict(
        status="BLOCKED_ASSETS",
        asset_root=str(asset_root),
        origin=str(ORIGIN),
        cases=cases,
        verified=[],
        missing=[],
        invalid=[],
        verified_corrections=0,
        correction_success_rate=None,
        historical_paths_modified=False,
    )

    def check(role, original, sha):
        record = dict(
            role=role, original=original, sha256=sha, relocated=str(relocate(asset_root, original))
        )
        try:
            verify_asset(asset_root, original, sha)
            report["verified"].append(record)
        except FileNotFoundError:
            report["missing"].append(record)
        except ValueError as exc:
            report["invalid"].append(dict(record, error=str(exc)))

    for role, folder, run_id in [
        ("baseline", "smolvla_seed42", BASELINE),
        ("teacher", "diversity_high_seed42", TEACHER),
    ]:
        receipt_path = root / f"reports/{folder}/training_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        cfg = receipt["config"]
        if (
            receipt["status"] != "COMPLETE"
            or receipt["training_run_id"] != run_id
            or cfg["experiment_type"] != "formal_training"
            or cfg["compute_protocol_identity"] != compute["identity"]
            or cfg["protocol_identity"] != data["identity"]
            or cfg["recipe"]["policy"] != "smolvla"
            or cfg["recipe"]["condition"]
            != ("baseline_full" if role == "baseline" else "diversity_high")
        ):
            raise ValueError("archived training ownership differs")
        rows = [r for r in receipt["verified_artifacts"] if r["role"] == "policy_checkpoint"]
        names = {PurePosixPath(r["path"]).name for r in rows}
        if (
            not {
                "model.safetensors",
                "config.json",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
            }
            <= names
        ):
            raise ValueError("incomplete checkpoint receipt")
        for row in rows:
            check(role + "_checkpoint", row["path"], row["sha256"])
        report[role] = dict(
            training_run_id=run_id,
            receipt_sha256=file_hash(receipt_path),
            processor=cfg["verified_sources"]["processor"],
        )

    for case in cases:
        reference = next(
            r
            for r in baseline["replays"]
            if (r["suite"], r["task_id"], r["prefix"])
            == (case["suite"], case["task_id"], case["prefix"])
        )
        reconstruction = next(
            r
            for r in prefix["rows"]
            if (r["suite"], r["prefix"]) == (case["suite"], case["prefix"])
        )
        check("gate0_reference", reference["raw_path"], reference["raw_sha256"])
        check("gate0_reconstruction", reconstruction["raw_path"], reconstruction["raw_sha256"])
    # Original commands are bound by the original read-only Registry and the
    # already hash-verified summary. Do not create a new empty Registry here.
    db_path = asset_root / "artifacts/experiments/registry.sqlite"
    if not db_path.is_file():
        report["missing"].append(dict(role="gate0_registry", relocated=str(db_path)))
    else:
        try:
            with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as db:
                parent = db.execute(
                    "SELECT status FROM runs WHERE run_id=?", (baseline["run_id"],)
                ).fetchone()
                summary = db.execute(
                    "SELECT sha256 FROM artifacts WHERE run_id=? AND role=?",
                    (baseline["run_id"], "gate0_summary"),
                ).fetchall()
                if (
                    parent != ("COMPLETE",)
                    or (file_hash(root / "reports/repair_gate0/summary_v2.json"),) not in summary
                ):
                    raise ValueError("unowned or incomplete Gate 0 summary")
                for suite in ("libero_10", "libero_goal"):
                    pair = next(
                        p for p in baseline["pairs"] if p["suite"] == suite and p["task_id"] == 0
                    )
                    original = pair["rollouts"][0]["trajectory_path"]
                    rows = db.execute(
                        "SELECT sha256 FROM artifacts WHERE run_id=? AND path=?",
                        (baseline["run_id"], original),
                    ).fetchall()
                    if len(rows) != 1:
                        raise ValueError("ambiguous Gate 0 trajectory ownership")
                    check("baseline_commands", original, rows[0][0])
        except (sqlite3.Error, ValueError) as exc:
            report["invalid"].append(dict(role="gate0_registry", error=str(exc)))
    for role, action in [
        ("data", lambda: verify_receipt(root / "data/libero", data["revision"])),
        (
            "processor",
            lambda: bind_processor(
                root / "data/models/smolvlm",
                "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
                report["teacher"]["processor"],
            ),
        ),
    ]:
        try:
            report[role + "_verification"] = action()
        except (FileNotFoundError, ValueError) as exc:
            report["missing" if isinstance(exc, FileNotFoundError) else "invalid"].append(
                dict(role=role, error=str(exc))
            )
    if not report["missing"] and not report["invalid"]:
        report["status"] = "ASSETS_VERIFIED_RUNTIME_PENDING"
    return report


def load_cases(root, asset_root, preflight):
    if preflight["status"] != "ASSETS_VERIFIED_RUNTIME_PENDING":
        raise ValueError("assets are not qualified")
    baseline, _ = load_gate0_evidence(Path(root) / "reports/repair_gate0")
    result = []
    for row in preflight["cases"]:
        pair = next(
            p for p in baseline["pairs"] if p["suite"] == row["suite"] and p["task_id"] == 0
        )
        rollout = pair["rollouts"][0]
        path = rollout["trajectory_path"]
        manifest = next(r for r in preflight["verified"] if r["original"] == path)
        with np.load(verify_asset(asset_root, path, manifest["sha256"]), allow_pickle=False) as z:
            commands = z["actions"][: row["prefix"]].copy()
        ref = next(
            r
            for r in baseline["replays"]
            if (r["suite"], r["prefix"], r["task_id"]) == (row["suite"], row["prefix"], 0)
        )
        with np.load(
            verify_asset(asset_root, ref["raw_path"], row["reference_sha256"]), allow_pickle=False
        ) as z:
            from .correction_records import FIELDS

            reference = {k: z["reference_" + k][0].copy() for k in FIELDS}
            reference["success"] = bool(reference["success"])
        result.append(
            dict(
                row,
                prefix_commands=commands,
                reference=reference,
                horizon=520 if row["suite"] == "libero_10" else 300,
                environment_seed=42,
                instruction=rollout["instruction"],
            )
        )
    return result
