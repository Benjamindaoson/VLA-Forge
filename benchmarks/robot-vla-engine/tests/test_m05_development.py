"""Guard M05 development evaluation against M06 leakage and stale receipts."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_m05_development.py"
PROTOCOL = Path(__file__).resolve().parents[3] / "configs/m05/formal_protocol.json"


def module():
    spec = importlib.util.spec_from_file_location("m05_development", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_only_frozen_development_cases_are_admitted():
    m = module()
    frozen = json.loads(PROTOCOL.read_text())
    assert len(m.validate_plan(frozen)) == 8
    leaked = json.loads(PROTOCOL.read_text())
    leaked["development"][0]["init"] = 40
    with pytest.raises(ValueError, match="M06"):
        m.validate_plan(leaked)


def test_resume_rejects_modified_rollout_artifact(tmp_path):
    m = module()
    payload = tmp_path / "real.bin"
    payload.write_bytes(b"true episode content")
    import hashlib

    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    entry = {"path": str(payload), "sha256": digest}
    receipt = {
        "status": "complete",
        "success": False,
        "checkpoint_sha256": "a" * 64,
        "video": entry,
        "trajectory": entry,
        "action_trace": entry,
    }
    assert m.receipt_valid(receipt, "a" * 64)
    payload.write_bytes(b"tampered")
    assert not m.receipt_valid(receipt, "a" * 64)
