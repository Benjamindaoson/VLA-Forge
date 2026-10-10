"""Fetch the frozen LIBERO asset dataset into the single VLA-Forge data root."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path("/data/VLA-Forge")
REPO = "lerobot/libero-assets"
REVISION = "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6"
DEST = ROOT / "Assets" / "LIBERO-assets"
RECEIPT = ROOT / "Evidence" / "M04" / "receipts" / "libero-assets.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    assert DEST.is_relative_to(ROOT) and RECEIPT.is_relative_to(ROOT)
    api = HfApi()
    info = api.dataset_info(REPO, revision=REVISION, files_metadata=False)
    if info.sha != REVISION:
        raise RuntimeError(f"asset revision mismatch: requested {REVISION}, resolved {info.sha}")
    expected = sorted(x.rfilename for x in info.siblings)
    DEST.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(DEST),
        max_workers=1,
        force_download=False,
    )
    actual = sorted(
        p.relative_to(DEST).as_posix()
        for p in DEST.rglob("*")
        if p.is_file() and ".cache" not in p.parts
    )
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise RuntimeError(json.dumps({"missing": missing[:20], "unexpected": unexpected[:20]}))
    files = [
        {"path": n, "bytes": (DEST / n).stat().st_size, "sha256": digest(DEST / n)} for n in actual
    ]
    record = {
        "repo_id": REPO,
        "revision": info.sha,
        "requested_revision": REVISION,
        "source": "hf-mirror.com serving pinned Hugging Face dataset commit; API commit SHA verified",
        "download_path": resolved,
        "file_count": len(files),
        "total_bytes": sum(x["bytes"] for x in files),
        "files": files,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "destination": str(DEST),
    }
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "files"}))


if __name__ == "__main__":
    main()
