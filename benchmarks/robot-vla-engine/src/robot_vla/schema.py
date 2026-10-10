"""Canonical metadata; unknown quantities are explicitly absent."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def content_hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Episode:
    dataset_id: str
    revision: str
    episode_id: str
    task: str
    embodiment: str
    frames: int
    source: str
    instruction: str | None = None
    action_dim: int | None = None
    action_semantics: str | None = None
    state_dim: int | None = None
    timestamp_basis: str = "unavailable"
    fps: float | None = None
    success: bool | None = None
    termination: str | None = None
    pair_id: str | None = None
    pair_evidence: str | None = None
    split: str | None = None
    source_sha256: str | None = None
    observations: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not all((self.dataset_id, self.revision, self.episode_id, self.embodiment, self.source)):
            raise ValueError("episode identity is incomplete")
        if self.frames <= 0:
            raise ValueError("frames must be positive")
        if self.action_dim is not None and (self.action_dim <= 0 or not self.action_semantics):
            raise ValueError("action dimensions require explicit action semantics")
        if self.pair_id is not None and not self.pair_evidence:
            raise ValueError("pair relationship requires source evidence")
        if self.timestamp_basis not in {"source", "index/fps", "unavailable"}:
            raise ValueError("invalid timestamp basis")
        if self.timestamp_basis == "index/fps" and (self.fps is None or self.fps <= 0):
            raise ValueError("synthetic timestamps require fps")
        if self.success is not None and type(self.success) is not bool:
            raise ValueError("success must be bool or null")
        canonical_json(asdict(self))

    def to_dict(self):
        return asdict(self)
