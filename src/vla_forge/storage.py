"""SQLite-backed immutable cases and append-only repair event audit log."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from vla_forge.models import RepairCase


class CaseConflictError(ValueError):
    pass


def canonical_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


class CaseRepository:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    policy_version TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    case_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (case_id) REFERENCES cases(case_id)
                );
                CREATE INDEX IF NOT EXISTS events_by_case ON events(case_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=20)
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("PRAGMA busy_timeout = 20000")
        return db

    def insert_case(self, case: RepairCase) -> str:
        content = canonical_json(case.model_dump(mode="json"))
        digest = hashlib.sha256(content.encode()).hexdigest()
        with self._connect() as db:
            old = db.execute(
                "SELECT content_sha256 FROM cases WHERE case_id=?",
                (case.failure.case_id,),
            ).fetchone()
            if old is not None:
                if old[0] != digest:
                    raise CaseConflictError("A case ID cannot be overwritten with different data.")
                return digest
            db.execute(
                "INSERT INTO cases VALUES (?, ?, ?, ?, ?)",
                (
                    case.failure.case_id, case.failure.policy_version,
                    case.failure.task_id, digest, content,
                ),
            )
        return digest

    def get_case(self, case_id: str) -> RepairCase | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM cases WHERE case_id=?", (case_id,)
            ).fetchone()
        return RepairCase.model_validate_json(row[0]) if row else None

    def list_cases(self, limit: int = 100) -> list[RepairCase]:
        if not 1 <= limit <= 1000:
            raise ValueError("List limit must be 1..1000.")
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM cases ORDER BY case_id LIMIT ?", (limit,)
            ).fetchall()
        return [RepairCase.model_validate_json(row[0]) for row in rows]

    def append_event(self, event_id: str, case_id: str, kind: str, payload: object) -> str:
        if not all(isinstance(value, str) and value.strip() for value in (event_id, case_id, kind)):
            raise ValueError("Event ID, case ID and kind must be nonempty.")
        content = canonical_json(payload)
        digest = hashlib.sha256(content.encode()).hexdigest()
        with self._connect() as db:
            row = db.execute(
                "SELECT case_id, kind, content_sha256 FROM events WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if row:
                if row != (case_id, kind, digest):
                    raise CaseConflictError("Existing audit event is immutable.")
                return digest
            db.execute(
                "INSERT INTO events (event_id, case_id, kind, content_sha256, payload)"
                " VALUES (?, ?, ?, ?, ?)", (event_id, case_id, kind, digest, content),
            )
        return digest

    def list_events(self, case_id: str) -> list[dict[str, object]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT event_id, kind, content_sha256, payload, created_at"
                " FROM events WHERE case_id=? ORDER BY created_at, event_id", (case_id,),
            ).fetchall()
        return [
            {
                "event_id": row[0], "kind": row[1], "sha256": row[2],
                "payload": json.loads(row[3]), "created_at": row[4],
            }
            for row in rows
        ]
