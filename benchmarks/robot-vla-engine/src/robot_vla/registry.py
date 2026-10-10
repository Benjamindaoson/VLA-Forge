"""Transactional experiment records; SQLite is the source of truth."""

import csv
import json
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .schema import canonical_json, content_hash, file_hash


def utcnow():
    return datetime.now(timezone.utc).isoformat()


class Registry:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.sqlite"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    timestamp_start TEXT NOT NULL, timestamp_end TEXT,
                    failure_reason TEXT, config_json TEXT NOT NULL, config_hash TEXT NOT NULL,
                    wall_seconds REAL);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
                    timestamp TEXT NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
                    role TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL, size INTEGER NOT NULL);
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        return db

    @contextmanager
    def run(self, config):
        encoded = canonical_json(config)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
        directory = self.root / "runs" / run_id
        directory.mkdir(parents=True, exist_ok=False)
        with self.connect() as db:
            db.execute(
                "INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                (run_id, "RUNNING", utcnow(), None, None, encoded, content_hash(config), None),
            )
        (directory / "config.json").write_text(encoded, encoding="utf-8")
        run = Run(self, run_id, directory)
        start = time.perf_counter()
        status, reason = "COMPLETE", None
        try:
            yield run
        except BaseException as exc:
            status, reason = "FAILED", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            with self.connect() as db:
                db.execute(
                    "UPDATE runs SET status=?,timestamp_end=?,failure_reason=?,wall_seconds=? WHERE run_id=?",
                    (status, utcnow(), reason, time.perf_counter() - start, run_id),
                )
            self.export_events(run_id)

    def get(self, run_id):
        with self.connect() as db:
            record = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if record is None:
            raise KeyError(run_id)
        return dict(record)

    def export_events(self, run_id):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM events WHERE run_id=? ORDER BY id", (run_id,)
            ).fetchall()
        path = self.root / "runs" / run_id / "events.jsonl"
        text = "".join(
            canonical_json(
                dict(
                    id=r["id"],
                    run_id=run_id,
                    timestamp=r["timestamp"],
                    kind=r["kind"],
                    payload=json.loads(r["payload_json"]),
                )
            )
            + "\n"
            for r in rows
        )
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    def export_csv(self, path):
        with self.connect() as db:
            cursor = db.execute("SELECT * FROM runs ORDER BY timestamp_start")
            fields = [c[0] for c in cursor.description]
            records = cursor.fetchall()
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(records)

    def verify_artifacts(self, run_id):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM artifacts WHERE run_id=?", (run_id,)).fetchall()
        return [
            r["path"]
            for r in rows
            if not Path(r["path"]).is_file() or file_hash(r["path"]) != r["sha256"]
        ]


class Run:
    def __init__(self, registry, run_id, directory):
        self.registry, self.run_id, self.directory = registry, run_id, directory

    def event(self, kind, payload):
        encoded = canonical_json(payload)
        with self.registry.connect() as db:
            db.execute(
                "INSERT INTO events(run_id,timestamp,kind,payload_json) VALUES (?,?,?,?)",
                (self.run_id, utcnow(), kind, encoded),
            )

    def artifact(self, path, role):
        path = Path(path).resolve()
        digest = file_hash(path)
        # Run directories are durable evidence. Shared reports/configs can be rerendered,
        # so snapshot their exact bytes before registering them. Cross-run checkpoint
        # references retain the existing immutable file without duplicating GB of weights.
        if not path.is_relative_to(self.registry.root / "runs"):
            original = path
            evidence = self.directory / "evidence"
            evidence.mkdir(exist_ok=True)
            path = evidence / f"{digest[:24]}{original.suffix}"
            if not path.exists():
                shutil.copyfile(original, path)
            if file_hash(path) != digest:
                raise ValueError("artifact changed while snapshotting")
            self.event(
                "artifact_origin", dict(original=str(original), snapshot=str(path), sha256=digest)
            )
        with self.registry.connect() as db:
            db.execute(
                "INSERT INTO artifacts(run_id,role,path,sha256,size) VALUES (?,?,?,?,?)",
                (self.run_id, role, str(path), digest, path.stat().st_size),
            )
        return digest
