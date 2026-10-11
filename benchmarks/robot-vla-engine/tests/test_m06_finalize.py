"""A resumed formal run can have SQLite telemetry before JSONL export."""
import importlib.util
import json
import sqlite3
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/finalize_m06.py'
spec = importlib.util.spec_from_file_location('finalize_m06', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_resource_payloads_use_sqlite_when_run_was_interrupted(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'OUT', tmp_path)
    root = tmp_path / 'registry'
    run_dir = root / 'runs' / 'run-a'
    run_dir.mkdir(parents=True)
    db = sqlite3.connect(root / 'registry.sqlite')
    db.execute('CREATE TABLE events (id INTEGER PRIMARY KEY, run_id TEXT, kind TEXT, payload_json TEXT)')
    db.execute('INSERT INTO events (run_id, kind, payload_json) VALUES (?, ?, ?)',
               ('run-a', 'resources', json.dumps({'process_cpu_percent': 23.0})))
    db.commit()
    db.close()
    assert module.resource_payloads('run-a') == [{'process_cpu_percent': 23.0}]
    (run_dir / 'events.jsonl').write_text(json.dumps({'kind': 'resources',
        'payload': {'process_cpu_percent': 25.0}}) + '\n')
    assert module.resource_payloads('run-a') == [{'process_cpu_percent': 25.0}]
