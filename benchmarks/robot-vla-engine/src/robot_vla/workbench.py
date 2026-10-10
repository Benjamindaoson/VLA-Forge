"""Read-only, loopback-only view of explicitly named project evidence."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .correction_assets import relocate
from .correction_sources import bind_cases, load_gate0_evidence

EVIDENCE = {
    "gate0": "reports/repair_gate0/summary_v2.json",
    "gate0-verification": "reports/repair_gate0/verification.json",
    "prefix": "reports/repair_gate0/prefix_replay.json",
    "source-audit": "reports/repair_gate1_sources/source_audit_v3.json",
    "source-verification": "reports/repair_gate1_sources/verification.json",
    "readiness": "reports/server_3090_20261010/qualification_v1.json",
    "transfer": "reports/server_3090_20261010/asset_transfer.json",
    "runtime": "reports/server_3090_20261010/runtime_verification.json",
    "preflight": "reports/server_3090_20261010/remote_preflight/preflight.json",
}


def validate_receipt(name, data):
    if not isinstance(data, dict):
        raise ValueError("receipt must be an object")
    if name not in ("readiness", "transfer"):
        return
    if not isinstance(data.get("status"), str):
        raise ValueError("receipt status is missing")
    key = "steps" if name == "readiness" else "assets"
    rows = data.get(key)
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("receipt rows are invalid")
    for row in rows:
        if name == "readiness":
            valid = (
                isinstance(row.get("name"), str)
                and type(row.get("exit_code")) is int
                and type(row.get("seconds")) in (int, float)
                and 0 <= row["seconds"] <= 1e12
            )
        else:
            valid = isinstance(row.get("path"), str) and all(
                type(row.get(k)) is int and row[k] >= 0 for k in ("files", "bytes")
            )
        if not valid:
            raise ValueError("receipt row fields are invalid")


class Workbench:
    def __init__(self, root, legacy_root=None):
        self.root = Path(root).resolve()
        self.legacy_root = Path(legacy_root).resolve() if legacy_root else self.root / "legacy"

    def evidence(self, name):
        relative = EVIDENCE[name]
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise KeyError(name)
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("receipt exceeds display limit")
        raw = path.read_bytes()
        data = json.loads(raw)
        json.dumps(data, allow_nan=False)
        validate_receipt(name, data)
        return dict(
            name=name,
            path=relative,
            sha256=hashlib.sha256(raw).hexdigest(),
            data=data,
        )

    def optional(self, name):
        try:
            return self.evidence(name)
        except KeyError:
            return dict(name=name, data=None, status="MISSING")
        except (ValueError, OSError):
            return dict(name=name, data=None, status="INVALID")

    def snapshot(self):
        result = dict(
            generated_utc=datetime.now(timezone.utc).isoformat(),
            source_status="INVALID",
            incidents=[],
            verified_corrections=0,
            correction_success_rate=None,
            read_only=True,
            scope="archived development cases and saved receipts; no live GPU monitoring",
            release=dict(
                status="INSUFFICIENT_EVIDENCE",
                can_release=False,
                missing=[
                    "经确认的纠正数据",
                    "修复后的候选权重",
                    "同条件旧能力回归",
                    "独立开发/测试评测",
                ],
                reason="当前工作台只展示归档和工程验收，未接入完整发布证据。",
            ),
        )
        try:
            for key in ("gate0", "prefix", "gate0-verification"):
                self.evidence(key)
            baseline, prefix = load_gate0_evidence(self.root / "reports/repair_gate0")
            cases = bind_cases(baseline, prefix)
            audit = self.evidence("source-audit")
            verification = self.evidence("source-verification")["data"]
            accepted = [r for r in verification["runs"] if r["role"] == "accepted"]
            if (
                verification["status"] != "PASS"
                or len(accepted) != 1
                or accepted[0]["summary_sha256"] != audit["sha256"]
                or accepted[0]["run_id"] != audit["data"]["run_id"]
            ):
                raise ValueError("source audit identity mismatch")
            for case in cases:
                pair = next(
                    p
                    for p in baseline["pairs"]
                    if p["suite"] == case["suite"] and p["task_id"] == case["task_id"]
                )
                trial = pair["rollouts"][0]
                current = relocate(self.legacy_root, trial["trajectory_path"])
                result["incidents"].append(
                    dict(
                        **case,
                        instruction=trial["instruction"],
                        original_success=False,
                        initial_state=0,
                        environment_seed=trial["seed"],
                        original_steps=trial["episode_length"],
                        remaining_steps=trial["episode_length"] - case["prefix"],
                        raw_trajectory_status="PRESENT_UNVERIFIED"
                        if current.is_file()
                        else "MISSING",
                        archived_reconstruction="PASS",
                        current_reconstruction="NOT_RUN",
                        observed_on="RTX 4090D / 归档开发实验",
                        baseline_weights_sha256=baseline["checkpoint"]["checkpoint_sha256"],
                        evidence_keys=["gate0", "prefix", "source-audit"],
                    )
                )
            result.update(
                source_status="VERIFIED_ARCHIVE",
                ordinary_candidates=verification["eligible_demonstration_candidates"],
                excluded_validation=verification["excluded_validation_episodes"],
            )
        except (OSError, KeyError, ValueError, TypeError, StopIteration):
            result.update(
                incidents=[],
                source_status="INVALID",
                source_error="归档缺失或未通过哈希/身份校验，已停止展示已验证案例。",
            )
        runtime = self.optional("readiness")
        data = runtime["data"]
        if data is not None and (not isinstance(data, dict) or "status" not in data):
            runtime = dict(status="INVALID", data=None)
            data = None
        result["runtime"] = dict(
            status=data["status"] if data else runtime["status"],
            observed_utc=data.get("ended_utc", data.get("started_utc")) if data else None,
            steps=data.get("steps", []) if data else [],
            live_monitoring=False,
            evidence_key="readiness",
        )
        result["transfer"] = self.optional("transfer")
        result["preflight"] = self.optional("preflight")
        result["evidence"] = [
            dict(key=name, available=self.optional(name)["data"] is not None) for name in EVIDENCE
        ]
        return result


def make_server(root, port=8765, legacy_root=None):
    view = Workbench(root, legacy_root)
    static = Path(__file__).parent / "web"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, content, content_type="application/json; charset=utf-8"):
            if not isinstance(content, bytes):
                content = json.dumps(content, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
            )
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            try:
                if urlsplit("//" + self.headers.get("Host", "")).hostname not in (
                    "127.0.0.1",
                    "localhost",
                ):
                    self.respond(421, dict(error="loopback host required"))
                    return
                path = unquote(urlsplit(self.path).path)
                if path == "/api/overview":
                    self.respond(200, view.snapshot())
                elif path.startswith("/api/incidents/"):
                    key = path.removeprefix("/api/incidents/")
                    row = next(
                        (r for r in view.snapshot()["incidents"] if r["case_id"] == key), None
                    )
                    if row is None:
                        raise KeyError(key)
                    self.respond(200, row)
                elif path.startswith("/api/evidence/"):
                    self.respond(200, view.evidence(path.removeprefix("/api/evidence/")))
                elif path in ("/", "/app.js", "/style.css"):
                    name, content_type = {
                        "/": ("index.html", "text/html; charset=utf-8"),
                        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                        "/style.css": ("style.css", "text/css; charset=utf-8"),
                    }[path]
                    self.respond(200, (static / name).read_bytes(), content_type)
                else:
                    raise KeyError(path)
            except (KeyError, FileNotFoundError):
                self.respond(404, dict(error="not found"))
            except (OSError, ValueError, TypeError):
                self.respond(422, dict(error="evidence unavailable or invalid"))

        def do_POST(self):
            self.respond(405, dict(error="read-only workbench"))

        do_PUT = do_DELETE = do_PATCH = do_POST

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = make_server(args.root, args.port, args.legacy_root)
    print(f"Read-only Reliability Lab: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
