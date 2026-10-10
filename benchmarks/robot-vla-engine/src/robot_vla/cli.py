"""Small evidence-producing CLI; no implicit training or cloud jobs."""

import argparse
import json
from pathlib import Path

from .audit import audit_hdf5
from .registry import Registry
from .telemetry import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Capture actual runtime and register evidence")
    doctor.add_argument("--output", default="artifacts/environment")
    audit = sub.add_parser("audit-hdf5", help="Stream a robomimic HDF5 quality audit")
    audit.add_argument("path")
    audit.add_argument("--output", default="artifacts/audit")
    audit.add_argument("--max-episodes", type=int)
    export = sub.add_parser("export-registry")
    export.add_argument("root")
    export.add_argument("--output", default="reports/registry.csv")
    args = parser.parse_args()
    if args.command == "export-registry":
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Registry(args.root).export_csv(args.output)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(
        experiment_name=args.command, experiment_type="engineering", environment=snapshot()
    )
    with Registry(output).run(config) as run:
        if args.command == "doctor":
            result, name = config["environment"], "environment.json"
        else:
            result, name = audit_hdf5(args.path, args.max_episodes), "audit.json"
        path = output / name
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        run.artifact(path, args.command)
        run.event("result", {"output": str(path), "status": "MEASURED"})
        print(json.dumps(dict(run_id=run.run_id, output=str(path))))


if __name__ == "__main__":
    main()
