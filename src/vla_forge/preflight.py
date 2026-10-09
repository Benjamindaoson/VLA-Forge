"""Read-only preflight for the first Linux GPU + LeRobot/LIBERO run.

This is an environment report, not a simulator, training or safety test.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_report(data_root: str | Path = ".") -> dict[str, object]:
    path = Path(data_root).expanduser().absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    if not path.is_dir():
        path = path.parent
    capacity = shutil.disk_usage(path)
    packages = {
        name: _version(name)
        for name in ("torch", "lerobot", "hf-libero", "mujoco", "torchcodec", "transformers")
    }

    cuda_available = False
    gpu_names: list[str] = []
    gpu_memory_gb: list[float] = []
    torch_error: str | None = None
    if packages["torch"] is not None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                for index in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(index)
                    gpu_names.append(props.name)
                    gpu_memory_gb.append(round(props.total_memory / 2**30, 2))
        except Exception as exc:
            torch_error = f"{type(exc).__name__}: {exc}"
    return {
        "report_kind": "read_only_environment_preflight",
        "platform": platform.system(),
        "python": sys.version.split()[0],
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "packages": packages,
        "cuda_available": cuda_available,
        "gpu_names": gpu_names,
        "gpu_memory_gb": gpu_memory_gb,
        "disk_root": str(path),
        "free_disk_gb": round(capacity.free / 10**9, 2),
        "torch_error": torch_error,
        "real_libero_executed": False,
        "policy_trained": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report LeRobot/LIBERO GPU readiness.")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--require-gpu", action="store_true")
    arguments = parser.parse_args(argv)
    report = collect_report(arguments.data_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    ready = (
        report["platform"] == "Linux"
        and report["cuda_available"] is True
        and report["packages"]["lerobot"] is not None
        and report["packages"]["hf-libero"] is not None
        and report["mujoco_gl"] == "egl"
    )
    if arguments.require_gpu and not ready:
        print(
            "P0 preflight not ready: check Linux, CUDA, LeRobot, hf-libero and MUJOCO_GL=egl.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
