"""Resource instrumentation respects container quotas and unavailable GPU fields."""

import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil


def command(args, cwd=None):
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=10)
        return dict(
            returncode=result.returncode, stdout=result.stdout.strip(), stderr=result.stderr.strip()
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(returncode=None, stdout=None, stderr=str(exc))


def cgroup_limits(root="/sys/fs/cgroup"):
    result = dict(cpu_quota_cores=None, memory_limit_bytes=None)
    root = Path(root)
    try:
        quota, period = (root / "cpu.max").read_text().split()
        if quota != "max":
            result["cpu_quota_cores"] = int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        memory = (root / "memory.max").read_text().strip()
        if memory != "max":
            result["memory_limit_bytes"] = int(memory)
    except (OSError, ValueError):
        pass
    return result


def snapshot(root="."):
    root = Path(root).resolve()
    process = psutil.Process()
    disk = shutil.disk_usage(root)
    io = psutil.disk_io_counters()
    gpu = command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )
    return dict(
        hostname=socket.gethostname(),
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        cpu_logical_host=psutil.cpu_count(),
        cpu_affinity=len(process.cpu_affinity()) if hasattr(process, "cpu_affinity") else None,
        container_limits=cgroup_limits(),
        ram_host_total_bytes=psutil.virtual_memory().total,
        process_rss_bytes=process.memory_info().rss,
        disk_total_bytes=disk.total,
        disk_free_bytes=disk.free,
        host_disk_io=io._asdict() if io else None,
        gpu=gpu,
        gpu_available=gpu["returncode"] == 0 and bool(gpu["stdout"]),
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        git_commit=command(["git", "rev-parse", "HEAD"], root),
        dirty_git_state=command(["git", "status", "--porcelain"], root),
        pid=os.getpid(),
    )


class Timer:
    """Use synchronize=torch.cuda.synchronize for a measured CUDA region."""

    def __init__(self, synchronize=None):
        self.synchronize = synchronize
        self.elapsed_ms = None

    def __enter__(self):
        if self.synchronize:
            self.synchronize()
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, *_):
        if self.synchronize:
            self.synchronize()
        self.elapsed_ms = (time.perf_counter_ns() - self.start) / 1e6
