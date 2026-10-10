"""Chunk-aware policy timing and periodic measured resource events."""

import threading
import time

import psutil

from .telemetry import Timer, command


class ChunkProfiler:
    def __init__(self, policy, synchronize=None):
        self.policy = policy
        self.synchronize = synchronize
        self.chunk_calls = []
        self.control_step = 0
        method = (
            "_get_action_chunk" if hasattr(policy, "_get_action_chunk") else "predict_action_chunk"
        )
        original = getattr(policy, method)

        def measured(*args, **kwargs):
            with Timer(self.synchronize) as timer:
                result = original(*args, **kwargs)
            self.chunk_calls.append(
                dict(control_step=self.control_step, latency_ms=timer.elapsed_ms)
            )
            return result

        setattr(policy, method, measured)

    def reset(self):
        self.chunk_calls.clear()
        self.control_step = 0
        self.policy.reset()

    def select_action(self, batch):
        result = self.policy.select_action(batch)
        self.control_step += 1
        return result


class ResourceMonitor:
    def __init__(self, run, interval=5.0, gpu=True):
        if interval <= 0:
            raise ValueError("monitor interval must be positive")
        self.run = run
        self.interval = interval
        self.gpu = gpu
        self.process = psutil.Process()
        self.process.cpu_percent()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self):
        io = self.process.io_counters() if hasattr(self.process, "io_counters") else None
        value = dict(
            monotonic_seconds=time.monotonic(),
            process_cpu_percent=self.process.cpu_percent(),
            process_rss_bytes=self.process.memory_info().rss,
            process_io=io._asdict() if io else None,
            disk_free_bytes=psutil.disk_usage(str(self.run.directory)).free,
            gpu=None,
        )
        if self.gpu:
            value["gpu"] = command(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ]
            )
        self.run.event("resources", value)

    def _loop(self):
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except Exception as exc:
                self.run.event("monitor_error", {"error": str(exc)})

    def __enter__(self):
        self.sample()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=12)
        if self.thread.is_alive():
            raise RuntimeError("resource monitor did not stop")
        self.sample()
