"""Measure a simulated allocator cap on RTX3090; never claim 8GB hardware deployment."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

import numpy as np
import torch

from robot_vla.m05_serving import PolicyEngine
from robot_vla.m05_training import EVIDENCE, atomic_json, sha256

INPUT = EVIDENCE / 'training_perf/data_pipeline/observation-corrective-0.npz'
OUTPUT = EVIDENCE / 'inference/allocator_8gib.json'


def main() -> None:
    with (EVIDENCE / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.cuda.get_device_properties(0)
        budget = 8 * 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, budget / device.total_memory), device=0)
        record = json.loads((EVIDENCE / 'modelops/jobs/m05-corrective-cam180-s5501/artifacts/result.json').read_text())
        engine = PolicyEngine('m05-corrective-cam180-s5501', Path(record['adapter']), 'eager')
        with np.load(INPUT, allow_pickle=False) as data:
            result = engine.predict(data['image1'], data['image2'], data['state'], str(data['task']), seed=991)
        evidence = {
            'scope': 'allocator-cap simulation on a physical RTX3090 24GB; not an 8GB-GPU deployment',
            'physical_gpu_name': device.name, 'physical_vram_bytes': device.total_memory,
            'allocator_cap_bytes': budget, 'model_sha256': engine.model_hash,
            'input_sha256': sha256(INPUT), 'action_shape': [len(result['actions']), len(result['actions'][0])],
            'allocated_peak_bytes': result['gpu_allocated_peak'],
            'reserved_peak_bytes': result['gpu_reserved_peak'],
            'valid': len(result['actions']) == 50 and len(result['actions'][0]) == 7,
        }
        atomic_json(OUTPUT, evidence)
        print(json.dumps(evidence, indent=2))
        if not evidence['valid']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
