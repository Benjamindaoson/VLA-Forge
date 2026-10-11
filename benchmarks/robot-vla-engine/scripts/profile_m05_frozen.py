"""Bounded 100-call frozen SmolVLA Eager/CUDA-Graph inference comparison."""
from __future__ import annotations

import fcntl
import json
import time

import numpy as np
from profile_m05_inference import INPUT, percentiles

from robot_vla.m05_serving import PolicyEngine
from robot_vla.m05_training import EVIDENCE, atomic_json, sha256

OUT = EVIDENCE / 'inference_frozen'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (EVIDENCE / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with np.load(INPUT, allow_pickle=False) as data:
            observation = (data['image1'], data['image2'], data['state'], str(data['task']))
        receipts = {}
        for mode in ('eager', 'cuda_graph'):
            path = OUT / f'{mode}.json'
            if path.exists():
                row = json.loads(path.read_text())
                if row['input_sha256'] != sha256(INPUT) or row['calls'] != 100:
                    raise ValueError('Frozen receipt does not match inputs')
            else:
                engine = PolicyEngine('m05-base-frozen', None, mode)
                cold = time.perf_counter()
                engine.predict(*observation, seed=991)
                first_predict_s = time.perf_counter() - cold
                for _ in range(99):
                    engine.predict(*observation, seed=991)
                measured = [engine.predict(*observation, seed=991) for _ in range(100)]
                action = np.asarray(measured[0]['actions'], dtype=np.float32)
                raw = np.asarray(measured[0]['raw_actions'], dtype=np.float32)
                if action.shape != (50, 7) or not np.isfinite(action).all():
                    raise ValueError('Invalid continuous action chunk')
                if any(not np.array_equal(action, np.asarray(r['actions'], dtype=np.float32)) for r in measured[1:]):
                    raise ValueError('Nondeterministic seeded actions')
                row = {
                    'mode': mode, 'model_version': 'm05-base-frozen', 'model_sha256': engine.model_hash,
                    'processor_sha256': engine.processor_hash, 'input_sha256': sha256(INPUT),
                    'calls': 100, 'warm_calls': 100, 'load_s': engine.load_s,
                    'first_predict_s': first_predict_s,
                    'capture_cold_s': engine.graph.cold_seconds if mode == 'cuda_graph' else None,
                    'chunk_latency': percentiles([r['chunk_s'] for r in measured]),
                    'end_to_end_latency': percentiles([r['predict_s'] for r in measured]),
                    'gpu_peak_allocated': max(r['gpu_allocated_peak'] for r in measured),
                    'gpu_peak_reserved': max(r['gpu_reserved_peak'] for r in measured),
                    'actions': action.tolist(), 'raw_actions': raw.tolist(),
                }
                atomic_json(path, row)
            receipts[mode] = row
        eager, graph = receipts['eager'], receipts['cuda_graph']
        delta = np.abs(np.asarray(eager['raw_actions']) - np.asarray(graph['raw_actions']))
        if delta.shape != (50, 7) or float(delta.max()) > 1e-6:
            raise ValueError('Frozen action parity failed')
        summary = {
            'scope': 'one real M04 observation, frozen official model, static inference only',
            'input_sha256': sha256(INPUT), 'model_sha256': eager['model_sha256'],
            'action_raw_linf': float(delta.max()),
            'eager_chunk_latency': eager['chunk_latency'], 'cuda_graph_chunk_latency': graph['chunk_latency'],
            'speedup_p50': eager['chunk_latency']['p50_s'] / graph['chunk_latency']['p50_s'],
            'graph_capture_cold_s': graph['capture_cold_s'], 'passes_static_correctness': float(delta.max()) <= 1e-6,
        }
        atomic_json(OUT / 'summary.json', summary)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
