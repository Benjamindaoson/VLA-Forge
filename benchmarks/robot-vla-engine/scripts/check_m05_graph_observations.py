"""Check CUDA-Graph freshness and action parity on multiple real M04 observations."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np

from robot_vla.m05_serving import PolicyEngine
from robot_vla.m05_training import EVIDENCE, atomic_json, sha256

OBSERVATIONS = [
    EVIDENCE / 'training_perf/data_pipeline/observation-corrective-0.npz',
    EVIDENCE / 'training_perf/data_pipeline/observation-corrective-300.npz',
    EVIDENCE / 'training_perf/data_pipeline/observation-standard-0.npz',
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('frozen', 'corrective'), required=True)
    args = parser.parse_args()
    if args.model == 'frozen':
        version, adapter = 'm05-base-frozen', None
    else:
        record = json.loads((EVIDENCE / 'modelops/jobs/m05-corrective-cam180-s5501/artifacts/result.json').read_text())
        version, adapter = 'm05-corrective-cam180-s5501', Path(record['adapter'])
    with (EVIDENCE / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        eager = PolicyEngine(version, adapter, 'eager')
        graph = PolicyEngine(version, adapter, 'cuda_graph')
        cases = []
        for idx, path in enumerate(OBSERVATIONS):
            with np.load(path, allow_pickle=False) as data:
                inp = (data['image1'], data['image2'], data['state'], str(data['task']))
            seed = 991 + idx
            reference = eager.predict(*inp, seed=seed)
            accelerated = graph.predict(*inp, seed=seed)
            raw_delta = np.abs(np.asarray(reference['raw_actions']) - np.asarray(accelerated['raw_actions']))
            executed_delta = np.abs(np.asarray(reference['actions']) - np.asarray(accelerated['actions']))
            if raw_delta.shape != (50, 7) or not np.isfinite(raw_delta).all():
                raise ValueError('Invalid multi-observation action output')
            cases.append({
                'input_path': str(path), 'input_sha256': sha256(path), 'seed': seed,
                'raw_action_linf': float(raw_delta.max()), 'raw_action_mae': float(raw_delta.mean()),
                'executed_action_linf': float(executed_delta.max()),
                'eager_chunk_s': reference['chunk_s'], 'graph_chunk_s': accelerated['chunk_s'],
                'graph_capture_cold_s': graph.graph.cold_seconds if idx == 0 else None,
            })
        try:
            graph.predict(inp[0], inp[1], np.zeros(6, dtype=np.float32), inp[3], seed=997)
        except ValueError:
            invalid_rejected = True
        else:
            invalid_rejected = False
        recovered = graph.predict(*inp, seed=991 + len(OBSERVATIONS) - 1)
        valid_after_error = np.array_equal(np.asarray(recovered['actions']), np.asarray(accelerated['actions']))
        record = {
            'scope': 'three real M04 observations, same model weights, seeded noise; static shape freshness',
            'model_version': version, 'model_sha256': graph.model_hash,
            'processor_sha256': graph.processor_hash,
            'cases': cases, 'invalid_shape_rejected': invalid_rejected,
            'valid_after_error_exact': valid_after_error,
            'passed': all(row['raw_action_linf'] <= 1e-6 for row in cases) and invalid_rejected and valid_after_error,
        }
        output = EVIDENCE / 'inference' / f'{args.model}_multi_observation_graph.json'
        atomic_json(output, record)
        print(json.dumps({'model': args.model, 'passed': record['passed'],
                          'max_linf': max(row['raw_action_linf'] for row in cases),
                          'capture_cold_s': cases[0]['graph_capture_cold_s']}, indent=2))
        if not record['passed']:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
