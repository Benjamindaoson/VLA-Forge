"""Bounded real-observation load test of the loopback SmolVLA action API."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

INPUT = Path('/data/VLA-Forge/Evidence/M05/training_perf/data_pipeline/observation-corrective-0.npz')
ROOT = Path('/data/VLA-Forge/Evidence/M05/serving')
CONCURRENCIES = (1, 2, 4, 8)


def encoded(image: np.ndarray) -> str:
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format='PNG')
    return base64.b64encode(stream.getvalue()).decode('ascii')


def percentiles(values: list[float]) -> dict:
    return {f'p{n}_s': float(np.percentile(values, n)) for n in (50, 95, 99)} if values else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:18769')
    parser.add_argument('--version', required=True)
    parser.add_argument('--output-label', help='Distinct receipt name for alternate backend')
    parser.add_argument('--requests-per-concurrency', type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.requests_per_concurrency <= 64:
        raise ValueError('Bounded service load is 1..64 requests per concurrency')
    ready = requests.get(args.url + '/ready', timeout=10).json()
    if ready.get('version') != args.version or not ready.get('ready'):
        raise ValueError('Active service version does not match benchmark')
    with np.load(INPUT, allow_pickle=False) as data:
        images = [encoded(data['image1']), encoded(data['image2'])]
        state, task = data['state'].tolist(), str(data['task'])
    ROOT.mkdir(parents=True, exist_ok=True)
    all_rows, groups = [], {}
    for workers in CONCURRENCIES:
        def call(index: int) -> dict:
            started = time.perf_counter()
            payload = {'scope': 'simulation', 'version': args.version,
                       'observed_at': time.time(), 'observation_id': f'{args.version}-{workers}-{index}',
                       'image1': images[0], 'image2': images[1], 'state': state,
                       'task': task, 'seed': 991, 'timeout_s': 30}
            try:
                response = requests.post(args.url + '/v1/action-chunks', json=payload, timeout=45)
                row = {'version': args.version, 'concurrency': workers, 'request_index': index,
                       'status_code': response.status_code, 'client_wall_s': time.perf_counter() - started}
                if response.ok:
                    result = response.json()
                    action = np.asarray(result['actions'], dtype=np.float32)
                    if action.shape != (50, 7) or not np.isfinite(action).all():
                        raise ValueError('Invalid action chunk in load test')
                    row.update({key: result[key] for key in ('request_s', 'chunk_s', 'queue_s',
                                'observation_age_s', 'gpu_allocated_peak', 'model_sha256')})
                else:
                    row['error'] = response.text[:200]
                return row
            except (requests.RequestException, ValueError, KeyError) as exc:
                return {'version': args.version, 'concurrency': workers, 'request_index': index,
                        'status_code': None, 'client_wall_s': time.perf_counter() - started,
                        'error': repr(exc)}

        begin = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(call, range(args.requests_per_concurrency)))
        wall = time.perf_counter() - begin
        successes = [row for row in rows if row['status_code'] == 200]
        groups[str(workers)] = {
            'requested': len(rows), 'successful': len(successes),
            'errors': len(rows) - len(successes), 'wall_s': wall,
            'successful_requests_per_s': len(successes) / wall,
            'request_latency': percentiles([row['request_s'] for row in successes]),
            'queue_latency': percentiles([row['queue_s'] for row in successes]),
            'chunk_latency': percentiles([row['chunk_s'] for row in successes]),
            'observation_age_max_s': max((row['observation_age_s'] for row in successes), default=None),
            'peak_gpu_allocated_bytes': max((row['gpu_allocated_peak'] for row in successes), default=None),
            'client_wall_mean_s': statistics.mean(row['client_wall_s'] for row in rows),
        }
        all_rows.extend(rows)
    receipt_name = args.output_label or args.version
    if '/' in receipt_name or '..' in receipt_name:
        raise ValueError('Output label must be a basename')
    raw_path = ROOT / f'{receipt_name}_requests.jsonl'
    raw_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in all_rows))
    summary = {'scope': 'loopback simulator-only, same one real M04 observation',
               'version': args.version, 'backend': ready['backend'], 'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
               'groups': groups, 'raw_requests': str(raw_path),
               'total_requested': len(all_rows),
               'total_successful': sum(row['status_code'] == 200 for row in all_rows),
               'total_errors': sum(row['status_code'] != 200 for row in all_rows)}
    (ROOT / f'{receipt_name}_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'version': args.version, 'groups': groups}, indent=2))


if __name__ == '__main__':
    main()
