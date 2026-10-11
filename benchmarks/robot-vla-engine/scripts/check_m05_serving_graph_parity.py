"""Verify live ModelOps CUDA Graph HTTP actions against frozen native Eager output."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import requests
from smoke_m05_managed_serving import image_payload

ROOT = Path('/data/VLA-Forge/Evidence/M05')
INPUT = ROOT / 'training_perf/data_pipeline/observation-corrective-0.npz'
REFERENCE = ROOT / 'inference_frozen/eager.json'
OUTPUT = ROOT / 'modelops/serving_graph_action_parity.json'
URL = 'http://127.0.0.1:18769'


def main() -> None:
    ready = requests.get(URL + '/ready', timeout=10).json()
    if ready.get('backend') != 'cuda_graph' or ready.get('version') != 'm05-base-frozen':
        raise ValueError('Wrong live backend or frozen version')
    with np.load(INPUT, allow_pickle=False) as data:
        payload = {'scope': 'simulation', 'version': 'm05-base-frozen',
                   'observed_at': time.time(), 'observation_id': 'graph-http-parity',
                   'image1': image_payload(data['image1']), 'image2': image_payload(data['image2']),
                   'state': data['state'].tolist(), 'task': str(data['task']),
                   'seed': 991, 'timeout_s': 30}
    response = requests.post(URL + '/v1/action-chunks', json=payload, timeout=45)
    response.raise_for_status()
    row = response.json()
    expected = json.loads(REFERENCE.read_text())
    actual_actions = np.asarray(row['actions'], dtype=np.float32)
    expected_actions = np.asarray(expected['actions'], dtype=np.float32)
    if actual_actions.shape != (50, 7) or row['model_sha256'] != expected['model_sha256']:
        raise ValueError('Action shape or model identity mismatch')
    diff = np.abs(actual_actions - expected_actions)
    receipt = {'scope': 'same real M04 observation, seed 991, frozen base, live loopback HTTP',
               'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
               'reference_sha256': hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
               'model_sha256': row['model_sha256'], 'backend': row['backend'],
               'action_shape': list(actual_actions.shape),
               'action_linf': float(diff.max()), 'action_mae': float(diff.mean()),
               'request_s': row['request_s'], 'chunk_s': row['chunk_s'],
               'exact': bool(np.array_equal(actual_actions, expected_actions))}
    OUTPUT.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    if not receipt['exact']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
