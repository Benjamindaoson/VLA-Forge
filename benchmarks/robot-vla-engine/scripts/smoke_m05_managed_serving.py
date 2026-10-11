"""Exercise ModelOps simulation policy serving, version switch, and rollback."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

INPUT = Path('/data/VLA-Forge/Evidence/M05/training_perf/data_pipeline/observation-corrective-0.npz')
OUTPUT = Path('/data/VLA-Forge/Evidence/M05/modelops/serving_smoke.json')


def image_payload(image: np.ndarray) -> str:
    stream = io.BytesIO()
    Image.fromarray(image).save(stream, format='PNG')
    return base64.b64encode(stream.getvalue()).decode('ascii')


def check_action(response: dict, version: str) -> np.ndarray:
    if response.get('version') != version:
        raise ValueError('Wrong model version in action response')
    actions = np.asarray(response.get('actions'), dtype=np.float32)
    if actions.shape != (50, 7) or not np.isfinite(actions).all() or np.abs(actions).max() > 1.00001:
        raise ValueError('Invalid continuous robot action chunk')
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:18769')
    parser.add_argument('--base-version', default='m05-base-frozen')
    parser.add_argument('--candidate-version', default='m05-standard-s5501')
    args = parser.parse_args()
    with np.load(INPUT, allow_pickle=False) as data:
        pixels = [image_payload(data['image1']), image_payload(data['image2'])]
        state = data['state'].tolist()
        task = str(data['task'])
    session = requests.Session()
    health = session.get(args.url + '/health', timeout=10)
    health.raise_for_status()
    ready = session.get(args.url + '/ready', timeout=10)
    ready.raise_for_status()
    if not ready.json().get('ready') or ready.json().get('version') != args.base_version:
        raise ValueError('ModelOps did not load the expected base')

    def predict(version: str, observation_id: str) -> tuple[np.ndarray, dict]:
        payload = {'scope': 'simulation', 'version': version, 'observed_at': time.time(),
                   'observation_id': observation_id, 'image1': pixels[0], 'image2': pixels[1],
                   'state': state, 'task': task, 'seed': 991, 'timeout_s': 30}
        response = session.post(args.url + '/v1/action-chunks', json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        return check_action(data, version), {key: data[key] for key in (
            'version', 'backend', 'model_sha256', 'processor_sha256', 'request_s',
            'chunk_s', 'queue_s', 'gpu_allocated_peak', 'observation_id')}

    base_actions, base = predict(args.base_version, 'base-before')
    invalid = session.post(args.url + '/v1/action-chunks', json={
        'scope': 'robot', 'version': args.base_version, 'observed_at': time.time(),
        'observation_id': 'real-robot-disallowed', 'image1': pixels[0], 'image2': pixels[1],
        'state': state, 'task': task, 'seed': 991}, timeout=10)
    if invalid.status_code != 422:
        raise ValueError('Real-robot scope was not rejected')
    switched = session.post(args.url + '/v1/development-version', json={
        'version': args.candidate_version, 'backend': 'eager', 'rollback': False}, timeout=120)
    switched.raise_for_status()
    candidate_actions, candidate = predict(args.candidate_version, 'candidate')
    stale = session.post(args.url + '/v1/action-chunks', json={
        'scope': 'simulation', 'version': args.base_version, 'observed_at': time.time(),
        'observation_id': 'stale', 'image1': pixels[0], 'image2': pixels[1],
        'state': state, 'task': task, 'seed': 991}, timeout=10)
    if stale.status_code != 422:
        raise ValueError('Stale model version was not rejected')
    rolled = session.post(args.url + '/v1/development-version', json={
        'version': args.base_version, 'backend': 'eager', 'rollback': True}, timeout=120)
    rolled.raise_for_status()
    after_actions, after = predict(args.base_version, 'base-after-rollback')
    same = bool(np.array_equal(base_actions, after_actions))
    if not same:
        raise ValueError('Rollback did not restore deterministic base action')
    active = session.get(args.url + '/modelops/active', timeout=10)
    active.raise_for_status()
    if active.json()['version'] != args.base_version:
        raise ValueError('Registry active revision not rolled back')
    record = {
        'scope': 'simulation_only', 'input_path': str(INPUT),
        'input_sha256': hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        'health': health.json(), 'initial_ready': ready.json(), 'base_before': base,
        'candidate': candidate, 'base_after_rollback': after,
        'candidate_vs_base_action_linf': float(np.abs(candidate_actions - base_actions).max()),
        'rollback_base_actions_exact': same,
        'real_robot_scope_http': invalid.status_code,
        'stale_version_http': stale.status_code,
        'switch': switched.json(), 'rollback': rolled.json(),
        'active_after_rollback': active.json(), 'requests_successful': 3,
        'request_failures_expected': 2,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: record[key] for key in (
        'candidate_vs_base_action_linf', 'rollback_base_actions_exact',
        'real_robot_scope_http', 'stale_version_http', 'requests_successful')}, indent=2))


if __name__ == '__main__':
    main()
