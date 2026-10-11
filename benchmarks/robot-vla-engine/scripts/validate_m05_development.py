"""Independently audit all M05 development videos, action traces, and states."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path('/data/VLA-Forge/Evidence/M05/development')
PROTOCOL = Path('configs/m05/formal_protocol.json')
CONDITIONS = ('frozen', 'standard', 'corrective', 'replay')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def check_receipt(path: Path, receipt: dict) -> dict:
    if receipt['status'] != 'complete' or type(receipt['success']) is not bool:
        raise ValueError('Episode has no terminal result')
    artifacts = {}
    for role in ('video', 'trajectory', 'action_trace'):
        item = receipt[role]
        source = Path(item['path'])
        if not source.is_file() or sha(source) != item['sha256']:
            raise ValueError(f'{role} source/hash mismatch')
        artifacts[role] = source
    steps = int(receipt['steps'])
    trace = json.loads(artifacts['action_trace'].read_text())
    if len(trace) != steps:
        raise ValueError('Action trace length mismatch')
    executed = np.asarray([row['executed'] for row in trace], dtype=np.float32)
    if executed.shape != (steps, 7) or not np.isfinite(executed).all() or np.abs(executed).max() > 1.00001:
        raise ValueError('Nonfinite or invalid executed 7D command')
    with np.load(artifacts['trajectory'], allow_pickle=False) as data:
        states = data['states']
        saved = data['actions']
        if states.shape != (steps + 1, 8) or not np.isfinite(states).all():
            raise ValueError('Nonfinite or misaligned canonical 8D states')
        if saved.shape != (steps, 7) or not np.array_equal(saved, executed):
            raise ValueError('Trajectory actions disagree with executed commands')
        for name in ('latency_preprocess_ms', 'latency_forward_ms', 'latency_postprocess_ms',
                     'latency_policy_ms', 'latency_environment_ms', 'latency_control_cycle_ms'):
            array = data[name]
            if array.shape != (steps,) or not np.isfinite(array).all() or (array < 0).any():
                raise ValueError(f'Latency evidence malformed: {name}')
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames',
                            '-of', 'json', str(artifacts['video'])],
                           capture_output=True, text=True, check=True)
    video = json.loads(probe.stdout)['streams'][0]
    if (video['codec_name'], video['width'], video['height'], video['r_frame_rate'], int(video['nb_frames'])) != ('h264', 512, 256, '20/1', steps):
        raise ValueError('Video codec/shape/fps/frame alignment mismatch')
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(artifacts['video']), '-f', 'null', '-'],
                             capture_output=True, text=True)
    if decoded.returncode:
        raise ValueError(f'Video full decode failed: {decoded.stderr[-200:]}')
    return {'receipt': str(path), 'steps': steps, 'success': receipt['success'],
            'artifact_hashes': {key: receipt[key]['sha256'] for key in artifacts},
            'decoded_frames': int(video['nb_frames'])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    args = parser.parse_args()
    if len(set(args.conditions)) != len(args.conditions):
        raise ValueError("Repeated conditions")
    protocol = json.loads(PROTOCOL.read_text())
    cases = {(case['suite'], int(case['task']), int(case['init'])) for case in protocol['development']}
    expected = {(condition, *case) for condition in args.conditions for case in cases}
    if len(expected) != 8 * len(args.conditions):
        raise ValueError("Frozen development protocol has duplicate episodes")
    seen, checks, errors = set(), {}, []
    for path in sorted((args.root / 'receipts').glob('*.json')):
        try:
            receipt = json.loads(path.read_text())
            key = (receipt['condition'], receipt['suite'], int(receipt['task']), int(receipt['init']))
            if key not in expected or key in seen:
                raise ValueError(f'Unexpected/duplicate case {key}')
            seen.add(key)
            checks[path.stem] = check_receipt(path, receipt)
        except Exception as exc:
            errors.append({'receipt': str(path), 'error': repr(exc)})
    missing = sorted(expected - seen)
    status = {'passed': not errors and not missing and len(checks) == len(expected),
              'scope': f'M05 {len(expected)} development episodes only; M06 not opened',
              'valid_count': len(checks), 'expected_count': len(expected),
              'conditions': list(args.conditions),
              'success': sum(row['success'] for row in checks.values()),
              'decoded_frames': sum(row['decoded_frames'] for row in checks.values()),
              'missing': missing, 'errors': errors, 'episodes': checks}
    (args.root / 'validation.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps({key: status[key] for key in ('passed', 'valid_count', 'expected_count',
                       'success', 'decoded_frames', 'missing', 'errors')}, indent=2))
    if not status['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
