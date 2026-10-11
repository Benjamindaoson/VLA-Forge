"""Derive reproducible training performance and energy evidence from step logs."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

ROOT = Path('/data/VLA-Forge/Evidence/M05')
OUTPUT = ROOT / 'training/training_summary.json'
RUNS = {
    'standard': ROOT / 'training/standard',
    'corrective': ROOT / 'modelops/jobs/m05-corrective-cam180-s5501/artifacts',
    'replay': ROOT / 'training/replay_cam180',
}
FIELDS = ('decode_s', 'processor_h2d_s', 'forward_s', 'backward_s', 'optimizer_s', 'update_wall_s')


def percentiles(values: list[float]) -> dict:
    return {f'p{n}_s': float(np.percentile(values, n)) for n in (50, 95, 99)}


def one_run(name: str, root: Path) -> dict:
    result = json.loads((root / 'result.json').read_text())
    steps = [json.loads(line) for line in (root / 'steps.jsonl').read_text().splitlines()]
    samples = [json.loads(line) for line in (root / 'resources.jsonl').read_text().splitlines()]
    if len(steps) != 500 or result['optimizer_updates'] != 500:
        raise ValueError(f'{name} missing formal 500-update run')
    if [row['step'] for row in steps] != list(range(1, 501)):
        raise ValueError(f'{name} missing or repeated optimizer updates')
    if not result['finite_gradients'] or result['reload_linf'] != 0:
        raise ValueError(f'{name} failed training/readback gate')
    process_wall = float(result['wall_s'])
    if name == 'corrective':
        pause = json.loads((root.parent / 'pause_receipt.json').read_text())
        first = pause['state']['result']
        if first['optimizer_updates'] != 250:
            raise ValueError('Corrective managed pause was not at step 250')
        process_wall += float(first['wall_s'])
    energy_wh = 0.0
    sampled_s = 0.0
    for first, second in zip(samples, samples[1:]):
        dt = second['utc_s'] - first['utc_s']
        if 0 < dt <= 2.5 and first.get('gpu_power_w') is not None:
            energy_wh += float(first['gpu_power_w']) * dt / 3600
            sampled_s += dt
    step_samples = sum(len(row['indices']) for row in steps)
    return {
        'run_id': result['run_id'], 'adapter_sha256': result['adapter_sha256'],
        'optimizer_updates': 500, 'sample_exposures': step_samples,
        'process_wall_s': process_wall, 'process_gpu_hours_upper_bound': process_wall / 3600,
        'updates_per_process_wall_s': 500 / process_wall,
        'loss_first': steps[0]['loss'], 'loss_last': steps[-1]['loss'],
        'loss_mean': statistics.mean(row['loss'] for row in steps),
        'gradient_norm_p95': float(np.percentile([row['gradient_norm'] for row in steps], 95)),
        'peak_allocated_bytes': result['allocated_peak'],
        'peak_reserved_bytes': result['reserved_peak'],
        'max_rss_bytes': max((row.get('rss') or 0 for row in samples), default=0),
        'gpu_util_mean_observed_percent': statistics.mean(row['gpu_util'] for row in samples if row.get('gpu_util') is not None) if samples else None,
        'gpu_power_mean_observed_w': statistics.mean(row['gpu_power_w'] for row in samples if row.get('gpu_power_w') is not None) if samples else None,
        'sampled_gpu_energy_Wh': energy_wh,
        'sampled_power_duration_s': sampled_s,
        'phase_latency': {field: percentiles([float(row[field]) for row in steps]) for field in FIELDS},
        'raw_step_log': str(root / 'steps.jsonl'), 'raw_resource_log': str(root / 'resources.jsonl'),
        'readback_linf': result['reload_linf'],
    }


def main() -> None:
    runs = {name: one_run(name, path) for name, path in RUNS.items()}
    ab = json.loads((ROOT / 'modelops/ab_comparison.json').read_text())
    if not ab['passed'] or not ab['adapter_sha256_exact'] or not ab['sample_indices_exact']:
        raise ValueError('Native/managed A/B correctness gate failed')
    record = {
        'scope': 'M05 formal training and short A/B; wall time is process occupancy, not a cloud rental invoice',
        'hardware': 'RTX3090 24GB', 'runs': runs,
        'native_vs_managed_short_ab': {
            'matched_updates': 20, 'adapter_sha256_exact': ab['adapter_sha256_exact'],
            'sample_indices_exact': ab['sample_indices_exact'],
            'wall_overhead_s': ab['wall_overhead_s'],
            'wall_overhead_fraction': ab['wall_overhead_fraction'],
            'evidence': str(ROOT / 'modelops/ab_comparison.json'),
        },
        'formal_process_gpu_hours_upper_bound': sum(r['process_gpu_hours_upper_bound'] for r in runs.values()),
        'billable_cloud_cost': None,
        'billable_cloud_cost_reason': 'No provider invoice or authorized hourly rate was supplied; no currency amount is invented.',
        'energy_method': 'Trapezoid-free left-hold integration of sampled GPU power; gaps above 2.5s excluded; not a cloud invoice.',
    }
    OUTPUT.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'runs': {name: {'updates': row['optimizer_updates'],
                      'wall_s': row['process_wall_s'], 'updates_per_s': row['updates_per_process_wall_s']}
                     for name, row in runs.items()},
                      'formal_process_gpu_hours_upper_bound': record['formal_process_gpu_hours_upper_bound']}, indent=2))


if __name__ == '__main__':
    main()
