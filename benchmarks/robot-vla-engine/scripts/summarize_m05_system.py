"""Compile M05 measured training, development, inference and cost receipts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from robot_vla.m05_training import EVIDENCE, atomic_json

OUT = EVIDENCE / 'summary'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'No rows for {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    training = json.loads((EVIDENCE / 'training/training_summary.json').read_text())
    dev = json.loads((EVIDENCE / 'development/metrics.json').read_text())
    infer = json.loads((EVIDENCE / 'inference/matrix.json').read_text())
    frozen = json.loads((EVIDENCE / 'inference_frozen/summary.json').read_text())
    ab = json.loads((EVIDENCE / 'modelops/ab_comparison.json').read_text())
    train_rows = []
    for condition, r in training['runs'].items():
        train_rows.append({
            'condition': condition, 'run_id': r['run_id'], 'updates': r['optimizer_updates'],
            'sample_exposures': r['sample_exposures'], 'process_wall_s': r['process_wall_s'],
            'updates_per_s': r['updates_per_process_wall_s'], 'loss_first': r['loss_first'],
            'loss_last': r['loss_last'], 'peak_allocated_bytes': r['peak_allocated_bytes'],
            'gpu_util_observed_percent': r['gpu_util_mean_observed_percent'],
            'energy_sampled_Wh': r['sampled_gpu_energy_Wh'], 'adapter_sha256': r['adapter_sha256'],
        })
    write_csv(OUT / 'training_comparison.csv', train_rows)
    infer_rows = []
    for mode, r in sorted(infer['modes'].items()):
        lat = r.get('chunk_latency', {})
        infer_rows.append({
            'model': 'corrective', 'mode': mode, 'status': r['status'],
            'warm_calls': r.get('warm_calls', 0), 'measured_calls': r.get('calls', 0),
            'chunk_p50_s': lat.get('p50_s'), 'chunk_p95_s': lat.get('p95_s'),
            'chunk_p99_s': lat.get('p99_s'), 'predict_p50_s': r.get('end_to_end_latency', {}).get('p50_s'),
            'action_linf_vs_eager': r.get('action_linf_vs_eager'),
            'correctness_gate': r.get('correctness_gate'),
            'gpu_peak_allocated_bytes': r.get('gpu_peak_allocated'),
            'error': r.get('error', ''),
        })
    for mode in ('eager', 'cuda_graph'):
        r = json.loads((EVIDENCE / 'inference_frozen' / f'{mode}.json').read_text())
        infer_rows.append({
            'model': 'frozen', 'mode': mode, 'status': 'completed',
            'warm_calls': r['warm_calls'], 'measured_calls': r['calls'],
            'chunk_p50_s': r['chunk_latency']['p50_s'],
            'chunk_p95_s': r['chunk_latency']['p95_s'],
            'chunk_p99_s': r['chunk_latency']['p99_s'],
            'predict_p50_s': r['end_to_end_latency']['p50_s'],
            'action_linf_vs_eager': 0.0 if mode == 'eager' else frozen['action_raw_linf'],
            'correctness_gate': True, 'gpu_peak_allocated_bytes': r['gpu_peak_allocated'], 'error': '',
        })
    write_csv(OUT / 'inference_comparison.csv', infer_rows)
    development = []
    for condition, r in sorted(dev['conditions'].items()):
        development.append({'condition': condition, 'success': r['success'],
                            'total': r['total'], 'errors': r['errors'],
                            'success_rate': r['success'] / r['total']})
    write_csv(OUT / 'development_comparison.csv', development)
    service_rows = []
    for label in ('m05-base-frozen', 'm05-standard-s5501', 'm05-base-frozen_cuda_graph'):
        record = json.loads((EVIDENCE / 'serving' / f'{label}_summary.json').read_text())
        for concurrency, group in record['groups'].items():
            service_rows.append({
                'version': record['version'], 'backend': record.get('backend', 'eager'),
                'concurrency': int(concurrency), 'requested': group['requested'],
                'successful': group['successful'], 'errors': group['errors'],
                'throughput_requests_s': group['successful_requests_per_s'],
                'request_p50_s': group['request_latency']['p50_s'],
                'request_p95_s': group['request_latency']['p95_s'],
                'request_p99_s': group['request_latency']['p99_s'],
                'queue_p95_s': group['queue_latency']['p95_s'],
                'gpu_peak_allocated_bytes': group['peak_gpu_allocated_bytes'],
            })
    write_csv(OUT / 'serving_comparison.csv', service_rows)
    receipts = [json.loads(p.read_text()) for p in (EVIDENCE / 'development/receipts').glob('*.json')]
    if len(receipts) != 32 or any(r['status'] != 'complete' for r in receipts):
        raise ValueError('Development coverage must be 32 complete episodes')
    train_wall = sum(r['process_wall_s'] for r in training['runs'].values())
    development_wall = sum(r['wall_s'] for r in receipts)
    graph_receipts = [json.loads(p.read_text()) for p in (EVIDENCE / 'development_graph/receipts').glob('*.json')]
    if len(graph_receipts) != 16 or any(r['status'] != 'complete' for r in graph_receipts):
        raise ValueError('Graph development coverage must be 16 complete episodes')
    graph_wall = sum(r['wall_s'] for r in graph_receipts)
    cost = {
        'hardware': 'RTX3090 24GB', 'currency_cloud_invoice': None,
        'currency_rental_rate': None,
        'why_currency_unavailable': 'No provider invoice or per-hour rate in authorized evidence.',
        'formal_training_process_wall_s': train_wall,
        'formal_training_gpu_hours_upper_bound': train_wall / 3600,
        'development_32_episode_wall_s': development_wall,
        'development_32_episode_gpu_hours_upper_bound': development_wall / 3600,
        'graph_16_episode_wall_s': graph_wall,
        'graph_16_episode_gpu_hours_upper_bound': graph_wall / 3600,
        'training_plus_all_development_process_gpu_hours_upper_bound': (train_wall + development_wall + graph_wall) / 3600,
        'sampled_training_gpu_energy_Wh': sum(r['sampled_gpu_energy_Wh'] for r in training['runs'].values()),
        'modelops_short_run_wall_overhead_s': ab['wall_overhead_s'],
        'modelops_short_run_wall_overhead_fraction': ab['wall_overhead_fraction'],
        'scope': 'Process wall time is an upper bound on occupied GPU-instance time for these phases; '
                 'not an invoice, total project cost, or GPU-compute utilization integral.',
        'excluded': ['M04 source collection', 'failed exploratory camera runs', 'idle rental',
                     'additional inference profiles', 'demo rendering', 'cloud storage/egress'],
    }
    atomic_json(OUT / 'cost_report.json', cost)
    logs = [EVIDENCE / 'training/training_summary.json', EVIDENCE / 'development/validation.json',
            EVIDENCE / 'inference/matrix.json', EVIDENCE / 'inference_frozen/summary.json',
            EVIDENCE / 'modelops/ab_comparison.json']
    for run in training['runs'].values():
        logs.extend([Path(run['raw_step_log']), Path(run['raw_resource_log'])])
    atomic_json(OUT / 'raw_evidence_index.json', [
        {'path': str(p), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in logs if p.is_file()
    ])
    print(json.dumps({'training_runs': len(train_rows), 'development_episodes': len(receipts),
                      'inference_modes': len(infer_rows), 'serving_cells': len(service_rows), 'training_gpu_hours_upper_bound': train_wall / 3600,
                      'development_gpu_hours_upper_bound': development_wall / 3600,
                      'graph_gpu_hours_upper_bound': graph_wall / 3600}, indent=2))


if __name__ == '__main__':
    main()
