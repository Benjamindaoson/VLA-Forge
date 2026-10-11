"""Record candidate optimization decisions without turning pilot timings into claims."""
from __future__ import annotations

import csv
import json

from robot_vla.m05_training import EVIDENCE, atomic_json

ROOT = EVIDENCE / 'training_perf'
OUT = EVIDENCE / 'summary'
DECISIONS = {
    'fp32_unprofiled': ('REFERENCE', 'Unprofiled FP32 reference; same adapter as profiled FP32.'),
    'bf16': ('REJECT_FORMAL', 'Finite eight-step pilot, but different adapter and no repeated speed evidence.'),
    'fp16': ('REJECT', 'Failed finite-gradient gate at update 2; no valid checkpoint.'),
    'tf32': ('REJECT_FORMAL', 'Different adapter and no repeated speed/correctness advantage.'),
    'fused': ('REJECT_FORMAL', 'Different adapter; no verified numerical parity benefit.'),
    'checkpoint': ('ACCEPT_MEMORY_ONLY', 'Exact adapter in eight-step pilot with lower peak allocation; no speed claim.'),
    'accumulation': ('REJECT_FORMAL', 'Changes optimization granularity; not identical update-matched comparison.'),
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ref = json.loads((ROOT / 'fp32_unprofiled/result.json').read_text())
    rows = []
    for name, (decision, rationale) in DECISIONS.items():
        file = ROOT / name / 'result.json'
        if file.is_file():
            d = json.loads(file.read_text())
            rows.append({
                'candidate': name, 'decision': decision, 'status': d['status'],
                'optimizer_updates': d['optimizer_updates'], 'process_wall_s': d['wall_s'],
                'setup_adjusted_wall_s': d['wall_s'] - d['model_load_s'] - d['checkpoint_save_s'] - d['dataset_load_s'],
                'peak_allocated_bytes': d['allocated_peak'],
                'same_adapter_sha_as_fp32': d['adapter_sha256'] == ref['adapter_sha256'],
                'finite_gradients': d['finite_gradients'], 'rationale': rationale,
            })
        else:
            rows.append({'candidate': name, 'decision': decision, 'status': 'FAILED_NO_RESULT',
                         'optimizer_updates': 1 if name == 'fp16' else None,
                         'process_wall_s': None, 'setup_adjusted_wall_s': None,
                         'peak_allocated_bytes': None, 'same_adapter_sha_as_fp32': None,
                         'finite_gradients': False if name == 'fp16' else None,
                         'rationale': rationale})
    with (OUT / 'optimization_matrix.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data = json.loads((ROOT / 'data_pipeline/results.json').read_text())
    pipeline = []
    for row in data['rows']:
        pipeline.append({k: row.get(k) for k in ('backend', 'workers', 'samples', 'wall_s', 'load_s', 'rss', 'error')})
    atomic_json(OUT / 'data_pipeline_matrix.json', {
        'ordering': data['ordering'], 'rows': pipeline,
        'decision': 'Retain PyAV; two-worker result is descriptive only; TorchCodec unavailable due binary compatibility.',
    })
    print(json.dumps({'optimization_rows': len(rows), 'pipeline_rows': len(pipeline)}, indent=2))


if __name__ == '__main__':
    main()
