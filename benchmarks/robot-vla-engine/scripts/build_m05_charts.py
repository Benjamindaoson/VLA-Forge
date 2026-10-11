"""Render evidence-linked M05 charts from actual training and GPU telemetry."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from robot_vla.m05_training import EVIDENCE, atomic_json

OUT = EVIDENCE / 'demo/charts'


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def style() -> None:
    plt.rcParams.update({'font.size': 13, 'figure.facecolor': '#f8fafb',
                         'axes.facecolor': '#ffffff', 'axes.edgecolor': '#7d909a',
                         'axes.labelcolor': '#15232c', 'text.color': '#15232c',
                         'xtick.color': '#33454f', 'ytick.color': '#33454f'})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    training = json.loads((EVIDENCE / 'training/training_summary.json').read_text())
    inputs: list[Path] = [EVIDENCE / 'training/training_summary.json']
    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    colors = {'standard': '#1265a8', 'corrective': '#c25b30', 'replay': '#208060'}
    for name, run in training['runs'].items():
        path = Path(run['raw_step_log'])
        inputs.append(path)
        data = rows(path)
        steps = [int(x['step']) for x in data]
        losses = np.array([float(x['loss']) for x in data])
        if len(steps) != 500:
            raise ValueError(f'{name} is not a 500-update training curve')
        ax.plot(steps, losses, lw=0.7, alpha=0.18, color=colors[name])
        smoothed = np.convolve(losses, np.ones(25) / 25, mode='valid')
        ax.plot(steps[24:], smoothed, lw=2.4, label=name, color=colors[name])
    ax.set(title='Real SmolVLA LoRA training loss (25-step moving average)',
           xlabel='Optimizer update', ylabel='Official Flow Matching loss')
    ax.grid(alpha=0.18)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'training_loss.png')
    plt.close(fig)

    matrix_path = EVIDENCE / 'inference/matrix.json'
    frozen_path = EVIDENCE / 'inference_frozen/summary.json'
    inputs.extend([matrix_path, frozen_path])
    matrix = json.loads(matrix_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    labels = ['Frozen Eager', 'Frozen Graph', 'Corrective Eager', 'Corrective Graph']
    values = [frozen['eager_chunk_latency']['p50_s'] * 1000,
              frozen['cuda_graph_chunk_latency']['p50_s'] * 1000,
              matrix['modes']['eager']['chunk_latency']['p50_s'] * 1000,
              matrix['modes']['cuda_graph']['chunk_latency']['p50_s'] * 1000]
    bars = ax.bar(labels, values, color=['#8d9fa8', '#197a69', '#8d9fa8', '#197a69'])
    ax.bar_label(bars, fmt='%.1f ms', padding=4)
    ax.set(title='100 measured action chunks per backend after 100 warm calls',
           ylabel='P50 action chunk latency (ms)', ylim=(0, max(values) * 1.2))
    ax.tick_params(axis='x', labelrotation=15)
    ax.grid(axis='y', alpha=0.18)
    fig.tight_layout()
    fig.savefig(OUT / 'inference_latency.png')
    plt.close(fig)

    path = Path(training['runs']['standard']['raw_resource_log'])
    inputs.append(path)
    resource = [r for r in rows(path) if r.get('phase') == 'training']
    if len(resource) < 5:
        raise ValueError('Insufficient GPU resource samples')
    t0 = float(resource[0]['monotonic_s'])
    t = [float(r['monotonic_s']) - t0 for r in resource]
    util = [float(r['gpu_util']) for r in resource]
    gpu_mem = [float(r['gpu_memory']) / 2**30 for r in resource]
    fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
    ax.plot(t, util, label='GPU utilization', color='#1265a8', lw=1.8)
    ax.set(xlabel='Elapsed train-phase seconds', ylabel='GPU utilization (%)',
           title='RTX 3090 sampled telemetry — standard 500-update run', ylim=(0, 105))
    ax2 = ax.twinx()
    ax2.plot(t, gpu_mem, label='GPU memory', color='#c25b30', lw=1.8)
    ax2.set(ylabel='GPU memory (GiB)')
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(OUT / 'gpu_telemetry.png')
    plt.close(fig)
    manifest = [{'path': str(path), 'sha256': sha(path)} for path in inputs]
    atomic_json(OUT / 'source_manifest.json', manifest)
    print(json.dumps({'charts': [p.name for p in OUT.glob('*.png')],
                      'source_count': len(manifest)}, indent=2))


if __name__ == '__main__':
    main()
