"""Pair actual Eager and CUDA Graph MuJoCo development rollouts by frozen init state."""
from __future__ import annotations

import csv
import json

import numpy as np

from robot_vla.m05_training import EVIDENCE, atomic_json


def load_receipts(folder: str) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in (EVIDENCE / folder / 'receipts').glob('*.json')}


def main() -> None:
    eager = load_receipts('development')
    graph = load_receipts('development_graph')
    if len(graph) != 16 or any(not key.startswith(('frozen_', 'corrective_')) for key in graph):
        raise ValueError('Require 16 frozen/corrective graph episodes')
    rows = []
    for key, g in sorted(graph.items()):
        b = eager[key]
        if any(b[k] != g[k] for k in ('suite', 'task', 'init', 'seed', 'checkpoint_sha256')):
            raise ValueError(f'Protocol mismatch for {key}')
        with np.load(b['trajectory']['path']) as x, np.load(g['trajectory']['path']) as y:
            ba, ga = x['actions'], y['actions']
            bs, gs = x['states'], y['states']
        if ba.shape[1:] != (7,) or ga.shape[1:] != (7,) or not np.isfinite(ba).all() or not np.isfinite(ga).all():
            raise ValueError(f'Invalid action trajectory for {key}')
        init_linf = float(np.max(np.abs(bs[0] - gs[0])))
        n = min(len(ba), len(ga))
        if n < 50 or init_linf > 1e-6:
            raise ValueError(f'Initial state mismatch or too-short trajectory: {key}, {init_linf}')
        first_linf = float(np.max(np.abs(ba[:50] - ga[:50])))
        row = {
            'case': key, 'suite': b['suite'], 'task': b['task'], 'init': b['init'],
            'condition': b['condition'], 'seed': b['seed'],
            'initial_robot_state_linf': init_linf, 'first_50_actions_linf': first_linf,
            'aligned_action_mae': float(np.mean(np.abs(ba[:n] - ga[:n]))),
            'aligned_robot_state_mae': float(np.mean(np.abs(bs[:n + 1] - gs[:n + 1]))),
            'baseline_success': b['success'], 'graph_success': g['success'],
            'baseline_steps': b['steps'], 'graph_steps': g['steps'],
            'baseline_episode_wall_s': b['wall_s'], 'graph_episode_wall_s': g['wall_s'],
            'baseline_chunk_mean_ms': b['chunk_mean_ms'], 'graph_chunk_mean_ms': g['chunk_mean_ms'],
            'baseline_video': b['video']['path'], 'graph_video': g['video']['path'],
            'baseline_trajectory_sha256': b['trajectory']['sha256'],
            'graph_trajectory_sha256': g['trajectory']['sha256'],
        }
        rows.append(row)
    out = EVIDENCE / 'summary'
    out.mkdir(exist_ok=True)
    with (out / 'graph_closed_loop_pairs.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    result = {
        'scope': 'same M05 fixed development cases, separate actual MuJoCo executions',
        'pairs': len(rows), 'all_initial_robot_states_exact': all(r['initial_robot_state_linf'] == 0 for r in rows),
        'first_50_actions_exact_pairs': sum(r['first_50_actions_linf'] == 0 for r in rows),
        'conditions': {}, 'individual_pairs': str(out / 'graph_closed_loop_pairs.csv'),
        'limitation': 'Eight-dimensional logged robot state does not prove full simulator microstate equality; '
                      'contact dynamics and reset variations can diverge after first chunk.',
    }
    for condition in ('frozen', 'corrective'):
        subset = [r for r in rows if r['condition'] == condition]
        result['conditions'][condition] = {
            'pairs': len(subset), 'baseline_success': sum(r['baseline_success'] for r in subset),
            'graph_success': sum(r['graph_success'] for r in subset),
            'baseline_episode_wall_s_sum': sum(r['baseline_episode_wall_s'] for r in subset),
            'graph_episode_wall_s_sum': sum(r['graph_episode_wall_s'] for r in subset),
            'graph_vs_baseline_wall_speedup': sum(r['baseline_episode_wall_s'] for r in subset) /
                                             sum(r['graph_episode_wall_s'] for r in subset),
            'mean_aligned_action_mae': float(np.mean([r['aligned_action_mae'] for r in subset])),
            'mean_aligned_robot_state_mae': float(np.mean([r['aligned_robot_state_mae'] for r in subset])),
        }
    atomic_json(out / 'graph_closed_loop_summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'individual_pairs'}, indent=2))
    if result['first_50_actions_exact_pairs'] != 16:
        raise SystemExit('Graph first-chunk actions differ from Eager')


if __name__ == '__main__':
    main()
