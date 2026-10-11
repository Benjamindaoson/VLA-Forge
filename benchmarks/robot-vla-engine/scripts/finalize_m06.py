"""Validate all 1,600 locked rollouts before any formal analysis or gate."""
from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import av
import numpy as np

from robot_vla.m06_protocol import CONDITIONS, EVIDENCE, SUITES, atomic_json, sha256
from robot_vla.m06_statistics import (
    holm_adjust,
    paired_target,
    release_decision,
    task_cluster_difference,
    wilson,
)

OUT = EVIDENCE / 'formal'

def pair_key(item):
    return (item['suite'], int(item['task_id']), int(item['init_state_id']))

def artifact(path_info: dict) -> Path:
    path = Path(path_info['path'])
    if not path.is_file() or sha256(path) != path_info['sha256']:
        raise ValueError(f'Artifact missing or hash mismatch: {path}')
    return path

def load_complete(protocol: dict) -> dict:
    paths = sorted((OUT / 'receipts').glob('*.json'))
    if len(paths) != 1600:
        raise ValueError(f'M06 is INCOMPLETE: {len(paths)}/1600 receipt files')
    rows = [json.loads(path.read_text()) for path in paths]
    if any(row.get('status') != 'complete' for row in rows):
        raise ValueError('M06 has incomplete/error receipts')
    keys = {(row['condition'], *pair_key(row)) for row in rows}
    expected = {(condition, *pair_key(case)) for condition in CONDITIONS for case in protocol['cases']}
    if len(keys) != 1600 or keys != expected:
        raise ValueError('M06 locked case/condition cross product differs')
    return {condition: {pair_key(row): row for row in rows if row['condition'] == condition}
            for condition in CONDITIONS}

def validate_outputs(protocol: dict, grouped: dict) -> tuple[dict, dict, dict, dict]:
    model_hashes = {name: ident.get('adapter_sha256', ident.get('model_sha256'))
                    for name, ident in protocol['model_identities'].items() if name in CONDITIONS}
    frames_total = 0
    steps_total = 0
    clipped = Counter()
    resource = defaultdict(list)
    chunk_latencies = defaultdict(list)
    actions_by_case = {}
    errors = []
    for condition in CONDITIONS:
        for case in protocol['cases']:
            key = pair_key(case)
            row = grouped[condition][key]
            if (row.get('checkpoint_sha256') != model_hashes[condition]
                    or row.get('protocol_sha256') != sha256(EVIDENCE / 'protocol_frozen.json')
                    or row.get('source_group') != case['source_group']
                    or row.get('seed') != case['seed']
                    or type(row.get('success')) is not bool
                    or row.get('floating_parameter_dtypes') != ['torch.float32']
                    or row.get('tf32_allowed') is not False):
                raise ValueError(f'Identity drift: {condition}/{key}')
            match = row.get('initial_match', {})
            if not match.get('matched') or not match.get('reference_sha256'):
                raise ValueError(f'Physical start unverified: {condition}/{key}')
            ref = Path(match['reference_path'])
            if not ref.is_file() or sha256(ref) != match['reference_sha256']:
                raise ValueError(f'Physical reference changed: {condition}/{key}')
            for field, maximum in (('sim', 1e-6), ('qpos', 1e-6), ('qvel', 1e-6), ('time', 1e-9)):
                if match['max_abs_difference'][field] > maximum:
                    raise ValueError(f'Physical start exceeds tolerance: {condition}/{key}')
            video = artifact(row['video'])
            trajectory = artifact(row['trajectory'])
            action_trace = artifact(row['action_trace'])
            physics = artifact(row['sim_states'])
            with np.load(trajectory, allow_pickle=False) as data:
                actions = data['actions']
                states = data['states']
                if (actions.shape != (row['steps'], 7) or states.shape != (row['steps'] + 1, 8)
                        or not np.isfinite(actions).all() or not np.isfinite(states).all()):
                    raise ValueError(f'Action/state shape or finite check failed: {condition}/{key}')
                if np.any(np.abs(actions) > 1 + 1e-6):
                    raise ValueError(f'Illegal executed action: {condition}/{key}')
                actions_by_case[(condition, *key)] = actions.copy()
            with np.load(physics, allow_pickle=False) as data:
                for field in ('sim', 'qpos', 'qvel', 'time'):
                    if len(data[field]) != row['steps'] + 1 or not np.isfinite(data[field]).all():
                        raise ValueError(f'Physical trace invalid: {condition}/{key}/{field}')
            trace = json.loads(action_trace.read_text())
            logged = np.asarray([a['executed'] for a in trace['actions']], dtype=np.float32)
            if not np.array_equal(actions, logged):
                raise ValueError(f'Actual command trace mismatch: {condition}/{key}')
            clipped[condition] += sum(a['command'] != a['executed'] for a in trace['actions'])
            if trace['input_shapes'] != {'observation.images.image': [1, 3, 256, 256],
                                        'observation.images.image2': [1, 3, 256, 256],
                                        'observation.state': [1, 8]}:
                raise ValueError(f'Model input shape drift: {condition}/{key}')
            with av.open(str(video)) as media:
                if (media.streams.video[0].codec_context.name != 'h264'
                        or media.streams.video[0].width != 512 or media.streams.video[0].height != 256):
                    raise ValueError(f'Video codec/shape invalid: {condition}/{key}')
                frames = sum(1 for _ in media.decode(video=0))
            if frames != row['steps']:
                raise ValueError(f'Video frame count mismatch: {condition}/{key}')
            frames_total += frames
            steps_total += row['steps']
            rollout_file = trajectory.parent / 'rollout.json'
            if not rollout_file.is_file():
                raise ValueError(f'Missing rollout metadata: {condition}/{key}')
            rollout = json.loads(rollout_file.read_text())
            calls = [float(v['latency_ms']) for v in rollout['chunk_calls']]
            if len(calls) != row['chunk_calls'] or not calls:
                raise ValueError(f'Chunk profile mismatch: {condition}/{key}')
            chunk_latencies[condition].extend(calls)
            resource[condition].append({'wall_s': float(row['wall_s']), 'steps': row['steps'],
                                        'gpu_peak_allocated': row['gpu_peak_allocated'],
                                        'gpu_peak_reserved': row['gpu_peak_reserved'],
                                        'latency': row['latency'], 'rollout_wall_s': float(rollout['wall_seconds']),
                                        'registry_run_id': row['registry_run_id']})
    attempts = OUT / 'attempts.jsonl'
    if attempts.exists():
        errors = [json.loads(line) for line in attempts.read_text().splitlines()
                  if json.loads(line).get('event') == 'error']
    validation = {'status': 'PASS', 'expected': 1600, 'valid': 1600,
                  'physical_starts_matched': 1600, 'decoded_video_frames': frames_total,
                  'executed_actions': steps_total, 'failed_attempts_preserved': len(errors),
                  'all_artifact_hashes_valid': True, 'all_checkpoint_hashes_valid': True,
                  'floating_parameter_dtypes': ['torch.float32'], 'tf32_allowed': False,
                  'source_cases': 400, 'illegal_executed_actions': 0,
                  'clipped_raw_commands': dict(clipped),
                  'license': 'Private visual evidence; no public redistribution clearance asserted'}
    return validation, dict(chunk_latencies), dict(resource), actions_by_case

def task_suite_tables(protocol: dict, grouped: dict):
    task_rows = []
    suites = {}
    for condition in CONDITIONS:
        suites[condition] = {}
        for suite in SUITES:
            suite_results = []
            for task_id in range(10):
                values = [grouped[condition][(suite, task_id, init)]['success'] for init in range(40, 50)]
                stats = wilson(sum(values), 10)
                task_rows.append({'condition': condition, 'suite': suite, 'task_id': task_id,
                                  'task_name': grouped[condition][(suite, task_id, 40)]['task_name'], **stats})
                suite_results.extend(values)
            suites[condition][suite] = wilson(sum(suite_results), 100)
        all_results = [row['success'] for row in grouped[condition].values()]
        suites[condition]['overall'] = wilson(sum(all_results), 400)
    return task_rows, suites

def comparisons(protocol: dict, grouped: dict, chunks: dict, actions: dict):
    base = grouped['frozen']
    target_keys = [('goal', 6, init) for init in range(40, 50)]
    target = {}
    skill = {}
    pair_rows = []
    pvalues = {}
    for candidate in CONDITIONS[1:]:
        result = paired_target([base[key]['success'] for key in target_keys],
                               [grouped[candidate][key]['success'] for key in target_keys])
        target[candidate] = result
        pvalues[candidate] = result['one_sided_exact_p']
        per_task = []
        degradations = []
        new_failures = 0
        for suite in SUITES:
            for task_id in range(10):
                if (suite, task_id) == ('goal', 6):
                    continue
                keys = [(suite, task_id, init) for init in range(40, 50)]
                frozen_score = sum(base[key]['success'] for key in keys)
                candidate_score = sum(grouped[candidate][key]['success'] for key in keys)
                delta = (candidate_score - frozen_score) / 10
                per_task.append(delta)
                degradations.append({'suite': suite, 'task_id': task_id,
                                     'frozen_success': frozen_score, 'candidate_success': candidate_score,
                                     'delta': delta})
                new_failures += sum(base[key]['success'] and not grouped[candidate][key]['success'] for key in keys)
        cluster = task_cluster_difference(per_task)
        skill[candidate] = {**cluster, 'new_failures': new_failures,
                            'worst_tasks': sorted(degradations, key=lambda row: row['delta'])[:5],
                            'task_differences': degradations}
    adjusted = holm_adjust(pvalues)
    for candidate in CONDITIONS[1:]:
        target[candidate]['holm_adjusted_p'] = adjusted[candidate]
    for case in protocol['cases']:
        key = pair_key(case)
        row = {**case, 'frozen_success': base[key]['success']}
        for candidate in CONDITIONS[1:]:
            current = grouped[candidate][key]
            row[f'{candidate}_success'] = current['success']
            row[f'{candidate}_delta'] = int(current['success']) - int(base[key]['success'])
            a = actions[('frozen', *key)]
            b = actions[(candidate, *key)]
            shared = min(len(a), len(b))
            row[f'{candidate}_action_l2_mean_shared_prefix'] = float(np.linalg.norm(a[:shared] - b[:shared], axis=1).mean())
            row[f'{candidate}_shared_action_steps'] = shared
        pair_rows.append(row)
    decisions = {'frozen': {'decision': 'PROMOTE', 'reason': 'unchanged_incumbent_reference',
                             'deployment_scope': 'authorized simulation only'}}
    for candidate in CONDITIONS[1:]:
        decisions[candidate] = release_decision(target=target[candidate], legacy=skill[candidate],
            holm_p=adjusted[candidate], candidate_chunk_p95_ms=float(np.percentile(chunks[candidate], 95)),
            frozen_chunk_p95_ms=float(np.percentile(chunks['frozen'], 95)),
            illegal_actions=0, missing_or_corrupt=0)
    return pair_rows, target, skill, decisions

def resource_payloads(run_id: str) -> list[dict]:
    """Read exported telemetry, or the SQLite source after an interrupted run."""
    registry = OUT / 'registry'
    event_file = registry / 'runs' / run_id / 'events.jsonl'
    if event_file.is_file():
        values = [json.loads(line)['payload'] for line in event_file.read_text().splitlines()
                  if json.loads(line)['kind'] == 'resources']
    else:
        db_path = registry / 'registry.sqlite'
        if not db_path.is_file():
            raise ValueError(f'Missing telemetry source for {run_id}')
        with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as db:
            values = [json.loads(row[0]) for row in db.execute(
                'SELECT payload_json FROM events WHERE run_id=? AND kind=? ORDER BY id',
                (run_id, 'resources'))]
    if not values:
        raise ValueError(f'No resource telemetry for {run_id}')
    return values

def performance_cost(chunks: dict, resource: dict, validation: dict):
    perf = {}
    costs = {}
    for condition in CONDITIONS:
        episodes = resource[condition]
        walls = np.asarray([row['wall_s'] for row in episodes])
        latencies = np.asarray(chunks[condition])
        total_s = float(walls.sum())
        phases = {name: float(sum(row['latency'][name]['mean_ms'] * row['latency'][name]['n'] for row in episodes) / 1000)
                  for name in ('preprocess', 'forward', 'postprocess', 'policy', 'environment', 'control_cycle')}
        video_loop_upper = float(sum(max(0.0, row['rollout_wall_s'] -
            row['latency']['control_cycle']['mean_ms'] * row['latency']['control_cycle']['n'] / 1000)
            for row in episodes))
        samples = []
        for run_id in {row['registry_run_id'] for row in episodes}:
            samples.extend(resource_payloads(run_id))
        cpu = [float(item['process_cpu_percent']) for item in samples]
        rss = [int(item['process_rss_bytes']) for item in samples]
        gpu = []
        for item in samples:
            value = item.get('gpu', {})
            if value.get('returncode') == 0:
                try:
                    gpu.append([float(x.strip()) for x in value['stdout'].split(',')])
                except (ValueError, TypeError):
                    pass
        perf[condition] = {'episodes': len(episodes), 'steps': sum(row['steps'] for row in episodes),
            'chunk_calls': len(latencies), 'chunk_latency_ms': {f'p{q}': float(np.percentile(latencies, q)) for q in (50, 95, 99)},
            'episode_wall_s': {'sum': total_s, 'mean': float(walls.mean()), 'p95': float(np.percentile(walls, 95))},
            'phase_sums_s': phases, 'video_encode_plus_loop_upper_bound_s': video_loop_upper,
            'phase_boundary': 'video duration is residual outside control_cycle; exact encoder CPU cost is not separately timed',
            'gpu_peak_allocated_bytes': max(row['gpu_peak_allocated'] for row in episodes),
            'gpu_peak_reserved_bytes': max(row['gpu_peak_reserved'] for row in episodes),
            'resource_samples': len(samples), 'process_cpu_percent_mean': float(np.mean(cpu)) if cpu else None,
            'process_rss_peak_bytes': max(rss) if rss else None,
            'gpu_utilization_percent_mean': float(np.mean([v[0] for v in gpu])) if gpu else None,
            'gpu_memory_used_mb_peak': max(v[1] for v in gpu) if gpu else None,
            'gpu_power_w_mean': float(np.mean([v[3] for v in gpu])) if gpu else None,
            'model_load_s': None, 'model_load_note': 'Not timed separately in the frozen formal runner',
            'first_chunk_ms': float(latencies[0])}
        costs[condition] = {'episode_gpu_hours': total_s / 3600,
            'episode_gpu_hours_per_100_rollouts': total_s / 3600 / 4,
            'scope': 'sum of measured episode walls only; model load/setup/idle/error overhead separate'}
    invalid_path = EVIDENCE / 'technical_attempts/bf16_native_20261011/worker_status.json'
    invalid_worker = json.loads(invalid_path.read_text()) if invalid_path.is_file() else None
    invalid_wall = ((invalid_worker['finished_unix'] - invalid_worker['started_unix']) / 3600
                    if invalid_worker and 'finished_unix' in invalid_worker else None)
    cost = {'by_condition': costs, 'total_measured_episode_gpu_hours': sum(v['episode_gpu_hours'] for v in costs.values()),
            'invalid_bf16_technical_attempt_gpu_wall_hours': invalid_wall,
            'invalid_attempt_note': '30 quarantined receipts, zero formal denominator contribution; worker wall is rental-time proxy, not an invoice',
            'rental_rate_per_hour': None, 'rental_invoice_cost': None,
            'rental_note': 'Unknown invoice/rate; no fabricated monetary estimate',
            'retry_errors': validation['failed_attempts_preserved'],
            'worker_status': json.loads((EVIDENCE / 'worker_status.json').read_text()) if (EVIDENCE / 'worker_status.json').exists() else None,
            'unattributed_wall_note': 'episode sums exclude model loading, startup, and idle; worker_status records the process interval'}
    return perf, cost

def csv_atomic(path: Path, rows: list[dict]):
    with path.with_suffix(path.suffix + '.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(path.suffix + '.tmp').replace(path)

def main():
    protocol = json.loads((EVIDENCE / 'protocol_frozen.json').read_text())
    grouped = load_complete(protocol)
    validation, chunks, resource, actions = validate_outputs(protocol, grouped)
    task_rows, suites = task_suite_tables(protocol, grouped)
    pairs, target, skill, decisions = comparisons(protocol, grouped, chunks, actions)
    perf, cost = performance_cost(chunks, resource, validation)
    shutil.copyfile(OUT / 'episodes.csv', EVIDENCE / 'episodes.csv')
    csv_atomic(EVIDENCE / 'paired_outcomes.csv', pairs)
    csv_atomic(EVIDENCE / 'task_metrics.csv', task_rows)
    for name, content in (('suite_metrics.json', suites), ('target_repair.json', target),
                          ('skill_retention.json', skill), ('performance.json', perf),
                          ('cost_report.json', cost), ('release_decisions.json', decisions),
                          ('validation.json', validation),
                          ('statistical_analysis.json', {'target': target, 'legacy': skill,
                            'method': protocol['rules']['statistics'], 'no_test_based_tuning': True})):
        atomic_json(EVIDENCE / name, content)
    print(json.dumps({'status': 'PASS', 'episodes': 1600,
                      'overall': {name: suites[name]['overall'] for name in CONDITIONS},
                      'decisions': decisions}, sort_keys=True))

if __name__ == '__main__':
    main()
