"""Independently reconcile M05 GPU receipts, visual evidence and ModelOps lifecycle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from robot_vla.m05_training import EVIDENCE, atomic_json

ROOT = EVIDENCE


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--expect-service', action='store_true')
    args = parser.parse_args()
    training = read('training/training_summary.json')['runs']
    dev = read('development/validation.json')
    graph = read('development_graph/validation.json')
    models = read('development/metrics.json')['conditions']
    infer = read('inference/matrix.json')['modes']
    frozen = read('inference_frozen/summary.json')
    graph_pairs = read('summary/graph_closed_loop_summary.json')
    lineage = read('data/lineage_gate.json')
    demo = read('demo/video_validation.json')
    sources = read('demo/video_sources.json')
    smoke_file = ROOT / 'modelops/serving_smoke.json'
    serving = {name: ROOT / f'serving/{name}_summary.json' for name in
               ('m05-base-frozen', 'm05-standard-s5501', 'm05-base-frozen_cuda_graph')}
    checks = {
        'formal_500_updates_each': all(x['optimizer_updates'] == 500 for x in training.values()) and len(training) == 3,
        'development_32_complete': sum(x['total'] for x in models.values()) == 32 and all(x['errors'] == 0 for x in models.values()),
        'development_integrity': bool(dev.get('passed')),
        'graph_16_complete': sum(x['total'] for x in read('development_graph/metrics.json')['conditions'].values()) == 16,
        'graph_integrity': bool(graph.get('passed')),
        'graph_closed_loop_exact_actions': graph_pairs['first_50_actions_exact_pairs'] == 16,
        'graph_no_dev_success_degradation': all(
            row['baseline_success'] == row['graph_success']
            for row in graph_pairs['conditions'].values()),
        'corrective_inference_5_modes_100_calls': sum(x['status'] == 'completed' and x.get('calls') == 100 and x.get('warm_calls') == 100 for x in infer.values()) == 5,
        'graph_exact_static_corrective': infer['cuda_graph']['action_linf_vs_eager'] == 0.0,
        'graph_exact_static_frozen': frozen['action_raw_linf'] == 0.0,
        'multi_observation_graph': all(read(f'inference/{name}_multi_observation_graph.json')['passed'] for name in ('frozen', 'corrective')),
        'source_identity_gate': bool(lineage['passed_recorded_identity_checks']),
        'demo_contract': demo['full_decode_exit_code'] == 0 and 180 <= demo['duration_s'] <= 240 and demo['codec'] == 'h264' and demo['width'] == 1920 and demo['height'] == 1080,
        'unique_robot_video_sources': sources['unique_source_count'] == sum(len(s['sources']) for s in sources['segments']),
        'modelops_native_managed_ab': bool(read('modelops/ab_comparison.json')['passed']),
        'bounded_compile_result_recorded': infer['compile']['status'] == 'failed',
        'component_profile_evidence': (ROOT / 'training_perf/component_profile/component_profile.json').is_file(),
        'component_profile_numerical_parity': bool(read('training_perf/component_profile/parity.json')['adapter_sha_exact']),
        'simulated_memory_budget_evidence': (ROOT / 'inference/allocator_8gib.json').is_file(),
        'real_service_switch_rollback': smoke_file.is_file() and bool(json.loads(smoke_file.read_text()).get('rollback_base_actions_exact')),
        'service_load_base_candidate_graph': all(
            p.is_file() and json.loads(p.read_text()).get('total_requested') == 128
            and json.loads(p.read_text()).get('total_errors') == 0 for p in serving.values()),
        'live_graph_http_exact_actions': (ROOT / 'modelops/serving_graph_action_parity.json').is_file()
            and bool(read('modelops/serving_graph_action_parity.json')['exact']),
    }
    if not args.expect_service:
        checks.pop('real_service_switch_rollback')
        checks.pop('service_load_base_candidate_graph')
        checks.pop('live_graph_http_exact_actions')
    source_files = [ROOT / rel for rel in (
        'training/training_summary.json', 'development/validation.json',
        'development_graph/validation.json', 'inference/matrix.json',
        'inference_frozen/summary.json', 'data/lineage_gate.json',
        'demo/video_sources.json', 'demo/video_validation.json',
        'modelops/ab_comparison.json', 'summary/cost_report.json',
        'summary/graph_closed_loop_summary.json',
        'training_perf/component_profile/component_profile.json',
        'training_perf/component_profile/parity.json',
        'inference/allocator_8gib.json',
    )]
    if smoke_file.exists():
        source_files.append(smoke_file)
    parity = ROOT / 'modelops/serving_graph_action_parity.json'
    if parity.exists():
        source_files.append(parity)
    source_files.extend(p for p in serving.values() if p.exists())
    receipt = {
        'scope': 'M05 development and private simulation serving; excludes M06 test results',
        'checks': checks, 'passed': all(checks.values()),
        'files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in source_files],
        'demo_video_sha256': sha(Path(demo['path'])),
        'limitations': ['HF expert episodes lack verified LIBERO init-state mapping',
                        '8 fixed development states are not a generalization claim',
                        'private robot images cannot be publicly redistributed without clearance',
                        'currency billing unavailable', 'Kubernetes/Argo untested'],
    }
    atomic_json(ROOT / 'validation.json', receipt)
    manifest = read('manifest.json')
    manifest['execution'] = {
        'validation_path': str(ROOT / 'validation.json'),
        'training_run_ids': {k: v['run_id'] for k, v in training.items()},
        'trained_adapter_sha256': {k: v['adapter_sha256'] for k, v in training.items()},
        'development_successes': {k: f"{v['success']}/{v['total']}" for k, v in models.items()},
        'demo_video_path': demo['path'], 'demo_video_sha256': receipt['demo_video_sha256'],
        'modelops_ab_evidence': str(ROOT / 'modelops/ab_comparison.json'),
        'distributable': False,
    }
    atomic_json(ROOT / 'manifest.json', manifest)
    print(json.dumps({'passed': receipt['passed'], 'checks': checks}, indent=2))
    if not receipt['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
