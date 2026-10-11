"""Bounded search for verified M04 corrections on additional M02 failed states.

Attempts use only recorded failed M02 goal-task-6 source episodes; no success is
counted until its exact action sequence passes two independent MuJoCo replays.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path('/data/VLA-Forge/project/robot-vla')
ROOT = Path('/data/VLA-Forge')
EVID = ROOT / 'Evidence/M04'
ATTEMPTS = EVID / 'artifacts/policy_attempts/attempts.jsonl'
VERIFIED = EVID / 'artifacts/verified_corrections'
M02 = ROOT / 'Evidence/M02/source/M02_libero40/registry/runs/20261010T030315-982cacc8515f/episodes'
SOURCE_IDS = (6,)
TRIES_PER_SOURCE = 3
TARGET_NEW = 3


def rows() -> list[dict]:
    return [json.loads(x) for x in ATTEMPTS.read_text().splitlines() if x.strip()]


def verified_receipts() -> dict[str, dict]:
    found = {}
    for p in VERIFIED.glob('*/summary.json'):
        d = json.loads(p.read_text())
        reps = d.get('replicates', [])
        if d.get('source_policy_run') and d.get('independent_replay_pass') is True and len(reps) == 2 and all(x.get('success') is True for x in reps):
            found[d['source_action_sha256']] = d
    return found


def run(command: list[str], env: dict[str, str], log: Path) -> int:
    with log.open('w', encoding='utf-8') as f:
        return subprocess.run(command, cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, check=False).returncode


def main() -> None:
    existing = verified_receipts()
    start_hashes = set(existing)
    result = {'schema': 'vla-forge.m04.diverse-source-seed-sweep.v1', 'source_policy': 'frozen M02 SmolVLA, wrist brightness gain 1.10 during first 50 observations', 'candidate_m02_failed_init_states': list(SOURCE_IDS), 'tries_per_source_max': TRIES_PER_SOURCE, 'target_new_verified_successes_across_two_or_more_sources': TARGET_NEW, 'attempts': []}
    out = EVID / 'logs/diverse-state6-sweep-summary.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    envbase = os.environ.copy()
    envbase.pop('LD_LIBRARY_PATH', None)
    envbase.update({'M04_SUITE': 'goal', 'M04_TASK_ID': '6', 'M04_POLICY_CONDITIONS': 'brightness_first50', 'M04_BRIGHTNESS_GAIN': '1.10', 'M04_INTERVENTION_STEPS': '50', 'M04_INTERVENTION_START': '0', 'M04_BRIGHTNESS_CAMERA': 'wrist', 'M04_ACTION_SCALE': '1.0'})
    for init_id in SOURCE_IDS:
        source_path = M02 / f'goal_task06_episode{init_id:02d}_attempt1/episode.json'
        source_ep = json.loads(source_path.read_text())
        if source_ep.get('success') is not False:
            raise RuntimeError(f'candidate is not an M02 failure: {source_path}')
        for offset in range(TRIES_PER_SOURCE):
            seed = 84120 + init_id * 10 + offset
            run_id = f'm04-diverse-goal-t06-init{init_id:02d}-seed{seed}'
            if any(x.get('episode_id') == run_id for x in rows()):
                result['attempts'].append({'run_id': run_id, 'status': 'SKIP_EXISTING_ID'})
                out.write_text(json.dumps(result, indent=2) + '\n')
                continue
            env = {**envbase, 'M04_SOURCE_EPISODE': str(init_id), 'M04_POLICY_SEED': str(seed), 'M04_RUN_ID': run_id}
            attempt_log = EVID / 'logs' / f'diverse-{run_id}-attempt.log'
            try:
                rc = run([sys.executable, 'benchmarks/robot-vla-engine/scripts/run_m04_policy_attempt.py'], env, attempt_log)
            except KeyboardInterrupt:
                result['attempts'].append({'run_id': run_id, 'source_m02_episode': source_ep.get('episode_id'), 'init_state_id': init_id, 'seed': seed, 'status': 'INTERRUPTED_DURING_SIM_RENDER', 'episode_counted': False, 'attempt_log': str(attempt_log)})
                result['status'] = 'INTERRUPTED'
                out.write_text(json.dumps(result, indent=2) + '\n')
                print(json.dumps({'summary': str(out), 'status': 'INTERRUPTED', 'run_id': run_id}))
                return
            matched = [x for x in rows() if x.get('episode_id') == run_id]
            if rc or not matched:
                result['attempts'].append({'run_id': run_id, 'status': 'RUNNER_ERROR', 'exit_code': rc, 'log': str(attempt_log)})
                out.write_text(json.dumps(result, indent=2) + '\n')
                continue
            row = matched[-1]
            item = {'run_id': run_id, 'source_m02_episode': source_ep.get('episode_id'), 'init_state_id': init_id, 'seed': seed, 'success': bool(row.get('success')), 'trajectory_sha256': row.get('trajectory', {}).get('sha256'), 'gpu_hours': row.get('gpu_hours_allocated'), 'attempt_log': str(attempt_log)}
            if not row.get('success'):
                item['status'] = 'POLICY_FAILURE_RETAINED'
                result['attempts'].append(item)
                out.write_text(json.dumps(result, indent=2) + '\n')
                continue
            replay_log = EVID / 'logs' / f'diverse-{run_id}-independent-replay.log'
            replay_rc = run([sys.executable, 'benchmarks/robot-vla-engine/scripts/verify_m04_policy_correction.py'], {**envbase, 'M04_SOURCE_EPISODE': str(init_id), 'M04_POLICY_RUN_ID': run_id}, replay_log)
            summary_path = VERIFIED / run_id / 'summary.json'
            if replay_rc or not summary_path.exists():
                item.update({'status': 'INDEPENDENT_REPLAY_FAILED', 'replay_exit_code': replay_rc, 'replay_log': str(replay_log)})
            else:
                summary = json.loads(summary_path.read_text())
                h = summary.get('source_action_sha256')
                if summary.get('independent_replay_pass') is not True or len(summary.get('replicates', [])) != 2 or not all(x.get('success') is True for x in summary['replicates']):
                    item['status'] = 'REPLAY_RECEIPT_FAILED'
                elif h in start_hashes:
                    item.update({'status': 'DUPLICATE_ACTION_SEQUENCE', 'action_sha256': h})
                else:
                    existing[h] = summary
                    item.update({'status': 'NEW_VERIFIED_UNIQUE_SEQUENCE', 'action_sha256': h})
                item['replay_log'] = str(replay_log)
            result['attempts'].append(item)
            out.write_text(json.dumps(result, indent=2) + '\n')
        new_verified = [d for h, d in existing.items() if h not in start_hashes]
        # Source IDs are confirmed by the M02 rows and policy run names.
        if len(new_verified) >= TARGET_NEW and len({x.get('init_state_id') for x in result['attempts'] if x.get('status') == 'NEW_VERIFIED_UNIQUE_SEQUENCE'}) >= 2:
            break
    distinct_init = sorted({x.get('init_state_id') for x in result['attempts'] if x.get('status') == 'NEW_VERIFIED_UNIQUE_SEQUENCE'})
    result.update({'new_verified_unique_successes': sum(x.get('status') == 'NEW_VERIFIED_UNIQUE_SEQUENCE' for x in result['attempts']), 'distinct_new_init_states': distinct_init, 'target_reached': sum(x.get('status') == 'NEW_VERIFIED_UNIQUE_SEQUENCE' for x in result['attempts']) >= TARGET_NEW and len(distinct_init) >= 2, 'scope_limit': 'No correction is counted without source M02 failure identity and two successful independent physical replays.'})
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'summary': str(out), 'attempts': len(result['attempts']), 'new_verified': result['new_verified_unique_successes'], 'distinct_init_states': distinct_init, 'target_reached': result['target_reached']}, indent=2))


if __name__ == '__main__':
    main()
