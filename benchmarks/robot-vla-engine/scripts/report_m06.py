"""Publish an evidence-bounded private M06 acceptance report after all gates."""
from __future__ import annotations

import json
import subprocess

from robot_vla.m06_protocol import CONDITIONS, EVIDENCE, sha256


def load(name):
    return json.loads((EVIDENCE / name).read_text())

def main():
    validation = load('validation.json')
    video = load('video_validation.json')
    suites = load('suite_metrics.json')
    target = load('target_repair.json')
    legacy = load('skill_retention.json')
    perf = load('performance.json')
    cost = load('cost_report.json')
    decisions = load('release_decisions.json')
    modelops = load('modelops/evaluations/summary.json')
    manifest = load('manifest.json')
    publication = load('github_publication.json')
    if (validation['status'] != 'PASS' or validation['valid'] != 1600
            or not 180 <= video['duration_s'] <= 240
            or any(decisions[name]['decision'] != modelops[name]['decision'] for name in CONDITIONS)):
        raise ValueError('M06 acceptance artifacts incomplete or cross-project gate differs')
    decision_text = ', '.join(f"{name}: {decisions[name]['decision']}" for name in CONDITIONS[1:])
    vla_commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    modelops_commit = subprocess.check_output(['git','-C','/data/ModelOps-Platform','rev-parse','HEAD'],text=True).strip()
    lines = ['# M06 Independent Robot Policy Regression & Release Evaluation', '',
             '**Engineering acceptance: PASS.** All four frozen policy identities have 400 real FP32 Eager LIBERO/MuJoCo rollouts each (1,600 total). The registered algorithm-quality decision is separate from engineering completeness.', '',
             f'Protocol SHA-256: `{manifest["protocol_sha256"]}`. Local VLA code commit `{vla_commit}`; local ModelOps commit `{modelops_commit}`. GPU: `{manifest["environment"]["gpu"]}`.',
             f'Published source: [VLA-Forge M06]({publication["vla"]["url"]}) and [ModelOps M06]({publication["modelops"]["url"]}). GitHub connector publication commits differ from the cloud-local commit IDs; the key source blob hashes were independently checked.', '',
             '## 40-task benchmark', '', '| Policy | Spatial | Object | Goal | Long | Overall | ModelOps decision |',
             '|---|---:|---:|---:|---:|---:|---|']
    for name in CONDITIONS:
        s = suites[name]
        lines.append('| '+name+' | '+' | '.join(f'{s[k]["success"]}/100' for k in ('spatial','object','goal','long'))+f' | {s["overall"]["success"]}/400 | {decisions[name]["decision"]} |')
    lines += ['', 'Every task has ten locked init states; task-level Wilson intervals are in `task_metrics.csv`. All failure episodes remain in the denominator and retain original video/action/physics evidence.', '',
              '## Target repair and old-skill retention', '',
              'Target: Goal task 6, ten matched states. One-sided exact paired discordance tests use Holm adjustment across three candidates; percent changes alone are not proof.', '',
              '| Candidate | Target Frozen→Candidate | Paired difference | Improved / worsened | Holm p | Legacy Δ | Task-cluster lower 95% | Decision |',
              '|---|---:|---:|---:|---:|---:|---:|---|']
    for name in CONDITIONS[1:]:
        t, legacy_row = target[name], legacy[name]
        lines.append(f'| {name} | {t["frozen_success"]}/10 → {t["candidate_success"]}/10 | {t["paired_difference"]:+.3f} | {t["improved"]}/{t["worsened"]} | {t["holm_adjusted_p"]:.4g} | {legacy_row["paired_difference"]:+.3f} | {legacy_row["one_sided_lower_95"]:+.3f} | {decisions[name]["decision"]} |')
    lines += ['', 'The legacy pool has 39 task clusters, not 390 independent tasks. The 3-point noninferiority margin, target significance rule, latency ceiling and safety rule were frozen before test opening. Worst task regressions and newly failed cases are in `skill_retention.json`; complete paired action differences are in `paired_outcomes.csv`.', '',
              'Historical M05 development scores (Frozen 3/8, Standard 6/8, Corrective 1/8, Replay 5/8) are exploratory and were never substituted for the M06 results.', '',
              '## Runtime and cost', '',
              '| Policy | Episodes | Episode GPU-hours | Chunk p50/p95/p99 (ms) | Peak allocated GPU (GiB) | Mean sampled GPU util |',
              '|---|---:|---:|---:|---:|---:|']
    for name in CONDITIONS:
        p, c = perf[name], cost['by_condition'][name]
        lat = p['chunk_latency_ms']
        util = p['gpu_utilization_percent_mean']
        lines.append(f'| {name} | {p["episodes"]} | {c["episode_gpu_hours"]:.3f} | {lat["p50"]:.1f}/{lat["p95"]:.1f}/{lat["p99"]:.1f} | {p["gpu_peak_allocated_bytes"]/2**30:.2f} | {util:.1f}% |' if util is not None else f'| {name} | {p["episodes"]} | {c["episode_gpu_hours"]:.3f} | {lat["p50"]:.1f}/{lat["p95"]:.1f}/{lat["p99"]:.1f} | {p["gpu_peak_allocated_bytes"]/2**30:.2f} | N/A |')
    lines += ['', f'Measured episode-wall total: {cost["total_measured_episode_gpu_hours"]:.3f} GPU-hours; this excludes startup, model loading, idle and failed retries. Model loading and pure encoder CPU time were not timed separately, so those phase values remain N/A; `performance.json` contains measured preprocessing, policy, MuJoCo, control-cycle and resource samples plus an explicitly labeled video/loop residual upper bound. The quarantined native-BF16 technical attempt consumed {cost["invalid_bf16_technical_attempt_gpu_wall_hours"]:.3f} additional GPU wall-hours and contributes zero formal episodes. Actual rental invoice/rate is unavailable; monetary cost remains N/A. Error/retry count: {validation["failed_attempts_preserved"]}.', '',
              '## Visual and reproducibility evidence', '',
              f'- Private film: `demo/M06_demo_200s.mp4` ({video["duration_s"]:.1f}s, 1920×1080 H.264, {video["bytes"]/2**20:.1f} MiB, SHA-256 `{video["sha256"]}`).',
              '- Offline dashboard: `visualization/index.html`. Source episode IDs, source time windows and hashes: `video_sources.json`.',
              '- `formal/receipts/`, `formal/attempts.jsonl`, `formal/registry/` retain real videos, executed actions, 8D observations, full available simulator state and resource events. `validation.json` binds all 1,600 artifacts.',
              '- Reproduce on the authorized GPU host: `/data/VLA-Forge/Evidence/M06/reproduce.sh`. The runner skips only receipts that still match the frozen model/protocol/artifact hashes.',
              '- Revalidate/aggregate: `PYTHONPATH=benchmarks/robot-vla-engine/src .venv/bin/python benchmarks/robot-vla-engine/scripts/finalize_m06.py`. ModelOps admission: `.venv/bin/python -m pipelines.robotics.ingest_m06_release --evidence /data/VLA-Forge/Evidence/M06 --registry /data/VLA-Forge/Evidence/M05/modelops/registry --output /data/VLA-Forge/Evidence/M06/modelops/evaluations` from `/data/ModelOps-Platform`.', '',
              '## Boundaries', '',
              '- The locked states are disjoint from recorded M02–M05 project development groups and exclude six retired contaminated source groups. LIBERO expert episode IDs have no verified mapping to fixed physical init IDs, and foundation-model pretraining exposure is unknown. Thus this is an independent project post-training regression test, not proof that the base model never saw related states.',
              '- All video and asset files remain private under the M04 source-license review. No public multimedia redistribution or real-robot deployment is asserted.',
              '- A candidate marked `INSUFFICIENT_EVIDENCE` did not prove noninferiority merely because an interval includes zero. Only `PROMOTE` is eligible for the authorized simulation service; ModelOps records gate decisions separately.', '',
              '## Resume bullet, based on measured evidence', '',
              f'> Built a hash-bound, resumable SmolVLA→LIBERO regression pipeline on RTX 3090; executed 1,600 matched MuJoCo rollouts across 40 tasks and four frozen policy versions, retained dual-camera video/action/physical-state traces, and connected paired repair and task-cluster retention analysis to a conservative ModelOps simulation release gate ({decision_text}).', '',
              'M06 stops here. New training or M07 requires a separate instruction.']
    # Keep the report self-contained and reveal no license-restricted images.
    (EVIDENCE / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'report': str(EVIDENCE/'REPORT.md'), 'sha256': sha256(EVIDENCE/'REPORT.md')}))

if __name__ == '__main__':
    main()
