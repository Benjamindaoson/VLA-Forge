"""Generate evidence-bounded M05 reports for VLA-Forge and ModelOps."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from robot_vla.m05_training import EVIDENCE

ROOT = EVIDENCE
VLA_REPO = Path('/data/VLA-Forge/project/robot-vla')
OPS_REPO = Path('/data/ModelOps-Platform')


def read(name: str) -> dict:
    return json.loads((ROOT / name).read_text())


def commit(repo: Path) -> str:
    return subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()


def main() -> None:
    validation = read('validation.json')
    if not validation['passed']:
        raise ValueError('M05 evidence validation is incomplete')
    training = read('training/training_summary.json')['runs']
    dev = read('development/metrics.json')['conditions']
    graph = read('summary/graph_closed_loop_summary.json')
    inf = read('inference/matrix.json')['modes']
    frozen = read('inference_frozen/summary.json')
    cost = read('summary/cost_report.json')
    optimization = list(csv.DictReader((ROOT / 'summary/optimization_matrix.csv').open()))
    video = read('demo/video_validation.json')
    sources = read('demo/video_sources.json')
    ab = read('modelops/ab_comparison.json')
    smoke = read('modelops/serving_smoke.json')
    component = read('training_perf/component_profile/parity.json')
    if not (component['adapter_sha_exact'] and component['loss_exact']):
        raise ValueError('Component profiling changed the numerical pilot')
    serving = {name: read(f'serving/{name}_summary.json') for name in
               ('m05-base-frozen', 'm05-standard-s5501', 'm05-base-frozen_cuda_graph')}
    vla_sha, ops_sha = commit(VLA_REPO), commit(OPS_REPO)
    lines = [
        '# VLA-Forge M05 — real SmolVLA post-training and inference systems', '',
        '**Technical validation: PASS.** All claimed GPU work below has raw receipts. This is an eight-state development evaluation; M06 formal 400-episode test remains unopened. Public visual/weight release is not authorized.', '',
        '**Gate 1 (real post-training): PASS.** Three frozen 500-update LoRA conditions saved reloadable adapters; 32/32 development MuJoCo receipts passed integrity checks.',
        '**Gate 2 (AI infrastructure): PASS.** Precision, memory, data and inference candidates have measured accept/reject decisions; real service, paired closed-loop rollouts and phase-bounded costs are audited.',
        '**Gate 3 (career evidence): PASS.** Private 235-second demo, offline HTML, checkpoint hashes, reproduction entrypoint, GitHub commits and evidence-bounded résumé bullets are present.', '',
        f'VLA code: `{vla_sha}` on `codex/m05-modelops-posttraining`. ModelOps code: `{ops_sha}` on `codex/smolvla-linux-adapter`.',
        f'Unified private evidence: `{ROOT}`. Frozen official model: `lerobot/smolvla_libero` with model SHA `{read("inference_frozen/eager.json")["model_sha256"]}`.', '',
        '## What actually ran', '',
        'Official LeRobot SmolVLA Flow Matching with LoRA r8/alpha16 and 371,328 trainable parameters (0.0824% of 450,417,504). Three 500-update FP32 runs used the same frozen recipe and last-checkpoint selection: standard expert demonstrations, M04 corrective data, and a replay mixture. Replay lineage is corrected by `data/replay_sources.json`; the immutable training identity file reflects its expert subset and is not silently rewritten.', '',
        '| Condition | Run ID | Updates | Loss first → last | Process wall | Adapter SHA256 |',
        '|---|---|---:|---:|---:|---|',
    ]
    for name, r in training.items():
        lines.append(f"| {name} | `{r['run_id']}` | {r['optimizer_updates']} | {r['loss_first']:.5f} → {r['loss_last']:.5f} | {r['process_wall_s']:.2f}s | `{r['adapter_sha256']}` |")
    lines += ['', '## Bounded training optimization pilots', '',
              'These are eight-update cold-process pilots on the same selected workload. '
              'Wall-derived throughput includes process overhead and is not steady-state training speed.', '',
              '| Candidate | Decision | Updates | Process wall | Cold effective updates/s | Peak allocated GiB |',
              '|---|---|---:|---:|---:|---:|']
    for row in optimization:
        steps = int(row['optimizer_updates'])
        wall = float(row['process_wall_s']) if row['process_wall_s'] else None
        peak = float(row['peak_allocated_bytes']) if row['peak_allocated_bytes'] else None
        lines.append(
            f"| {row['candidate']} | {row['decision']} | {steps} | "
            f"{f'{wall:.2f}s' if wall else 'N/A'} | "
            f"{f'{steps / wall:.3f}' if wall else 'N/A'} | "
            f"{f'{peak / 2**30:.3f}' if peak else 'N/A'} |"
        )
    lines += ['', 'BF16 and TF32 completed bounded runs with finite gradients but '
              'did not establish a speed/correctness benefit; FP16 failed its finite-gradient '
              'gate at update 2 and has no valid checkpoint. Checkpointing reduced peak '
              'allocation in its pilot but did not improve speed. Candidate decisions and '
              'data-loader observations are in `summary/optimization_matrix.csv` and '
              '`summary/data_pipeline_matrix.json`.', '']
    lines += ['', 'Each run retained per-step loss, gradient, LR, data/forward/backward/optimizer timing and sampled RTX 3090 utilization/power. Adapter save and reload checks passed. Component instrumentation was compared with a new same-camera, same-seed eight-step reference and produced the exact same adapter hash and final loss; an older pre-camera-fix FP32 pilot is not treated as a valid parity reference.', '',
              '## Real MuJoCo development evaluation', '',
              'Eight frozen initial states span LIBERO Goal task 6 (4), Goal task 0 (2), and Object task 0 (2). Each condition used actual model inference, dual-camera MuJoCo rollout, raw actions, states, video and independently audited file hashes. The locked M06 init IDs 40–49 were not used.', '',
              '| Policy | Success / 8 | Difference versus frozen on these states |',
              '|---|---:|---:|']
    base = dev['frozen']['success']
    for name in ('frozen', 'standard', 'corrective', 'replay'):
        score = dev[name]['success']
        lines.append(f'| {name} | {score}/8 | {score-base:+d} episode(s) |')
    lines += ['', 'The corrective-only model scored 1/8 and was **not** promoted. The standard model scored 6/8 and replay 5/8, but eight fixed states provide no cross-seed or generalization claim. No algorithmic gain is inferred from training loss alone.', '',
              '## Inference and closed-loop performance', '',
              '| Corrective model mode | 100-call chunk P50 / P95 / P99 | Raw action L∞ vs Eager | Decision |',
              '|---|---:|---:|---|']
    for name in ('eager', 'cuda_graph', 'bf16', 'fp16', 'tf32', 'compile'):
        r = inf[name]
        if r['status'] == 'completed':
            t = r['chunk_latency']
            value = f"{t['p50_s']*1000:.1f} / {t['p95_s']*1000:.1f} / {t['p99_s']*1000:.1f} ms"
            delta = f"{r['raw_action_linf_vs_eager']:.6f}"
            decision = 'ACCEPT static output/speed' if name == 'cuda_graph' else ('reference' if name == 'eager' else 'REJECT speed')
        else:
            value, delta, decision = '180s compile budget exceeded', 'N/A', 'REJECT'
        lines.append(f'| {name} | {value} | {delta} | {decision} |')
    lines += ['', f"Frozen base static action-chunk P50 was {frozen['eager_chunk_latency']['p50_s']*1000:.1f} ms Eager and {frozen['cuda_graph_chunk_latency']['p50_s']*1000:.1f} ms CUDA Graph ({frozen['speedup_p50']:.2f}×). Corrective static warm P50 speedup was {inf['eager']['chunk_latency']['p50_s']/inf['cuda_graph']['chunk_latency']['p50_s']:.2f}×. Both produced exactly matching 50×7 raw actions on three different real observations; graph capture cold-start was separately recorded. A physical RTX 3090 was constrained to an 8 GiB PyTorch allocator budget for one valid action request; that is not evidence of deployment on an 8 GiB GPU.", '',
              f"The **16 separate real MuJoCo Eager/Graph pairs** had matching 8D initial robot state and exact first 50 actions (16/16); all aligned logged actions/states matched. Frozen success remained {graph['conditions']['frozen']['baseline_success']}/8 and corrective {graph['conditions']['corrective']['baseline_success']}/8. Full Episode cumulative wall speedup was only {graph['conditions']['frozen']['graph_vs_baseline_wall_speedup']:.3f}× frozen and {graph['conditions']['corrective']['graph_vs_baseline_wall_speedup']:.3f}× corrective. Logged 8D state is not a full simulator microstate claim.", '',
              '## ModelOps value and cost', '',
              f"A native/managed 20-update A/B matched the same final adapter SHA, sample indices, loss, gradient norm and LR. Native wall was {ab['native']['wall_s']:.3f}s; deliberate managed pause/resume used {ab['managed']['wall_s_including_two_processes']:.3f}s, adding {ab['wall_overhead_s']:.3f}s ({100*ab['wall_overhead_fraction']:.1f}% of native short-job wall). This overhead is real and should not be hidden behind the lineage benefit.", '',
              f"The simulation-only ModelOps API loaded the base, switched to the standard candidate, rejected real-robot scope and stale versions (HTTP {smoke['real_robot_scope_http']}), then rolled back to exact base actions. All three bounded service benchmarks completed 128/128 requests without error. At concurrency 8, frozen Eager throughput was {serving['m05-base-frozen']['groups']['8']['successful_requests_per_s']:.2f} req/s and frozen CUDA Graph was {serving['m05-base-frozen_cuda_graph']['groups']['8']['successful_requests_per_s']:.2f} req/s on one repeated real observation. The Graph HTTP action chunk also matched native Eager exactly. This synthetic serving throughput does **not** describe MuJoCo rollout throughput.", '',
              f"Measured formal training process occupancy: {cost['formal_training_process_wall_s']/3600:.4f} GPUh upper bound. Eager 32 development episodes: {cost['development_32_episode_gpu_hours_upper_bound']:.4f} GPUh; Graph 16 episodes: {cost['graph_16_episode_gpu_hours_upper_bound']:.4f} GPUh. Combined measured-phase process occupancy: {cost['training_plus_all_development_process_gpu_hours_upper_bound']:.4f} GPUh. This excludes setup/exploration, service, idle rental, storage and video encoding; no cloud invoice or hourly rate was available, so currency cost is N/A.", '',
              '## Private visual evidence and reproduction', '',
              f"`{video['path']}` — {video['duration_s']:.2f}s, {video['width']}×{video['height']}, {video['codec']}/{video['pixel_format']}, {video['bytes']/2**20:.1f} MiB, full FFmpeg decode passed. It uses {sources['unique_source_count']} distinct hashed real MuJoCo source recordings, plus measured training/inference/GPU charts. Original recordings are separately retained. Open `demo/visualization/index.html` through the authorized loopback evidence server; `demo/video_sources.json` maps every cut to its original recording and hash.", '',
              '```bash', 'cd /data/VLA-Forge/project/robot-vla', 'bash benchmarks/robot-vla-engine/scripts/reproduce_m05.sh verify', 'bash benchmarks/robot-vla-engine/scripts/reproduce_m05.sh managed-service', '```', '',
              'Source config, environment pins, replay lineage and receipt hashes are in `manifest.json`, `dependencies_vla.json`, `data/lineage_gate.json`, and `validation.json`. Run the GPU stages in `reproduce.sh` one at a time under the shared GPU lock. The private footage, corrected data, and checkpoints remain off public GitHub pending file-level redistribution clearance.', '',
              '## Limits and next gate', '',
              'The eleven M04 corrections originated from two source states of one Goal task; the corrective-only result is negative. HF expert episode IDs do not map to verified LIBERO fixed init IDs, so the recorded group split cannot prove physical-state independence from M06. This work does not validate Kubernetes/Argo, real hardware safety, or a frozen M06 policy release. M06 should remain closed until the user explicitly authorizes its protocol.', '',
              '**Robot Learning résumé bullet:** Built a reproducible SmolVLA Flow Matching post-training and MuJoCo evaluation pipeline on one RTX 3090, comparing three 500-update LoRA adapters on eight fixed LIBERO development states (frozen 3/8, standard 6/8, corrective 1/8, replay 5/8); verified exact Eager/CUDA Graph action trajectories across 16 real paired rollouts while reducing full-rollout wall time by 5.2–5.7%.', '',
              '**AI Infrastructure résumé bullet:** Integrated real SmolVLA training with a single-host ModelOps lifecycle, preserving exact 20-step native/managed adapter parity through pause/resume and immutable version rollback; served 384/384 simulator action requests without error and measured CUDA Graph versus Eager throughput at concurrency 8 (15.81 vs 3.23 req/s), while exposing 93% short-job managed restart overhead.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))
    ops = [
        '# ModelOps-Platform × VLA-Forge M05 — real single-GPU integration', '',
        '**Technical validation: PASS for single-host simulation development.** Kubernetes, Argo and real robot control were not tested.', '',
        f'ModelOps code `{ops_sha}`; VLA code `{vla_sha}`. Shared evidence `{ROOT}`. Exact training Run ID `m05-corrective-cam180-s5501`, dataset selection `data/selection.json`, adapter SHA `{training["corrective"]["adapter_sha256"]}`.', '',
        'ModelOps launches the VLA-Forge LeRobot Flow Matching trainer in the VLA `.venv`; it does not replace it with text SFT. It controls start/stop/resume/query, stores immutable version records, MLflow metrics and raw JSONL references, gates dev evaluations by real MuJoCo receipts, and serves only loopback simulator 50×7 action chunks with dual RGB cameras, 8D state and task text.', '',
        f'The managed corrective job completed 500 updates across two attempts with a deliberate pause at 250. MLflow file store is `modelops/mlruns`; raw job query is `modelops/managed_job_query_final.json`. The 20-step native/managed comparison produced the exact same adapter SHA and sample indices, but managed two-process wall was {ab["managed"]["wall_s_including_two_processes"]:.3f}s versus native {ab["native"]["wall_s"]:.3f}s. The {ab["wall_overhead_s"]:.3f}s restart overhead is a cost of this short validation run, not a steady-state speed claim.', '',
        f'The registry bound base, standard, corrective and replay versions to processor, normalizer, dataset, run and checkpoint hashes. Corrective 1/8 versus base 3/8 was registered and evaluated but not promoted. Standard 6/8 was switched into the simulator action API and rolled back to base; deterministic base actions matched exactly after rollback. The current active revision after benchmark rollback is {read("modelops/registry/active.json")["revision"]} (`m05-base-frozen`).', '',
        '| Service version/backend | Requests | Concurrency-8 throughput | Request P95 | Queue P95 |',
        '|---|---:|---:|---:|---:|']
    for name in ('m05-base-frozen', 'm05-standard-s5501', 'm05-base-frozen_cuda_graph'):
        r = serving[name]
        g = r['groups']['8']
        ops.append(f"| {r['version']} / {r.get('backend','eager')} | {r['total_successful']}/{r['total_requested']} | {g['successful_requests_per_s']:.2f} req/s | {g['request_latency']['p95_s']:.3f}s | {g['queue_latency']['p95_s']:.3f}s |")
    ops += ['', 'The Graph HTTP action matched frozen native Eager exactly on the same real M04 observation. At higher concurrency, throughput saturated and queue latency increased; the reported figures are one-GPU loopback loads on repeated observations, not deployment SLOs or hardware robot safety approval.', '',
            'Reproduction: run the VLA `reproduce_m05.sh` `verify`, `managed-service`, `serving-smoke` and `serving-benchmark VERSION` stages. ModelOps `pipelines/robotics/README.md` documents the independent adapter and registry CLI. Private reports and videos remain under the shared M05 evidence directory. Full source hashes are in `validation.json`.', '',
            '**AI Infrastructure résumé bullet:** Delivered a single-host ModelOps adapter for real SmolVLA GPU training, immutable checkpoint lineage and loopback continuous-action serving; demonstrated exact 20-step native/managed weight parity, pause/resume and rollback, plus 384 successful requests across three model/backend loads, and quantified 18.7s short-job restart overhead.', '']
    (ROOT / 'modelops/REPORT.md').write_text('\n'.join(ops))
    print(json.dumps({'vla_report': str(ROOT / 'REPORT.md'),
                      'modelops_report': str(ROOT / 'modelops/REPORT.md'),
                      'vla_commit': vla_sha, 'modelops_commit': ops_sha}, indent=2))


if __name__ == '__main__':
    main()
