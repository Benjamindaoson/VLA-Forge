"""Resumable real LIBERO independent regression; never opens training data."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import random
import time
import traceback
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ['LIBERO_CONFIG_PATH'] = str(Path(__file__).resolve().parents[3] / 'work/libero')

import av
import numpy as np
import torch
from lerobot.envs.libero import LiberoEnv
from libero.libero import benchmark

from robot_vla.instrumentation import ChunkProfiler, ResourceMonitor
from robot_vla.m01 import fit_libero_state_to_checkpoint
from robot_vla.m05_training import load_policy
from robot_vla.m06_protocol import CONDITIONS, ENGINE, EVIDENCE, SUITES, atomic_json, sha256
from robot_vla.observations import libero_batch
from robot_vla.preflight import create_libero_env
from robot_vla.registry import Registry
from robot_vla.rollout import record_episode

PROTO = EVIDENCE / 'protocol_frozen.json'
LOCK = Path('/data/VLA-Forge/Evidence/M05/gpu.lock')

class CapturedEnv:
    """Capture real physical state after reset and each executed control step."""
    def __init__(self, env, verifier=None):
        self.env = env
        self.samples = []
        self.verifier = verifier
        self.initial_match = None

    def _capture(self):
        sim = self.env._env.sim
        physical = self.env._env.get_sim_state()
        self.samples.append({
            'sim': np.asarray(physical, dtype=np.float64).copy(),
            'qpos': np.asarray(sim.data.qpos, dtype=np.float64).copy(),
            'qvel': np.asarray(sim.data.qvel, dtype=np.float64).copy(),
            'time': float(sim.data.time),
        })

    def reset(self, *, seed):
        result = self.env.reset(seed=seed)
        self.samples.clear()
        self._capture()
        if self.verifier is not None:
            self.initial_match = self.verifier(self.samples[0])
        return result

    def step(self, action):
        result = self.env.step(action)
        self._capture()
        return result

    def close(self):
        self.env.close()


def _state_arrays(samples):
    return {key: np.stack([s[key] for s in samples]) if key != 'time'
            else np.asarray([s[key] for s in samples], dtype=np.float64)
            for key in ('sim', 'qpos', 'qvel', 'time')}


def compare_reference(path: Path, sample: dict, tolerance: dict, *, create: bool) -> dict:
    """Compare reset physics; first frozen episode creates a write-once reference."""
    arrays = _state_arrays([sample])
    if not path.exists():
        if not create:
            raise ValueError('Frozen physical reference absent; evaluate frozen policy first')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, **arrays)
        path.with_suffix('.npz.tmp').replace(path)
    with np.load(path, allow_pickle=False) as source:
        reference = {key: source[key][0] for key in arrays}
    differences = {}
    for key, threshold in (('sim', 'sim_vector_max_abs'), ('qpos', 'qpos_max_abs'), ('qvel', 'qvel_max_abs')):
        if arrays[key][0].shape != reference[key].shape:
            raise ValueError(f'Physical vector shape mismatch: {key}')
        differences[key] = float(np.max(np.abs(arrays[key][0] - reference[key])))
        if differences[key] > tolerance[threshold]:
            raise ValueError(f'Unmatched initial physics {key}: {differences[key]}')
    differences['time'] = abs(float(arrays['time'][0] - reference['time']))
    if differences['time'] > tolerance['time_abs_s']:
        raise ValueError(f'Unmatched initial sim time: {differences["time"]}')
    return {'reference_path': str(path), 'reference_sha256': sha256(path),
            'max_abs_difference': differences, 'matched': True}


def valid_receipt(receipt: dict, case: dict, condition: str, model_sha: str, protocol_sha: str) -> bool:
    if (receipt.get('status') != 'complete' or receipt.get('condition') != condition
            or receipt.get('floating_parameter_dtypes') != ['torch.float32']
            or receipt.get('tf32_allowed') is not False
            or receipt.get('source_group') != case['source_group']
            or receipt.get('checkpoint_sha256') != model_sha
            or receipt.get('protocol_sha256') != protocol_sha
            or type(receipt.get('success')) is not bool or not receipt.get('initial_match', {}).get('matched')):
        return False
    for kind in ('video', 'trajectory', 'action_trace', 'sim_states'):
        item = receipt.get(kind, {})
        path = Path(item.get('path', ''))
        if not path.is_file() or sha256(path) != item.get('sha256'):
            return False
    return True


def append_attempt(path: Path, item: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(item, allow_nan=False, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def write_summary(out: Path):
    rows = [json.loads(p.read_text()) for p in sorted((out / 'receipts').glob('*.json'))]
    keys = ('condition', 'suite', 'task_id', 'init_state_id', 'seed', 'source_group', 'status',
            'success', 'steps', 'wall_s', 'chunk_p50_ms', 'chunk_p95_ms',
            'gpu_peak_allocated', 'checkpoint_sha256', 'video_sha256', 'error')
    target = out / 'episodes.csv'
    with target.with_suffix('.csv.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        for item in rows:
            writer.writerow({**item, 'video_sha256': item.get('video', {}).get('sha256')})
    target.with_suffix('.csv.tmp').replace(target)
    atomic_json(out / 'progress.json', {'scope': 'M06-independent', 'expected': 1600,
        'complete': sum(r['status'] == 'complete' for r in rows),
        'errors': sum(r['status'] == 'error' for r in rows),
        'by_condition': {name: {'complete': sum(r['condition'] == name and r['status'] == 'complete' for r in rows),
                               'success': sum(r['condition'] == name and r['status'] == 'complete' and r['success'] for r in rows)}
                         for name in CONDITIONS}, 'updated_unix': time.time()})


def run_condition(condition: str, cases: list[dict], out: Path, protocol: dict, max_new: int | None):
    identity = protocol['model_identities'][condition]
    adapter = identity.get('adapter_path')
    digest = identity.get('adapter_sha256', identity.get('model_sha256'))
    model_path = Path(adapter) if adapter else Path(protocol['model_identities']['frozen']['model_path'])
    model_file = model_path / ('adapter_model.safetensors' if adapter else 'model.safetensors')
    if sha256(model_file) != digest:
        raise ValueError(f'Model drift: {condition}')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    policy, pre, post = load_policy(Path(adapter) if adapter else None, train=False, lora=False)
    policy = policy.to(dtype=torch.float32)
    floating_tensors = list(policy.parameters()) + list(policy.buffers())
    floating_dtypes = sorted({str(tensor.dtype) for tensor in floating_tensors
                              if tensor.is_floating_point()})
    if floating_dtypes != ['torch.float32'] or torch.is_autocast_enabled('cuda'):
        raise ValueError(f'Formal FP32 Eager runtime not established: {floating_dtypes}')
    base = policy.get_base_model() if adapter else policy
    measured = ChunkProfiler(base, torch.cuda.synchronize)
    settings = json.loads((ENGINE / 'configs/data_protocol_v1.json').read_text())
    settings['evaluation']['initial_state_indices'] = list(range(40, 50)) if out.name == 'formal' else [cases[0]['init_state_id']]
    settings['evaluation']['episodes_per_task'] = 10 if out.name == 'formal' else 1
    if settings['evaluation']['control_freq'] != 20:
        raise ValueError('Control frequency drift')
    suite_classes = benchmark.get_benchmark_dict()
    suites = {short: suite_classes[full](task_order_index=settings['task_order_index']) for short, full in SUITES.items()}
    registry = Registry(out / 'registry')
    with registry.run({'scope': 'M06-independent' if out.name == 'formal' else 'M06-dev-smoke',
                       'condition': condition, 'backend': 'fp32_eager', 'checkpoint_sha256': digest,
                       'protocol_sha256': sha256(PROTO)}) as registry_run, \
         ResourceMonitor(registry_run, interval=2.0, gpu=True), torch.inference_mode():
        new = 0
        for case in cases:
            name = f"{condition}_{case['suite']}_t{case['task_id']:02d}_i{case['init_state_id']:02d}"
            receipt_path = out / 'receipts' / f'{name}.json'
            if receipt_path.exists():
                prior = json.loads(receipt_path.read_text())
                if valid_receipt(prior, case, condition, digest, sha256(PROTO)):
                    continue
                if prior.get('status') != 'error':
                    raise ValueError(f'Invalid completed receipt, do not silently overwrite: {receipt_path}')
            if max_new is not None and new >= max_new:
                break
            new += 1
            suite = suites[case['suite']]
            task = suite.get_task(int(case['task_id']))
            seed = int(case['seed'])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
            horizon = settings['evaluation']['horizons'][SUITES[case['suite']]]
            if out.name != 'formal':
                horizon = min(8, horizon)
            video_path = registry_run.directory / 'videos' / f'{name}.mp4'
            video_path.parent.mkdir(parents=True, exist_ok=True)
            env = container = stream = None
            action_trace = []
            input_shapes = {}
            started = time.perf_counter()
            append_attempt(out / 'attempts.jsonl', {'case': name, 'event': 'started', 'unix': time.time()})
            try:
                reference = out / 'initial_references' / f"{case['suite']}_t{case['task_id']:02d}_i{case['init_state_id']:02d}.npz"
                env = CapturedEnv(create_libero_env(LiberoEnv, suite, case['task_id'],
                    SUITES[case['suite']], settings, case['init_state_id'], 256, horizon),
                    verifier=lambda sample: compare_reference(reference, sample,
                        protocol['rules']['state_match_tolerance'], create=condition == 'frozen'))
                container = av.open(str(video_path), 'w')
                def process(obs):
                    batch = libero_batch(obs)
                    batch['observation.state'], _ = fit_libero_state_to_checkpoint(
                        batch['observation.state'], expected_dim=6, normalizer_dim=8)
                    batch['task'] = [task.language]
                    if not input_shapes:
                        input_shapes.update({key: list(value.shape) for key, value in batch.items()
                                             if hasattr(value, 'shape')})
                    return pre(batch)
                def output_action(action):
                    raw = action.detach().float().cpu().numpy()[0]
                    command = post(action).detach().float().cpu().numpy()[0]
                    clipped = np.clip(command, -1, 1).astype(np.float32)
                    action_trace.append({'raw': raw.tolist(), 'command': command.tolist(), 'executed': clipped.tolist()})
                    return clipped
                def video_sink(obs, step):
                    nonlocal stream
                    frames = [np.flip(obs['pixels'][key], axis=(0, 1)).copy() for key in ('image', 'image2')]
                    frame = np.concatenate(frames, axis=1)
                    if stream is None:
                        stream = container.add_stream('libx264', rate=settings['evaluation']['control_freq'])
                        stream.width, stream.height, stream.pix_fmt = frame.shape[1], frame.shape[0], 'yuv420p'
                    for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
                        container.mux(packet)
                result = record_episode(env, measured, registry_run,
                    suite=case['suite'], task_id=case['task_id'], episode_id=case['init_state_id'], seed=seed,
                    instruction=task.language, max_steps=horizon, preprocess=process, postprocess=output_action,
                    synchronize=torch.cuda.synchronize,
                    state_extract=lambda obs: libero_batch(obs)['observation.state'][0].numpy(),
                    observation_sink=video_sink,
                    environment_config={'control_mode': 'relative', 'initial_state': case['init_state_id'],
                                        'suite': SUITES[case['suite']]},
                    policy_metadata={'condition': condition, 'backend': 'fp32_eager', 'checkpoint_sha256': digest})
                if stream is None or not result.get('chunk_calls'):
                    raise ValueError('No real model chunk or RGB frames')
                for packet in stream.encode():
                    container.mux(packet)
                container.close()
                container = None
                if len(env.samples) != result['episode_length'] + 1:
                    raise ValueError('Physical state capture missing steps')
                match = env.initial_match
                if not match or not match['matched']:
                    raise ValueError('Initial physical state was not matched before actions')
                state_path = registry_run.directory / f'{name}_sim_states.npz'
                np.savez_compressed(state_path, **_state_arrays(env.samples))
                trace_path = registry_run.directory / f'{name}_actions.json'
                atomic_json(trace_path, {'actions': action_trace, 'input_shapes': input_shapes,
                                         'action_chunk_size': 50})
                with np.load(result['trajectory_path'], allow_pickle=False) as trace:
                    actions, states = trace['actions'], trace['states']
                if (not np.array_equal(actions, np.asarray([a['executed'] for a in action_trace], dtype=np.float32))
                        or states.shape != (len(actions) + 1, 8)):
                    raise ValueError('Executed action/state trajectory mismatch')
                chunks = [v['latency_ms'] for v in result['chunk_calls']]
                receipt = {**case, 'condition': condition, 'status': 'complete',
                    'success': bool(result['success']), 'steps': len(actions),
                    'termination_reason': result['termination_reason'], 'instruction': task.language,
                    'wall_s': time.perf_counter() - started,
                    'chunk_p50_ms': float(np.percentile(chunks, 50)),
                    'chunk_p95_ms': float(np.percentile(chunks, 95)),
                    'chunk_calls': len(chunks), 'gpu_peak_allocated': int(torch.cuda.max_memory_allocated()),
                    'gpu_peak_reserved': int(torch.cuda.max_memory_reserved()),
                    'checkpoint_sha256': digest, 'protocol_sha256': sha256(PROTO),
                    'floating_parameter_dtypes': floating_dtypes,
                    'tf32_allowed': False, 'registry_run_id': registry_run.run_id, 'initial_match': match,
                    'input_shapes': input_shapes, 'latency': result['latency'],
                    'video': {'path': str(video_path), 'sha256': sha256(video_path)},
                    'trajectory': {'path': result['trajectory_path'], 'sha256': sha256(Path(result['trajectory_path']))},
                    'action_trace': {'path': str(trace_path), 'sha256': sha256(trace_path)},
                    'sim_states': {'path': str(state_path), 'sha256': sha256(state_path)}}
                atomic_json(receipt_path, receipt)
                append_attempt(out / 'attempts.jsonl', {'case': name, 'event': 'complete',
                               'success': receipt['success'], 'steps': receipt['steps'], 'unix': time.time()})
                print(json.dumps({'case': name, 'success': receipt['success'], 'steps': receipt['steps']}), flush=True)
            except Exception as exc:
                atomic_json(receipt_path, {**case, 'condition': condition, 'status': 'error',
                    'success': None, 'checkpoint_sha256': digest, 'protocol_sha256': sha256(PROTO),
                    'error': repr(exc), 'traceback': traceback.format_exc()})
                append_attempt(out / 'attempts.jsonl', {'case': name, 'event': 'error',
                               'error': repr(exc), 'unix': time.time()})
                print(json.dumps({'case': name, 'error': repr(exc)}), flush=True)
                raise
            finally:
                if container is not None:
                    container.close()
                if env is not None:
                    env.close()
                write_summary(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-smoke', action='store_true')
    parser.add_argument('--dev-init', type=int, choices=range(10, 14), default=10)
    parser.add_argument('--max-new-per-condition', type=int)
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    args = parser.parse_args()
    protocol = json.loads(PROTO.read_text())
    if protocol['status'] != 'FROZEN_NOT_EXECUTED' or len(protocol['cases']) != 400:
        raise ValueError('M06 frozen protocol missing or invalid')
    out = EVIDENCE / ('dev_smoke_fp32' if args.dev_smoke else 'formal')
    out.joinpath('receipts').mkdir(parents=True, exist_ok=True)
    cases = ([{'suite': 'goal', 'task_id': 6, 'init_state_id': args.dev_init,
               'source_group': f'goal:6:{args.dev_init}', 'seed': 515500 + args.dev_init,
               'task_name': 'put_the_cream_cheese_in_the_bowl'}]
             if args.dev_smoke else protocol['cases'])
    with LOCK.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for condition in args.conditions:
            run_condition(condition, cases, out, protocol, args.max_new_per_condition)
    write_summary(out)

if __name__ == '__main__':
    main()
