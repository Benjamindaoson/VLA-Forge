"""Independently audit the exported M04 LeRobot v3 correction dataset."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import av
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.env_processor import LiberoProcessorStep

ROOT = Path('/data/VLA-Forge')
EVIDENCE = ROOT / 'Evidence/M04'
DATASET = EVIDENCE / 'dataset/lerobot_v3'
SOURCE_NPZ = EVIDENCE / 'artifacts/first_correction/m04-first-correction-replay-01.npz'
SOURCE_JSON = EVIDENCE / 'artifacts/first_correction/m04-first-correction-replay-01.json'
OUT = EVIDENCE / 'dataset/lerobot_v3_validation.json'
REPO_ID = 'vla_forge/m04_verified_corrections'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def decode(path: Path) -> tuple[int, tuple[int, int]]:
    count = 0
    shape = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            arr = frame.to_ndarray(format='rgb24')
            shape = arr.shape[:2]
            count += 1
    return count, shape


def main() -> None:
    source = json.loads(SOURCE_JSON.read_text())
    if source.get('success') is not True:
        raise AssertionError('source episode is not physically success-verified')
    with np.load(SOURCE_NPZ) as z:
        frames = z['dual_camera_frames'].copy()
        actions = z['actions'].copy()
        positions = z['eef_pos'].copy()
        quats = z['eef_quat'].copy()
        grippers = z['gripper_qpos'].copy()
    start = int(source['prefix_steps'])
    n = int(source['correction_actions_executed'])
    processor = LiberoProcessorStep()
    axis_angle = processor._quat2axisangle(torch.from_numpy(quats[start:start+n].astype(np.float32))).cpu().numpy()
    expected_state = np.concatenate([positions[start:start+n], axis_angle, grippers[start:start+n]], axis=-1).astype(np.float32)
    ds = LeRobotDataset(REPO_ID, root=DATASET, download_videos=False, video_backend='pyav', return_uint8=True)
    if len(ds) != n or ds.num_episodes != 1:
        raise AssertionError(f'frame coverage mismatch: {len(ds)} != {n}')
    state_err = action_err = 0.0
    image_error = {'agentview': [], 'wrist': []}
    success_indices, done_indices = [], []
    checks = []
    for i in range(n):
        row = ds[i]
        state = np.asarray(row['observation.state'], dtype=np.float32)
        action = np.asarray(row['action'], dtype=np.float32)
        state_e = float(np.max(np.abs(state - expected_state[i])))
        action_e = float(np.max(np.abs(action - actions[start+i])))
        state_err = max(state_err, state_e)
        action_err = max(action_err, action_e)
        if state_e > 1e-6 or action_e > 1e-6:
            raise AssertionError(f'numeric roundtrip failed at {i}: state={state_e} action={action_e}')
        success = bool(np.asarray(row['next.success']).item())
        done = bool(np.asarray(row['next.done']).item())
        if success:
            success_indices.append(i)
        if done:
            done_indices.append(i)
        for key, src, label in [
            ('observation.images.agentview_image', 0, 'agentview'),
            ('observation.images.robot0_eye_in_hand_image', 1, 'wrist'),
        ]:
            got = np.asarray(row[key])
            if got.shape != (3, 256, 256) or got.dtype != np.uint8:
                raise AssertionError(f'{key} has invalid shape/dtype: {got.shape}/{got.dtype}')
            expected = frames[start+i, :, src*256:(src+1)*256, :].transpose(2, 0, 1)
            err = float(np.abs(got.astype(np.float32) - expected.astype(np.float32)).mean())
            image_error[label].append(err)
            if not np.any(got):
                raise AssertionError(f'{key} all-black at {i}')
        if i in (0, n//2, n-1):
            checks.append({'index': i, 'state_linf': state_e, 'action_linf': action_e,
                           'success': success, 'done': done})
    expected_success = [n-1]
    if success_indices != expected_success or done_indices != [n-1]:
        raise AssertionError(f'wrong terminal markers: success={success_indices}, done={done_indices}')
    video_receipts = []
    for key in ['observation.images.agentview_image', 'observation.images.robot0_eye_in_hand_image']:
        path = DATASET / 'videos' / key / 'chunk-000/file-000.mp4'
        count, shape = decode(path)
        if count != n or shape != (256, 256):
            raise AssertionError(f'video decode failed for {key}: {count} frames, {shape}')
        video_receipts.append({'feature': key, 'path': str(path), 'frames': count, 'frame_shape_hw': shape,
                               'bytes': path.stat().st_size, 'sha256': sha(path)})
    result = {
        'schema': 'vla-forge.m04.lerobot-v3-independent-validation.v1',
        'repo_id': REPO_ID, 'dataset_root': str(DATASET), 'codebase_version': ds.meta.info.get('codebase_version'),
        'source_episode_id': source['episode_id'], 'source_npz_sha256': sha(SOURCE_NPZ),
        'episodes': ds.num_episodes, 'transitions': len(ds), 'fps': ds.meta.info.fps,
        'state_max_linf': state_err, 'action_max_linf': action_err,
        'image_mean_abs_error_vs_source_uint8': {k: float(np.mean(v)) for k,v in image_error.items()},
        'success_indices': success_indices, 'done_indices': done_indices,
        'sample_checks': checks, 'decoded_videos': video_receipts,
        'pass': True,
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
