"""Export a physically replay-verified M04 correction as a local LeRobot v3 dataset."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor.env_processor import LiberoProcessorStep

ROOT = Path('/data/VLA-Forge')
EVIDENCE = ROOT / 'Evidence/M04'
SOURCE_NPZ = EVIDENCE / 'artifacts/first_correction/m04-first-correction-replay-01.npz'
SOURCE_JSON = EVIDENCE / 'artifacts/first_correction/m04-first-correction-replay-01.json'
DEST = EVIDENCE / 'dataset/lerobot_v3'
REPO_ID = 'vla_forge/m04_verified_corrections'
FPS = 20


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    result=json.loads(SOURCE_JSON.read_text())
    if result.get('success') is not True:
        raise RuntimeError('source rollout did not end with verified success')
    if result.get('correction_actions_executed') != 99 or result.get('prefix_steps') != 100:
        raise RuntimeError('unexpected source transition boundaries')
    with np.load(SOURCE_NPZ) as z:
        frames=z['dual_camera_frames'].copy()
        actions=z['actions'].copy()
        eef_pos=z['eef_pos'].copy()
        eef_quat=z['eef_quat'].copy()
        gripper=z['gripper_qpos'].copy()
    start=int(result['prefix_steps'])
    n=int(result['correction_actions_executed'])
    if frames.shape[0] != actions.shape[0]+1 or frames.shape[0] != eef_pos.shape[0]:
        raise RuntimeError('transition/observation alignment is not T actions and T+1 observations')
    if start+n != actions.shape[0] or frames.shape[1:] != (256,512,3):
        raise RuntimeError('source episode dimensions differ from the audited contract')
    # LeRobot's pinned official LIBERO processor defines (x,y,z,axis-angle xyz,gripper qpos[2]).
    processor=LiberoProcessorStep()
    quat=torch.from_numpy(eef_quat[start:start+n].astype(np.float32))
    aa=processor._quat2axisangle(quat).cpu().numpy()
    state=np.concatenate([eef_pos[start:start+n].astype(np.float32),aa,gripper[start:start+n].astype(np.float32)],axis=-1)
    obs_frames=frames[start:start+n]
    action_rows=actions[start:start+n].astype(np.float32)
    if state.shape != (n,8) or action_rows.shape != (n,7):
        raise RuntimeError(f'bad state/action shape: {state.shape}, {action_rows.shape}')
    if not np.isfinite(state).all() or not np.isfinite(action_rows).all():
        raise RuntimeError('nonfinite dataset values')
    if DEST.exists():
        raise FileExistsError(f'refusing to overwrite existing dataset: {DEST}')
    features={
        'observation.state': {'dtype':'float32','shape':(8,), 'names':None},
        'observation.images.agentview_image': {'dtype':'video','shape':(256,256,3),'names':['height','width','channel']},
        'observation.images.robot0_eye_in_hand_image': {'dtype':'video','shape':(256,256,3),'names':['height','width','channel']},
        'action': {'dtype':'float32','shape':(7,), 'names':None},
        'next.success': {'dtype':'bool','shape':(1,), 'names':None},
        'next.done': {'dtype':'bool','shape':(1,), 'names':None},
    }
    DEST.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    ds=LeRobotDataset.create(repo_id=REPO_ID,root=DEST,fps=FPS,features=features,use_videos=True,image_writer_threads=2,video_backend='pyav')
    for i in range(n):
        frame={
            'observation.state':state[i],
            'observation.images.agentview_image':obs_frames[i,:,:256,:],
            'observation.images.robot0_eye_in_hand_image':obs_frames[i,:,256:,:],
            'action':action_rows[i],
            'next.success':np.asarray([i==n-1],dtype=np.bool_),
            'next.done':np.asarray([i==n-1],dtype=np.bool_),
            'task':result['instruction'],
        }
        ds.add_frame(frame)
    ds.save_episode(parallel_encoding=True)
    encode_seconds=time.perf_counter()-started
    # Close writer resources, then independently load through the public dataset reader.
    if hasattr(ds,'finalize'):
        ds.finalize()
    del ds
    loaded=LeRobotDataset(REPO_ID,root=DEST,download_videos=False,video_backend='pyav',return_uint8=True)
    if len(loaded)!=n or loaded.num_episodes!=1:
        raise RuntimeError(f'LeRobot v3 reload mismatch: rows={len(loaded)} episodes={loaded.num_episodes}')
    sampled=[0,n//2,n-1]
    checks=[]
    for i in sampled:
        row=loaded[i]
        got=np.asarray(row['action']).reshape(-1)
        err=float(np.max(np.abs(got-action_rows[i])))
        if err>1e-6:
            raise RuntimeError(f'action roundtrip mismatch at {i}: {err}')
        checks.append({'index':i,'action_linf':err,'success':bool(np.asarray(row['next.success']).reshape(-1)[0]),'done':bool(np.asarray(row['next.done']).reshape(-1)[0])})
    output_bytes=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())
    receipt={
        'schema':'vla-forge.m04.lerobot-v3-export.v1','repo_id':REPO_ID,'root':str(DEST),'dataset_version':loaded.meta.info.get('codebase_version'),'episodes':loaded.num_episodes,'transitions':len(loaded),'fps':FPS,
        'source':{'episode_id':result['episode_id'],'source_m02_episode':result['source_m02_episode'],'source_m02_action_trace_sha256':result['source_m02_trace_sha256'],'m03_correction_episode':result['correction_source_m03_episode'],'m03_action_sha256':result['correction_source_m03_action_sha256'],'m04_replay_npz':str(SOURCE_NPZ),'m04_replay_npz_sha256':sha256(SOURCE_NPZ),'m04_replay_json_sha256':sha256(SOURCE_JSON),'transition_range':{'source_action_start':start,'count':n,'observations':'pre-action frames 100..198','success_terminal_observation_index':start+n}},
        'feature_contract':{k:{'dtype':v['dtype'],'shape':list(v['shape'])} for k,v in features.items()},
        'alignment':'T correction actions with T pre-action observations; final action row carries measured terminal success/done; all observations and actions are from independent M04 MuJoCo replay.',
        'validation':{'reloaded':True,'sample_roundtrip':checks,'video_backend':'pyav','dataset_bytes':output_bytes,'write_and_encode_seconds':encode_seconds,'transitions_per_second':n/encode_seconds},
        'evidence_boundary':'one unique verified corrected trajectory; no duplicate replay added as a second sample; correction actions originate in the M03 frozen SmolVLA brightness treatment and were physically re-executed in M04.'
    }
    (DEST/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    main()
