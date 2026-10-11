"""Render an audited, private M05 demo exclusively from real MuJoCo recordings."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from build_m04_demo import m02_clips, m04_clips

EVIDENCE = Path('/data/VLA-Forge/Evidence/M05')
OUT = EVIDENCE / 'demo'
VIDEO = OUT / 'M05_demo.mp4'
W, H, FPS = 1920, 1080, 20
CONDITIONS = ('frozen', 'standard', 'corrective', 'replay')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict:
    result = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,pix_fmt,r_frame_rate',
        '-show_entries', 'format=duration', '-of', 'json', str(path),
    ], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return {**data['streams'][0], 'duration_s': float(data['format']['duration'])}


def m05_receipts() -> dict:
    root = EVIDENCE / 'development/receipts'
    protocol = json.loads(Path('configs/m05/formal_protocol.json').read_text())
    expected = {(c['suite'], int(c['task']), int(c['init'])) for c in protocol['development']}
    if len(expected) != 8:
        raise ValueError('Expected exactly eight frozen development cases')
    grouped = {}
    for path in root.glob('*.json'):
        row = json.loads(path.read_text())
        key = (row['suite'], int(row['task']), int(row['init']))
        if row['status'] != 'complete' or key not in expected:
            raise ValueError(f'Invalid development receipt: {path}')
        source = Path(row['video']['path'])
        if not source.is_file() or sha(source) != row['video']['sha256']:
            raise ValueError(f'Video missing or hash mismatch: {path}')
        group_key = (row['condition'], key)
        if group_key in grouped:
            raise ValueError(f'Duplicate development case: {group_key}')
        grouped[group_key] = {**row, 'source_id': path.stem, 'source_path': str(source),
                              'origin': 'M05 same-init development rollout'}
    if len(grouped) != 32 or any((condition, key) not in grouped for key in expected for condition in CONDITIONS):
        raise ValueError(f'32/32 development videos required; currently {len(grouped)}')
    return grouped


def source_entry(raw: dict, section: str) -> dict:
    path = Path(raw['source_path'])
    meta = probe(path)
    if meta['codec_name'] not in ('h264', 'mpeg4') or meta['duration_s'] < 2:
        raise ValueError(f'Invalid raw recording: {path}')
    return {
        'source_id': raw['source_id'], 'origin': raw['origin'],
        'source_path': str(path), 'source_sha256': sha(path),
        'duration_s': meta['duration_s'], 'fps': meta['r_frame_rate'],
        'suite': raw.get('suite'), 'task': raw.get('task', raw.get('task_id')),
        'init': raw.get('init'), 'seed': raw.get('seed'),
        'checkpoint_sha256': raw.get('checkpoint_sha256'),
        'instruction': raw.get('instruction'), 'success': bool(raw['success']),
        'condition': raw.get('condition'), 'section': section,
    }


def plan() -> list[dict]:
    grouped = m05_receipts()
    used_hashes: set[str] = set()
    segments = []

    def add_solo(raw: dict, section: str, max_seconds: float) -> None:
        # Some genuine M04 attempts terminate in under two seconds; retain them
        # as raw evidence but omit them from the narrated demo.
        if probe(Path(raw['source_path']))['duration_s'] < 2:
            return
        entry = source_entry(raw, section)
        if entry['source_sha256'] in used_hashes:
            return
        used_hashes.add(entry['source_sha256'])
        length = min(max_seconds, entry['duration_s'] - 0.05)
        if length < 2:
            return
        start = max(0.0, (entry['duration_s'] - length) / 2)
        segments.append({'kind': 'solo', 'section': section, 'seconds': length,
                         'sources': [{**entry, 'clip_start_s': start, 'clip_duration_s': length}]})

    for raw in m02_clips():
        add_solo(raw, 'M02 frozen-policy benchmark context', 5.5)
    for raw in m04_clips():
        add_solo(raw, 'M04 verified corrections and bounded failures', 6.5)
    protocol = json.loads(Path('configs/m05/formal_protocol.json').read_text())
    for first, second in (('frozen', 'standard'), ('corrective', 'replay')):
        for case in protocol['development']:
            key = (case['suite'], int(case['task']), int(case['init']))
            left = source_entry(grouped[(first, key)], 'M05 matched development comparison')
            right = source_entry(grouped[(second, key)], 'M05 matched development comparison')
            if left['source_sha256'] in used_hashes or right['source_sha256'] in used_hashes:
                raise ValueError('M05 comparison would repeat a source recording')
            used_hashes.update((left['source_sha256'], right['source_sha256']))
            length = min(7.0, left['duration_s'] - 0.05, right['duration_s'] - 0.05)
            if length < 2:
                raise ValueError('Matched recording is too short')
            left['clip_start_s'] = max(0.0, (left['duration_s'] - length) / 2)
            right['clip_start_s'] = max(0.0, (right['duration_s'] - length) / 2)
            for entry in (left, right):
                entry['clip_duration_s'] = length
            segments.append({'kind': 'pair', 'section': 'M05 matched development comparison',
                             'seconds': length, 'sources': [left, right]})
    # Fill missing time with *new* M02 recordings, never replaying a clip.
    if sum(s['seconds'] for s in segments) < 184:
        run = Path('/data/VLA-Forge/Evidence/M02/source/M02_libero40/registry/runs/20261010T030315-982cacc8515f')
        for path in sorted((run / 'episodes').glob('*/episode.json')):
            meta = json.loads(path.read_text())
            video = run / 'videos' / f'{path.parent.name}.mp4'
            if not video.is_file():
                continue
            raw = {'source_id': path.parent.name, 'origin': 'M02 real LIBERO baseline context',
                   'source_path': str(video), 'suite': meta['suite'], 'task_id': meta['task_id'],
                   'success': bool(meta['success']), 'condition': 'frozen SmolVLA'}
            add_solo(raw, 'M02 benchmark context', 5.5)
            if sum(s['seconds'] for s in segments) >= 190:
                break
    # Three evidence plots accompany three additional, unique moving MuJoCo clips.
    source_root = Path('/data/VLA-Forge/Evidence/M02/source/M02_libero40/registry/runs/20261010T030315-982cacc8515f')
    episode_files = iter(sorted((source_root / 'episodes').glob('*/episode.json')))
    for chart in ('training_loss', 'inference_latency', 'gpu_telemetry'):
        for episode_file in episode_files:
            episode = json.loads(episode_file.read_text())
            source = source_root / 'videos' / f'{episode_file.parent.name}.mp4'
            if not source.is_file() or probe(source)['duration_s'] < 5.6:
                continue
            raw = {'source_id': episode_file.parent.name, 'origin': 'M02 real LIBERO chart context',
                   'source_path': str(source), 'suite': episode['suite'],
                   'task_id': episode['task_id'], 'success': bool(episode['success']),
                   'condition': 'frozen SmolVLA'}
            entry = source_entry(raw, 'M05 real measured systems results')
            if entry['source_sha256'] in used_hashes:
                continue
            used_hashes.add(entry['source_sha256'])
            entry['clip_start_s'] = max(0.0, (entry['duration_s'] - 5.5) / 2)
            entry['clip_duration_s'] = 5.5
            segments.append({'kind': 'quant', 'chart': chart,
                             'section': 'M05 real measured systems results',
                             'seconds': 5.5, 'sources': [entry]})
            break
        else:
            raise ValueError(f'No unused real MuJoCo clip for chart {chart}')
    duration = sum(round(s['seconds'] * FPS) / FPS for s in segments)
    if not 180 <= duration <= 240:
        raise ValueError(f'Honest unique-footage duration {duration:.1f}s outside 180-240s')
    return segments


def label(frame: np.ndarray, message: str, y: int, scale: float = 0.75) -> None:
    cv2.putText(frame, message[:95], (45, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (240, 245, 245), 2, cv2.LINE_AA)


def compose(frames: list[np.ndarray], segment: dict, stats: dict) -> np.ndarray:
    canvas = np.full((H, W, 3), (34, 38, 43), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (W, 170), (18, 22, 27), -1)
    label(canvas, 'VLA-FORGE M05  |  real SmolVLA + LIBERO / MuJoCo', 67, 0.95)
    label(canvas, segment['section'], 126, 0.72)
    if segment['kind'] == 'quant':
        source = segment['sources'][0]
        canvas[245:875, 35:955] = cv2.resize(frames[0], (920, 630), interpolation=cv2.INTER_AREA)
        canvas[245:875, 990:1875] = cv2.resize(stats['charts'][segment['chart']],
                                                (885, 630), interpolation=cv2.INTER_AREA)
        label(canvas, f"Real {source['suite']} task {source['task']} | measured {segment['chart']}", 940, 0.72)
        label(canvas, stats['quant_lines'][segment['chart']], 1000, 0.62)
    elif segment['kind'] == 'solo':
        source = segment['sources'][0]
        view = cv2.resize(frames[0], (1720, 860), interpolation=cv2.INTER_AREA)
        canvas[180:1040, 100:1820] = view
        detail = f"{source['source_id']} | {source['suite']} task {source['task']} | {'SUCCESS' if source['success'] else 'FAILURE'}"
        cv2.rectangle(canvas, (0, 1015), (W, H), (18, 22, 27), -1)
        label(canvas, detail, 1055, 0.67)
    else:
        left, right = segment['sources']
        for idx, frame in enumerate(frames):
            view = cv2.resize(frame, (900, 450), interpolation=cv2.INTER_AREA)
            canvas[300:750, 30 + idx * 960:930 + idx * 960] = view
        label(canvas, f"{left['condition'].upper()}  |  {'SUCCESS' if left['success'] else 'FAILURE'}", 210, 0.70)
        cv2.putText(canvas, f"{right['condition'].upper()}  |  {'SUCCESS' if right['success'] else 'FAILURE'}",
                    (1005, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (240, 245, 245), 2, cv2.LINE_AA)
        label(canvas, f"Matched {left['suite']} task {left['task']} / init {left['init']}  |  same frozen evaluation protocol", 925, 0.70)
        label(canvas, 'Development set only; paired observations do not establish cross-task generalization.', 1000, 0.58)
    label(canvas, stats['footer'], H - 18, 0.52)
    return canvas


def read_frames(source: dict, count: int):
    cap = cv2.VideoCapture(source['source_path'])
    if not cap.isOpened():
        raise ValueError(f"Cannot decode {source['source_path']}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(source['clip_start_s'] * fps))
    try:
        for index in range(count):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Premature EOF {source['source_id']} at {index}/{count}")
            yield frame
    finally:
        cap.release()


def render(segments: list[dict]) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'visualization').mkdir(exist_ok=True)
    chart_dir = OUT / 'charts'
    dev = json.loads((EVIDENCE / 'development/metrics.json').read_text())['conditions']
    inference = json.loads((EVIDENCE / 'inference/matrix.json').read_text())['modes']
    eager_ms = inference['eager']['chunk_latency']['p50_s'] * 1000
    graph_ms = inference['cuda_graph']['chunk_latency']['p50_s'] * 1000
    stats = {
        'quant_lines': {
            'training_loss': '500 real optimizer updates per adapter; fixed development set only.',
            'inference_latency': f'Corrective warm action-chunk P50: {eager_ms:.0f} ms Eager / {graph_ms:.0f} ms Graph.',
            'gpu_telemetry': 'Development success: ' + ', '.join(
                f"{name} {dev[name]['success']}/{dev[name]['total']}"
                for name in ('frozen', 'standard', 'corrective', 'replay')),
        },
        'footer': 'M05 development: 8 fixed initial states x 4 strategies; M06 locked 400 test episodes remain unopened.',
        'charts': {name: cv2.imread(str(chart_dir / f'{name}.png'))
                   for name in ('training_loss', 'inference_latency', 'gpu_telemetry')},
    }
    if any(image is None for image in stats['charts'].values()):
        raise ValueError('Generate evidence charts before video rendering')
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
               '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-an', '-r', '30', '-c:v', 'libx264',
               '-preset', 'veryfast', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(VIDEO)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    cursor = 0.0
    for segment in segments:
        count = round(segment['seconds'] * FPS)
        streams = [read_frames(source, count) for source in segment['sources']]
        for i in range(count):
            frame = compose([next(stream) for stream in streams], segment, stats)
            if cursor == 0 and i == 10:
                cv2.imwrite(str(OUT / 'M05_overview.png'), frame)
            if segment['kind'] == 'pair' and not (OUT / 'M05_comparison.png').exists() and i == 10:
                cv2.imwrite(str(OUT / 'M05_comparison.png'), frame)
            encoder.stdin.write(frame.tobytes())
        segment['output_start_s'] = cursor
        cursor += count / FPS
        segment['output_end_s'] = cursor
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('ffmpeg H.264 encoder failed')
    meta = probe(VIDEO)
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(VIDEO), '-f', 'null', '-'],
                             capture_output=True, text=True)
    if decoded.returncode or not 180 <= meta['duration_s'] <= 240 or meta['codec_name'] != 'h264' or meta['pix_fmt'] != 'yuv420p' or meta['width'] != W or meta['height'] != H:
        raise ValueError(f"Demo contract failed: {meta}; decode={decoded.stderr[-500:]}")
    source_count = len({s['source_sha256'] for segment in segments for s in segment['sources']})
    if source_count != sum(len(s['sources']) for s in segments):
        raise ValueError('Source recording reused in demo')
    charts = [{'path': str(chart_dir / f'{name}.png'),
               'sha256': sha(chart_dir / f'{name}.png')}
              for name in ('training_loss', 'inference_latency', 'gpu_telemetry')]
    sources = {'segments': segments, 'unique_source_count': source_count,
               'charts': charts, 'chart_source_manifest': str(chart_dir / 'source_manifest.json'),
               'scope': 'Private evidence. All robot pixels derive from hashed LIBERO/MuJoCo recordings.'}
    (OUT / 'video_sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    validation = {'path': str(VIDEO), 'sha256': sha(VIDEO), 'bytes': VIDEO.stat().st_size,
                  'duration_s': meta['duration_s'], 'width': meta['width'], 'height': meta['height'],
                  'codec': meta['codec_name'], 'pixel_format': meta['pix_fmt'],
                  'fps': meta['r_frame_rate'], 'full_decode_exit_code': decoded.returncode,
                  'unique_source_count': source_count, 'segment_count': len(segments)}
    (OUT / 'video_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    bundle = OUT / 'visualization/sources'
    bundle.mkdir(exist_ok=True)
    for segment in segments:
        for source in segment['sources']:
            bundled = bundle / f"{source['source_sha256'][:24]}.mp4"
            if not bundled.exists():
                shutil.copy2(source['source_path'], bundled)
            if sha(bundled) != source['source_sha256']:
                raise ValueError(f'Bundled source differs from original: {bundled}')
            source['bundle_path'] = str(bundled)
    (OUT / 'video_sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    rows = []
    for idx, segment in enumerate(segments, 1):
        ids = ', '.join(f"<a href='sources/{source['source_sha256'][:24]}.mp4'>{html.escape(source['source_id'])}</a>" for source in segment['sources'])
        rows.append(f"<tr><td>{idx}</td><td>{html.escape(segment['section'])}</td><td>{ids}</td><td>{segment['output_start_s']:.1f}-{segment['output_end_s']:.1f}s</td></tr>")
    page = """<!doctype html><meta charset='utf-8'><title>VLA-Forge M05 private demo</title><style>body{font:16px system-ui;background:#f5f7f8;color:#17232c;max-width:1120px;margin:36px auto}main{background:white;padding:28px;border-radius:12px}video,img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style><main><h1>VLA-Forge M05 — real development evidence</h1><p>Private, eight fixed LIBERO development initial states. This is not an M06 release result. Each robot shot is traced to a hashed source recording.</p><video controls preload='metadata' src='../M05_demo.mp4'></video><p><a href='../video_sources.json'>Source manifest</a> · <a href='../video_validation.json'>Validation</a></p><h2>Matched comparison</h2><img src='../M05_comparison.png'><h2>Measurements from raw logs</h2><p><img src='../charts/training_loss.png' alt='Training loss'><img src='../charts/inference_latency.png' alt='Inference latency'><img src='../charts/gpu_telemetry.png' alt='GPU telemetry'></p><h2>Timeline and original videos</h2><table><tr><th>#</th><th>Section</th><th>Source IDs</th><th>Time</th></tr>""" + ''.join(rows) + '</table></main>'
    (OUT / 'visualization/index.html').write_text(page)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    segments = plan()
    planned = sum(round(s['seconds'] * FPS) for s in segments) / FPS
    print(json.dumps({'planned_duration_s': planned, 'segments': len(segments),
                      'unique_sources': sum(len(s['sources']) for s in segments)}, indent=2))
    if not args.plan_only:
        print(json.dumps(render(segments), indent=2))


if __name__ == '__main__':
    main()
