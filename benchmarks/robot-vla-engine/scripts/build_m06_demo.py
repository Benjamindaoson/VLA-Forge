"""Assemble a private 180-240s technical film from unique real M06 videos."""
from __future__ import annotations

import csv
import html
import json
import os
import subprocess
from pathlib import Path

from robot_vla.m06_protocol import CONDITIONS, EVIDENCE, SUITES, atomic_json, sha256

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
DEMO = EVIDENCE / 'demo'
VIS = EVIDENCE / 'visualization'

def run(command: list[str]):
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

def probe(path: Path) -> dict:
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
        '-show_format', '-of', 'json', str(path)]))

def select_cases(protocol: dict, receipts: dict) -> list[dict]:
    candidates = []
    for case in protocol['cases']:
        key = (case['suite'], case['task_id'], case['init_state_id'])
        rows = [receipts[(condition, *key)] for condition in CONDITIONS]
        duration = min(row['steps'] for row in rows) / 20.0
        if duration < 4:
            continue
        variety = len({row['success'] for row in rows})
        candidates.append({'case': case, 'rows': rows, 'duration': min(10.0, duration - .05),
                           'variety': variety})
    used, chosen = set(), []
    def take(item):
        key = (item['case']['suite'], item['case']['task_id'], item['case']['init_state_id'])
        if key not in used:
            chosen.append(item)
            used.add(key)
    target = sorted((x for x in candidates if x['case']['suite'] == 'goal' and x['case']['task_id'] == 6),
                    key=lambda x: (-x['variety'], -x['duration']))
    for item in target[:3]:
        take(item)
    for suite in SUITES:
        for task in range(10):
            options = [x for x in candidates if x['case']['suite'] == suite and x['case']['task_id'] == task]
            if options:
                take(max(options, key=lambda x: (x['variety'], x['duration'])))
            if sum(x['duration'] for x in chosen) >= 190:
                break
        if sum(x['duration'] for x in chosen) >= 190:
            break
    for item in sorted(candidates, key=lambda x: (-x['variety'], -x['duration'])):
        if sum(x['duration'] for x in chosen) >= 190:
            break
        take(item)
    total = sum(x['duration'] for x in chosen)
    if not 180 <= total <= 240 or len(chosen) < 18:
        raise ValueError(f'Unique real material insufficient for a substantive demo: {len(chosen)} cases, {total:.1f}s')
    return chosen

def make_segment(item: dict, index: int, stats: dict, decisions: dict) -> dict:
    case = item['case']
    rows = item['rows']
    length = item['duration']
    folder = DEMO / 'segments'
    folder.mkdir(parents=True, exist_ok=True)
    labels = []
    for condition, row in zip(CONDITIONS, rows, strict=True):
        label = folder / f'{index:03d}_{condition}.txt'
        label.write_text(f'{condition.upper()}  {"SUCCESS" if row["success"] else "FAIL"}  {row["steps"]} steps', encoding='utf-8')
        labels.append(label)
    title = folder / f'{index:03d}_title.txt'
    title.write_text(f'LIBERO {case["suite"].upper()} / task {case["task_id"]} / init {case["init_state_id"]}  |  Frozen vs 3 LoRA policies', encoding='utf-8')
    caption = folder / f'{index:03d}_caption.txt'
    if (case['suite'], case['task_id']) == ('goal', 6):
        sentence = 'TARGET REPAIR | Same verified physical start | 10 locked initial states'
    else:
        sentence = 'LEGACY SKILL | Matched start | Four FP32 Eager rollouts'
    if index >= 15:
        result = stats['frozen']['overall']
        sentence += f' | Incumbent {result["success"]}/400 | M06 gate: ' + ', '.join(
            f'{name} {decisions[name]["decision"]}' for name in CONDITIONS[1:])
    caption.write_text(sentence, encoding='utf-8')
    streams = []
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y']
    for row in rows:
        command += ['-ss', '0', '-t', f'{length:.3f}', '-i', row['video']['path']]
    for i, label in enumerate(labels):
        streams.append(f'[{i}:v]fps=30,scale=960:480:flags=lanczos,pad=960:540:0:0:black,'
                       f'drawtext=fontfile={FONT}:textfile={label}:x=16:y=492:fontsize=25:'
                       'fontcolor=white:box=1:boxcolor=black@0.75[v'+str(i)+']')
    streams.append('[v0][v1]hstack=inputs=2[top]')
    streams.append('[v2][v3]hstack=inputs=2[bottom]')
    streams.append('[top][bottom]vstack=inputs=2,'
                   f'drawtext=fontfile={FONT}:textfile={title}:x=20:y=8:fontsize=27:'
                   'fontcolor=white:box=1:boxcolor=black@0.78,'
                   f'drawtext=fontfile={FONT}:textfile={caption}:x=20:y=1030:fontsize=22:'
                   'fontcolor=white:box=1:boxcolor=black@0.78[out]')
    segment = folder / f'segment_{index:03d}.mp4'
    command += ['-filter_complex', ';'.join(streams), '-map', '[out]', '-an',
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
                '-pix_fmt', 'yuv420p', '-r', '30', '-movflags', '+faststart', str(segment)]
    run(command)
    sources = [{'condition': condition, 'episode_id': f'{condition}_{case["suite"]}_t{case["task_id"]:02d}_i{case["init_state_id"]:02d}',
                'path': row['video']['path'], 'sha256': row['video']['sha256'],
                'start_s': 0, 'end_s': length, 'success': row['success']} for condition, row in zip(CONDITIONS, rows, strict=True)]
    return {'segment': str(segment), 'segment_sha256': sha256(segment), 'duration_s': length,
            'suite': case['suite'], 'task_id': case['task_id'], 'init_state_id': case['init_state_id'],
            'sources': sources}

def build_html(segments: list[dict], stats: dict, decisions: dict):
    VIS.mkdir(parents=True, exist_ok=True)
    with (EVIDENCE / 'task_metrics.csv').open(newline='') as stream:
        task_rows = list(csv.DictReader(stream))
    task_map = {(row['condition'], row['suite'], int(row['task_id'])): row for row in task_rows}
    table = []
    for suite in SUITES:
        for task in range(10):
            cells = []
            for condition in CONDITIONS:
                count = int(task_map[(condition, suite, task)]['success'])
                cells.append(f'<td><span class="bar" style="width:{count*10}%"></span>{count}/10</td>')
            table.append(f'<tr><td>{html.escape(suite)}</td><td>{task}</td>{"".join(cells)}</tr>')
    links = []
    for segment in segments:
        first = segment['sources'][0]
        rel = os.path.relpath(first['path'], VIS)
        links.append(f'<li>{html.escape(segment["suite"])} task {segment["task_id"]} init {segment["init_state_id"]} — <a href="{html.escape(rel)}">original Frozen video</a></li>')
    cards = ''.join(f'<article><b>{html.escape(name)}</b><span>{stats[name]["overall"]["success"]}/400</span><small>{html.escape(decisions[name]["decision"])}</small></article>' for name in CONDITIONS)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VLA-Forge M06 Independent LIBERO Regression</title><style>body{{font:16px system-ui;margin:auto;max-width:1240px;padding:32px;color:#1c2833;background:#f7f8fa}}h1{{margin-bottom:4px}}video{{width:100%;background:#111}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}}article{{background:white;padding:16px;border:1px solid #dfe3e8}}article span{{display:block;font-size:28px;margin:8px 0}}small{{color:#52616b}}table{{width:100%;border-collapse:collapse;background:white}}.bar{{display:inline-block;background:#6c88ba;height:7px;margin-right:8px}}th,td{{padding:10px;border-bottom:1px solid #e1e5e9;text-align:left}}a{{color:#2557a7}}.note{{background:#fff3d6;padding:14px;border-left:4px solid #c78600}}</style><h1>M06 Independent Robot Policy Regression</h1><p>40 LIBERO tasks × 10 frozen initial states × 4 unchanged policies; matched real MuJoCo rollouts.</p><p class="note">Private license-restricted visual evidence. Independent of recorded M02–M05 project states; foundation-model pretraining overlap remains unknown. No real-robot deployment approval.</p><div class="cards">{cards}</div><video controls preload="metadata" src="../demo/M06_demo_200s.mp4"></video><p><a href="../protocol_frozen.json">Frozen protocol</a> · <a href="../validation.json">Evidence validation</a> · <a href="../statistical_analysis.json">Statistics</a> · <a href="../release_decisions.json">Release decisions</a></p><h2>Task coverage</h2><p>Per-task Success/Total and confidence intervals are in <a href="../task_metrics.csv">task_metrics.csv</a>. Each cell gives true task-level successes among the ten frozen initial states; Wilson intervals are in the CSV.</p><table><thead><tr><th>Suite</th><th>Task</th><th>Frozen</th><th>Standard</th><th>Corrective</th><th>Replay</th></tr></thead><tbody>{''.join(table)}</tbody></table><h2>Original clips</h2><ul>{''.join(links)}</ul></html>'''
    (VIS / 'index.html').write_text(page, encoding='utf-8')

def main():
    validation = json.loads((EVIDENCE / 'validation.json').read_text())
    if validation['status'] != 'PASS' or validation['valid'] != 1600:
        raise ValueError('The M06 demo requires 1600 independently validated episodes')
    protocol = json.loads((EVIDENCE / 'protocol_frozen.json').read_text())
    receipts = {}
    for p in (EVIDENCE / 'formal/receipts').glob('*.json'):
        row = json.loads(p.read_text())
        receipts[(row['condition'], row['suite'], row['task_id'], row['init_state_id'])] = row
    stats = json.loads((EVIDENCE / 'suite_metrics.json').read_text())
    decisions = json.loads((EVIDENCE / 'release_decisions.json').read_text())
    selected = select_cases(protocol, receipts)
    DEMO.mkdir(parents=True, exist_ok=True)
    manifest = [make_segment(item, i, stats, decisions) for i, item in enumerate(selected)]
    concat = DEMO / 'concat.txt'
    concat.write_text(''.join(f"file '{row['segment']}'\n" for row in manifest))
    output = DEMO / 'M06_demo_200s.mp4'
    run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
         '-i', str(concat), '-c', 'copy', '-movflags', '+faststart', str(output)])
    metadata = probe(output)
    video = next(x for x in metadata['streams'] if x['codec_type'] == 'video')
    duration = float(metadata['format']['duration'])
    if not 180 <= duration <= 240 or video['width'] != 1920 or video['height'] != 1080 or video['codec_name'] != 'h264' or video['pix_fmt'] != 'yuv420p':
        raise ValueError('M06 demo encoding/duration gate failed')
    run(['ffmpeg', '-v', 'error', '-i', str(output), '-f', 'null', '-'])
    used = [source['episode_id'] for segment in manifest for source in segment['sources']]
    if len(set(used)) != len(used):
        raise ValueError('Source episode video reused to pad duration')
    atomic_json(EVIDENCE / 'video_sources.json', {'segments': manifest, 'unique_source_episodes': len(used)})
    atomic_json(EVIDENCE / 'video_validation.json', {'duration_s': duration, 'resolution': [1920,1080],
        'codec': 'h264', 'pixel_format': 'yuv420p', 'decoded_without_error': True,
        'bytes': output.stat().st_size, 'sha256': sha256(output),
        'unique_source_episodes': len(used), 'segments': len(manifest)})
    run(['ffmpeg', '-v', 'error', '-y', '-ss', '2', '-i', str(output), '-frames:v', '1', str(DEMO / 'M06_overview.png')])
    build_html(manifest, stats, decisions)
    print(json.dumps({'video': str(output), 'duration_s': duration, 'sha256': sha256(output),
                      'html': str(VIS / 'index.html')}))

if __name__ == '__main__':
    main()
