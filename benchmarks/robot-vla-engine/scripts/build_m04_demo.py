"""Create a source-audited offline M04 demo from unique real LIBERO/MuJoCo clips."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

EVIDENCE = Path("/data/VLA-Forge/Evidence/M04")
OUT = EVIDENCE / "demo"
SOURCES = OUT / "source_videos"
VIS = OUT / "visualization"
W, H, FPS = 1920, 1080, 20
CLIP_MAX_S = 10.0
M03_DUPLICATE_SHAM_VIDEOS: list[dict] = []


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def probe(path: Path) -> dict:
    r = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    d = json.loads(r.stdout)
    return {**d["streams"][0], "duration_s": float(d["format"]["duration"])}


def m02_clips() -> list[dict]:
    run = Path(
        "/data/VLA-Forge/Evidence/M02/source/M02_libero40/registry/runs/20261010T030315-982cacc8515f"
    )
    episodes = []
    for meta_path in sorted((run / "episodes").glob("*/episode.json")):
        m = json.loads(meta_path.read_text())
        video = run / "videos" / f"{meta_path.parent.name}.mp4"
        if video.exists():
            episodes.append((m, video))
    selected = []
    for suite in ("spatial", "object", "goal", "long"):
        for success in (True, False):
            candidates = [
                (m, v) for m, v in episodes if m["suite"] == suite and bool(m["success"]) is success
            ]
            if not candidates:
                raise RuntimeError(f"M02 source missing {suite} success={success}")
            m, v = candidates[0]
            selected.append(
                {
                    "source_id": f"M02-{suite}-t{m['task_id']:02d}-e{m['episode_id']:02d}",
                    "origin": "M02 LIBERO 40 real rollout",
                    "suite": suite,
                    "task_id": m["task_id"],
                    "task_name": m["task_name"],
                    "instruction": m["instruction"],
                    "success": bool(m["success"]),
                    "condition": "frozen SmolVLA baseline",
                    "source_path": str(v),
                }
            )
    return selected


def m03_clips() -> list[dict]:
    root = Path("/data/vla-forge-artifacts/M03_interventional_diagnosis")
    rows = []
    for meta_path in sorted((root / "episodes").glob("*.json")):
        video = root / "videos" / f"{meta_path.stem}.mp4"
        if video.exists():
            m = json.loads(meta_path.read_text())
            rows.append((meta_path.stem, video, m))
    groups: dict[str, list[tuple[str, Path, dict]]] = {}
    for row in rows:
        group = row[2].get("group_id", row[0].split("_")[0])
        groups.setdefault(group, []).append(row)
    ordered = sorted(groups, key=lambda x: ("g09" not in x, x))
    selected = []
    seen_hashes = set()
    condition_order = {"baseline": 0, "sham": 1, "brightness": 2, "occlusion": 3}
    for group in ordered:
        items = sorted(
            groups[group], key=lambda row: condition_order.get(row[2].get("condition", ""), 9)
        )
        for name, video, m in items:
            video_hash = sha(video)
            if video_hash in seen_hashes:
                M03_DUPLICATE_SHAM_VIDEOS.append(
                    {
                        "source_id": name,
                        "sha256": video_hash,
                        "reason": "byte-identical no-op clip omitted to prevent duplicate footage",
                    }
                )
                continue
            seen_hashes.add(video_hash)
            selected.append(
                {
                    "source_id": name,
                    "origin": "M03 real MuJoCo matched replay",
                    "suite": m.get("suite"),
                    "task_id": m.get("task_id"),
                    "task_name": m.get("task_name"),
                    "instruction": m.get("instruction"),
                    "success": bool(m.get("success")),
                    "condition": m.get("condition"),
                    "source_path": str(video),
                }
            )
            if len(selected) == 16:
                return selected
    raise RuntimeError(f"M03 source only supplied {len(selected)} unique episode videos")


def m04_clips() -> list[dict]:
    registry = EVIDENCE / "artifacts/policy_attempts/attempts.jsonl"
    attempts = [json.loads(line) for line in registry.read_text().splitlines() if line.strip()]
    verified = json.loads(
        (
            EVIDENCE
            / "artifacts/verified_corrections/m04-goal-t06-e07-brightness115-first50-seed84075/summary.json"
        ).read_text()
    )
    source_episode_id = verified["source_policy_run"]
    success = next(
        (
            row
            for row in attempts
            if row.get("episode_id") == source_episode_id and row.get("success")
        ),
        None,
    )
    if success is None:
        raise RuntimeError("verified fresh M04 policy success is absent from registry")
    chosen = [success]
    for suite in ("spatial", "long"):
        failure = next(
            (
                row
                for row in reversed(attempts)
                if row.get("suite") == suite and not row.get("success")
            ),
            None,
        )
        if failure is not None:
            chosen.append(failure)
    out = []
    for row in chosen:
        video = row.get("video", {}).get("path")
        if not video or not Path(video).is_file():
            raise RuntimeError(f"M04 episode video missing for {row['episode_id']}")
        out.append(
            {
                "source_id": row["episode_id"],
                "origin": "M04 fresh policy inference; real LIBERO/MuJoCo episode",
                "suite": row.get("suite"),
                "task_id": row.get("task_id"),
                "task_name": row.get("task_name"),
                "instruction": row.get("instruction"),
                "success": bool(row.get("success")),
                "condition": row.get("condition"),
                "source_path": video,
            }
        )
    return out


def text(
    frame: np.ndarray,
    value: str,
    xy: tuple[int, int],
    scale: float = 1.0,
    color=(255, 255, 255),
    thick: int = 2,
) -> None:
    cv2.putText(
        frame, value[:140], xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 4, cv2.LINE_AA
    )
    cv2.putText(frame, value[:140], xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def card_frames(title: str, lines: list[str], seconds: int) -> list[np.ndarray]:
    frames = []
    for i in range(seconds * FPS):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:] = (28, 35, 43)
        cv2.rectangle(frame, (0, 0), (W, 14), (96, 164, 226), -1)
        text(
            frame, "VLA-FORGE 2.0  |  M04 REAL CLOUD EVIDENCE", (95, 170), 1.35, (236, 243, 250), 3
        )
        text(frame, title, (100, 340), 1.65, (115, 190, 255), 3)
        y = 455
        for line in lines:
            text(frame, line, (110, y), 0.9, (230, 235, 240), 2)
            y += 85
        cv2.rectangle(frame, (100, 965), (W - 100, 977), (70, 77, 85), -1)
        cv2.rectangle(
            frame,
            (100, 965),
            (100 + int((W - 200) * (i + 1) / (seconds * FPS)), 977),
            (96, 164, 226),
            -1,
        )
        frames.append(frame)
    return frames


def clip_frames(entry: dict, target: Path) -> tuple[np.ndarray, dict]:
    source = Path(entry["source_path"])
    meta = probe(source)
    duration = min(CLIP_MAX_S, meta["duration_s"])
    if duration < 2:
        raise RuntimeError(f"too-short source video {source}: {duration}s")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    src_hash = sha(source)
    copied_hash = sha(target)
    if src_hash != copied_hash:
        raise RuntimeError("source video copy hash mismatch")
    start = max(0.0, (meta["duration_s"] - duration) / 2.0)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode source video {source}")
    cap.set(cv2.CAP_PROP_POS_MSEC, (start + duration / 2) * 1000)
    ok, raw = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot extract overview frame: {source}")
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    ih, iw = raw.shape[:2]
    scale = min((W - 100) / iw, 900 / ih)
    resized = cv2.resize(raw, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)
    x, y = (W - resized.shape[1]) // 2, 100 + (900 - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 0), (W, 90), (15, 20, 26), -1)
    status = "SUCCESS" if entry["success"] else "FAILURE"
    text(
        canvas,
        f"{entry['origin']} | {entry.get('source_id', '')} | {status}",
        (45, 58),
        0.72,
        (115, 210, 140) if entry["success"] else (110, 170, 255),
        2,
    )
    return canvas, {
        "source_id": entry["source_id"],
        "origin": entry["origin"],
        "source_original_path": str(source),
        "copied_path": str(target),
        "source_sha256": src_hash,
        "copied_sha256": copied_hash,
        "source_duration_s": meta["duration_s"],
        "clip_start_s": start,
        "clip_duration_s": duration,
        "suite": entry.get("suite"),
        "task_id": entry.get("task_id"),
        "task_name": entry.get("task_name"),
        "instruction": entry.get("instruction"),
        "success": entry["success"],
        "condition": entry.get("condition"),
        "width": meta["width"],
        "height": meta["height"],
        "codec": meta["codec_name"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)
    VIS.mkdir(parents=True, exist_ok=True)
    entries = m02_clips() + m03_clips() + m04_clips()
    seen = set()
    for item in entries:
        h = sha(Path(item["source_path"]))
        if h in seen:
            raise RuntimeError(f"duplicate clip source hash: {item['source_path']}")
        seen.add(h)
    episode_entries = []
    for i, entry in enumerate(entries):
        target = SOURCES / f"{i + 1:02d}_{entry['source_id'].replace('/', '_')}.mp4"
        frame, source_receipt = clip_frames(entry, target)
        episode_entries.append({**entry, **source_receipt})
        if i == 0:
            cv2.imwrite(str(OUT / "M04_overview.png"), frame)
    cards = [
        (
            "Project question",
            [
                "Can a frozen VLA resume from a real failed LIBERO prefix?",
                "SmolVLA + LeRobot + LIBERO / MuJoCo + RTX 3090",
            ],
            8,
        ),
        (
            "Dataset reality",
            [
                "2 unique verified correction episodes | 200 transitions | LeRobot v3",
                "Both are Goal task 6 / init state 7; not task-diverse and not public-ready",
            ],
            6,
        ),
        (
            "Inference and cost",
            [
                "FP32 retained: BF16 and TF32 failed speed / fidelity gates",
                "RTX 3090 peak allocated memory ~0.96 GiB; failed attempts remain in the cost denominator",
            ],
            6,
        ),
        (
            "Release gate",
            [
                "No Hugging Face upload: current sample quality and upstream rights are insufficient",
                "Raw videos and action/state evidence remain in this cloud Evidence/M04 package",
            ],
            6,
        ),
        (
            "Conclusion",
            [
                "One fresh SmolVLA correction succeeded and replayed twice from a matched prefix",
                "M04 remains INCOMPLETE: at least 10 independent, diverse corrections are still required",
            ],
            8,
        ),
    ]
    video = OUT / "M04_demo_200s.mp4"
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        stdin=subprocess.PIPE,
    )
    # Compose from precisely the documented sequence: intro, 16 clips, 4 interstitials, final card.
    sequence = (
        [cards[0]]
        + [(e, episode_entries[i]) for i, e in enumerate(entries)]
        + cards[1:4]
        + [cards[4]]
    )
    for part in sequence:
        if isinstance(part, tuple) and isinstance(part[1], dict):
            path = Path(part[1]["copied_path"])
            cap = cv2.VideoCapture(str(path))
            cap.set(cv2.CAP_PROP_POS_MSEC, part[1]["clip_start_s"] * 1000)
            clip_frames_list = []
            for _ in range(int(part[1]["clip_duration_s"] * FPS)):
                ok, frame = cap.read()
                if not ok:
                    break
                ih, iw = frame.shape[:2]
                scale = min((W - 100) / iw, 900 / ih)
                resized = cv2.resize(frame, (int(iw * scale), int(ih * scale)))
                canvas = np.zeros((H, W, 3), dtype=np.uint8)
                x, y = (W - resized.shape[1]) // 2, 100 + (900 - resized.shape[0]) // 2
                canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
                status = "SUCCESS" if part[1]["success"] else "FAILURE"
                cv2.rectangle(canvas, (0, 0), (W, 90), (15, 20, 26), -1)
                cv2.rectangle(canvas, (0, H - 90), (W, H), (15, 20, 26), -1)
                text(
                    canvas,
                    f"{part[1]['origin']} | {part[1]['source_id']} | {status}",
                    (45, 58),
                    0.72,
                    (115, 210, 140) if part[1]["success"] else (110, 170, 255),
                    2,
                )
                text(
                    canvas,
                    f"{part[1].get('suite', '')} task {part[1].get('task_id', '')} | {part[1].get('task_name', '')} | {part[1].get('condition', '')}",
                    (45, H - 35),
                    0.60,
                    (240, 240, 240),
                    2,
                )
                clip_frames_list.append(canvas)
            cap.release()
            for frame in clip_frames_list:
                proc.stdin.write(frame.tobytes())
        else:
            title, lines, seconds = part
            for frame in card_frames(title, lines, seconds):
                proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    rc = proc.wait()
    if rc:
        raise RuntimeError(f"ffmpeg encoder failed with exit {rc}")
    vmeta = probe(video)
    dec = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    if dec.returncode:
        raise RuntimeError(f"full video decode failed: {dec.stderr[-1000:]}")
    receipt = {
        "path": str(video),
        "sha256": sha(video),
        "bytes": video.stat().st_size,
        "duration_s": vmeta["duration_s"],
        "width": vmeta["width"],
        "height": vmeta["height"],
        "codec": vmeta["codec_name"],
        "pixel_format": vmeta["pix_fmt"],
        "fps": vmeta["r_frame_rate"],
        "full_decode_exit_code": dec.returncode,
        "unique_episode_source_count": len(episode_entries),
        "episode_sources": episode_entries,
        "omitted_byte_identical_m03_videos": M03_DUPLICATE_SHAM_VIDEOS,
        "cards": [x[0] for x in cards],
    }
    if (
        not 180 <= receipt["duration_s"] <= 240
        or receipt["codec"] != "h264"
        or receipt["pixel_format"] != "yuv420p"
        or receipt["width"] != W
        or receipt["height"] != H
    ):
        raise RuntimeError(f"video contract failed: {receipt}")
    (OUT / "video_sources.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (OUT / "video_validation.json").write_text(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "path",
                    "sha256",
                    "bytes",
                    "duration_s",
                    "width",
                    "height",
                    "codec",
                    "pixel_format",
                    "fps",
                    "full_decode_exit_code",
                    "unique_episode_source_count",
                )
            },
            indent=2,
        )
        + "\n"
    )
    # Real-frame before/after comparison from one M03 matched Goal task group.
    comp_names = ["m03-g09-goal-t06-init07_baseline", "m03-g09-goal-t06-init07_brightness"]
    comp_frames = []
    for nm in comp_names:
        p = SOURCES / next(x.name for x in SOURCES.glob("*.mp4") if nm in x.name)
        cap = cv2.VideoCapture(str(p))
        cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
        ok, f = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"could not extract comparison frame {p}")
        f = cv2.resize(f, (960, 480))
        cv2.rectangle(f, (0, 0), (960, 65), (15, 20, 26), -1)
        text(f, nm, (20, 43), 0.55)
        comp_frames.append(f)
    comparison = np.vstack([np.hstack(comp_frames), np.zeros((120, 1920, 3), dtype=np.uint8)])
    text(
        comparison,
        "Matched M03 group g09 | source task/init fixed | baseline vs brightness treatment; treatment success is case-specific",
        (25, 1010),
        0.7,
    )
    cv2.imwrite(str(OUT / "M04_comparison.png"), comparison)
    cards_dir = VIS / "cards"
    cards_dir.mkdir(exist_ok=True)
    summary = {
        "status": "INCOMPLETE",
        "verified_unique_corrections": 2,
        "new_m04_policy_sequences": 1,
        "dataset_episodes": 2,
        "dataset_transitions": 200,
        "public_release": False,
        "reason": "insufficient task diversity and upstream redistribution review pending",
    }
    (VIS / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>VLA-Forge M04 Evidence</title><style>body{font:16px system-ui;max-width:1100px;margin:36px auto;background:#f4f6f8;color:#17212b}article{background:white;padding:22px;margin:18px 0;border-radius:8px}video{width:100%;max-width:900px;background:#111}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style></head><body><h1>VLA-Forge M04 — real correction evidence</h1><article><p><b>Status: INCOMPLETE.</b> One fresh policy-generated correction was independently replayed twice. Two unique correction sequences exist overall, both from one Goal task and initial state. No generalization or public release is claimed.</p><video controls src='../M04_demo_200s.mp4'></video></article><article><h2>Matched intervention view</h2><img width='100%' src='../M04_comparison.png'></article><article><h2>Copied source clips</h2><table><tr><th>Source</th><th>Suite/task</th><th>Result</th><th>Condition</th><th>Clip</th></tr>"
    ]
    for e in episode_entries:
        rel = Path(e["copied_path"]).relative_to(OUT)
        html.append(
            f"<tr><td>{e['source_id']}</td><td>{e.get('suite')} / {e.get('task_id')}</td><td>{'Success' if e['success'] else 'Failure'}</td><td>{e.get('condition')}</td><td><a href='../{rel.as_posix()}'>open source MP4</a></td></tr>"
        )
    html += [
        "</table></article><article><h2>Dataset and metrics</h2><pre>"
        + json.dumps(summary, indent=2)
        + "</pre><p>See ../REPORT.md, ../metrics.json, ../dataset/lerobot_v3_verified_r2/provenance.json, and ../status.json.</p></article></body></html>"
    ]
    (VIS / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "duration_s",
                    "bytes",
                    "width",
                    "height",
                    "codec",
                    "pixel_format",
                    "fps",
                    "unique_episode_source_count",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
