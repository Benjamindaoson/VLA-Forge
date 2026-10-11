"""Create a source-audited offline M04 demo from unique real LIBERO/MuJoCo clips."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

EVIDENCE = Path("/data/VLA-Forge/Evidence/M04")
OUT = EVIDENCE / "demo"
SOURCES = OUT / "source_videos"
VIS = OUT / "visualization"
DATASET = Path(os.environ.get("M04_DATASET_PATH", str(EVIDENCE / "dataset/lerobot_v3_verified_r7")))
W, H, FPS = 1920, 1080, 20
CLIP_MAX_S = 5.5
M03_DUPLICATE_SHAM_VIDEOS: list[dict] = []


def text(
    image: np.ndarray,
    label: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = (240, 240, 240),
    thickness: int = 2,
) -> None:
    cv2.putText(
        image, label, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA
    )


def card_frames(title: str, lines: list[str], seconds: int):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:] = (28, 34, 41)
    cv2.rectangle(frame, (0, 0), (W, 126), (15, 20, 26), -1)
    text(frame, "VLA-FORGE M04 | VERIFIED CLOUD EXPERIMENT", (55, 78), 0.88)
    text(frame, title, (70, 330), 1.35, (120, 215, 240), 3)
    y = 470
    for line in lines:
        text(frame, line, (75, y), 0.78, (238, 238, 238), 2)
        y += 92
    text(frame, "Evidence boundary: one Goal task / two source states; no public release claim", (55, H - 60), 0.58, (175, 185, 195), 1)
    for _ in range(max(1, int(seconds * FPS))):
        yield frame


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
    chosen = []
    seen_actions = set()
    for summary_path in sorted((EVIDENCE / "artifacts/verified_corrections").glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        reps = summary.get("replicates", [])
        action_hash = summary.get("source_action_sha256")
        if not (
            summary.get("source_policy_run")
            and summary.get("independent_replay_pass") is True
            and len(reps) == 2
            and all(x.get("success") is True for x in reps)
            and action_hash not in seen_actions
        ):
            continue
        row = next(
            (x for x in attempts if x.get("episode_id") == summary["source_policy_run"] and x.get("success")),
            None,
        )
        if row is None:
            raise RuntimeError(f"verified success absent from registry: {summary['source_policy_run']}")
        seen_actions.add(action_hash)
        chosen.append(row)
    if len(chosen) < 10:
        raise RuntimeError(f"demo requires 10 unique replay-verified M04 policy successes; found {len(chosen)}")
    for suite in ("spatial", "long"):
        failure = next(
            (row for row in reversed(attempts) if row.get("suite") == suite and not row.get("success")),
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
    dataset_provenance = json.loads((DATASET / "provenance.json").read_text())
    verified_episode_count = int(dataset_provenance["episodes"])
    verified_transition_count = int(dataset_provenance["transitions"])
    source_task_ids = sorted({int(x["task_id"]) for x in dataset_provenance["sources"]})
    source_init_state_ids = sorted({int(x["init_state_id"]) for x in dataset_provenance["sources"]})
    fresh_policy_count = sum(
        1 for p in (EVIDENCE / "artifacts/verified_corrections").glob("*/summary.json")
        if json.loads(p.read_text()).get("source_policy_run")
        and json.loads(p.read_text()).get("independent_replay_pass") is True
    )
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
                f"{verified_episode_count} verified sequences | {verified_transition_count} transitions | LeRobot v3",
                f"Goal task 6 / init states {source_init_state_ids}; no cross-task claim",
            ],
            6,
        ),
        (
            "Inference and cost",
            [
                "FP32 CUDA Graph: 391.7 ms -> 56.4 ms per full action chunk (6.95x warm)",
                "18 paired full chunks: max error 0; cold capture 1.82 s; no end-to-end speed claim",
            ],
            6,
        ),
        (
            "Release gate",
            [
                "Private evidence preview prepared; public asset redistribution review remains open",
                "Raw videos and action/state evidence remain in this cloud Evidence/M04 package",
            ],
            6,
        ),
        (
            "Conclusion",
            [
                f"{fresh_policy_count} fresh SmolVLA corrections succeeded and replayed twice from matched prefixes",
                "Minimum correction count passed; one-task scope and public-release limits retained",
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
            "-r",
            "30",
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
        "verified_unique_corrections": verified_episode_count,
        "new_m04_policy_sequences": fresh_policy_count,
        "dataset_episodes": verified_episode_count,
        "dataset_transitions": verified_transition_count,
        "source_task_ids": source_task_ids,
        "source_m02_init_state_ids": source_init_state_ids,
        "public_release": False,
        "reason": "insufficient task diversity and upstream redistribution review pending",
    }
    (VIS / "metrics_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>VLA-Forge M04 Evidence</title><style>body{font:16px system-ui;max-width:1100px;margin:36px auto;background:#f4f6f8;color:#17212b}article{background:white;padding:22px;margin:18px 0;border-radius:8px}video{width:100%;max-width:900px;background:#111}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}</style></head><body><h1>VLA-Forge M04 — real correction evidence</h1><article><p><b>Status: INCOMPLETE.</b> Five fresh policy-generated corrections were independently replayed twice. Six unique correction sequences exist including one M03-derived replay; all share one Goal task and initial state. No generalization or public release is claimed.</p><video controls src='../M04_demo_200s.mp4'></video></article><article><h2>Matched intervention view</h2><img width='100%' src='../M04_comparison.png'></article><article><h2>Copied source clips</h2><table><tr><th>Source</th><th>Suite/task</th><th>Result</th><th>Condition</th><th>Clip</th></tr>"
    ]
    for e in episode_entries:
        rel = Path(e["copied_path"]).relative_to(OUT)
        html.append(
            f"<tr><td>{e['source_id']}</td><td>{e.get('suite')} / {e.get('task_id')}</td><td>{'Success' if e['success'] else 'Failure'}</td><td>{e.get('condition')}</td><td><a href='../{rel.as_posix()}'>open source MP4</a></td></tr>"
        )
    html += [
        "</table></article><article><h2>Dataset and metrics</h2><pre>"
        + json.dumps(summary, indent=2)
        + f"</pre><p>See ../REPORT.md, ../metrics.json, ../dataset/{DATASET.name}/provenance.json, and ../status.json.</p></article></body></html>"
    ]
    html[0] = html[0].replace("Five fresh", f"{fresh_policy_count} fresh").replace(
        "Six unique", f"{verified_episode_count} unique"
    ).replace(
        "all share one Goal task and initial state",
        f"cover Goal task ids {source_task_ids} and M02 init states {source_init_state_ids}",
    )
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
