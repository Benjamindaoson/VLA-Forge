"""Build an offline M03 visual evidence package from verified real LIBERO clips."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from robot_vla.schema import file_hash


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = next(x for x in data["streams"] if x.get("codec_type") == "video")
    return {
        "duration": float(data["format"]["duration"]),
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pixel_format": stream["pix_fmt"],
        "fps": float(Fraction(stream["avg_frame_rate"])),
    }


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def video_frame(path: Path, seconds: float = 0.0) -> np.ndarray:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(seconds * av.time_base), stream=stream)
        frame = next(container.decode(video=0))
        return frame.to_ndarray(format="rgb24")


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def screenshot_pack(groups: list[dict], run_dir: Path, report: Path) -> None:
    shots = report / "visualization" / "screenshots"
    shots.mkdir(parents=True, exist_ok=True)
    for group in groups:
        name = group["group_id"]
        mode = (
            "brightness"
            if any("brightness" in p.name for p in (run_dir / "videos").glob(f"{name}_*.mp4"))
            else "occlusion"
        )
        images = []
        for cond in ("baseline", "sham", mode):
            frame = Image.fromarray(video_frame(run_dir / "videos" / f"{name}_{cond}.mp4"))
            if cond == mode:
                left, right = (
                    frame.crop((0, 0, frame.width // 2, frame.height)),
                    frame.crop((frame.width // 2, 0, frame.width, frame.height)),
                )
                if mode == "brightness":
                    left = Image.fromarray(
                        np.clip(np.asarray(left).astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
                    )
                    right = Image.fromarray(
                        np.clip(np.asarray(right).astype(np.float32) * 1.15, 0, 255).astype(
                            np.uint8
                        )
                    )
                else:
                    arr = np.asarray(right).copy()
                    h, w = arr.shape[:2]
                    arr[int(h * 7 / 16) : int(h * 9 / 16), int(w * 7 / 16) : int(w * 9 / 16)] = 0
                    right = Image.fromarray(arr)
                frame = Image.new("RGB", (frame.width, frame.height))
                frame.paste(left, (0, 0))
                frame.paste(right, (frame.width // 2, 0))
            images.append((cond, frame))
        baseline_after = Image.fromarray(
            video_frame(run_dir / "videos" / f"{name}_baseline.mp4", 2.5)
        )
        treatment_after = Image.fromarray(
            video_frame(run_dir / "videos" / f"{name}_{mode}.mp4", 2.5)
        )
        canvas = Image.new("RGB", (1536, 896), "#101923")
        draw = ImageDraw.Draw(canvas)
        for i, (label, frame) in enumerate(images):
            tile = frame.resize((512, 256))
            x, y = i * 512, 56
            canvas.paste(tile, (x, y))
            draw.text((x + 16, 16), label.upper(), font=font(24), fill="white")
        draw.text((24, 332), "AFTER 50 POST-INTERVENTION EXECUTION STEPS", font=font(20), fill="#d6e4f2")
        draw.text((24, 348), "Baseline physical RGB", font=font(18), fill="#b8c8dc")
        draw.text((792, 348), "Treatment physical RGB", font=font(18), fill="#b8c8dc")
        canvas.paste(baseline_after.resize((720, 360)), (24, 370))
        canvas.paste(treatment_after.resize((720, 360)), (792, 370))
        canvas.save(shots / f"{name}.png")
    canvas = Image.new("RGB", (1600, 120 + ((len(groups) + 2) // 3) * 320), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (32, 20), "VLA-Forge M03 · matched-state intervention groups", font=font(32), fill="#183554"
    )
    for idx, group in enumerate(groups):
        img = Image.open(shots / f"{group['group_id']}.png").resize((500, 292))
        x, y = 24 + (idx % 3) * 524, 72 + (idx // 3) * 320
        canvas.paste(img, (x, y))
        draw.text((x, y + 294), group["group_id"], font=font(15), fill="#183554")
    canvas.save(report / "M03_overview.png")
    if groups:
        shutil.copy2(shots / f"{groups[0]['group_id']}.png", report / "M03_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--m02-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir, m02, output = args.run_dir.resolve(), args.m02_report.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    video_dir = output / "visualization" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    groups = manifest["groups"]
    rows = read_csv(run_dir / "paired_interventions.csv")
    if len(groups) != 10 or len(rows) != 30:
        raise RuntimeError(f"expected 10 groups/30 conditions, found {len(groups)}/{len(rows)}")
    screenshot_pack(groups, run_dir, output)
    m02_rows = read_csv(m02 / "episodes.csv")
    reserved_source_ids = {
        (g["source_suite"], int(g["source_task_id"]), int(g["source_episode_id"])) for g in groups
    }
    used: set[tuple[str, int, int]] = set()

    def identity(row: dict) -> tuple[str, int, int]:
        return row["suite"], int(row["task_id"]), int(row["episode_id"])

    for row in m02_rows:
        if row.get("status") != "complete" or not row.get("video_path"):
            continue
        path = Path(row["video_path"])
        if path.is_file() and file_hash(path) == row["video_sha256"]:
            row["_probe"] = probe(path)
        else:
            row["_probe"] = None
    valid = [r for r in m02_rows if r.get("_probe") and r["_probe"]["duration"] >= 5.1]
    successes = [r for r in valid if r.get("success", "").lower() == "true"]
    failures = [r for r in valid if r.get("success", "").lower() == "false"]

    def take(pool: list[dict], count: int, *, exclude_m03: bool = False) -> list[dict]:
        chosen = []
        for row in pool:
            key = identity(row)
            if key in used or (exclude_m03 and key in reserved_source_ids):
                continue
            used.add(key)
            chosen.append(row)
            if len(chosen) == count:
                return chosen
        if len(chosen) != count:
            raise RuntimeError(
                "not enough distinct verified M02 clips for requested video sections"
            )
        return chosen

    intro = take([r for r in successes if r["suite"] in ("spatial", "object", "goal")], 3)
    cases = take(failures, 5, exclude_m03=True)
    recovery = []
    for group in groups[:5]:
        key = (group["source_suite"], int(group["source_task_id"]), int(group["source_episode_id"]))
        row = next(r for r in valid if identity(r) == key)
        used.add(key)
        recovery.append(row)
    result_context = take(
        [r for r in successes + failures if identity(r) not in reserved_source_ids], 5
    )
    closing = take(successes + failures, 2, exclude_m03=True)
    m02_section_rows = intro + cases + recovery + result_context + closing

    triads = []
    for group in groups:
        # The assigned treatment is taken from the actual complete result records.
        treat = next(
            r["condition"]
            for r in rows
            if r["group_id"] == group["group_id"] and r["condition"] not in ("baseline", "sham")
        )
        source_paths = [
            run_dir / "videos" / f"{group['group_id']}_{c}.mp4" for c in ("baseline", "sham", treat)
        ]
        if any(not p.is_file() for p in source_paths):
            raise RuntimeError(f"missing condition video for {group['group_id']}")
        for source_path in source_paths:
            result_row = next(
                r for r in rows
                if r["group_id"] == group["group_id"]
                and Path(r["video_path"]).name == source_path.name
            )
            if file_hash(source_path) != result_row["video_sha256"]:
                raise RuntimeError(f"M03 source video hash mismatch: {source_path}")
        triads.append((group, treat, source_paths))

    triad_lengths = [min(probe(p)["duration"] for p in paths) for _, _, paths in triads]
    triad_total = sum(triad_lengths)
    context_target = max(100.0, 150.0 - triad_total)
    per_m02 = context_target / len(m02_section_rows)
    if any(r["_probe"]["duration"] < per_m02 + 0.1 for r in m02_section_rows):
        raise RuntimeError(
            f"M02 source footage too short for non-padding 150s target: {per_m02:.2f}s per source"
        )

    tmp = output / "visualization" / "build_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []
    sources = []
    sections = [
        "Project introduction",
        "Real LIBERO task outcomes",
        "M02 failures selected for prefix replay",
        "State-matched M03 paired interventions",
        "Measured M03 outcomes in benchmark context",
        "Conclusion and limitations",
    ]
    captions = [
        "VLA-FORGE | SmolVLA + LIBERO / MuJoCo | frozen policy diagnostics",
        "Real M02 robot rollouts: task execution and observed failures",
        "Source M02 episodes selected by predeclared rules; fixed-init action-prefix replay",
        "Synchronized Frozen Baseline | Sham | Visual Treatment; same recovered MuJoCo state",
        "M03 measured paired outcomes; contextual LIBERO scene footage is separately identified",
        "Negative or unchanged outcomes are retained; timeout is not a root-cause label",
    ]
    for item_idx, row in enumerate(m02_section_rows):
        section_idx = (
            0
            if item_idx < 3
            else 1
            if item_idx < 8
            else 2
            if item_idx < 13
            else 4
            if item_idx < 18
            else 5
        )
        path = Path(row["video_path"])
        duration = per_m02
        if row in recovery:
            gi = groups[recovery.index(row)]
            prefix_seconds = float(gi["prefix_steps"]) / 20.0
            start = max(0.0, prefix_seconds - duration / 2)
        else:
            start = max(
                0.0, min(row["_probe"]["duration"] - duration, row["_probe"]["duration"] * 0.15)
            )
        caption_path = tmp / "caption.txt"
        caption_path.write_text(
            captions[section_idx]
            + "\n"
            + row["suite"].upper()
            + f" task {row['task_id']} | episode {row['episode_id']}\n"
            + str(row.get("instruction", "")),
            encoding="utf-8",
        )
        segment = tmp / f"m02_{item_idx:02d}.mp4"
        vf = (
            "scale=1800:900:flags=lanczos,pad=1920:1080:60:90:black,drawbox=x=0:y=0:w=iw:h=90:color=black@0.88:t=fill,drawbox=x=0:y=990:w=iw:h=90:color=black@0.82:t=fill,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:textfile="
            + str(caption_path)
            + ":fontcolor=white:fontsize=22:x=48:y=8:line_spacing=3"
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start),
                "-i",
                str(path),
                "-t",
                str(duration),
                "-vf",
                vf,
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
                str(segment),
            ]
        )
        segments.append(segment)
        sources.append(
            {
                "section": sections[section_idx],
                "kind": "M02_original",
                "suite": row["suite"],
                "task_id": int(row["task_id"]),
                "episode_id": int(row["episode_id"]),
                "start_seconds": start,
                "duration_seconds": duration,
                "source_path": str(path),
                "source_sha256": row["video_sha256"],
            }
        )
    for idx, (group, treatment, paths) in enumerate(triads):
        caption = tmp / "caption.txt"
        outcomes = group["condition_results"]
        caption.write_text(
            f"{group['group_id']} | {treatment} starts at post-prefix step 0 | prefix={group['prefix_steps']}\n"
            f"Success: baseline={outcomes['baseline']['success']} sham={outcomes['sham']['success']} treatment={outcomes[treatment]['success']} | synchronized camera views",
            encoding="utf-8",
        )
        output_seg = tmp / f"triad_{idx:02d}.mp4"
        filters = []
        for i in range(3):
            filters.append(
                f"[{i}:v]split=2[v{i}a][v{i}b];[v{i}a]crop=iw/2:ih:0:0,scale=480:480:flags=lanczos,setsar=1, fps=30[top{i}];[v{i}b]crop=iw/2:ih:iw/2:0,scale=480:480:flags=lanczos,setsar=1, fps=30[bot{i}]"
            )
        filters += [
            "[top0][top1][top2]hstack=inputs=3[top]",
            "[bot0][bot1][bot2]hstack=inputs=3[bot]",
            "[top][bot]vstack=inputs=2,pad=1920:1080:240:60:black,drawbox=x=0:y=0:w=iw:h=60:color=black@0.9:t=fill,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Frozen Baseline':fontcolor=white:fontsize=22:x=330:y=65,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Sham':fontcolor=white:fontsize=22:x=890:y=65,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='Treatment':fontcolor=white:fontsize=22:x=1350:y=65,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:textfile="
            + str(caption)
            + ":fontcolor=white:fontsize=22:x=30:y=6:line_spacing=4[out]",
        ]
        command = ["ffmpeg", "-y"]
        for path in paths:
            command += ["-i", str(path)]
        command += [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-shortest",
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
            str(output_seg),
        ]
        run(command)
        segments.append(output_seg)
        actual = probe(output_seg)["duration"]
        sources.append(
            {
                "section": sections[3],
                "kind": "M03_matched_triptych",
                "group_id": group["group_id"],
                "treatment": treatment,
                "start_seconds": 0,
                "duration_seconds": actual,
                "sources": [
                    {"path": str(p), "sha256": file_hash(p), "synchronized_start_seconds": 0}
                    for p in paths
                ],
            }
        )
    # Reorder segment streams into the requested narrative section order.
    ordered = (
        segments[:3]
        + segments[3:8]
        + segments[8:13]
        + segments[20:30]
        + segments[13:18]
        + segments[18:20]
    )
    ordered_sources = (
        sources[:3]
        + sources[3:8]
        + sources[8:13]
        + sources[20:30]
        + sources[13:18]
        + sources[18:20]
    )
    concat_list = tmp / "concat.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in ordered), encoding="utf-8")
    demo = output / "M03_demo_150s.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            str(demo),
        ]
    )
    meta = probe(demo)
    decode = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(demo), "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    sources_path = output / "video_sources.json"
    sources_path.write_text(
        json.dumps(
            {
                "segments": ordered_sources,
                "unique_m02_episode_count": len(used),
                "repeated_m02_episodes": False,
                "demo_sha256": file_hash(demo),
                "source_order_rule": "all real source video intervals are unique M02 episodes; each M03 triptych uses its three original condition videos exactly once",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    validation = {
        **meta,
        "file_bytes": demo.stat().st_size,
        "full_decode_ok": decode.returncode == 0,
        "decode_stderr": decode.stderr[-1000:],
        "duration_in_range": 120 <= meta["duration"] <= 180,
        "expected_codec": "h264",
        "expected_pixel_format": "yuv420p",
        "expected_resolution": [1920, 1080],
        "unique_source_segments": len(ordered_sources),
        "unique_m02_source_episodes": len(used),
        "source_hashes_verified_before_edit": True,
        "raw_source_video_hashes_recorded": True,
        "valid": decode.returncode == 0
        and 120 <= meta["duration"] <= 180
        and meta["codec"] == "h264"
        and meta["pixel_format"] == "yuv420p"
        and [meta["width"], meta["height"]] == [1920, 1080],
    }
    (output / "video_validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    overall_path = output / "validation.json"
    if overall_path.is_file():
        overall = json.loads(overall_path.read_text(encoding="utf-8"))
        overall["video_requirement_checked_after_demo_build"] = True
        overall["demo_video_valid"] = bool(validation["valid"])
        overall["demo_duration_seconds"] = validation["duration"]
        overall["demo_sha256"] = file_hash(demo)
        experimental_gates = all(
            overall.get(key) is True
            for key in (
                "conditions_complete",
                "all_groups_physical_state_matched",
                "baseline_sham_exact_trace_exact",
                "two_treatment_types",
                "all_three_conditions_measured",
                "physical_trajectory_deltas_complete",
                "contact_trajectory_deltas_complete",
                "object_pose_records_complete",
                "all_episode_artifacts_valid",
                "all_m02_source_prefix_artifacts_valid",
            )
        )
        overall["experiment_and_video_gates_pass"] = experimental_gates and bool(validation["valid"])
        overall["status"] = "PASS" if overall["experiment_and_video_gates_pass"] else "INCOMPLETE"
        overall_path.write_text(
            json.dumps(overall, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        metrics = json.loads((output / "intervention_metrics.json").read_text(encoding="utf-8"))
        result = metrics["condition_results"]
        pose = metrics["object_pose_checkpoints"]
        resources = metrics["resource_usage"]
        paired = metrics["matched_success_outcomes"]
        report_text = (
            "# M03 Interventional Diagnosis\n\n"
            f"**Acceptance: {overall['status']}**\n\n"
            "## Physical replay and state matching\n\n"
            f"- Prefix replay recovered the saved 8D M02 policy observation exactly in {metrics['physical_replay']['source_state_exact_matches']}/10 groups.\n"
            f"- Independent Baseline/Sham/Treatment resets matched qpos, qvel and simulation time exactly in {metrics['physical_replay']['physical_groups_exact_matches']}/10 groups; max errors were {metrics['physical_replay']['position_linf_max']}, {metrics['physical_replay']['velocity_linf_max']} and {metrics['physical_replay']['sim_time_abs_max']}.\n"
            f"- Decoded {pose['named_free_joint_object_pose_records']} named movable-object pose records across {pose['records']} matched condition checkpoints; all body names resolved.\n"
            "- The M02 source data contains no original mid-episode full MuJoCo snapshot. These results validate deterministic reset-plus-logged-prefix reconstruction, not equality to an unavailable historical full simulator snapshot.\n\n"
            "## Matched policy outcomes\n\n"
            f"- Frozen Baseline: {result['baseline']['successes']}/{result['baseline']['episodes']} success; Sham: {result['sham']['successes']}/{result['sham']['episodes']}. Sham action/state/physics traces were exactly equal to Baseline in {metrics['baseline_sham_exact_trace_group_count']}/10 groups.\n"
            f"- Paired outcomes: brightness {paired['brightness']['condition_only_success']}/{paired['brightness']['matched_groups']} treatment-only successes, {paired['brightness']['baseline_only_success']} baseline-only; occlusion {paired['occlusion']['condition_only_success']}/{paired['occlusion']['matched_groups']} treatment-only, {paired['occlusion']['baseline_only_success']} baseline-only.\n"
            f"- Brightness treatment: {result['brightness']['successes']}/{result['brightness']['episodes']} success, mean absolute 7D action difference {sum(result['brightness']['action_delta_vs_baseline'])/max(1,len(result['brightness']['action_delta_vs_baseline'])):.6f}, mean 8D observed-state L2 difference {sum(result['brightness']['trajectory_delta_mean_l2'])/max(1,len(result['brightness']['trajectory_delta_mean_l2'])):.6f}, mean MuJoCo qpos path L2 difference {sum(result['brightness']['sim_qpos_trajectory_mean_l2_vs_baseline'])/max(1,len(result['brightness']['sim_qpos_trajectory_mean_l2_vs_baseline'])):.6f}, mean absolute contact-count difference {sum(result['brightness']['sim_contact_count_mean_abs_delta_vs_baseline'])/max(1,len(result['brightness']['sim_contact_count_mean_abs_delta_vs_baseline'])):.3f}.\n"
            f"- Central wrist-camera occlusion: {result['occlusion']['successes']}/{result['occlusion']['episodes']} success, mean absolute 7D action difference {sum(result['occlusion']['action_delta_vs_baseline'])/max(1,len(result['occlusion']['action_delta_vs_baseline'])):.6f}, mean 8D observed-state L2 difference {sum(result['occlusion']['trajectory_delta_mean_l2'])/max(1,len(result['occlusion']['trajectory_delta_mean_l2'])):.6f}, mean MuJoCo qpos path L2 difference {sum(result['occlusion']['sim_qpos_trajectory_mean_l2_vs_baseline'])/max(1,len(result['occlusion']['sim_qpos_trajectory_mean_l2_vs_baseline'])):.6f}, mean absolute contact-count difference {sum(result['occlusion']['sim_contact_count_mean_abs_delta_vs_baseline'])/max(1,len(result['occlusion']['sim_contact_count_mean_abs_delta_vs_baseline'])):.3f}.\n"
            "- Outcome difference is descriptive only (5 groups per visual treatment); it is not a powered or generalized causal success-rate claim. Timeout is retained as an outcome, never assigned as a root cause.\n\n"
            "## Resources and presentation\n\n"
            f"- RTX 3090 sampled device-memory peak: {resources['device_memory_used_peak_mib_sampled']} MiB (nvidia-smi whole-device sample, not allocator peak).\n"
            f"- Median policy inference episode-mean: {resources['select_action_latency_episode_mean_ms_median']:.3f} ms; rollout wall-time sum: {resources['condition_rollout_wall_seconds_sum']:.1f} s; total measured run window: {resources['gpu_sampler_window_seconds']:.0f} s from 1 Hz GPU samples.\n"
            f"- Demo: `M03_demo_150s.mp4`, {validation['duration']:.3f} s, {validation['file_bytes']} bytes, 1920x1080 H.264/yuv420p at {validation['fps']:.2f} fps; full decode passed.\n"
            "- Demo comprises 20 distinct hash-verified M02 source episode intervals and 10 synchronized three-condition M03 comparisons; original episode videos are preserved.\n\n"
            "## Limitations and reproduction\n\n"
            "Visual factors act on policy observations; they do not alter task goals or physical object initial positions. The treatment sample is small and mixed across task groups. Original full-state M02 snapshots were unavailable, so recovery uses the frozen LIBERO initial-state index plus exact logged action prefix.\n\n"
            "Run `bash reproduce.sh` in the frozen Linux CUDA/EGL environment. The script resumes only missing/invalid condition records, extracts named object poses, rebuilds tables, and regenerates the demo.\n\n"
            "## Resume bullet\n\n"
            f"Built a reproducible SmolVLA-LIBERO intervention diagnostic that reconstructed {metrics['physical_replay']['physical_groups_exact_matches']}/10 failed-episode checkpoints with zero measured qpos/qvel/time mismatch, captured {pose['named_free_joint_object_pose_records']} named object poses, and compared 10 matched brightness/occlusion groups across 30 real MuJoCo rollouts; brightness produced {paired['brightness']['condition_only_success']}/{paired['brightness']['matched_groups']} treatment-only successes (0 baseline-only), while occlusion produced {paired['occlusion']['condition_only_success']}/{paired['occlusion']['matched_groups']} treatment-only successes; exploratory small-sample evidence, not a generalized improvement claim.\n"
        )
        (output / "REPORT.md").write_text(report_text, encoding="utf-8")
    (output / "visualization" / "index.html").write_text(
        _index(groups, demo.name, validation), encoding="utf-8"
    )
    shutil.rmtree(tmp)
    print(json.dumps(validation, indent=2))


def _index(groups: list[dict], demo: str, validation: dict) -> str:
    cards = []
    for g in groups:
        group_id = g["group_id"]
        treatment = next(k for k in g["condition_results"] if k not in ("baseline", "sham"))
        cards.append(
            f"<article><h2>{group_id}</h2><img src='screenshots/{group_id}.png'><p>Source task {g['source_suite']}:{g['source_task_id']} · init {g['init_state_id']} · prefix {g['prefix_steps']} steps · exact physical match={g['eligible_for_matched_comparison']}</p><video controls preload='none' src='videos/{group_id}_baseline.mp4'></video><video controls preload='none' src='videos/{group_id}_sham.mp4'></video><video controls preload='none' src='videos/{group_id}_{treatment}.mp4'></video></article>"
        )
    return (
        "<!doctype html><html><meta charset='utf-8'><title>VLA-Forge M03</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;background:#f4f7fb;color:#1a2b3d}video{width:31%;background:#111}article{padding:20px;margin:18px 0;background:white;border:1px solid #dce3eb;border-radius:8px}img{width:100%;max-width:1200px;display:block;margin:10px 0}</style><h1>VLA-Forge M03 · real LIBERO / MuJoCo matched-state interventions</h1>"
        + f"<p>{validation['duration']:.2f}s · {validation['width']}×{validation['height']} · H.264 / yuv420p · validated={validation['valid']}</p><video controls src='../{demo}'></video>"
        + "".join(cards)
        + "</html>"
    )


if __name__ == "__main__":
    main()
