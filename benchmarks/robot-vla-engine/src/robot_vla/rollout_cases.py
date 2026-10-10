"""Outcome-stratified illustrations of owned formal rollouts, not a new metric."""

import json
import shutil
from pathlib import Path

from .schema import file_hash


def key(row):
    return str(row["suite"]), int(row["task_id"]), int(row["episode_id"])


def verify_metadata_binding(rows, artifacts):
    metadata_paths = [
        str(Path(r["path"]).resolve()) for r in artifacts if r["role"] == "rollout_metadata"
    ]
    references = [str(Path(r["metadata_path"]).resolve()) for r in rows]
    if (
        len(metadata_paths) != len(set(metadata_paths))
        or len(references) != len(set(references))
        or set(metadata_paths) != set(references)
    ):
        raise ValueError("metadata references must match unique registered artifacts")
    for row in rows:
        if json.loads(Path(row["metadata_path"]).read_text(encoding="utf-8")) != row:
            raise ValueError("metadata self-reference differs from the selected trial")


def associate_videos(rows, artifacts, directory, first_initial_state):
    directory = Path(directory).resolve()
    expected = {
        str(
            (directory / f"{r['suite']}_{int(r['task_id'])}_{first_initial_state}.mp4").resolve()
        ): key(r)
        for r in rows
        if int(r["episode_id"]) == first_initial_state
    }
    videos = [r for r in artifacts if r["role"] == "rollout_video"]
    paths = [str(Path(r["path"]).resolve()) for r in videos]
    if len(paths) != len(set(paths)) or set(paths) != set(expected):
        raise ValueError("registered video identities do not match first initial-state coverage")
    return {expected[p]: r for p, r in zip(paths, videos)}


def select_cases(rows, videos):
    result = []
    for suite in sorted({r["suite"] for r in rows}):
        for success in [True, False]:
            group = [r for r in rows if r["suite"] == suite and r["success"] is success]
            available = sorted([r for r in group if key(r) in videos], key=key)
            selected = available[0] if available else None
            result.append(
                dict(
                    suite=suite,
                    outcome="success" if success else "failure",
                    all_rollouts=len(group),
                    recorded_candidates=len(available),
                    selected=selected,
                    video=videos[key(selected)] if selected else None,
                )
            )
    return result


def copy_verified(source, destination, expected_sha256):
    source, destination = Path(source), Path(destination)
    if file_hash(source) != expected_sha256:
        raise ValueError("case source hash changed")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if file_hash(destination) != expected_sha256:
        raise ValueError("case destination hash differs")
    return destination


def verify_video(path, frames, resolution, control_frequency):
    import av

    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError("expected one video stream")
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        decoded = 0
        for frame in container.decode(stream):
            if [frame.width, frame.height] != list(resolution):
                raise ValueError("video resolution mismatch")
            decoded += 1
    if decoded != frames or rate != control_frequency:
        raise ValueError("video frame count or nominal rate mismatch")
    return dict(
        decoded_frames=decoded,
        resolution=list(resolution),
        nominal_fps=rate,
        frame_semantics="pre-action observations; final post-action state has no video frame",
        time_semantics="simulator playback time, not wall-clock throughput",
    )
