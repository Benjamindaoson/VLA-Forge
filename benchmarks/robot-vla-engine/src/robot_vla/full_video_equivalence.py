"""Bounded full-frame RGB/PTS comparison, independent of policy capability."""

import hashlib
from itertools import islice

import numpy as np


def compare_stream(reference, candidate, *, expected_frames, height, width, batch_frames=32):
    if any(type(v) is not int or v < 1 for v in [expected_frames, height, width, batch_frames]):
        raise ValueError("positive integer stream dimensions required")
    digests = [hashlib.sha256(), hashlib.sha256()]
    seen = bad_frames = bad_channels = max_delta = 0
    max_pts_delta = 0.0
    examples = []
    previous = [None, None]
    first_pts = [None, None]
    reference = iter(reference)
    while True:
        batch = list(islice(reference, batch_frames))
        if not batch:
            break
        count = len(batch)
        if seen + count > expected_frames:
            raise ValueError("reference contains extra frames")
        left = np.stack([v[0] for v in batch])
        lpts = np.array([v[1] for v in batch], dtype=np.float64)
        right, rpts = candidate(list(range(seen, seen + count)))
        right, rpts = np.asarray(right), np.asarray(rpts, dtype=np.float64)
        for side, pixels, pts in [(0, left, lpts), (1, right, rpts)]:
            if pixels.dtype != np.uint8 or pixels.shape != (count, height, width, 3):
                raise ValueError("RGB dtype/shape mismatch")
            if pts.shape != (count,) or not np.isfinite(pts).all() or (pts < 0).any():
                raise ValueError("invalid presentation timestamps")
            if (np.diff(pts) <= 0).any() or previous[side] is not None and pts[0] <= previous[side]:
                raise ValueError("presentation timestamps are not strictly increasing")
            if first_pts[side] is None:
                first_pts[side] = float(pts[0])
            previous[side] = float(pts[-1])
            digests[side].update(pixels.tobytes())
        max_pts_delta = max(max_pts_delta, float(np.max(np.abs(lpts - rpts))))
        unequal = left != right
        mismatch = np.flatnonzero(unequal.reshape(count, -1).any(axis=1))
        if len(mismatch):
            bad_frames += len(mismatch)
            bad_channels += int(unequal.sum())
            max_delta = max(
                max_delta, int(np.abs(left.astype(np.int16) - right.astype(np.int16)).max())
            )
            examples.extend((seen + int(i)) for i in mismatch[: max(0, 50 - len(examples))])
        seen += count
    if seen != expected_frames:
        raise ValueError("reference frame count differs from candidate metadata")
    return dict(
        frames=seen,
        rgb_exact=bad_frames == 0,
        pts_equivalent=max_pts_delta <= 1e-6,
        pyav_rgb_sha256=digests[0].hexdigest(),
        torchcodec_rgb_sha256=digests[1].hexdigest(),
        mismatched_frames=bad_frames,
        mismatched_channels=bad_channels,
        max_abs_channel_difference=max_delta,
        max_pts_difference_seconds=max_pts_delta,
        mismatch_frame_examples=examples,
        first_pts_seconds=first_pts,
        last_pts_seconds=previous,
        pts_tolerance_seconds=1e-6,
        batch_frames=batch_frames,
    )


def compare_file(path, *, height=256, width=256, fps=10.0):
    import av
    from lerobot.datasets.video_utils import VideoDecoderCache

    cache = VideoDecoderCache(max_size=1)
    try:
        decoder = cache.get_decoder(str(path))
        meta = decoder.metadata
        if (
            meta.height != height
            or meta.width != width
            or not np.isclose(meta.average_fps, fps, rtol=0, atol=1e-9)
        ):
            raise ValueError("video metadata differs from expected source geometry/fps")
        with av.open(str(path)) as container:
            stream = container.streams.video[0]

            def reference():
                for frame in container.decode(stream):
                    if frame.pts is None:
                        raise ValueError("reference video contains missing PTS")
                    yield frame.to_ndarray(format="rgb24"), float(frame.pts * stream.time_base)

            def candidate(indices):
                batch = decoder.get_frames_at(indices=indices)
                return batch.data.numpy().transpose(0, 2, 3, 1), batch.pts_seconds.numpy()

            return compare_stream(
                reference(), candidate, expected_frames=meta.num_frames, height=height, width=width
            )
    finally:
        cache.clear()
