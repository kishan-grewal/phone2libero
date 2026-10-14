"""Frame access by presentation timestamp (phone video is variable frame rate)."""

from __future__ import annotations

from collections.abc import Iterator

import av
import numpy as np


def iter_frames(path: str, every: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, time_s, RGB uint8 HxWx3) for every `every`-th frame."""
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, fr in enumerate(c.decode(s)):
            if i % every == 0:
                yield i, float(fr.pts * s.time_base), fr.to_ndarray(format="rgb24")


def frame_at(path: str, t: float) -> np.ndarray:
    for _, ts, img in iter_frames(path):
        if ts >= t:
            return img
    raise ValueError(f"{path} has no frame at {t}s")


def video_info(path: str) -> dict:
    with av.open(path) as c:
        s = c.streams.video[0]
        return {
            "container_metadata": dict(c.metadata),
            "codec": s.codec_context.name,
            "profile": s.codec_context.profile,
            "width": s.width,
            "height": s.height,
            "n_frames": s.frames,
            "duration_s": float(s.duration * s.time_base),
            "nominal_fps": float(s.base_rate),
            "mean_fps": s.frames / float(s.duration * s.time_base),
        }
