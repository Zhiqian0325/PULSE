"""Explicit training sampling, separate from the native evaluation interval."""

from random import Random

import numpy as np


def tiled_clips(indices: range, length: int = 8):
    ids = tuple(indices)
    if not ids:
        raise ValueError("Cannot sample an empty interval")
    starts = sorted(
        {min(i, max(0, len(ids) - length)) for i in range(0, len(ids), length)}
    )
    return [ids[start : start + length] for start in starts]


def refinement_clip(frame_count: int, ed: int, es: int, seed: int, length: int = 20):
    first, last = sorted((ed, es))
    if not 0 <= first < last < frame_count or length < 2:
        raise ValueError("Invalid source length or annotated endpoints")
    if frame_count < length:
        return tuple(range(frame_count))
    if last - first + 1 <= length:
        start = Random(seed).randint(
            max(0, last - length + 1), min(first, frame_count - length)
        )
        return tuple(range(start, start + length))
    return tuple(np.rint(np.linspace(first, last, length)).astype(int).tolist())
