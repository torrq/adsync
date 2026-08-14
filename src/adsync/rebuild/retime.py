"""Retime AD audio segments according to the segment map."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from adsync.models import SegmentMap
from adsync.utils.mathx import catmull_rom_interp

log = logging.getLogger("adsync")

_BLOCK_SECONDS = 30


def retime_segment(
    y: NDArray[np.floating],
    sr: int,
    seg: SegmentMap,
) -> NDArray[np.float32]:
    """Cut [src_start, src_end] from *y* and resample to seg.stretch ratio.

    Accepts 1-D (mono) or 2-D (channels, samples) input and returns the same
    layout.  Cubic interpolation in 30-second blocks keeps memory bounded
    on full-length films.  At typical stretch values (0.99–1.01) the ~17-cent
    pitch shift is imperceptible.

    Returns an owned float32 array — the caller is free to mutate it.
    """
    start_sample = int(seg.src_start * sr)
    end_sample = int(seg.src_end * sr)
    src = y[..., start_sample:end_sample]
    src_len = src.shape[-1]

    if src_len == 0:
        return _empty_like(y, 0)

    if abs(seg.stretch - 1.0) <= 1e-6:
        return src.astype(np.float32, copy=True)

    target_len = int(src_len * seg.stretch)
    if target_len == 0:
        return _empty_like(y, 0)

    log.debug(
        "Resampling [%.1f–%.1f] stretch=%.6f (%d → %d samples)",
        seg.src_start, seg.src_end, seg.stretch, src_len, target_len,
    )

    src = np.ascontiguousarray(src, dtype=np.float32)
    out = _empty_like(y, target_len)
    scale = (src_len - 1) / max(target_len - 1, 1)
    block = sr * _BLOCK_SECONDS

    for start in range(0, target_len, block):
        end = min(start + block, target_len)
        pos = np.arange(start, end, dtype=np.float64)
        pos *= scale
        catmull_rom_interp(src, pos, out[..., start:end])

    return out


def _empty_like(y: NDArray[np.floating], n: int) -> NDArray[np.float32]:
    if y.ndim == 1:
        return np.zeros(n, dtype=np.float32) if n == 0 else np.empty(n, dtype=np.float32)
    shape = (y.shape[0], n)
    return np.zeros(shape, dtype=np.float32) if n == 0 else np.empty(shape, dtype=np.float32)
