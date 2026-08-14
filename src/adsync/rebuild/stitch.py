"""Stitch retimed segments with crossfades and silence fills."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from adsync.models import SegmentMap
from adsync.rebuild.retime import retime_segment

log = logging.getLogger("adsync")


def stitch_segments(
    y: NDArray[np.floating],
    sr: int,
    segments: list[SegmentMap],
    video_duration: float,
    *,
    crossfade_ms: int = 80,
) -> NDArray[np.float32]:
    """Build the final synced AD waveform from segments with crossfades.

    Inserts silence for gaps, crossfades joins, and clamps the peak.  Accepts
    1-D (mono) or 2-D (channels, samples) input and returns the same layout.
    """
    if not segments:
        return np.asarray(y, dtype=np.float32)

    segments = sorted(segments, key=lambda s: s.dst_start)
    crossfade_samples = int(crossfade_ms * sr / 1000)
    output_len = int(video_duration * sr)

    chunks: list[tuple[int, NDArray[np.float32]]] = []
    for seg in segments:
        retimed = retime_segment(y, sr, seg)
        if retimed.shape[-1] == 0:
            continue
        retimed -= retimed.mean(axis=-1, keepdims=True).astype(np.float32)
        chunks.append((int(seg.dst_start * sr), retimed))

    output_shape = (y.shape[0], output_len) if y.ndim == 2 else (output_len,)
    if not chunks:
        return np.zeros(output_shape, dtype=np.float32)

    output = np.zeros(output_shape, dtype=np.float32)

    for i, (dst_start, chunk) in enumerate(chunks):
        end = dst_start + chunk.shape[-1]

        if dst_start < 0:
            chunk = chunk[..., -dst_start:]
            dst_start = 0
        if end > output_len:
            chunk = chunk[..., : output_len - dst_start]
            end = dst_start + chunk.shape[-1]

        if chunk.shape[-1] == 0:
            continue

        if i > 0:
            prev_end = min(chunks[i - 1][0] + chunks[i - 1][1].shape[-1], output_len)
            overlap = prev_end - dst_start
            if overlap > 0:
                xf_len = min(crossfade_samples, overlap, chunk.shape[-1])
                if xf_len > 0:
                    fade_in = np.linspace(0.0, 1.0, xf_len, dtype=np.float32)
                    fade_out = np.linspace(1.0, 0.0, xf_len, dtype=np.float32)
                    output[..., dst_start: dst_start + xf_len] *= fade_out
                    chunk[..., :xf_len] *= fade_in
                # A clamped stretch can overshoot the next segment's start by
                # far more than the crossfade; that tail would play doubled
                # under the new segment.  The new segment's anchors own this
                # range — silence the stale audio before adding.
                stale_end = min(prev_end, end)
                if stale_end > dst_start + xf_len:
                    output[..., dst_start + xf_len: stale_end] = 0.0

        output[..., dst_start:end] += chunk

    peak = float(np.max(np.abs(output)))
    if peak > 0.99:
        output *= np.float32(0.99 / peak)
        log.debug("Normalized output peak from %.3f to 0.99", peak)

    return output
