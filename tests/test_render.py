"""Rendering correctness: interpolation fidelity, no dropped AD at segment
boundaries, and no doubled audio where segment video claims overlap."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator

from adsync.align.warp_fit import fit_warp_function
from adsync.models import SegmentMap, WarpPoint
from adsync.rebuild.stitch import stitch_segments
from adsync.rebuild.warp_render import render_from_warp
from adsync.utils.mathx import catmull_rom_interp

SR = 8000


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def test_catmull_rom_exact_at_integers() -> None:
    y = np.random.default_rng(0).normal(size=500).astype(np.float32)
    out = np.empty(500, dtype=np.float32)
    catmull_rom_interp(y, np.arange(500, dtype=np.float64), out)
    assert np.max(np.abs(out - y)) < 1e-6


def test_catmull_rom_beats_linear_at_half_sample() -> None:
    sr = 48000
    t = np.arange(sr) / sr
    y = np.sin(2 * np.pi * 5000.0 * t).astype(np.float32)
    pos = np.arange(100, sr - 100, dtype=np.float64) + 0.5
    ref = np.sin(2 * np.pi * 5000.0 * (pos / sr))

    out = np.empty(len(pos), dtype=np.float32)
    catmull_rom_interp(y, pos, out)
    idx = pos.astype(np.int64)
    linear = y[idx] * (1.0 - (pos - idx)) + y[idx + 1] * (pos - idx)

    err_cubic = _rms(out - ref)
    err_linear = _rms(linear - ref)
    assert err_cubic < 0.25 * err_linear, (err_cubic, err_linear)


def _decoded_path(offset_fn, ad_dur: float, step: float = 2.0) -> list[WarpPoint]:
    pts = []
    t = step
    while t < ad_dur - step:
        pts.append(WarpPoint(
            source_time=t, target_time=t + offset_fn(t), confidence=0.5,
        ))
        t += step
    return pts


def test_no_ad_dropped_at_discontinuity() -> None:
    """A cut splits the path into two segments; the AD content between the
    last window of one and the first window of the next must still render."""
    ad_dur, vid_dur = 60.0, 75.0
    path = _decoded_path(lambda t: 0.0 if t < 30.0 else 15.0, ad_dur)
    warp_fns, ranges, _ = fit_warp_function(path, ad_dur, vid_dur)
    assert len(warp_fns) == 2

    # Ranges must tile the AD timeline with no gap.
    for (_, a_end), (b_start, _) in zip(ranges, ranges[1:]):
        assert b_start <= a_end + 1e-9, (a_end, b_start)

    y = 0.3 * np.random.default_rng(1).normal(size=int(ad_dur * SR)).astype(np.float32)
    out = render_from_warp(y, SR, warp_fns, ranges, vid_dur, crossfade_ms=80)

    loud = _rms(y)
    # Video seconds that must carry AD audio: right up to the jump (AD ~29 s
    # maps to video ~29 s) and right after it (AD ~29 s maps to video ~44 s).
    for lo, hi in ((27.5, 28.5), (28.5, 29.4), (44.6, 45.5), (46.0, 47.0)):
        seg = out[int(lo * SR): int(hi * SR)]
        assert _rms(seg) > 0.3 * loud, f"AD audio missing in video [{lo}, {hi}]"
    # The video-only gap stays silent.
    gap = out[int(31.0 * SR): int(43.0 * SR)]
    assert _rms(gap) < 0.02 * loud


def test_no_doubled_audio_on_overlapping_segments() -> None:
    """Adjacent segments whose video claims overlap by ~1 s (chain tolerance)
    must not both play at full level past the crossfade."""
    ad_dur, vid_dur = 60.0, 60.0
    y = 0.3 * np.random.default_rng(2).normal(size=int(ad_dur * SR)).astype(np.float32)

    fn_a = PchipInterpolator([0.0, 30.0], [0.0, 30.0])          # offset 0
    fn_b = PchipInterpolator([30.0, 60.0], [29.0, 59.0])        # offset -1
    out = render_from_warp(
        y, SR, [fn_a, fn_b], [(0.0, 30.0), (30.0, 60.0)], vid_dur,
        crossfade_ms=80,
    )

    base = _rms(out[int(26.0 * SR): int(28.5 * SR)])
    overlap = _rms(out[int(29.2 * SR): int(29.95 * SR)])
    assert overlap < 1.25 * base, (overlap, base)


def test_stitch_overshoot_not_doubled() -> None:
    """A clamped stretch overshooting the next segment's start must fade out,
    not play doubled under it."""
    ad_dur = 60.0
    y = 0.3 * np.random.default_rng(3).normal(size=int(ad_dur * SR)).astype(np.float32)
    segs = [
        SegmentMap(src_start=0.0, src_end=30.0, dst_start=0.0, dst_end=30.0,
                   offset=0.0, stretch=1.0, confidence=0.9),
        SegmentMap(src_start=30.0, src_end=60.0, dst_start=25.0, dst_end=55.0,
                   offset=-5.0, stretch=1.0, confidence=0.9),
    ]
    out = stitch_segments(y, SR, segs, 60.0, crossfade_ms=80)

    base = _rms(out[int(20.0 * SR): int(24.0 * SR)])
    overlap = _rms(out[int(25.5 * SR): int(29.5 * SR)])
    assert overlap < 1.25 * base, (overlap, base)
