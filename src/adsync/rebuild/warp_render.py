"""Render the synced AD track from a continuous warp function.

For each output sample the renderer maps t_video → t_ad through the inverse
of the warp and interpolates the source.  Done in 30-second blocks so peak
memory stays bounded regardless of film length.  Mono and multi-channel AD
sources are both supported — the same warp is applied to every channel.

Plain interpolation behaves like a varispeed tape: wherever the warp's local
slope deviates from 1.0, pitch follows it.  Regions where the deviation would
be audible are re-rendered with WSOLA (frames copied unresampled and
overlap-added at warp-tracking positions) so stretch never touches pitch.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator

from adsync.utils.mathx import catmull_rom_interp

log = logging.getLogger("adsync")

_BLOCK_SECONDS = 30

# ── Pitch preservation (WSOLA) ───────────────────────────────────────────────
_SLOPE_TOL = 0.005            # ≈ 9 cents — engage WSOLA beyond this deviation
_SLOPE_SCAN_STEP = 0.25       # seconds between slope samples when scanning
_REGION_MARGIN = 0.4          # seconds of padding around detected regions
_REGION_MERGE_GAP = 1.0       # seconds — merge regions closer than this
_WSOLA_WIN_MS = 40            # analysis/synthesis window
_WSOLA_SEARCH_MS = 10         # ± similarity search tolerance
_WSOLA_SPAN_SECONDS = 120     # bounded sub-spans for long regions
_EDGE_XFADE_MS = 30           # crossfade between varispeed and WSOLA renders


def render_from_warp(
    y_ad: NDArray[np.floating],
    sr: int,
    warp_fns: list[PchipInterpolator],
    segment_ranges: list[tuple[float, float]],
    video_duration: float,
    *,
    crossfade_ms: int = 200,
) -> NDArray[np.floating]:
    """Render the synced AD waveform using continuous warp function(s)."""
    output_len = int(video_duration * sr)
    y_ad = np.ascontiguousarray(y_ad, dtype=np.float32)
    ad_len = y_ad.shape[-1]

    if y_ad.ndim == 1:
        output = np.zeros(output_len, dtype=np.float32)
    else:
        output = np.zeros((y_ad.shape[0], output_len), dtype=np.float32)

    if len(warp_fns) == 1:
        _render_single_segment(
            y_ad, sr, warp_fns[0], segment_ranges[0],
            video_duration, output, ad_len,
        )
    else:
        _render_multi_segment(
            y_ad, sr, warp_fns, segment_ranges,
            video_duration, output, ad_len, crossfade_ms,
        )

    dc = output.mean(axis=-1, keepdims=True)
    if np.any(np.abs(dc) > 1e-6):
        output -= dc.astype(np.float32)

    peak = float(np.max(np.abs(output)))
    if peak > 0.99:
        output *= np.float32(0.99 / peak)
        log.debug("Normalized output peak from %.3f to 0.99", peak)

    return output


def _render_single_segment(
    y_ad: NDArray[np.float32],
    sr: int,
    warp_fn: PchipInterpolator,
    segment_range: tuple[float, float],
    video_duration: float,
    output: NDArray[np.float32],
    ad_len: int,
) -> None:
    inverse_fn = _invert_pchip(warp_fn, segment_range, video_duration)

    vid_start = max(0.0, float(warp_fn(segment_range[0])))
    vid_end = min(video_duration, float(warp_fn(segment_range[1])))
    valid_lo = max(0, math.ceil(vid_start * sr))
    valid_hi = min(output.shape[-1], math.floor(vid_end * sr) + 1)

    block = sr * _BLOCK_SECONDS
    for start in range(valid_lo, valid_hi, block):
        end = min(start + block, valid_hi)
        _interp_into(y_ad, sr, inverse_fn, output[..., start:end], start, end)

    _apply_pitch_regions(y_ad, sr, inverse_fn, ad_len, output, valid_lo, valid_hi)


def _render_multi_segment(
    y_ad: NDArray[np.float32],
    sr: int,
    warp_fns: list[PchipInterpolator],
    segment_ranges: list[tuple[float, float]],
    video_duration: float,
    output: NDArray[np.float32],
    ad_len: int,
    crossfade_ms: int,
) -> None:
    crossfade_samples = int(crossfade_ms * sr / 1000)
    output_len = output.shape[-1]
    block = sr * _BLOCK_SECONDS
    buf_shape = (y_ad.shape[0], block) if y_ad.ndim == 2 else (block,)
    buf = np.empty(buf_shape, dtype=np.float32)
    placed: list[tuple[int, int]] = []

    for warp_fn, seg_range in zip(warp_fns, segment_ranges):
        src_start, src_end = seg_range
        vid_start = max(0.0, min(video_duration, float(warp_fn(src_start))))
        vid_end = max(0.0, min(video_duration, float(warp_fn(src_end))))
        if vid_end <= vid_start:
            continue

        out_start = max(0, min(output_len, math.ceil(vid_start * sr)))
        out_end = max(0, min(output_len, math.floor(vid_end * sr) + 1))
        if out_end <= out_start:
            continue

        inverse_fn = _invert_pchip(warp_fn, seg_range, video_duration)

        xf_len = 0
        fade_in: NDArray[np.float32] | None = None
        if placed:
            overlap = placed[-1][1] - out_start
            if overlap > 0:
                xf_len = min(crossfade_samples, overlap, out_end - out_start)
                if xf_len > 0:
                    t = np.linspace(0.0, np.pi, xf_len, dtype=np.float32)
                    output[..., out_start:out_start + xf_len] *= 0.5 * (1.0 + np.cos(t))
                    fade_in = 0.5 * (1.0 - np.cos(t))
                # Segment video claims may overlap past the crossfade (chain
                # tolerance); the stale tail would play doubled under this
                # segment's render.  This segment owns the range — silence it.
                stale_end = min(placed[-1][1], out_end)
                if stale_end > out_start + xf_len:
                    output[..., out_start + xf_len: stale_end] = 0.0

        for start in range(out_start, out_end, block):
            end = min(start + block, out_end)
            chunk = buf[..., : end - start]
            _interp_into(y_ad, sr, inverse_fn, chunk, start, end)
            if fade_in is not None and start == out_start:
                chunk[..., :xf_len] *= fade_in
            output[..., start:end] += chunk

        placed.append((out_start, out_end))

        # Fix pitch inside stretch regions (skip the inter-segment crossfade
        # zone — its faded contribution can't be reconstructed exactly).
        _apply_pitch_regions(
            y_ad, sr, inverse_fn, ad_len, output, out_start + xf_len, out_end,
        )


def _interp_into(
    y_ad: NDArray[np.float32],
    sr: int,
    inverse_fn: PchipInterpolator,
    dest: NDArray[np.float32],
    out_start: int,
    out_end: int,
) -> None:
    t = np.arange(out_start, out_end, dtype=np.float64)
    t /= sr
    src = inverse_fn(t)
    src *= sr
    catmull_rom_interp(y_ad, src, dest)


def _invert_pchip(
    forward_fn: PchipInterpolator,
    segment_range: tuple[float, float],
    video_duration: float,
    grid_step: float = 0.01,
) -> PchipInterpolator:
    """Invert a forward warp (AD→video) to get the inverse (video→AD)."""
    src_start, src_end = segment_range
    n_points = max(10, int((src_end - src_start) / grid_step))
    ad_times = np.linspace(src_start, src_end, n_points)
    vid_times = forward_fn(ad_times)

    # Guard against tiny numerical noise that can break strict monotonicity.
    for i in range(1, len(vid_times)):
        if vid_times[i] <= vid_times[i - 1]:
            vid_times[i] = vid_times[i - 1] + 1e-9

    return PchipInterpolator(vid_times, ad_times)


# ── Pitch-preserving re-render of stretch regions ────────────────────────────


def _apply_pitch_regions(
    y_ad: NDArray[np.float32],
    sr: int,
    inverse_fn: PchipInterpolator,
    ad_len: int,
    output: NDArray[np.float32],
    lo: int,
    hi: int,
) -> None:
    """Re-render stretch regions pitch-preserved and splice them into *output*.

    Detects spans where the warp slope deviates enough from 1.0 to audibly
    shift pitch under plain interpolation, renders those spans with WSOLA,
    and adds ``wsola − interp`` so the varispeed contribution is replaced in
    place while audio from other segments already mixed into the same span is
    preserved.  Edges crossfade back to the interpolated render.
    """
    win_len = int(sr * _WSOLA_WIN_MS / 1000)
    if hi <= lo or ad_len <= 4 * win_len:
        return
    regions = _stretch_regions(inverse_fn, sr, lo, hi)
    if not regions:
        return

    xf = max(1, int(sr * _EDGE_XFADE_MS / 1000))
    total_s = 0.0
    for reg_lo, reg_hi in regions:
        reg_len = reg_hi - reg_lo
        if reg_len < 2 * win_len:
            continue
        e = min(xf, reg_len // 2)
        ramp_up = 0.5 * (1.0 - np.cos(np.pi * np.arange(e, dtype=np.float32) / e))
        ramp_down = ramp_up[::-1]

        for sub_lo, sub_hi, wsola in _wsola_spans(
            y_ad, sr, inverse_fn, ad_len, reg_lo, reg_hi,
        ):
            n = sub_hi - sub_lo
            interp = np.empty_like(wsola)
            _interp_into(y_ad, sr, inverse_fn, interp, sub_lo, sub_hi)
            np.subtract(wsola, interp, out=wsola)

            head = sub_lo - reg_lo  # sub-span offset within the region
            # Fade-in over region-relative [0, e).
            m = min(max(e - head, 0), n)
            if m > 0:
                wsola[..., :m] *= ramp_up[head:head + m]
            # Fade-out over region-relative [reg_len − e, reg_len).
            fo = (reg_len - e) - head
            j = max(fo, 0)
            if j < n:
                wsola[..., j:] *= ramp_down[j - fo : (j - fo) + (n - j)]
            output[..., sub_lo:sub_hi] += wsola

        total_s += reg_len / sr

    if total_s > 0:
        log.info(
            "Pitch-preserving WSOLA re-render: %d region(s), %.1f s total",
            len(regions), total_s,
        )


def _stretch_regions(
    inverse_fn: PchipInterpolator,
    sr: int,
    lo: int,
    hi: int,
) -> list[tuple[int, int]]:
    """Output-sample ranges where the warp slope would audibly shift pitch."""
    t0, t1 = lo / sr, hi / sr
    if t1 - t0 < 2 * _SLOPE_SCAN_STEP:
        return []
    n = int((t1 - t0) / _SLOPE_SCAN_STEP) + 2
    ts = np.linspace(t0, t1, n)
    slope = inverse_fn.derivative()(ts)
    dev = np.abs(slope - 1.0) > _SLOPE_TOL
    if not bool(np.any(dev)):
        return []

    flags = dev.astype(np.int8)
    d = np.diff(flags)
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if flags[0]:
        starts.insert(0, 0)
    if flags[-1]:
        ends.append(n)

    merge_gap = int(_REGION_MERGE_GAP * sr)
    regions: list[tuple[int, int]] = []
    for a, b in zip(starts, ends):
        ra = max(lo, int((ts[a] - _REGION_MARGIN) * sr))
        rb = min(hi, int((ts[min(b, n - 1)] + _REGION_MARGIN) * sr))
        if rb <= ra:
            continue
        if regions and ra - regions[-1][1] < merge_gap:
            regions[-1] = (regions[-1][0], rb)
        else:
            regions.append((ra, rb))
    return regions


def _wsola_spans(
    y_ad: NDArray[np.float32],
    sr: int,
    inverse_fn: PchipInterpolator,
    ad_len: int,
    out_lo: int,
    out_hi: int,
) -> Iterator[tuple[int, int, NDArray[np.float32]]]:
    """Yield ``(lo, hi, rendered)`` WSOLA sub-spans covering [out_lo, out_hi).

    Frames are copied from the source *unresampled* — constant pitch — at
    positions tracking the inverse warp, each nudged within ±_WSOLA_SEARCH_MS
    to maximise waveform similarity with the natural continuation of the
    previous frame, then Hann overlap-added.  Long regions are processed in
    bounded sub-spans; OLA tails and the similarity state carry across, so
    sub-span boundaries are seamless.
    """
    N = max(int(sr * _WSOLA_WIN_MS / 1000) & ~1, 128)
    hop = N // 2
    search = max(1, int(sr * _WSOLA_SEARCH_MS / 1000))
    tmpl_len = min(hop, max(64, int(sr * 0.015)))

    idx = np.arange(N, dtype=np.float32)
    win = np.sin(np.pi * idx / N).astype(np.float32) ** 2  # periodic Hann: COLA at N/2

    multi = y_ad.ndim == 2
    n_total = out_hi - out_lo
    span = max(8 * hop, (int(sr * _WSOLA_SPAN_SECONDS) // hop) * hop)

    carry: NDArray[np.float32] | None = None
    carry_w: NDArray[np.float32] | None = None
    prev_src = -1

    for s0 in range(0, n_total, span):
        s1 = min(s0 + span, n_total)
        n_sub = s1 - s0
        buf_shape = (y_ad.shape[0], n_sub + N) if multi else (n_sub + N,)
        buf = np.zeros(buf_shape, dtype=np.float32)
        wsum = np.zeros(n_sub + N, dtype=np.float32)
        if carry is not None and carry_w is not None:
            buf[..., :N] += carry
            wsum[:N] += carry_w

        # Frame starts on the global hop grid within [s0, s1).
        ks = np.arange(s0 // hop, -(-s1 // hop))
        starts = ks * hop
        centers = (out_lo + starts + hop) / sr
        tgt = np.clip(
            inverse_fn(centers) * sr - hop, 0.0, ad_len - N - 1.0,
        ).astype(np.int64)

        # Mono guide covering every candidate read in this sub-span.
        g_lo = max(0, int(tgt.min()) - (search + N))
        g_hi = min(ad_len, int(tgt.max()) + 2 * N + search)
        guide = y_ad[..., g_lo:g_hi]
        if multi:
            guide = guide.mean(axis=0, dtype=np.float32)
        guide = np.ascontiguousarray(guide, dtype=np.float32)
        cs = np.empty(guide.shape[0] + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(np.square(guide, dtype=np.float64), out=cs[1:])

        for f in range(len(tgt)):
            t = int(tgt[f])
            if prev_src < 0:
                src = t
            else:
                nat = prev_src + hop  # perfect-continuation position
                lo_s = max(0, t - search)
                hi_s = min(ad_len - N, t + search)
                if hi_s <= lo_s or nat < g_lo or nat + tmpl_len > g_hi:
                    src = t
                else:
                    src = lo_s + _best_offset(
                        guide, cs, g_lo, nat, lo_s, hi_s, tmpl_len,
                    )
            s_local = int(starts[f]) - s0
            buf[..., s_local:s_local + N] += y_ad[..., src:src + N] * win
            wsum[s_local:s_local + N] += win
            prev_src = src

        rendered = buf[..., :n_sub]
        rendered /= np.maximum(wsum[:n_sub], np.float32(1e-2))
        carry = buf[..., n_sub:].copy()
        carry_w = wsum[n_sub:].copy()
        yield out_lo + s0, out_lo + s1, rendered


def _best_offset(
    guide: NDArray[np.float32],
    cs: NDArray[np.float64],
    g_lo: int,
    nat: int,
    lo_s: int,
    hi_s: int,
    tmpl_len: int,
) -> int:
    """Offset in [0, hi_s − lo_s] whose frame best continues the previous one.

    Maximises normalised cross-correlation between each candidate frame
    ``guide[lo_s+o : lo_s+o+tmpl_len]`` and the natural-continuation template
    ``guide[nat : nat+tmpl_len]``.  Indices are absolute AD-sample positions;
    *guide* starts at *g_lo* and *cs* is the zero-prefixed cumulative energy
    of *guide*.
    """
    tpl = guide[nat - g_lo : nat - g_lo + tmpl_len]
    a = lo_s - g_lo
    b = hi_s - g_lo + tmpl_len
    num = np.correlate(guide[a:b], tpl)
    off = np.arange(num.shape[0])
    energy = cs[a + off + tmpl_len] - cs[a + off]
    t_norm = float(np.dot(tpl, tpl)) ** 0.5
    denom = np.sqrt(np.maximum(energy, 1e-12)) * max(t_norm, 1e-6)
    return int(np.argmax(num / denom))
