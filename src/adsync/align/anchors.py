"""Mode 3 — Piecewise anchor search for cut-difference cases.

Uses FFT-based normalized cross-correlation for fast sliding window
matching, following the same pattern as ``drift.py`` and
``global_offset.py``.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from adsync.models import Anchor, FeatureBundle

log = logging.getLogger("adsync")


def find_anchors(
    video_features: FeatureBundle,
    ad_features: FeatureBundle,
    *,
    window_sec: float = 8.0,
    step_sec: float = 2.0,
    search_radius_sec: float = 30.0,
    min_score: float = 0.4,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Anchor]:
    """Scan the AD track in windows and find matching positions in the video.

    *on_progress(current, total)* is called after each window if provided.

    Returns anchors sorted by source_time.
    """
    hop = video_features.hop_length
    sr = video_features.sr
    spf = hop / sr  # seconds per feature frame

    vid = video_features.onset.astype(np.float64)
    ad = ad_features.onset.astype(np.float64)

    win_frames = int(window_sec / spf)
    step_frames = int(step_sec / spf)
    search_frames = int(search_radius_sec / spf)

    raw_anchors: list[Anchor] = []
    ad_positions = list(range(0, len(ad) - win_frames, step_frames))
    total_windows = len(ad_positions)

    for step_i, ad_start in enumerate(ad_positions):
        if on_progress is not None:
            on_progress(step_i, total_windows)
        ad_seg = ad[ad_start : ad_start + win_frames].copy()
        ad_seg -= np.mean(ad_seg)
        ad_energy = np.sqrt(np.sum(ad_seg ** 2))
        if ad_energy < 1e-10:
            continue

        ad_center_sec = (ad_start + win_frames / 2) * spf

        # Search region in video — centered on same nominal position
        nominal_vid_start = ad_start
        search_start = max(0, nominal_vid_start - search_frames)
        search_end = min(len(vid), nominal_vid_start + search_frames + win_frames)

        if search_end - search_start < win_frames:
            continue

        v_region = vid[search_start:search_end].astype(np.float64)
        v_region = v_region - np.mean(v_region)

        # FFT-based sliding cross-correlation
        raw_corr = fftconvolve(v_region, ad_seg[::-1], mode="valid")
        n_pos = len(raw_corr)
        if n_pos == 0:
            continue

        # Per-position energy normalization via cumulative sum of squares
        v_sq = v_region ** 2
        cs = np.empty(len(v_sq) + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(v_sq, out=cs[1:])
        v_norms = np.sqrt(np.maximum(cs[win_frames:win_frames + n_pos] - cs[:n_pos], 1e-20))
        norm_corr = raw_corr / (ad_energy * v_norms)

        peak_idx = int(np.argmax(norm_corr))
        best_score = float(norm_corr[peak_idx])

        if best_score < min_score:
            continue

        # Sub-frame parabolic interpolation for improved precision
        sub = 0.0
        if 0 < peak_idx < len(norm_corr) - 1:
            y_prev = float(norm_corr[peak_idx - 1])
            y_peak = float(norm_corr[peak_idx])
            y_next = float(norm_corr[peak_idx + 1])
            denom = y_prev - 2.0 * y_peak + y_next
            if abs(denom) > 1e-12:
                sub = 0.5 * (y_prev - y_next) / denom
                best_score = float(y_peak - 0.25 * (y_prev - y_next) * sub)

        best_vid_start = search_start + peak_idx + sub
        vid_center_sec = (best_vid_start + win_frames / 2) * spf
        raw_anchors.append(Anchor(
            source_time=ad_center_sec,
            target_time=vid_center_sec,
            score=best_score,
            window=window_sec,
        ))

    # Filter: keep only forward-moving, monotonic anchors
    anchors = _monotonic_filter(raw_anchors)

    log.info("Anchor search: %d raw → %d monotonic", len(raw_anchors), len(anchors))
    return anchors


def _monotonic_filter(anchors: list[Anchor]) -> list[Anchor]:
    """Keep the maximum-score strictly increasing subsequence of target_time.

    A greedy scan would make the first anchor mandatory, so one early false
    match far ahead in the video silently discards every genuine anchor that
    follows; the DP lets any coherent majority outvote it.
    """
    if not anchors:
        return []

    sorted_a = sorted(anchors, key=lambda a: a.source_time)
    n = len(sorted_a)
    targets = np.array([a.target_time for a in sorted_a])
    scores = np.array([a.score for a in sorted_a])

    dp = scores.copy()
    prev = np.full(n, -1, dtype=np.int64)
    for i in range(1, n):
        ok = np.nonzero(targets[:i] < targets[i])[0]
        if len(ok):
            j = ok[np.argmax(dp[ok])]
            cand = dp[j] + scores[i]
            if cand > dp[i]:
                dp[i] = cand
                prev[i] = j

    best = int(np.argmax(dp))
    kept: list[Anchor] = []
    while best != -1:
        kept.append(sorted_a[best])
        best = int(prev[best])
    kept.reverse()
    return kept
