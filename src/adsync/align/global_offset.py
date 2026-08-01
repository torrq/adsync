"""Mode 1 — Global offset detection via cross-correlation on raw audio.

Uses downsampled raw waveforms for a robust match immune to periodic
onset patterns (musical beats), then refines with onset features.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from adsync.models import FeatureBundle

log = logging.getLogger("adsync")


def estimate_global_offset(
    video_features: FeatureBundle,
    ad_features: FeatureBundle,
    *,
    max_offset_sec: float = 120.0,
    y_vid: NDArray | None = None,
    y_ad: NDArray | None = None,
    sr: int = 16000,
) -> tuple[float, float, list[float]]:
    """Estimate a single global offset (seconds) and a confidence score.

    Returns (offset_seconds, confidence, window_offsets) where a **positive**
    offset means the AD track should be shifted *later* relative to the video
    timeline.  *window_offsets* holds the per-probe-window offsets — their
    spread tells downstream stages how far offsets wander (e.g. across an
    edit discontinuity), which sizes the warp search radius.

    When raw audio arrays *y_vid* and *y_ad* are provided, the offset is
    computed on downsampled raw waveforms (robust to musical periodicity).
    Falls back to onset features if raw audio is not available.
    """
    hop = video_features.hop_length
    feat_sr = video_features.sr
    sec_per_frame = hop / feat_sr

    if y_vid is not None and y_ad is not None:
        offset_sec, best_score, window_offsets = _raw_audio_offset(
            y_vid, y_ad, sr, max_offset_sec,
        )
        # Verify with onset features at the detected offset
        vid_onset = video_features.onset.astype(np.float64)
        ad_onset = ad_features.onset.astype(np.float64)
        best_lag = int(round(offset_sec / sec_per_frame))
        region_scores = _verify_across_regions(
            _znorm(vid_onset), _znorm(ad_onset), best_lag, n_regions=5,
        )
        mean_region = float(np.mean(region_scores)) if region_scores else 0.0
        confidence = 0.6 * best_score + 0.4 * mean_region
    else:
        # Fallback: onset-only when raw audio isn't provided.
        vid_onset = video_features.onset.astype(np.float64)
        ad_onset = ad_features.onset.astype(np.float64)
        vid_onset = _znorm(vid_onset)
        ad_onset = _znorm(ad_onset)
        max_lag = int(max_offset_sec / sec_per_frame)
        corr = _norm_cross_correlation(vid_onset, ad_onset, max_lag)
        best_lag = int(np.argmax(corr)) - max_lag
        best_score = float(corr[best_lag + max_lag])
        offset_sec = best_lag * sec_per_frame
        region_scores = _verify_across_regions(vid_onset, ad_onset, best_lag, n_regions=5)
        mean_region = float(np.mean(region_scores)) if region_scores else 0.0
        confidence = 0.5 * best_score + 0.5 * mean_region
        window_offsets = [offset_sec]

    confidence = max(0.0, min(1.0, confidence))

    log.info(
        "Global offset: %.3f s  (peak=%.3f, regions=%.3f, conf=%.3f)",
        offset_sec, best_score, mean_region, confidence,
    )
    return offset_sec, confidence, window_offsets


def _raw_audio_offset(
    y_vid: NDArray,
    y_ad: NDArray,
    sr: int,
    max_offset_sec: float,
) -> tuple[float, float, list[float]]:
    """Find offset using multi-window normalized cross-correlation at full SR.

    Takes several 30-second windows from the video audio and searches for
    each in the AD audio over the full ±max_offset range.  Uses per-position
    energy normalization and FFT-based convolution for speed.
    """
    win_sec = 30.0
    n_windows = 5
    win = int(win_sec * sr)

    # Place windows evenly across the shorter track
    usable = min(len(y_vid), len(y_ad)) - win
    if usable < win:
        return 0.0, 0.0, []

    offsets: list[float] = []
    scores_list: list[float] = []

    test_starts = np.linspace(win, usable - win, n_windows).astype(int)

    for v_start in test_starts:
        v_seg = y_vid[v_start:v_start + win].astype(np.float64)
        v_seg -= np.mean(v_seg)
        v_energy = np.sqrt(np.sum(v_seg ** 2))
        if v_energy < 1e-10:
            continue

        # AD search region: cover all offsets from 0 to +max_offset
        # offset = v_start - ad_match_pos → ad_match_pos = v_start - offset
        # For offset in [0, max_offset]: ad ranges from v_start to v_start - max_offset
        a_start = max(0, v_start - int(max_offset_sec * sr))
        a_end = min(len(y_ad), v_start + win + int(max_offset_sec * sr))
        if a_end - a_start < win:
            continue

        a_region = y_ad[a_start:a_end].astype(np.float64)
        a_region -= np.mean(a_region)

        # FFT-based sliding cross-correlation
        raw_corr = fftconvolve(a_region, v_seg[::-1], mode="valid")
        npos = len(raw_corr)
        if npos == 0:
            continue

        # Per-position energy normalization (critical for accuracy)
        a_sq = a_region ** 2
        cs = np.empty(len(a_sq) + 1, dtype=np.float64)
        cs[0] = 0.0
        np.cumsum(a_sq, out=cs[1:])
        a_norms = np.sqrt(np.maximum(cs[win:win + npos] - cs[:npos], 1e-20))
        norm_corr = raw_corr / (v_energy * a_norms)

        peak_idx = int(np.argmax(norm_corr))
        score = float(norm_corr[peak_idx])

        if score < 0.1:
            continue

        # Sub-sample parabolic refinement
        sub = 0.0
        if 0 < peak_idx < len(norm_corr) - 1:
            y_prev = float(norm_corr[peak_idx - 1])
            y_peak = float(norm_corr[peak_idx])
            y_next = float(norm_corr[peak_idx + 1])
            denom = y_prev - 2.0 * y_peak + y_next
            if abs(denom) > 1e-12:
                sub = 0.5 * (y_prev - y_next) / denom

        ad_match_pos = a_start + peak_idx + sub
        local_offset = (v_start - ad_match_pos) / sr
        offsets.append(local_offset)
        scores_list.append(score)
        log.debug("  Window at t=%.0fs: offset=%.4f s, score=%.4f",
                  v_start / sr, local_offset, score)

    if not offsets:
        return 0.0, 0.0, []

    offsets_arr = np.array(offsets)
    scores_arr = np.array(scores_list)

    # ── Scatter detection ────────────────────────────────────────────
    # When windows land on different offsets (e.g. a timing discontinuity
    # in the source material), the weighted mean is meaningless.  Detect
    # this by checking the spread of per-window offsets and, if they
    # cluster into distinct groups, pick the dominant cluster instead of
    # averaging across the gap.
    offset_std = float(np.std(offsets_arr))
    if offset_std > 1.0 and len(offsets_arr) >= 3:
        # Try to find the dominant cluster: group offsets within 0.5 s
        sorted_idx = np.argsort(offsets_arr)
        best_cluster: list[int] = []
        for i in range(len(sorted_idx)):
            cluster = [int(sorted_idx[i])]
            for j in range(i + 1, len(sorted_idx)):
                if offsets_arr[sorted_idx[j]] - offsets_arr[sorted_idx[i]] < 0.5:
                    cluster.append(int(sorted_idx[j]))
                else:
                    break
            if len(cluster) > len(best_cluster):
                best_cluster = cluster

        cluster_scores = scores_arr[best_cluster]
        cluster_offsets = offsets_arr[best_cluster]
        w_cluster = cluster_scores ** 2
        final_offset = float(np.average(cluster_offsets, weights=w_cluster))
        best_score = float(np.max(cluster_scores))

        # Penalise confidence — a split indicates a timing discontinuity
        # that offset mode cannot represent; piecewise should handle it.
        penalty = len(best_cluster) / len(offsets_arr)
        best_score *= penalty

        log.warning(
            "Offset scatter detected (std=%.2f s): windows disagree — "
            "likely a timing discontinuity. Using dominant cluster "
            "(%d/%d windows, offset=%.4f s). Confidence penalised to %.3f.",
            offset_std, len(best_cluster), len(offsets_arr),
            final_offset, best_score,
        )
    else:
        # Normal case: consistent windows
        w = scores_arr ** 2
        final_offset = float(np.average(offsets_arr, weights=w))
        best_score = float(np.max(scores_arr))

    log.debug("Raw audio offset: %.4f s (best_score=%.4f, %d windows)",
              final_offset, best_score, len(offsets))
    return final_offset, best_score, offsets


def _znorm(x: NDArray) -> NDArray:
    """Zero-mean, unit-variance normalization."""
    std = np.std(x)
    if std < 1e-10:
        return x - np.mean(x)
    return (x - np.mean(x)) / std


def _norm_cross_correlation(a: NDArray, b: NDArray, max_lag: int) -> NDArray:
    """Normalised cross-correlation over [-max_lag, +max_lag]."""
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]

    full_corr = fftconvolve(a, b[::-1], mode="full")
    # The zero-lag position in 'full' mode is at index len(b)-1
    center = len(b) - 1
    start = max(center - max_lag, 0)
    end = min(center + max_lag + 1, len(full_corr))

    segment = full_corr[start:end]
    norm = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if norm < 1e-10:
        return np.zeros(2 * max_lag + 1)

    result = segment / norm

    # Pad to exactly 2*max_lag+1 if edge clipping occurred
    if len(result) < 2 * max_lag + 1:
        padded = np.zeros(2 * max_lag + 1)
        offset = (center - max_lag) - start
        padded[max(0, -offset):max(0, -offset) + len(result)] = result
        result = padded

    return result


def _verify_across_regions(
    vid: NDArray, ad: NDArray, lag: int, n_regions: int = 5
) -> list[float]:
    """Check that the detected lag holds across *n_regions* evenly spaced sections."""
    n = min(len(vid), len(ad))
    region_len = n // (n_regions + 1)
    if region_len < 100:
        return []

    scores: list[float] = []
    for i in range(1, n_regions + 1):
        start = i * region_len - region_len // 2
        end = start + region_len
        if end > n:
            break
        v_seg = vid[start:end]
        a_start = start - lag
        a_end = end - lag
        if a_start < 0 or a_end > len(ad):
            continue
        a_seg = ad[a_start:a_end]
        if len(v_seg) != len(a_seg):
            continue
        dot = np.dot(v_seg, a_seg)
        norm = np.sqrt(np.sum(v_seg ** 2) * np.sum(a_seg ** 2))
        if norm > 1e-10:
            scores.append(float(dot / norm))
    return scores
