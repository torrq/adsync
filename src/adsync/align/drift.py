"""Mode 2 — Drift estimation (linear timing mismatch)."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.signal import fftconvolve

from adsync.models import Anchor, FeatureBundle

log = logging.getLogger("adsync")


def estimate_drift(
    video_features: FeatureBundle,
    ad_features: FeatureBundle,
    *,
    n_test_points: int = 11,
    window_sec: float = 30.0,
    search_sec: float = 5.0,
    offset_hint: float | None = None,
    y_vid: NDArray | None = None,
    y_ad: NDArray | None = None,
    audio_sr: int = 16000,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[float, float, list[Anchor], float]:
    """Measure offset at several points and fit a linear drift model.

    Returns (drift_ppm, confidence, anchors, intercept).
    drift_ppm is positive when the AD track is running *faster* than video.
    intercept is the base offset at t=0 from the linear fit.
    If *offset_hint* is provided, local searches are seeded around that offset
    for better accuracy.
    When *y_vid* and *y_ad* are provided, uses raw audio waveforms instead of
    onset features for more reliable local cross-correlation.
    """
    # Use raw audio if available, else fall back to onset features
    if y_vid is not None and y_ad is not None:
        # Downsample to ~4 kHz for fast local cross-correlation
        ds_factor = max(1, audio_sr // 4000)
        ds_sr = audio_sr / ds_factor

        def _ds(y: NDArray) -> NDArray:
            n = len(y) - len(y) % ds_factor
            return np.mean(y[:n].reshape(-1, ds_factor), axis=1)

        vid = _ds(y_vid).astype(np.float64)
        ad = _ds(y_ad).astype(np.float64)
        sec_per_sample = 1.0 / ds_sr
        use_raw = True
    else:
        vid = video_features.onset.astype(np.float64)
        ad = ad_features.onset.astype(np.float64)
        hop = video_features.hop_length
        sr = video_features.sr
        sec_per_sample = hop / sr
        use_raw = False

    vid_dur = len(vid) * sec_per_sample
    ad_dur = len(ad) * sec_per_sample
    min_dur = min(vid_dur, ad_dur)

    # sample roughly evenly-spaced test points, avoiding edges
    margin = window_sec
    if min_dur < 2 * margin + window_sec:
        log.warning("Tracks too short for reliable drift estimation")
        return 0.0, 0.0, [], 0.0

    test_times = np.linspace(margin, min_dur - margin, n_test_points)

    # Convert offset_hint to samples so local search can be seeded.
    # Negate because in _local_offset, positive lag = later in AD,
    # but a positive offset means AD content is EARLIER in its timeline.
    hint_samples = -int(offset_hint / sec_per_sample) if offset_hint else 0

    anchors: list[Anchor] = []
    for i, t in enumerate(test_times):
        if on_progress is not None:
            on_progress(i, n_test_points)
        anchor = _local_offset(
            vid, ad, float(t), sec_per_sample, window_sec, search_sec,
            hint_frames=hint_samples,
        )
        if anchor is not None:
            anchors.append(anchor)

    if len(anchors) < 3:
        log.warning("Too few anchors (%d) for drift estimation", len(anchors))
        return 0.0, 0.0, anchors, 0.0

    # Fit offset(t) = intercept + slope * t on source - target (positive = AD
    # shifts later, matches global_offset), weighted by anchor score so
    # high-quality matches dominate the fit.
    times = np.array([a.source_time for a in anchors])
    offsets = np.array([a.source_time - a.target_time for a in anchors])

    for i, a in enumerate(anchors):
        log.info(
            "  Anchor %2d: t=%7.1fs  offset=%+.4fs  score=%.3f",
            i + 1, a.source_time, a.source_time - a.target_time, a.score,
        )
    weights = np.array([a.score for a in anchors])

    # Weighted least squares via polyfit
    coeffs = np.polyfit(times, offsets, 1, w=weights)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    # drift in ppm
    drift_ppm = slope * 1e6

    # Residuals for confidence (weighted)
    predicted = intercept + slope * times
    residuals = np.abs(offsets - predicted)
    mean_residual = float(np.average(residuals, weights=weights))

    # ── Discontinuity detection ────────────────────────────────────
    # A step-function in offsets (e.g. different edits between AD and
    # video) looks like huge drift + large residuals.  Detect this by
    # checking if offsets cluster at distinct values.
    offset_std = float(np.std(offsets))
    if offset_std > 1.0 and len(offsets) >= 4:
        log.warning(
            "Drift anchors show high offset spread (std=%.2f s) — "
            "likely a timing discontinuity, not linear drift. "
            "Piecewise mode recommended.",
            offset_std,
        )

    # Confidence: small residuals + consistent scores
    score_mean = float(np.mean(weights))
    residual_factor = max(0.0, 1.0 - mean_residual / 0.5)
    confidence = 0.5 * score_mean + 0.5 * residual_factor
    confidence = max(0.0, min(1.0, confidence))

    # Extra penalty for very large residuals (step-function pattern)
    if mean_residual > 1.0:
        confidence *= 0.3
        log.warning(
            "Mean residual %.2f s far exceeds linear model — "
            "forcing low confidence (%.3f) to trigger piecewise fallback.",
            mean_residual, confidence,
        )

    log.info(
        "Drift: %.1f ppm (slope=%.6f, intercept=%.3f s), mean residual=%.4f s, conf=%.3f",
        drift_ppm, slope, intercept, mean_residual, confidence,
    )
    return drift_ppm, confidence, anchors, intercept


def _local_offset(
    vid: NDArray, ad: NDArray,
    center_sec: float,
    sec_per_frame: float,
    window_sec: float,
    search_sec: float,
    hint_frames: int = 0,
) -> Anchor | None:
    """Find best local offset near *center_sec* with sub-frame precision.

    Uses vectorized cross-correlation for speed, then parabolic interpolation
    around the peak for sub-frame accuracy (~1-3 ms instead of ~32 ms).
    """
    win_frames = int(window_sec / sec_per_frame)
    search_frames = int(search_sec / sec_per_frame)

    center_frame = int(center_sec / sec_per_frame)
    v_start = center_frame - win_frames // 2
    v_end = v_start + win_frames

    if v_start < 0 or v_end > len(vid):
        return None

    v_seg = vid[v_start:v_end].copy()
    v_seg -= np.mean(v_seg)
    v_norm = np.sqrt(np.sum(v_seg ** 2))
    if v_norm < 1e-10:
        return None

    # AD search region: centred on (v_start + hint_frames), ±search_frames
    a_region_start = v_start + hint_frames - search_frames
    a_region_end = v_end + hint_frames + search_frames
    a_region_start = max(0, a_region_start)
    a_region_end = min(len(ad), a_region_end)
    if a_region_end - a_region_start < win_frames:
        return None

    a_region = ad[a_region_start:a_region_end].copy()
    a_region -= np.mean(a_region)

    # FFT-based sliding cross-correlation (identical to np.correlate "valid",
    # but O(n log n) — direct correlation at these sizes costs seconds per anchor)
    full_corr = fftconvolve(a_region, v_seg[::-1], mode="valid")
    if len(full_corr) == 0:
        return None

    # Per-position normalization via running sum-of-squares
    a_sq = a_region ** 2
    cumsum = np.empty(len(a_sq) + 1, dtype=np.float64)
    cumsum[0] = 0.0
    np.cumsum(a_sq, out=cumsum[1:])
    n_pos = len(full_corr)
    a_norms_sq = cumsum[win_frames:win_frames + n_pos] - cumsum[:n_pos]
    a_norms = np.sqrt(np.maximum(a_norms_sq, 1e-20))

    norm_corr = full_corr / (v_norm * a_norms)

    # Find integer peak
    best_idx = int(np.argmax(norm_corr))
    best_score = float(norm_corr[best_idx])

    if best_score < 0.2:
        return None

    # Sub-frame parabolic interpolation for ~1-3 ms precision
    sub_offset = 0.0
    if 0 < best_idx < len(norm_corr) - 1:
        y_prev = float(norm_corr[best_idx - 1])
        y_peak = float(norm_corr[best_idx])
        y_next = float(norm_corr[best_idx + 1])
        denom = y_prev - 2.0 * y_peak + y_next
        if abs(denom) > 1e-12:
            sub_offset = 0.5 * (y_prev - y_next) / denom
            best_score = float(y_peak - 0.25 * (y_prev - y_next) * sub_offset)

    # Convert index to lag in frames relative to v_start
    lag_frames = (a_region_start + best_idx) - v_start + sub_offset

    source_time = center_sec
    target_time = center_sec + lag_frames * sec_per_frame

    return Anchor(
        source_time=source_time,
        target_time=target_time,
        score=best_score,
        window=window_sec,
    )
