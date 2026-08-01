"""Confidence scoring for alignment results."""

from __future__ import annotations

import logging

import numpy as np

from adsync.models import Anchor, SegmentMap, WarpPath

log = logging.getLogger("adsync")


def compute_confidence(
    anchors: list[Anchor],
    segments: list[SegmentMap],
    ad_duration: float,
    video_duration: float,
    mode: str = "piecewise",
    warp_path: WarpPath | None = None,
    fp_residual_p95_ms: float | None = None,
) -> tuple[float, list[str]]:
    """Compute an overall confidence score and list of warnings.

    *mode* adjusts the scoring strategy:
    - ``"offset"``: relies on segment confidence, coverage, and duration ratio
      (no anchor-based factors since offset mode produces zero anchors).
    - ``"drift"``: includes anchor quality but skips density (drift uses a
      handful of sparse test points by design, not dense anchors).
    - ``"piecewise"``: uses all five factors (original behaviour).
    - ``"warp"``: uses warp path mean confidence, anchor density, and
      duration ratio.

    Returns (confidence, warnings).
    """
    # ── Warp mode: use WarpPath metrics directly ──────────────────────
    if mode == "warp" and warp_path is not None:
        return _warp_confidence(
            warp_path, anchors, ad_duration, video_duration,
            fp_residual_p95_ms=fp_residual_p95_ms,
        )
    warnings: list[str] = []
    factors: list[float] = []

    # 1. Mean anchor score (skip for offset mode — no anchors expected)
    if mode != "offset":
        if anchors:
            anchor_scores = [a.score for a in anchors]
            mean_anchor = float(np.mean(anchor_scores))
            factors.append(mean_anchor)
            if mean_anchor < 0.5:
                warnings.append(f"Low average anchor quality ({mean_anchor:.2f})")
        else:
            factors.append(0.0)
            warnings.append("No anchors found")

    # 2. Anchor density (only meaningful for piecewise mode)
    if mode == "piecewise":
        if anchors and ad_duration > 0:
            density = len(anchors) / (ad_duration / 60.0)  # anchors per minute
            density_score = min(1.0, density / 10.0)  # 10+ per min = 1.0
            factors.append(density_score)
            if density < 2.0:
                warnings.append(f"Low anchor density ({density:.1f}/min)")
        else:
            factors.append(0.0)

    # 3. Segment confidence
    if segments:
        seg_scores = [s.confidence for s in segments]
        factors.append(float(np.mean(seg_scores)))
    else:
        factors.append(0.0)

    # 4. Coverage — fraction of AD timeline covered by segments
    if segments and ad_duration > 0:
        covered = sum(s.src_end - s.src_start for s in segments)
        coverage = min(1.0, covered / ad_duration)
        factors.append(coverage)
        if coverage < 0.8:
            warnings.append(f"Only {coverage:.0%} of AD track covered by segments")
    else:
        factors.append(0.0)

    # 5. Duration ratio sanity
    if ad_duration > 0 and video_duration > 0:
        ratio = ad_duration / video_duration
        if ratio < 0.5 or ratio > 2.0:
            warnings.append(f"Unusual duration ratio (AD/video = {ratio:.2f})")
            factors.append(0.3)
        else:
            factors.append(0.9)

    confidence = float(np.mean(factors)) if factors else 0.0
    confidence = max(0.0, min(1.0, confidence))

    # Extra heuristic warnings
    if segments:
        stretches = [abs(s.stretch - 1.0) for s in segments]
        if max(stretches) > 0.005:
            warnings.append(f"Large stretch detected ({max(stretches):.4f})")

    # Discontinuity detection: check if piecewise anchors show offset jumps
    if anchors and len(anchors) >= 10:
        anchor_offsets = np.array([a.target_time - a.source_time for a in anchors])
        # Look for jumps > 2 seconds between consecutive anchors
        diffs = np.abs(np.diff(anchor_offsets))
        jumps = np.where(diffs > 2.0)[0]
        if len(jumps) > 0:
            jump_times = [anchors[int(j)].source_time for j in jumps]
            jump_sizes = [float(diffs[int(j)]) for j in jumps]
            for t, sz in zip(jump_times, jump_sizes):
                mins, secs = divmod(int(t), 60)
                warnings.append(
                    f"Timing discontinuity at AD {mins:02d}:{secs:02d} "
                    f"(offset jumps by {sz:+.1f} s) — source material "
                    f"has different edits between video and AD"
                )
            log.warning(
                "Detected %d timing discontinuit%s in anchor offsets",
                len(jumps), "y" if len(jumps) == 1 else "ies",
            )

    log.info("Confidence: %.3f  (%d warnings)", confidence, len(warnings))
    return confidence, warnings


def _warp_confidence(
    warp_path: WarpPath,
    anchors: list[Anchor],
    ad_duration: float,
    video_duration: float,
    fp_residual_p95_ms: float | None = None,
) -> tuple[float, list[str]]:
    """Confidence scoring for warp mode.

    Raw-audio cross-correlation scores are naturally lower than onset-based
    scores (0.3-0.6 vs 0.7-0.9), so we scale path confidence accordingly.
    The key quality indicator for warp mode is path *consistency* (few jumps,
    high coverage) rather than raw correlation magnitude.
    """
    warnings: list[str] = []
    factors: list[float] = []

    # 1. Warp path consistency — fraction of points with any candidates
    coverage = 0.0
    if warp_path.points:
        # Points with real (non-synthetic) confidence
        real_points = [p for p in warp_path.points if p.confidence > 0.05]
        coverage = len(real_points) / len(warp_path.points)
        factors.append(coverage)
        if coverage < 0.5:
            warnings.append(f"Low warp coverage ({coverage:.0%} of windows matched)")
    else:
        factors.append(0.0)
        warnings.append("Empty warp path")

    # 2. Path smoothness — penalize remaining jumps (after decoder)
    if len(warp_path.points) >= 2:
        offsets = np.array([p.target_time - p.source_time for p in warp_path.points])
        diffs = np.abs(np.diff(offsets))
        # Fraction of transitions that are smooth (< 0.5s jump)
        smooth_frac = float(np.mean(diffs < 0.5))
        factors.append(smooth_frac)
        n_jumps = int(np.sum(diffs > 2.0))
        if n_jumps > 0:
            warnings.append(f"{n_jumps} timing discontinuit{'y' if n_jumps == 1 else 'ies'} detected")
    else:
        factors.append(0.5)

    # 3. Anchor density
    if warp_path.anchor_points and ad_duration > 0:
        density = len(warp_path.anchor_points) / (ad_duration / 60.0)
        density_score = min(1.0, density / 10.0)
        factors.append(density_score)
    else:
        factors.append(0.0)

    # 4. Duration ratio sanity
    if ad_duration > 0 and video_duration > 0:
        ratio = ad_duration / video_duration
        if ratio < 0.5 or ratio > 2.0:
            warnings.append(f"Unusual duration ratio (AD/video = {ratio:.2f})")
            factors.append(0.3)
        else:
            factors.append(0.9)

    # 5. Path evidence quality — mean decoded-candidate confidence.
    # Known-good warp runs land around 0.29-0.31; paths threaded through
    # spurious correlations sit near 0.15.
    if warp_path.points:
        mean_path_conf = float(np.mean([p.confidence for p in warp_path.points]))
        factors.append(min(1.0, mean_path_conf / 0.25))

    # 6. Bridged time — AD stretches with no usable evidence coast on their
    # flanking offsets.  A little is normal (quiet scenes, differing credits);
    # a lot means the map is interpolation.
    if ad_duration > 0:
        bridged_frac = warp_path.bridged_sec / ad_duration
        factors.append(max(0.0, 1.0 - 2.0 * bridged_frac))
        if warp_path.bridged_sec > 0.05 * ad_duration:
            warnings.append(
                f"{warp_path.bridged_sec:.0f} s of AD coasts on flanking offsets "
                "(no usable evidence there)"
            )
    if warp_path.dropped_segments:
        warnings.append(
            f"{warp_path.dropped_segments} excursion segment(s) rejected — "
            "repeated content matched at physically impossible offsets"
        )

    # 7. Independent verification — the fitted warp evaluated against the
    # fingerprint's inlier matches.  p95 under ~100 ms is a verified lock;
    # anything approaching 2 s means the map disagrees with the evidence.
    if fp_residual_p95_ms is not None:
        res_factor = float(np.clip(1.0 - (fp_residual_p95_ms - 100.0) / 1900.0, 0.0, 1.0))
        factors.append(res_factor)
        if fp_residual_p95_ms > 250.0:
            warnings.append(
                f"Verification residual p95 is {fp_residual_p95_ms:.0f} ms against "
                "fingerprint matches — parts of the map may be off"
            )

    confidence = float(np.mean(factors)) if factors else 0.0

    # A path where most windows had no real candidates is interpolation, not
    # measurement — gate the score so it cannot quietly clear the release
    # threshold no matter how smooth the interpolated path looks.
    if coverage < 0.5:
        confidence *= coverage / 0.5
        warnings.append(
            f"Confidence gated by low evidence coverage ({coverage:.0%})"
        )

    confidence = max(0.0, min(1.0, confidence))

    log.info("Warp confidence: %.3f  (%d warnings)", confidence, len(warnings))
    return confidence, warnings


def fingerprint_residuals(
    warp_fns,
    segment_ranges: list[tuple[float, float]],
    match_t_ad,
    match_t_vid,
) -> tuple[float, float] | None:
    """Evaluate the fitted warp against fingerprint inlier matches.

    Returns (p50_ms, p95_ms) of |warp(t_ad) − t_vid| over all matches that
    fall inside a fitted segment range, or None when nothing is scoreable.
    This is an independent check: the matches located content globally,
    the warp was decoded from windowed correlation — agreement between the
    two is what "verified" means.
    """
    residuals: list[np.ndarray] = []
    t_ad = np.asarray(match_t_ad, dtype=np.float64)
    t_vid = np.asarray(match_t_vid, dtype=np.float64)
    for fn, (lo, hi) in zip(warp_fns, segment_ranges):
        mask = (t_ad >= lo) & (t_ad <= hi)
        if not np.any(mask):
            continue
        residuals.append(np.abs(fn(t_ad[mask]) - t_vid[mask]) * 1000.0)
    if not residuals:
        return None
    all_res = np.concatenate(residuals)
    return float(np.percentile(all_res, 50)), float(np.percentile(all_res, 95))
