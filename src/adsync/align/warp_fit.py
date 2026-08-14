"""Fit a smooth, monotone warp function through the decoded path.

Takes the Viterbi-decoded path (one WarpPoint per analysis window),
selects high-confidence anchor points, enforces monotonicity, and fits
a PchipInterpolator per contiguous segment.  PCHIP is shape-preserving
and monotone on monotone data, avoiding overshoot problems of cubic splines.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import PchipInterpolator

from adsync.models import WarpPath, WarpPoint

log = logging.getLogger("adsync")

# ── Segment vetting ──────────────────────────────────────────────────────────
_MIN_SEG_SEC = 8.0        # shorter runs cannot establish an offset on their own
_MIN_SEG_REAL_POINTS = 3  # decoded points above the synthetic floor
# Real evidence scores >= 0.3 -> confidence >= 0.15; synthetic pass-throughs
# score 0.1 -> confidence <= 0.10.  0.12 separates them cleanly.
_SYNTH_CONF = 0.12
_SEG_CHAIN_TOL = 1.0      # video-time regression allowed between chained segments


def fit_warp_function(
    path: list[WarpPoint],
    ad_duration: float,
    video_duration: float,
    *,
    anchor_fraction: float = 0.6,
    min_anchor_density: float = 1 / 30,  # per second
    discontinuity_threshold: float = 2.0,
) -> tuple[list[PchipInterpolator], list[tuple[float, float]], WarpPath]:
    """Fit monotone PCHIP warp function(s) through the decoded path.

    Parameters
    ----------
    path : list[WarpPoint]
        Decoded path from :func:`decode_warp_path`.
    ad_duration, video_duration : float
        Total durations in seconds.
    anchor_fraction : float
        Fraction of points (by confidence rank) to use as PCHIP anchors.
    min_anchor_density : float
        Minimum anchor density in points per second.
    discontinuity_threshold : float
        Offset jump (seconds) between consecutive points to split segments.

    Returns
    -------
    warp_fns : list[PchipInterpolator]
        One per contiguous segment; maps AD time → video time.
    segment_ranges : list[tuple[float, float]]
        Source-time (AD) ranges for each segment.
    warp_path : WarpPath
        Report-friendly summary of the path and anchor selection.
    """
    if not path:
        # Fallback: identity warp
        fn = PchipInterpolator([0.0, ad_duration], [0.0, ad_duration])
        wp = WarpPath(points=[], anchor_points=[], path_cost=0.0, mean_confidence=0.0)
        return [fn], [(0.0, ad_duration)], wp

    # ── Step 1: Segment the path at discontinuities ────────────────────
    segments = _split_at_discontinuities(path, discontinuity_threshold, log_jumps=False)

    # ── Step 1.5: Vet segments globally and bridge the rejects ────────
    # A valid playback map is monotone in video time; sustained excursions
    # (repeated content matched coherently) lose to the maximum-evidence
    # monotone chain, and their AD range coasts on the flanking offsets.
    path, dropped_segments, bridged_sec = _vet_and_bridge(segments)
    segments = _split_at_discontinuities(path, discontinuity_threshold)
    log.info(
        "Warp fit: %d contiguous segment(s) from %d points%s",
        len(segments), len(path),
        (f" ({dropped_segments} excursion segment(s) dropped, "
         f"{bridged_sec:.0f} s bridged)") if dropped_segments else "",
    )

    warp_fns: list[PchipInterpolator] = []
    segment_ranges: list[tuple[float, float]] = []
    all_anchor_points: list[WarpPoint] = []

    for seg_idx, seg_points in enumerate(segments):
        if len(seg_points) < 2:
            # Single point — extend to a tiny segment with constant offset
            p = seg_points[0]
            offset = p.target_time - p.source_time
            src_start = max(0.0, p.source_time - 5.0)
            src_end = min(ad_duration, p.source_time + 5.0)
            fn = PchipInterpolator(
                [src_start, src_end],
                [src_start + offset, src_end + offset],
            )
            warp_fns.append(fn)
            segment_ranges.append((src_start, src_end))
            p_copy = p.model_copy(update={"is_anchor": True})
            all_anchor_points.append(p_copy)
            continue

        # ── Step 2: Select anchors by confidence ──────────────────────
        anchors = _select_anchors(seg_points, anchor_fraction, min_anchor_density)

        # ── Step 3: Enforce monotonicity ──────────────────────────────
        anchors = _enforce_monotonicity(anchors)

        if len(anchors) < 2:
            # Couldn't get 2 monotone anchors — use endpoints
            anchors = [seg_points[0], seg_points[-1]]
            anchors = _enforce_monotonicity(anchors)

        # ── Step 4: Add virtual boundary points ───────────────────────
        anchors = _add_boundaries(
            anchors, seg_idx, len(segments),
            ad_duration, video_duration,
        )

        # ── Step 5: Fit PCHIP ─────────────────────────────────────────
        src_times = np.array([a.source_time for a in anchors])
        tgt_times = np.array([a.target_time for a in anchors])

        # Deduplicate source times (PCHIP requires strictly increasing x)
        src_times, unique_idx = np.unique(src_times, return_index=True)
        tgt_times = tgt_times[unique_idx]

        if len(src_times) < 2:
            offset = tgt_times[0] - src_times[0]
            src_times = np.array([max(0, src_times[0] - 5), min(ad_duration, src_times[0] + 5)])
            tgt_times = src_times + offset

        fn = PchipInterpolator(src_times, tgt_times)
        warp_fns.append(fn)
        segment_ranges.append((float(src_times[0]), float(src_times[-1])))

        for a in anchors:
            if not a.is_anchor:
                a = a.model_copy(update={"is_anchor": True})
            all_anchor_points.append(a)

    # ── Extend segment ranges to cover full AD duration ───────────────
    if segment_ranges:
        # First segment starts at 0
        s0_start, s0_end = segment_ranges[0]
        segment_ranges[0] = (0.0, s0_end)
        # Last segment extends to ad_duration
        sN_start, sN_end = segment_ranges[-1]
        segment_ranges[-1] = (sN_start, ad_duration)
        # Interior ranges end at decoded window centres, one step apart —
        # left as-is, the AD audio between two segments belongs to neither
        # and vanishes from the render.  Meet at the midpoint instead; the
        # true edit point is only localized to within a window anyway.
        for i in range(len(segment_ranges) - 1):
            a_start, a_end = segment_ranges[i]
            b_start, b_end = segment_ranges[i + 1]
            if b_start > a_end:
                mid = 0.5 * (a_end + b_start)
                segment_ranges[i] = (a_start, mid)
                segment_ranges[i + 1] = (mid, b_end)

    mean_conf = float(np.mean([p.confidence for p in path])) if path else 0.0
    path_cost = 0.0  # Filled by decoder if needed

    warp_path = WarpPath(
        points=path,
        anchor_points=all_anchor_points,
        path_cost=path_cost,
        mean_confidence=mean_conf,
        dropped_segments=dropped_segments,
        bridged_sec=bridged_sec,
        n_segments=len(warp_fns),
    )

    log.info(
        "Warp fit complete: %d PCHIP segments, %d anchor points, mean_conf=%.3f",
        len(warp_fns), len(all_anchor_points), mean_conf,
    )
    return warp_fns, segment_ranges, warp_path


def _split_at_discontinuities(
    path: list[WarpPoint],
    threshold: float,
    log_jumps: bool = True,
) -> list[list[WarpPoint]]:
    """Split path into segments where consecutive offset jumps exceed threshold."""
    if len(path) <= 1:
        return [path]

    segments: list[list[WarpPoint]] = [[path[0]]]
    for i in range(1, len(path)):
        prev_offset = path[i - 1].target_time - path[i - 1].source_time
        cur_offset = path[i].target_time - path[i].source_time
        if abs(cur_offset - prev_offset) > threshold:
            if log_jumps:
                log.info(
                    "Discontinuity at t=%.1fs: offset jumps %.2f → %.2f s",
                    path[i].source_time, prev_offset, cur_offset,
                )
            segments.append([])
        segments[-1].append(path[i])

    return [s for s in segments if s]


def _real_points(seg: list[WarpPoint]) -> list[WarpPoint]:
    return [p for p in seg if p.confidence > _SYNTH_CONF]


def _strip_synthetic_edges(
    segments: list[list[WarpPoint]],
) -> list[list[WarpPoint]]:
    """Split leading/trailing synthetic runs off each segment.

    Synthetic pass-through points at a segment's edges carry the offset the
    decoder coasted on, not evidence — left inside the segment they extend
    its video-time claim past what the evidence supports, which corrupts the
    monotone arbitration between segments.  Interior synthetics are fine:
    they sit between real points at the same offset.
    """
    out: list[list[WarpPoint]] = []
    for seg in segments:
        head = 0
        while head < len(seg) and seg[head].confidence <= _SYNTH_CONF:
            head += 1
        tail = len(seg)
        while tail > head and seg[tail - 1].confidence <= _SYNTH_CONF:
            tail -= 1
        for piece in (seg[:head], seg[head:tail], seg[tail:]):
            if piece:
                out.append(piece)
    return out


def _vet_and_bridge(
    segments: list[list[WarpPoint]],
) -> tuple[list[WarpPoint], int, float]:
    """Keep the maximum-evidence monotone chain of segments; bridge the rest.

    Eligible segments (long enough, enough real evidence) compete in a
    weighted longest-increasing-subsequence over the video-time intervals of
    their **real** points; weight is summed real-point confidence.  Rejected
    segments and synthetic edge runs are re-timed at low confidence:

    - When the flanking evidence leaves enough video room for the bridged AD
      content, it coasts on the left offset up to the gap midpoint and on
      the right offset after (a plain hole in the video is correct when the
      video carries content the AD lacks).
    - When there is **no video room** (AD-only content, e.g. a recap
      insert), the bridge ramps video time monotonically between the flank
      evidence points, squeezing the unplaceable content instead of letting
      it overlap either neighbour.

    Returns the rebuilt point list, the dropped-segment count, and the
    bridged AD seconds.
    """
    segments = _strip_synthetic_edges(segments)
    n = len(segments)
    if n <= 1:
        return [p for seg in segments for p in seg], 0, 0.0

    reals: list[list[WarpPoint]] = [_real_points(seg) for seg in segments]
    masses = [sum(p.confidence for p in real) for real in reals]

    eligible = [
        i for i in range(n)
        if len(reals[i]) >= _MIN_SEG_REAL_POINTS
        and reals[i][-1].source_time - reals[i][0].source_time >= _MIN_SEG_SEC
    ]
    if not eligible:
        candidates = [i for i in range(n) if reals[i]] or list(range(n))
        eligible = [max(candidates, key=lambda i: masses[i])]

    # Weighted LIS over real-evidence video intervals (segments are in AD order).
    m = len(eligible)
    vid_lo = [
        (reals[i][0] if reals[i] else segments[i][0]).target_time for i in eligible
    ]
    vid_hi = [
        (reals[i][-1] if reals[i] else segments[i][-1]).target_time for i in eligible
    ]
    best = [0.0] * m
    prev = [-1] * m
    for a in range(m):
        best[a] = masses[eligible[a]]
        for b in range(a):
            if vid_lo[a] >= vid_hi[b] - _SEG_CHAIN_TOL:
                cand = best[b] + masses[eligible[a]]
                if cand > best[a]:
                    best[a] = cand
                    prev[a] = b
    a = int(np.argmax(best))
    keep: set[int] = set()
    while a >= 0:
        keep.add(eligible[a])
        a = prev[a]

    n_rejected = len(eligible) - len(keep)
    if len(keep) == n:
        return [p for seg in segments for p in seg], 0, 0.0

    kept_sorted = sorted(keep)
    new_points: list[WarpPoint] = []
    bridged_sec = 0.0
    for idx, seg in enumerate(segments):
        if idx in keep:
            new_points.extend(seg)
            continue
        left = next((j for j in reversed(kept_sorted) if j < idx), None)
        right = next((j for j in kept_sorted if j > idx), None)
        left_p = reals[left][-1] if left is not None else None
        right_p = reals[right][0] if right is not None else None
        if left_p is None and right_p is None:
            new_points.extend(seg)
            continue

        if left_p is not None and right_p is not None:
            ad_span = right_p.source_time - left_p.source_time
            video_room = right_p.target_time - left_p.target_time
            if video_room >= ad_span - 0.5:
                # Video has room: coast on each flank's offset, hole at the
                # midpoint.
                mid = 0.5 * (left_p.source_time + right_p.source_time)
                left_off = left_p.target_time - left_p.source_time
                right_off = right_p.target_time - right_p.source_time
                for p in seg:
                    off = left_off if p.source_time <= mid else right_off
                    new_points.append(WarpPoint(
                        source_time=p.source_time,
                        target_time=p.source_time + off,
                        confidence=0.05,
                    ))
            else:
                # AD-only content, no video room: squeeze monotonically.
                rise = max(0.0, video_room)
                for p in seg:
                    frac = (p.source_time - left_p.source_time) / max(ad_span, 1e-9)
                    frac = min(1.0, max(0.0, frac))
                    new_points.append(WarpPoint(
                        source_time=p.source_time,
                        target_time=left_p.target_time + frac * rise,
                        confidence=0.05,
                    ))
        else:
            flank = left_p if left_p is not None else right_p
            off = flank.target_time - flank.source_time
            for p in seg:
                new_points.append(WarpPoint(
                    source_time=p.source_time,
                    target_time=p.source_time + off,
                    confidence=0.05,
                ))
        bridged_sec += seg[-1].source_time - seg[0].source_time

    return new_points, n_rejected, bridged_sec


def _select_anchors(
    points: list[WarpPoint],
    fraction: float,
    min_density: float,
) -> list[WarpPoint]:
    """Select top-fraction points by confidence, with minimum density."""
    n = len(points)
    if n <= 2:
        return list(points)

    duration = points[-1].source_time - points[0].source_time
    min_count = max(2, int(duration * min_density))
    target_count = max(min_count, int(n * fraction))
    target_count = min(target_count, n)

    # Always include first and last
    ranked = sorted(range(n), key=lambda i: points[i].confidence, reverse=True)
    selected_idx = {0, n - 1}

    for idx in ranked:
        if len(selected_idx) >= target_count:
            break
        selected_idx.add(idx)

    # Sort by source_time
    selected = sorted(selected_idx)
    return [points[i].model_copy(update={"is_anchor": True}) for i in selected]


def _enforce_monotonicity(anchors: list[WarpPoint]) -> list[WarpPoint]:
    """Remove points that violate strict target_time monotonicity.

    Maximum-confidence increasing subsequence on target_time via an O(n^2)
    DP (n is small — typically < 1000), so a run of high-confidence points
    beats a longer run of low-confidence violators.  This guarantees the
    result is strictly increasing in both source_time and target_time.
    """
    if len(anchors) <= 1:
        return anchors

    # Sort by source_time
    sorted_a = sorted(anchors, key=lambda p: p.source_time)

    n = len(sorted_a)
    targets = [p.target_time for p in sorted_a]
    conf = [p.confidence for p in sorted_a]

    # dp[i] = best summed confidence of an increasing subseq ending at i
    dp = list(conf)
    prev_idx = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if targets[j] < targets[i] and dp[j] + conf[i] > dp[i]:
                dp[i] = dp[j] + conf[i]
                prev_idx[i] = j

    # Backtrack to find the actual subsequence
    best_end = int(np.argmax(dp))
    lis_indices: list[int] = []
    idx = best_end
    while idx != -1:
        lis_indices.append(idx)
        idx = prev_idx[idx]
    lis_indices.reverse()

    result = [sorted_a[i] for i in lis_indices]
    if len(result) < len(sorted_a):
        log.debug(
            "Monotonicity: kept %d of %d points (removed %d violators)",
            len(result), len(sorted_a), len(sorted_a) - len(result),
        )
    return result


def _add_boundaries(
    anchors: list[WarpPoint],
    seg_idx: int,
    n_segments: int,
    ad_duration: float,
    video_duration: float,
) -> list[WarpPoint]:
    """Add virtual boundary anchors at start/end using local slope."""
    if len(anchors) < 2:
        return anchors

    result = list(anchors)

    # Extrapolate to t=0 for the first segment
    if seg_idx == 0 and result[0].source_time > 0.1:
        first = result[0]
        second = result[1]
        slope = (second.target_time - first.target_time) / max(
            second.source_time - first.source_time, 1e-6,
        )
        t0_target = first.target_time - first.source_time * slope
        t0_target = max(0.0, t0_target)
        result.insert(0, WarpPoint(
            source_time=0.0,
            target_time=t0_target,
            confidence=first.confidence * 0.8,
            is_anchor=True,
        ))

    # Extrapolate to t=ad_duration for the last segment
    if seg_idx == n_segments - 1 and result[-1].source_time < ad_duration - 0.1:
        last = result[-1]
        prev = result[-2]
        slope = (last.target_time - prev.target_time) / max(
            last.source_time - prev.source_time, 1e-6,
        )
        tend_target = last.target_time + (ad_duration - last.source_time) * slope
        tend_target = min(video_duration, tend_target)
        result.append(WarpPoint(
            source_time=ad_duration,
            target_time=tend_target,
            confidence=last.confidence * 0.8,
            is_anchor=True,
        ))

    return result
