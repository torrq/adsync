"""Warp-fit segment vetting: excursion segments are dropped and bridged.

The decoded path can still contain sustained excursions (repeated content
matched coherently for many windows).  A valid playback map is monotone in
video time, so segments are vetted globally: excursions lose to the
maximum-evidence monotone chain and their AD range coasts on the flanking
segments' offsets, which also guarantees the renderer never places two
segments over the same video range.
"""

from __future__ import annotations

import pytest

from adsync.align.warp_fit import fit_warp_function
from adsync.models import WarpPoint


def _run(t0: float, t1: float, offset: float, conf: float = 0.5, step: float = 2.0):
    pts = []
    t = t0
    while t < t1:
        pts.append(WarpPoint(source_time=t, target_time=t + offset, confidence=conf))
        t += step
    return pts


def _warp_at(fns, ranges, t: float) -> float:
    best = None
    best_dist = float("inf")
    for fn, (lo, hi) in zip(fns, ranges):
        dist = max(lo - t, t - hi, 0.0)
        if dist < best_dist:
            best_dist = dist
            best = (fn, lo, hi)
    assert best is not None and best_dist <= 2.5, f"t={t} far from all ranges: {ranges}"
    fn, lo, hi = best
    return float(fn(min(max(t, lo), hi)))


def test_backwards_excursion_dropped_and_bridged() -> None:
    """A 30 s excursion to -900 s between agreeing +46 flanks must be bridged."""
    path = _run(0, 100, 46.0) + _run(100, 130, -900.0, conf=0.3) + _run(130, 200, 46.0)
    fns, ranges, wp = fit_warp_function(path, 200.0, 260.0)

    assert wp.dropped_segments >= 1
    assert wp.bridged_sec == pytest.approx(30.0, abs=6.0)
    assert _warp_at(fns, ranges, 115.0) == pytest.approx(115.0 + 46.0, abs=1.0)
    # Bridged flat at the flank offset -> one merged segment, no discontinuity
    assert len(fns) == 1


def test_real_edit_segments_kept() -> None:
    """Two genuine plateaus (forward jump) must both survive vetting."""
    path = _run(0, 100, 44.0) + _run(100, 200, 70.0)
    fns, ranges, wp = fit_warp_function(path, 200.0, 300.0)

    assert wp.dropped_segments == 0
    assert len(fns) == 2
    assert _warp_at(fns, ranges, 50.0) == pytest.approx(94.0, abs=1.0)
    assert _warp_at(fns, ranges, 150.0) == pytest.approx(220.0, abs=1.0)


def test_chaos_region_bridged_flat() -> None:
    """Short wild segments between agreeing flanks collapse onto the plateau."""
    chaos = []
    wild = [70.0, -10.0, 30.0, 55.0, -25.0, 12.0, 80.0, -40.0, 22.0, 61.0]
    for i, off in enumerate(wild):
        t = 100.0 + 2.0 * i
        chaos.append(WarpPoint(source_time=t, target_time=t + off, confidence=0.4))
    path = _run(0, 100, 46.0) + chaos + _run(120, 200, 46.0)
    fns, ranges, wp = fit_warp_function(path, 200.0, 260.0)

    assert len(fns) == 1
    assert _warp_at(fns, ranges, 110.0) == pytest.approx(110.0 + 46.0, abs=1.0)


def test_tail_excursion_coasts_on_left_flank() -> None:
    """A backwards tail excursion has no right flank — coast at the left offset."""
    path = _run(0, 180, 46.0) + _run(180, 200, -500.0, conf=0.3)
    fns, ranges, wp = fit_warp_function(path, 200.0, 260.0)

    assert wp.dropped_segments >= 1
    assert _warp_at(fns, ranges, 195.0) == pytest.approx(195.0 + 46.0, abs=1.0)


def test_recap_insert_squeezed_not_overlapped() -> None:
    """AD-only content with zero video room (a recap insert) must be squeezed.

    The decoder rides bridge synthetics through the insert at the left
    flank's offset; those riders must not extend the left segment's video
    claim — the right plateau starts at the same video time, and content
    outside the insert must stay exact.
    """
    a = _run(0, 300, 0.02)
    riders = [
        WarpPoint(source_time=float(t), target_time=float(t) + 0.02, confidence=0.075)
        for t in range(300, 315, 2)
    ]
    b = _run(315, 610, -14.98)
    fns, ranges, wp = fit_warp_function(a + riders + b, 610.0, 620.0)

    # Outside the insert: exact on both sides
    assert _warp_at(fns, ranges, 200.0) == pytest.approx(200.02, abs=0.5)
    assert _warp_at(fns, ranges, 400.0) == pytest.approx(385.02, abs=0.5)
    # The insert itself maps into (almost) zero video room — never backwards,
    # never claiming the right plateau's video range
    v_start = _warp_at(fns, ranges, 301.0)
    v_end = _warp_at(fns, ranges, 314.0)
    assert 297.5 <= v_start <= 302.0, v_start
    assert 297.5 <= v_end <= 302.0, v_end
    assert v_end >= v_start - 0.1, (v_start, v_end)
    # And globally monotone
    prev = -1.0
    t = 0.0
    while t <= 609.0:
        v = _warp_at(fns, ranges, t)
        assert v >= prev - 0.75, f"video regresses at t={t}"
        prev = v
        t += 1.0


def test_video_time_stays_monotone_overall() -> None:
    """After vetting, evaluating the warp over the full AD must never regress."""
    path = (_run(0, 80, 46.0) + _run(80, 110, -300.0, conf=0.45)
            + _run(110, 160, 48.0) + _run(160, 200, 71.0))
    fns, ranges, wp = fit_warp_function(path, 200.0, 300.0)

    prev = -1.0
    t = 0.0
    while t <= 199.0:
        v = _warp_at(fns, ranges, t)
        assert v >= prev - 0.75, f"video time regresses at t={t}: {v} < {prev}"
        prev = v
        t += 1.0
