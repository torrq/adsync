"""Monotonicity filters and decoder return-shape edge cases."""

from __future__ import annotations

from adsync.align.anchors import _monotonic_filter
from adsync.align.warp_decode import decode_warp_path
from adsync.align.warp_fit import _enforce_monotonicity
from adsync.models import Anchor, WarpPoint


def _anchor(src: float, tgt: float, score: float = 0.6) -> Anchor:
    return Anchor(source_time=src, target_time=tgt, score=score, window=8.0)


def test_monotonic_filter_survives_bad_first_anchor() -> None:
    """One early false match far ahead in the video must not discard the
    coherent majority that follows (a greedy scan did exactly that)."""
    anchors = [_anchor(0.0, 500.0, score=0.9)]
    anchors += [_anchor(2.0 * i, 2.0 * i + 10.0) for i in range(1, 20)]
    kept = _monotonic_filter(anchors)
    assert len(kept) >= 15
    assert all(a.target_time < 200.0 for a in kept)


def test_monotonic_filter_still_monotone() -> None:
    anchors = [
        _anchor(0.0, 10.0), _anchor(2.0, 14.0), _anchor(4.0, 12.0),
        _anchor(6.0, 16.0), _anchor(8.0, 15.0), _anchor(10.0, 18.0),
    ]
    kept = _monotonic_filter(anchors)
    tgts = [a.target_time for a in kept]
    assert tgts == sorted(tgts) and len(set(tgts)) == len(tgts)
    assert len(kept) == 4


def test_enforce_monotonicity_prefers_confidence() -> None:
    """A high-confidence point outweighs a longer chain of near-noise points
    (an unweighted LIS would keep the longer chain and drop it)."""
    strong = WarpPoint(source_time=5.0, target_time=100.0, confidence=0.9)
    weak = [
        WarpPoint(source_time=2.0 * i, target_time=10.0 + 2.0 * i, confidence=0.06)
        for i in range(6)
    ]
    kept = _enforce_monotonicity(weak + [strong])
    assert strong in kept


def test_decode_warp_path_empty_lattice_unpacks() -> None:
    points, cost = decode_warp_path([])
    assert points == [] and cost == 0.0
