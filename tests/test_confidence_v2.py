"""Confidence v2: bridged time, dropped excursions, and fingerprint verification
must move the score; the summary must name warp segments and stay
screen-reader friendly (no symbol glyphs).
"""

from __future__ import annotations

from adsync.align.confidence import compute_confidence
from adsync.models import SyncReport, WarpPath, WarpPoint
from adsync.report.json_report import console, print_summary


def _path(n: int = 200, offset: float = 46.0, conf: float = 0.5) -> list[WarpPoint]:
    return [
        WarpPoint(source_time=2.0 * i, target_time=2.0 * i + offset, confidence=conf)
        for i in range(n)
    ]


def _wp(**kw) -> WarpPath:
    pts = kw.pop("points", _path())
    return WarpPath(points=pts, anchor_points=pts[::2], mean_confidence=0.5, **kw)


AD_DUR = 400.0
VID_DUR = 460.0


def test_bridged_time_penalized() -> None:
    clean, _ = compute_confidence([], [], AD_DUR, VID_DUR, mode="warp", warp_path=_wp())
    bridged, warns = compute_confidence(
        [], [], AD_DUR, VID_DUR, mode="warp",
        warp_path=_wp(bridged_sec=120.0, dropped_segments=3),
    )
    assert bridged < clean - 0.03
    assert any("coast" in w.lower() or "bridged" in w.lower() for w in warns), warns
    assert any("excursion" in w.lower() for w in warns), warns


def test_fp_residual_moves_confidence() -> None:
    good, _ = compute_confidence(
        [], [], AD_DUR, VID_DUR, mode="warp", warp_path=_wp(), fp_residual_p95_ms=40.0,
    )
    bad, warns = compute_confidence(
        [], [], AD_DUR, VID_DUR, mode="warp", warp_path=_wp(), fp_residual_p95_ms=1500.0,
    )
    assert good > bad + 0.03
    assert any("residual" in w.lower() for w in warns), warns


def test_coverage_excludes_synthetic_points() -> None:
    """Synthetic pass-throughs (score 0.1 → confidence <= 0.10) are not
    evidence; a path that is mostly synthetics must trip the coverage gate
    even though their confidence sits above the old 0.05 cutoff."""
    real = _path(30, conf=0.5)
    synthetic = [
        WarpPoint(source_time=60.0 + 2.0 * i, target_time=106.0 + 2.0 * i, confidence=0.08)
        for i in range(170)
    ]
    wp = WarpPath(points=real + synthetic, anchor_points=real, mean_confidence=0.2)
    conf, warns = compute_confidence([], [], AD_DUR, VID_DUR, mode="warp", warp_path=wp)
    assert any("coverage" in w.lower() for w in warns), warns

    solid, warns_solid = compute_confidence(
        [], [], AD_DUR, VID_DUR, mode="warp", warp_path=_wp(points=_path(200, conf=0.15)),
    )
    assert not any("coverage" in w.lower() for w in warns_solid), warns_solid
    assert conf < solid


def test_summary_names_warp_segments_and_avoids_glyphs() -> None:
    report = SyncReport(
        mode="warp",
        confidence=0.95,
        global_offset=44.4,
        warnings=["something to know"],
        warp_path=_wp(n_segments=6),
        fp_residual_p50_ms=12.0,
        fp_residual_p95_ms=55.0,
    )
    with console.capture() as cap:
        print_summary(report)
    out = cap.get()
    assert "6" in out and "segment" in out.lower()
    assert "⚠" not in out and "✓" not in out and "✗" not in out
    assert "Warning" in out
    assert "55" in out  # verification p95 shown


def test_summary_prints_timing_map() -> None:
    pts = _path(60, offset=44.0) + [
        WarpPoint(source_time=2.0 * i, target_time=2.0 * i + 70.0, confidence=0.5)
        for i in range(60, 100)
    ]
    report = SyncReport(
        mode="warp", confidence=0.95,
        warp_path=WarpPath(points=pts, anchor_points=pts[::2], n_segments=2),
    )
    with console.capture() as cap:
        print_summary(report)
    out = cap.get()
    assert "+44.0" in out and "+70.0" in out, out
