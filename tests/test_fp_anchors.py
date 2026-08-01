"""Fingerprint-anchored alignment: landmark matches fill a starved lattice."""

from __future__ import annotations

import numpy as np
import pytest

from adsync.align.candidate_lattice import augment_lattice_with_fingerprint
from adsync.align.confidence import compute_confidence
from adsync.align.fingerprint import FingerprintResult, FingerprintSpan
from adsync.align.warp_decode import decode_warp_path
from adsync.align.warp_fit import fit_warp_function
from adsync.models import CandidateWindow

AD_DUR = 200.0
VID_DUR = 215.0
STEP_T = 100.0
OFF_A = 5.0
OFF_B = 12.4


def _fake_fp(seed: int = 3) -> FingerprintResult:
    """Dense match pairs at offset 5.0 then 12.4, with jitter and outliers."""
    rng = np.random.default_rng(seed)
    t_ad = np.arange(0.5, AD_DUR - 0.5, 0.05)
    offset = np.where(t_ad < STEP_T, OFF_A, OFF_B)
    t_vid = t_ad + offset + rng.normal(0.0, 0.02, len(t_ad))
    # 5% wild outliers, as raw hash matching always produces
    n_out = len(t_ad) // 20
    idx = rng.choice(len(t_ad), n_out, replace=False)
    t_vid[idx] = rng.uniform(0.0, VID_DUR, n_out)
    order = np.argsort(t_ad)
    fp = FingerprintResult(
        spans=[
            FingerprintSpan(ad_start=0.0, ad_end=STEP_T, offset=OFF_A, matches=2000),
            FingerprintSpan(ad_start=STEP_T, ad_end=AD_DUR, offset=OFF_B, matches=2000),
        ],
        n_matches=len(t_ad),
    )
    fp.match_t_ad = t_ad[order].astype(np.float32)
    fp.match_t_vid = t_vid[order].astype(np.float32)
    return fp


def _starved_lattice() -> list[CandidateWindow]:
    return [
        CandidateWindow(source_center=float(c), candidates=[], speech_score=0.5, energy=1.0)
        for c in np.arange(1.0, AD_DUR - 1.0, 2.0)
    ]


def test_augment_fills_starved_windows() -> None:
    lattice = _starved_lattice()
    fp = _fake_fp()
    n_aug = augment_lattice_with_fingerprint(lattice, fp)
    assert n_aug > 0.8 * len(lattice)
    for w in lattice:
        if not w.candidates:
            continue
        expected = OFF_A if w.source_center < STEP_T - 2.0 else (
            OFF_B if w.source_center > STEP_T + 2.0 else None)
        if expected is None:
            continue
        best = w.candidates[0]
        assert best.source == "fingerprint"
        assert best.offset_sec == pytest.approx(expected, abs=0.06)


def test_decoded_path_follows_fingerprint() -> None:
    lattice = _starved_lattice()
    fp = _fake_fp()
    augment_lattice_with_fingerprint(lattice, fp)
    points, _cost = decode_warp_path(
        lattice, offset_hint_fn=lambda t: fp.offset_at(t) or 0.0,
    )
    assert len(points) == len(lattice)
    for p in points:
        if p.source_time < STEP_T - 4.0:
            assert p.target_time - p.source_time == pytest.approx(OFF_A, abs=0.08)
        elif p.source_time > STEP_T + 4.0:
            assert p.target_time - p.source_time == pytest.approx(OFF_B, abs=0.08)


def test_fp_anchored_confidence_clears_gate() -> None:
    """Dense fingerprint evidence must yield a releasable confidence."""
    lattice = _starved_lattice()
    fp = _fake_fp()
    augment_lattice_with_fingerprint(lattice, fp)
    points, _cost = decode_warp_path(
        lattice, offset_hint_fn=lambda t: fp.offset_at(t) or 0.0,
    )
    _fns, _ranges, warp_path = fit_warp_function(points, AD_DUR, VID_DUR)
    confidence, _warnings = compute_confidence(
        [], [], AD_DUR, VID_DUR, mode="warp", warp_path=warp_path,
    )
    assert confidence >= 0.70
