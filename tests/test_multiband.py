"""Multi-band correlation scoring in the candidate lattice.

Narration mixed over a quiet scene contaminates the speech bands and can
dilute a full-band correlation below the candidate floor.  Splitting the
correlation into frequency bands, normalizing per band, and down-weighting
the narration bands keeps the clean bands' vote — windows that used to
starve now produce the true offset candidate.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from adsync.align.candidate_lattice import build_candidate_lattice
from adsync.features.extract_basic import extract_basic_features

SR = 16000
OFFSET = -5.0  # AD delayed 5 s relative to video


def _make_pair(narration: bool):
    """Video = broadband noise; AD = same content delayed, optionally with a
    loud speech-band masker across the whole track."""
    rng = np.random.default_rng(7)
    dur = 60.0
    n = int(dur * SR)
    vid = rng.normal(0.0, 0.3, n).astype(np.float32)

    delay = int(5.0 * SR)
    ad = np.zeros(n, dtype=np.float32)
    ad[delay:] = vid[:-delay]

    if narration:
        masker = rng.normal(0.0, 1.0, n).astype(np.float32)
        sos = butter(4, [300.0, 900.0], btype="bandpass", fs=SR, output="sos")
        masker = sosfilt(sos, masker).astype(np.float32)
        # ~5x the energy of the shared content, confined to speech bands
        masker *= 5.0 * float(np.std(ad)) / max(float(np.std(masker)), 1e-9)
        ad = ad + masker

    return vid, ad


def _lattice(vid, ad, multiband: bool):
    feat_v = extract_basic_features(vid, SR)
    feat_a = extract_basic_features(ad, SR)
    return build_candidate_lattice(
        feat_v, feat_a,
        y_vid=vid, y_ad=ad, audio_sr=SR,
        search_radius_sec=10.0,
        multiband=multiband,
    )


def _hit_rate(lattice) -> float:
    """Fraction of candidate-bearing windows whose best candidate is the truth."""
    hits = total = 0
    for w in lattice:
        if w.source_center < 6.0:  # AD before the delay has no video counterpart
            continue
        total += 1
        if w.candidates and abs(w.candidates[0].offset_sec - OFFSET) < 0.2:
            hits += 1
    return hits / max(1, total)


def test_clean_pair_unaffected() -> None:
    vid, ad = _make_pair(narration=False)
    assert _hit_rate(_lattice(vid, ad, multiband=True)) > 0.9


def test_narration_masked_windows_recovered() -> None:
    vid, ad = _make_pair(narration=True)
    single = _hit_rate(_lattice(vid, ad, multiband=False))
    multi = _hit_rate(_lattice(vid, ad, multiband=True))
    assert multi > 0.8, f"multiband hit rate only {multi:.2f}"
    assert multi > single + 0.25, (
        f"multiband ({multi:.2f}) must clearly beat full-band ({single:.2f})"
    )


def test_offset_precision_preserved() -> None:
    """Band-splitting must not blur the sub-sample peak refinement."""
    vid, ad = _make_pair(narration=False)
    lattice = _lattice(vid, ad, multiband=True)
    errs = [
        abs(w.candidates[0].offset_sec - OFFSET)
        for w in lattice
        if w.source_center >= 6.0 and w.candidates
        and abs(w.candidates[0].offset_sec - OFFSET) < 0.2
    ]
    assert errs, "no true-offset candidates found"
    assert float(np.median(errs)) < 0.01, f"median error {np.median(errs) * 1000:.1f} ms"
