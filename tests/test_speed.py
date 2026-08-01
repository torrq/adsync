"""Speed detection: PAL-style rate shifts found, measured, and corrected."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import resample_poly

from adsync.align.fingerprint import FingerprintSpan, fingerprint_align
from adsync.align.speed import SpeedDetection, detect_speed, fit_speed_slope

SR = 16000
PAL = 25.0 / 24.0


def _tone_bed(duration: float, seed: int = 7) -> np.ndarray:
    """Landmark-friendly synthetic bed: dense random tone pips over noise."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR)
    y = rng.normal(0.0, 0.02, n).astype(np.float64)
    t = 0.25
    while t < duration - 0.2:
        f = rng.uniform(150.0, 4500.0)
        amp = rng.uniform(0.3, 1.0)
        pip_len = int(0.06 * SR)
        i0 = int(t * SR)
        tt = np.arange(pip_len) / SR
        y[i0:i0 + pip_len] += amp * np.sin(2 * np.pi * f * tt) * np.hanning(pip_len)
        t += rng.uniform(0.25, 0.45)
    return y / np.max(np.abs(y))


# ── fit_speed_slope ──────────────────────────────────────────────────────────


def test_fit_speed_slope_recovers_ramp() -> None:
    spans = [
        FingerprintSpan(ad_start=t, ad_end=t + 10.0, offset=2.0 + 0.004 * (t + 5.0), matches=100)
        for t in range(0, 3000, 10)
    ]
    slope = fit_speed_slope(spans)
    assert slope == pytest.approx(0.004, abs=2e-4)


def test_fit_speed_slope_ignores_edit_step() -> None:
    """A real edit step must not leak into the slope (least squares would)."""
    spans = []
    for t in range(0, 3000, 10):
        off = 2.0 + 0.004 * (t + 5.0)
        if t >= 1500:
            off += 7.5  # edit in the source material
        spans.append(FingerprintSpan(ad_start=t, ad_end=t + 10.0, offset=off, matches=100))
    slope = fit_speed_slope(spans)
    assert slope == pytest.approx(0.004, abs=5e-4)


# ── detect_speed ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def bed() -> np.ndarray:
    return _tone_bed(300.0)


def test_detect_speed_finds_pal_ratio(bed: np.ndarray) -> None:
    """AD resampled to PAL speed (4.2% fast) is detected and measured."""
    y_ad = resample_poly(bed, 24, 25)  # PAL-fast: shorter, pitch up
    baseline = fingerprint_align(bed, y_ad, SR)
    det = detect_speed(bed, y_ad, SR, baseline=baseline)
    assert det is not None
    assert det.stretch == pytest.approx(PAL, abs=1e-3)
    assert det.fp.strong


def test_detect_speed_refines_beyond_pal(bed: np.ndarray) -> None:
    """PAL plus a transfer error refines to the true combined ratio."""
    true_stretch = 2500.0 / 2397.0  # ≈ PAL × 1.00125
    y_ad = resample_poly(bed, 2397, 2500)
    baseline = fingerprint_align(bed, y_ad, SR)
    det = detect_speed(bed, y_ad, SR, baseline=baseline)
    assert det is not None
    assert det.stretch == pytest.approx(true_stretch, abs=3e-4)


def test_detect_speed_no_false_trigger(bed: np.ndarray) -> None:
    """A same-speed pair must not be 'corrected'."""
    rng = np.random.default_rng(11)
    y_ad = bed + rng.normal(0.0, 0.01, len(bed))
    baseline = fingerprint_align(bed, y_ad, SR)
    det = detect_speed(bed, y_ad, SR, baseline=baseline)
    assert det is None
