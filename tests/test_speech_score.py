"""Speech-likeness scoring must actually discriminate.

Regression guard for two related bugs: the mel spectrogram is stored in dB
(power_to_db, ref=max, values <= 0), and the score once clamped those dB
values against a tiny positive linear floor — flattening every window to the
same constant.  The narration band was also mapped assuming mel bins are
linear in Hz, which silently turned "100-4000 Hz" into "0-1700 Hz".
"""

from __future__ import annotations

import librosa
import numpy as np

from adsync.align.candidate_lattice import _compute_speech_score, build_candidate_lattice
from adsync.features.extract_basic import extract_basic_features

SR = 16000
N_MELS = 64


def _narration_band(n_mels: int = N_MELS, sr: int = SR) -> tuple[int, int]:
    freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2.0)
    lo = int(np.searchsorted(freqs, 100.0))
    hi = max(lo + 1, int(np.searchsorted(freqs, 4000.0)))
    return lo, hi


def _score(y: np.ndarray, center: float = 6.0, window: float = 8.0) -> float:
    feat = extract_basic_features(y.astype(np.float32), SR, n_mels=N_MELS)
    lo, hi = _narration_band()
    return _compute_speech_score(
        feat.mel, feat.hop_length / feat.sr, center, window, lo, hi,
    )


def _harmonic_stack(f0: float, dur: float = 12.0, n_harm: int = 10) -> np.ndarray:
    t = np.arange(int(dur * SR)) / SR
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, n_harm + 1))
    y *= 0.6 + 0.4 * np.sin(2 * np.pi * 4.0 * t)  # syllabic-rate AM
    return (0.2 * y / np.max(np.abs(y))).astype(np.float32)


def test_speech_beats_noise() -> None:
    speech = _score(_harmonic_stack(150.0))
    rng = np.random.default_rng(0)
    noise = _score(0.2 * rng.normal(size=int(12 * SR)).astype(np.float32))
    assert speech > noise + 0.3, (speech, noise)


def test_score_is_not_constant() -> None:
    """The dB-clamp bug froze every window at exactly 0.25."""
    rng = np.random.default_rng(1)
    scores = [
        _score(_harmonic_stack(180.0)),
        _score(0.2 * rng.normal(size=int(12 * SR)).astype(np.float32)),
        _score(_harmonic_stack(3000.0, n_harm=1)),
    ]
    assert len({round(s, 3) for s in scores}) > 1, scores


def test_narration_band_uses_mel_spacing() -> None:
    """3 kHz is inside 100-4000 Hz; a linear-Hz bin mapping puts it outside.

    Tonal content at 3 kHz and at 6.5 kHz has near-identical flatness, so only
    the band-ratio half separates them — which requires the band edges to sit
    at the true mel positions of 100 and 4000 Hz.
    """
    in_band = _score(_harmonic_stack(3000.0, n_harm=1))
    out_band = _score(_harmonic_stack(6500.0, n_harm=1))
    assert in_band > out_band + 0.25, (in_band, out_band)

    lo, hi = _narration_band()
    freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0.0, fmax=SR / 2.0)
    assert freqs[lo] >= 50.0 and freqs[lo] <= 250.0
    assert freqs[hi - 1] >= 3300.0 and freqs[hi - 1] <= 4000.0


def test_bounds_and_silent_windows() -> None:
    for y in (_harmonic_stack(200.0), np.zeros(int(12 * SR), dtype=np.float32)):
        s = _score(y)
        assert 0.0 <= s <= 1.0 and np.isfinite(s)

    # Silent windows never reach the scorer: the lattice energy gate zeroes them.
    y = np.zeros(int(30 * SR), dtype=np.float32)
    y[: 10 * SR] = _harmonic_stack(200.0, dur=10.0)
    feat = extract_basic_features(y, SR, n_mels=N_MELS)
    lattice = build_candidate_lattice(
        feat, feat, y_vid=y, y_ad=y, audio_sr=SR,
        search_radius_sec=5.0, multiband=False,
    )
    tail = [w for w in lattice if w.source_center > 16.0]
    assert tail and all(w.speech_score == 0.0 for w in tail)


def test_lattice_windows_carry_varying_scores() -> None:
    """End-to-end: lattice windows over mixed content get finite, non-constant
    speech scores in [0, 1]."""
    rng = np.random.default_rng(7)
    dur = 40.0
    n = int(dur * SR)
    y = 0.15 * rng.normal(size=n).astype(np.float32)
    seg = _harmonic_stack(160.0, dur=10.0)
    y[10 * SR:20 * SR] = seg[: 10 * SR]

    feat = extract_basic_features(y, SR, n_mels=N_MELS)
    lattice = build_candidate_lattice(
        feat, feat, y_vid=y, y_ad=y, audio_sr=SR,
        search_radius_sec=5.0, multiband=False,
    )
    scores = [w.speech_score for w in lattice if w.candidates]
    assert scores
    assert all(0.0 <= s <= 1.0 and np.isfinite(s) for s in scores)
    assert max(scores) - min(scores) > 0.1, "speech scores should vary with content"
    speech_windows = [w.speech_score for w in lattice if 12.0 <= w.source_center <= 18.0]
    noise_windows = [w.speech_score for w in lattice if w.source_center <= 8.0 or w.source_center >= 24.0]
    assert np.mean(speech_windows) > np.mean(noise_windows), (
        np.mean(speech_windows), np.mean(noise_windows),
    )
