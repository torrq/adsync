"""Tier-1 (coarse / cheap) feature extraction."""

from __future__ import annotations

import logging

import librosa
import numpy as np
from numpy.typing import NDArray

from adsync.models import FeatureBundle

log = logging.getLogger("adsync")


def extract_basic_features(
    y: NDArray[np.floating],
    sr: int,
    *,
    hop_length: int = 512,
    n_fft: int = 2048,
    n_mels: int = 64,
    n_mfcc: int = 13,
) -> FeatureBundle:
    """Compute Tier-1 features: RMS, onset strength, low-res mel, and MFCC."""
    duration = float(len(y) / sr)

    # RMS envelope (time-domain, cheap)
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]

    # One power spectrogram feeds mel, onset, and MFCC — previously each of
    # them recomputed the same STFT, which dominated this step's cost.
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    np.square(S, out=S)

    # Mel spectrogram (low-res summary)
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Full-resolution mel in dB — the exact intermediate that onset_strength
    # and mfcc would build internally (128 mels, power_to_db with default ref).
    # Note: onset previously always used n_fft=2048 regardless of config; it
    # now follows n_fft like the other features (identical for the default).
    mel128_db = librosa.power_to_db(librosa.feature.melspectrogram(S=S, sr=sr))
    del S

    onset = librosa.onset.onset_strength(S=mel128_db, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(S=mel128_db, sr=sr, n_mfcc=n_mfcc)

    log.info("Features: %.1f s, %d frames (hop=%d)", duration, len(rms), hop_length)

    return FeatureBundle(
        sr=sr,
        hop_length=hop_length,
        duration=duration,
        rms=rms,
        onset=onset,
        mel=mel_db,
        mfcc=mfcc,
    )
