"""Load audio from WAV files."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
from numpy.typing import NDArray

log = logging.getLogger("adsync")


def load_wav(path: Path | str, *, sr: int = 16000, mono: bool = True) -> tuple[NDArray[np.floating], int]:
    """Load a WAV file. Returns (samples, sample_rate).

    The file is loaded at the requested *sr*.  If *mono* is True the result is
    always 1-D.
    """
    y, sr_out = librosa.load(str(path), sr=sr, mono=mono)
    log.debug("Loaded %s — %.2f s, sr=%d", Path(path).name, y.shape[-1] / sr_out, sr_out)
    return y, sr_out
