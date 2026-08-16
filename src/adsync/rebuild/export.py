"""Export synced AD audio to file."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from adsync.utils.subprocesses import run_ffmpeg

log = logging.getLogger("adsync")


def export_wav(y: NDArray[np.floating], sr: int, path: Path) -> Path:
    """Write the synced AD waveform to a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # soundfile expects (frames, channels); we store (channels, frames).
    data = y.T if y.ndim == 2 else y
    sf.write(str(path), data, sr, subtype="PCM_16")
    log.info("Exported WAV: %s (%.1f s)", path.name, y.shape[-1] / sr)
    return path


def export_aac(wav_path: Path, output_path: Path, *, bitrate: str = "192k") -> Path:
    """Encode a WAV to AAC in an M4A container."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "-i", str(wav_path),
        "-c:a", "aac",
        "-b:a", bitrate,
        str(output_path),
    ]
    run_ffmpeg(args)
    log.info("Exported AAC: %s", output_path.name)
    return output_path
