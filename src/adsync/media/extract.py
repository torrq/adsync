"""Extract audio from media files to analysis WAVs."""

from __future__ import annotations

import logging
from pathlib import Path

from adsync.models import MediaInfo
from adsync.utils.subprocesses import run_ffmpeg

log = logging.getLogger("adsync")


def extract_audio(
    info: MediaInfo,
    output_path: Path,
    *,
    stream_index: int | None = None,
    sr: int = 16000,
    mono: bool = True,
    speed_ratio: float | None = None,
    sample_fmt: str = "s16",
) -> Path:
    """Extract an audio stream to a standardized WAV file.

    Parameters
    ----------
    info:
        Probed media info.
    output_path:
        Destination WAV path.
    stream_index:
        Audio stream index to extract. If *None* the first audio stream is used.
    sr:
        Target sample rate.
    mono:
        Convert to mono.
    speed_ratio:
        When set, stretch the audio by this factor (>1 = slower and longer)
        with pitch following — a plain resample, the exact inverse of a
        PAL-style transfer.  Rate-normalize first so the factor applies
        regardless of the source sample rate.
    sample_fmt:
        "s16" for analysis WAVs, "f32" for rebuild sources that should not
        pick up an extra quantization stage.
    """
    if not info.audio_streams:
        raise ValueError(f"No audio streams found in {info.path}")

    if stream_index is None:
        stream_index = info.audio_streams[0].index

    channels_args = ["-ac", "1"] if mono else []

    filter_args: list[str] = []
    if speed_ratio is not None and abs(speed_ratio - 1.0) > 1e-9:
        shifted = round(sr / speed_ratio)
        filter_args = ["-af", f"aresample={sr},asetrate={shifted},aresample={sr}:filter_size=256:cutoff=0.985"]

    codecs = {"s16": "pcm_s16le", "f32": "pcm_f32le"}
    if sample_fmt not in codecs:
        raise ValueError(f"Unsupported sample_fmt {sample_fmt!r} (use 's16' or 'f32')")

    args = [
        "-i", info.path,
        "-map", f"0:{stream_index}",
        *filter_args,
        "-ar", str(sr),
        *channels_args,
        "-c:a", codecs[sample_fmt],
        "-rf64", "auto",
        "-f", "wav",
        str(output_path),
    ]

    run_ffmpeg(args)
    log.info(
        "Extracted audio → %s (%d Hz, %s%s)",
        output_path.name, sr, "mono" if mono else "stereo",
        f", speed ×{speed_ratio:.6f}" if speed_ratio else "",
    )
    return output_path
