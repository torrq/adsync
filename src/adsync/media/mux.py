"""Mux streams into the final container."""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from adsync.utils.subprocesses import run_ffmpeg_streamed

# ~2 MB per chunk at 4 bytes / sample → constant memory overhead
_CHUNK_SAMPLES = 512 * 1024

log = logging.getLogger("adsync")


def mux_ad_track(
    video_path: str | Path,
    synced_y: NDArray[np.floating],
    sr: int,
    output_path: str | Path,
    *,
    codec: str = "libopus",
    bitrate: str = "96k",
    language: str = "eng",
    title: str = "Audio Description",
    n_existing_audio: int | None = None,
) -> Path:
    """Mux original video with AD audio encoded directly into the MKV.

    Streams raw PCM in chunks to FFmpeg via stdin — never materialises the
    full byte array.  Channel count is inferred from *synced_y* (1-D = mono,
    2-D = (channels, samples)).  Uses Opus by default.
    """
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if synced_y.ndim == 1:
        n_channels = 1
        total_samples = len(synced_y)
    else:
        n_channels, total_samples = synced_y.shape

    n_audio = n_existing_audio if n_existing_audio is not None else _count_audio_streams(video_path)

    args = [
        "-i", str(video_path),
        "-f", "f32le", "-ar", str(sr), "-ac", str(n_channels), "-i", "pipe:0",
        "-map", "0",
        "-map", "1:a:0",
        "-c", "copy",
        f"-c:a:{n_audio}", codec,
        f"-b:a:{n_audio}", bitrate,
        f"-metadata:s:a:{n_audio}", f"language={language}",
        f"-metadata:s:a:{n_audio}", f"title={title}",
        str(output_path),
    ]

    proc = run_ffmpeg_streamed(args)
    assert proc.stdin is not None

    stderr_buf: list[bytes] = []
    def _drain_stderr() -> None:
        if proc.stderr:
            stderr_buf.append(proc.stderr.read())
    drain = threading.Thread(target=_drain_stderr, daemon=True)
    drain.start()

    from rich.progress import Progress, BarColumn, TimeRemainingColumn

    total_chunks = (total_samples + _CHUNK_SAMPLES - 1) // _CHUNK_SAMPLES

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Encoding AD track", total=total_chunks)
        for start in range(0, total_samples, _CHUNK_SAMPLES):
            chunk = synced_y[..., start : start + _CHUNK_SAMPLES]
            # Float straight through to the encoder — no 16-bit quantization
            # stage. The renderers guarantee peak <= 0.99; clip is a belt.
            pcm = np.clip(chunk, -1.0, 1.0).astype(np.float32, copy=False)
            if pcm.ndim == 2:
                pcm = np.ascontiguousarray(pcm.T)
            proc.stdin.write(pcm.tobytes())
            progress.advance(task)

    proc.stdin.close()
    proc.wait()
    drain.join()

    if proc.returncode != 0:
        stderr = b"".join(stderr_buf)
        raise subprocess.CalledProcessError(
            proc.returncode, "ffmpeg", stderr=stderr,
        )

    log.info("Muxed output → %s", output_path)
    return output_path


def mux_ad_file(
    video_path: str | Path,
    ad_path: str | Path,
    output_path: str | Path,
    *,
    codec: str = "libopus",
    bitrate: str = "96k",
    language: str = "eng",
    title: str = "Audio Description",
) -> Path:
    """Mux a pre-synced AD audio *file* into the video container.

    Unlike :func:`mux_ad_track` (which streams PCM from memory), this takes
    an on-disk audio file as the second input — intended for the ``adsync mux``
    CLI command where the user supplies an already-synced file.
    """
    from adsync.utils.subprocesses import run_ffmpeg

    video_path = Path(video_path)
    ad_path = Path(ad_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_audio = _count_audio_streams(video_path)

    args = [
        "-i", str(video_path),
        "-i", str(ad_path),
        "-map", "0",
        "-map", "1:a:0",
        "-c", "copy",
        f"-c:a:{n_audio}", codec,
        f"-b:a:{n_audio}", bitrate,
        f"-metadata:s:a:{n_audio}", f"language={language}",
        f"-metadata:s:a:{n_audio}", f"title={title}",
        str(output_path),
    ]

    run_ffmpeg(args)
    log.info("Muxed output → %s", output_path)
    return output_path


def _count_audio_streams(video_path: Path) -> int:
    """Quick helper to count audio streams in the original file."""
    from adsync.media.probe import probe
    info = probe(video_path)
    return len(info.audio_streams)
