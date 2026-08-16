"""Probe media files with ffprobe."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from adsync.models import MediaInfo, StreamInfo
from adsync.utils.subprocesses import run_ffprobe

log = logging.getLogger("adsync")


def probe(path: str | Path) -> MediaInfo:
    """Run ffprobe on *path* and return a structured :class:`MediaInfo`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    result = run_ffprobe([
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ])

    data = json.loads(result.stdout)

    fmt = data.get("format", {})
    streams = data.get("streams", [])

    video_streams: list[StreamInfo] = []
    audio_streams: list[StreamInfo] = []

    for s in streams:
        codec_type = s.get("codec_type", "")
        tags = s.get("tags", {})

        si = StreamInfo(
            index=s.get("index", 0),
            codec_type=codec_type,
            codec_name=s.get("codec_name"),
            channels=_int_or_none(s.get("channels")),
            channel_layout=s.get("channel_layout"),
            sample_rate=_int_or_none(s.get("sample_rate")),
            language=tags.get("language"),
            title=tags.get("title"),
            duration=_float_or_none(s.get("duration") or fmt.get("duration")),
        )

        if codec_type == "video":
            video_streams.append(si)
        elif codec_type == "audio":
            audio_streams.append(si)

    info = MediaInfo(
        path=str(path),
        format_name=fmt.get("format_name"),
        duration=_float_or_none(fmt.get("duration")),
        video_streams=video_streams,
        audio_streams=audio_streams,
    )
    log.info(
        "Probed [bold]%s[/bold]: %.1f s, %d video, %d audio",
        path.name,
        info.duration or 0,
        len(video_streams),
        len(audio_streams),
    )
    return info


def _int_or_none(v: object) -> int | None:
    try:
        return int(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _float_or_none(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
