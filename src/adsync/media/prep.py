"""Preprocess a source video for AD syncing.

Keeps the video, one language's audio (downmixed to stereo when needed), and
all subtitles; drops every other audio track.  This replaces the ad-hoc
per-project ffmpeg prep scripts.

Speed notes, measured on a 2 h film (HE-AAC 5.1 source, NVMe):

- Video and subtitles are always stream-copied (~1500x realtime alone).
- If the kept track is already stereo/mono it is stream-copied too, and the
  whole prep runs at remux speed.
- Downmixing re-encodes only the audio.  FFmpeg's native ``aac`` encoder is
  the classic trap here: single-threaded ~33x realtime, and the muxer paces
  the video copy to it, so the *whole file* crawls.  ``libopus`` measures
  ~128x on the same input — that is the default.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from adsync.models import MediaInfo, StreamInfo
from adsync.utils.subprocesses import FFmpegError, _find_binary

log = logging.getLogger("adsync")

# Center-forward stereo downmix, matching the original From/Obsession prep on
# 5.1 sources: full-gain centre (dialog first), attenuated fronts + surrounds,
# LFE dropped to keep rumble out of the stereo mix.
_FRONT_GAIN = 0.707
_SURROUND_GAIN = 0.707
_WIDE_GAIN = 0.5

# ffmpeg channel-layout name → channel names present in that layout.
# Only the names matter for pan; order is ffmpeg's canonical order.
_LAYOUT_CHANNELS: dict[str, list[str]] = {
    "mono": ["FC"],
    "stereo": ["FL", "FR"],
    "2.1": ["FL", "FR", "LFE"],
    "3.0": ["FL", "FR", "FC"],
    "3.0(back)": ["FL", "FR", "BC"],
    "3.1": ["FL", "FR", "FC", "LFE"],
    "4.0": ["FL", "FR", "FC", "BC"],
    "quad": ["FL", "FR", "BL", "BR"],
    "quad(side)": ["FL", "FR", "SL", "SR"],
    "4.1": ["FL", "FR", "FC", "LFE", "BC"],
    "5.0": ["FL", "FR", "FC", "BL", "BR"],
    "5.0(side)": ["FL", "FR", "FC", "SL", "SR"],
    "5.1": ["FL", "FR", "FC", "LFE", "BL", "BR"],
    "5.1(side)": ["FL", "FR", "FC", "LFE", "SL", "SR"],
    "6.0": ["FL", "FR", "FC", "BC", "SL", "SR"],
    "6.1": ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
    "7.0": ["FL", "FR", "FC", "BL", "BR", "SL", "SR"],
    "7.1": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    "7.1(wide)": ["FL", "FR", "FC", "LFE", "FLC", "FRC", "BL", "BR"],
    "7.1(wide-side)": ["FL", "FR", "FC", "LFE", "FLC", "FRC", "SL", "SR"],
}

# channel name → (gain, destination): destination is "L", "R", or "both"
_CHANNEL_MIX: dict[str, tuple[float, str]] = {
    "FC": (1.0, "both"),
    "FL": (_FRONT_GAIN, "L"),
    "FR": (_FRONT_GAIN, "R"),
    "BL": (_SURROUND_GAIN, "L"),
    "BR": (_SURROUND_GAIN, "R"),
    "SL": (_SURROUND_GAIN, "L"),
    "SR": (_SURROUND_GAIN, "R"),
    "BC": (_WIDE_GAIN, "both"),
    "FLC": (_WIDE_GAIN, "L"),
    "FRC": (_WIDE_GAIN, "R"),
    # LFE intentionally absent — never mixed in
}

# The downmix gains can sum past full scale on correlated loud passages
# (measured ~+1.2 dB overs on a theatrical climax).  A look-ahead limiter with
# no makeup gain tames only those rare peaks; latency=1 compensates the
# look-ahead delay so A/V sync is untouched.
_LIMITER = "alimiter=limit=0.98:level=false:latency=1"

_DEFAULT_BITRATE = {"libopus": "192k", "aac": "256k"}


@dataclass
class PrepResult:
    output_path: Path
    elapsed_sec: float
    picked: StreamInfo
    downmixed: bool
    audio_codec: str  # "copy" or the encoder name


def build_downmix_pan(layout: str | None) -> str | None:
    """Build a stereo pan filter for *layout*, or None when unknown/not needed.

    For "5.1" this reproduces the original prep formula exactly:
    ``pan=stereo|FL=FC+0.707*FL+0.707*BL|FR=FC+0.707*FR+0.707*BR``
    """
    if not layout:
        return None
    channels = _LAYOUT_CHANNELS.get(layout)
    if channels is None:
        return None

    left: list[str] = []
    right: list[str] = []
    # Centre channels first so the 5.1 output is byte-identical to the
    # original prep formula (FC+0.707*FL+0.707*BL), then sides in layout order.
    for pass_dest in ("both", "side"):
        for ch in channels:
            mix = _CHANNEL_MIX.get(ch)
            if mix is None:
                continue  # LFE and anything exotic stays out of the stereo mix
            gain, dest = mix
            if (dest == "both") != (pass_dest == "both"):
                continue
            term = ch if gain == 1.0 else f"{gain:g}*{ch}"
            if dest in ("L", "both"):
                left.append(term)
            if dest in ("R", "both"):
                right.append(term)

    if not left or not right:
        return None
    return f"pan=stereo|FL={'+'.join(left)}|FR={'+'.join(right)}"


def pick_audio_stream(
    info: MediaInfo,
    *,
    language: str | None = "eng",
    audio_index: int | None = None,
    strict: bool = True,
) -> StreamInfo:
    """Choose which audio stream to keep.

    An explicit *audio_index* wins.  Otherwise streams are matched on the
    first two letters of the language tag ("eng" matches "en"); among matches
    the one with the most channels wins (main mix beats stereo commentary),
    ties broken by file order.

    With ``strict=False`` an ambiguous no-match falls back to the first audio
    stream with a warning instead of raising — for callers (like the sync
    pipeline) where any stream still works, just less well.
    """
    streams = info.audio_streams
    if not streams:
        raise ValueError(f"No audio streams in {info.path}")

    if audio_index is not None:
        for s in streams:
            if s.index == audio_index:
                return s
        raise ValueError(
            f"Stream index {audio_index} is not an audio stream.\n{_format_streams(streams)}"
        )

    if language:
        want = language.lower()[:2]
        matches = [s for s in streams if (s.language or "").lower()[:2] == want]
        if matches:
            return max(matches, key=lambda s: s.channels or 0)
        if len(streams) == 1:
            log.warning(
                "No %r audio track found; keeping the only audio stream (lang=%s)",
                language, streams[0].language or "untagged",
            )
            return streams[0]
        if not strict:
            log.warning(
                "No %r audio track found among %d streams; falling back to the first "
                "(lang=%s)", language, len(streams), streams[0].language or "untagged",
            )
            return streams[0]
        raise ValueError(
            f"No {language!r} audio track in {Path(info.path).name} and several to pick from.\n"
            f"{_format_streams(streams)}\n"
            "Re-run with --audio-index <n> to choose one explicitly."
        )

    return max(streams, key=lambda s: s.channels or 0)


def _format_streams(streams: list[StreamInfo]) -> str:
    lines = ["Audio streams:"]
    for s in streams:
        lines.append(
            f"  index {s.index}: {s.codec_name or '?'}  lang={s.language or 'untagged'}  "
            f"{s.channels or '?'}ch ({s.channel_layout or 'unknown layout'})"
            + (f"  title={s.title!r}" if s.title else "")
        )
    return "\n".join(lines)


def prep_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    *,
    language: str | None = "eng",
    audio_index: int | None = None,
    codec: str = "libopus",
    bitrate: str | None = None,
    limiter: bool = True,
    on_progress: Callable[[float, float, float], None] | None = None,
) -> PrepResult:
    """Prep *video_path* → a new MKV with one stereo audio track.

    on_progress, when given, is called as ``(done_sec, total_sec, speed_x)``
    for every ffmpeg progress tick (~2/s).
    """
    from adsync.media.probe import probe

    video_path = Path(video_path)
    if output_path is None:
        output_path = video_path.parent / (video_path.stem + ".prepped.mkv")
    output_path = Path(output_path)

    info = probe(video_path)
    total_sec = info.duration or 0.0
    picked = pick_audio_stream(info, language=language, audio_index=audio_index)

    needs_downmix = (picked.channels or 0) > 2
    lang_tag = (picked.language or language or "und")[:3]

    if needs_downmix:
        pan = build_downmix_pan(picked.channel_layout)
        if pan is None:
            log.warning(
                "Unknown channel layout %r — using ffmpeg's default downmix",
                picked.channel_layout,
            )
        af_chain = ",".join(filter(None, [pan, _LIMITER if limiter else None]))
        audio_args = ["-af", af_chain] if af_chain else []
        if pan is None:
            audio_args += ["-ac", "2"]
        audio_args += ["-c:a", codec, "-b:a", bitrate or _DEFAULT_BITRATE.get(codec, "192k")]
        if codec == "libopus":
            audio_args += ["-ar", "48000"]
        title = "English Stereo" if lang_tag.startswith("en") else "Stereo Downmix"
        audio_args += ["-metadata:s:a:0", f"title={title}"]
        codec_used = codec
    else:
        # Already stereo/mono → pure remux, runs at I/O speed
        audio_args = ["-c:a", "copy"]
        codec_used = "copy"

    log.info(
        "Keeping audio stream %d: %s %dch (%s) lang=%s → %s",
        picked.index, picked.codec_name, picked.channels or 0,
        picked.channel_layout or "?", picked.language or "untagged",
        "stereo downmix" if needs_downmix else "stream copy",
    )

    args = [
        "-i", str(video_path),
        # 0:V (capital) excludes attached pictures, so embedded cover art can
        # never be picked as "the movie"
        "-map", "0:V:0",
        "-map", f"0:{picked.index}",
        "-map", "0:s?",
        "-map", "0:t?",
        "-c:v", "copy",
        "-c:s", "copy",
        *audio_args,
        "-metadata:s:a:0", f"language={lang_tag}",
        "-disposition:a:0", "default",
        str(output_path),
    ]

    t0 = time.monotonic()
    _run_ffmpeg_with_progress(args, total_sec, on_progress)
    elapsed = time.monotonic() - t0

    # Sanity: the output should exist and cover the source duration.
    out_info = probe(output_path)
    if total_sec and out_info.duration and abs(out_info.duration - total_sec) > 5.0:
        log.warning(
            "Prepped duration %.1f s differs from source %.1f s — check the output",
            out_info.duration, total_sec,
        )

    log.info(
        "Prepped → %s  (%.0f s, %.0fx realtime, audio=%s)",
        output_path.name, elapsed, (total_sec / elapsed) if elapsed > 0 else 0.0,
        codec_used,
    )
    return PrepResult(
        output_path=output_path,
        elapsed_sec=elapsed,
        picked=picked,
        downmixed=needs_downmix,
        audio_codec=codec_used,
    )


def _run_ffmpeg_with_progress(
    args: list[str],
    total_sec: float,
    on_progress: Callable[[float, float, float], None] | None,
) -> None:
    """Run ffmpeg emitting machine-readable progress on stdout and parse it."""
    binary = _find_binary("ffmpeg")
    cmd = [
        binary, "-hide_banner", "-y", "-nostats", "-loglevel", "error",
        "-progress", "pipe:1", *args,
    ]
    log.debug("ffmpeg %s", " ".join(args))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None

    stderr_buf: list[str] = []

    def _drain_stderr() -> None:
        if proc.stderr:
            stderr_buf.append(proc.stderr.read())

    drain = threading.Thread(target=_drain_stderr, daemon=True)
    drain.start()

    done_sec = 0.0
    speed = 0.0
    for line in proc.stdout:
        key, _, value = line.strip().partition("=")
        if key == "out_time_us" and value.lstrip("-").isdigit():
            done_sec = max(done_sec, int(value) / 1e6)
        elif key == "speed":
            try:
                speed = float(value.rstrip("x"))
            except ValueError:
                pass
        elif key == "progress":
            if on_progress is not None:
                on_progress(done_sec, total_sec, speed)

    proc.wait()
    drain.join()
    if proc.returncode != 0:
        raise FFmpegError(proc.returncode, "".join(stderr_buf))
