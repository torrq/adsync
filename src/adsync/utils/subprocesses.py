"""Subprocess helpers for running FFmpeg / ffprobe."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("adsync")


def _find_binary(name: str) -> str:
    """Locate *name* on PATH or raise."""
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            f"{name} not found on PATH. Install FFmpeg and ensure it is accessible."
        )
    return path


def check_dependencies() -> None:
    """Verify that ffmpeg and ffprobe are available on PATH.

    Raises :class:`FileNotFoundError` with a user-friendly message if either
    binary is missing.
    """
    missing: list[str] = []
    for name in ("ffmpeg", "ffprobe"):
        if shutil.which(name) is None:
            missing.append(name)
    if missing:
        names = " and ".join(missing)
        raise FileNotFoundError(
            f"{names} not found on PATH. "
            "Install FFmpeg (https://ffmpeg.org) and ensure it is accessible."
        )


class FFmpegError(RuntimeError):
    """User-friendly wrapper around FFmpeg failures."""

    def __init__(self, returncode: int, stderr: str) -> None:
        self.returncode = returncode
        self.stderr = stderr
        summary = _parse_ffmpeg_error(stderr)
        super().__init__(f"FFmpeg failed (exit {returncode}): {summary}")


def _parse_ffmpeg_error(stderr: str) -> str:
    """Extract the most relevant error line from FFmpeg's stderr."""
    for line in reversed(stderr.strip().splitlines()):
        lower = line.lower()
        if any(kw in lower for kw in ("error", "invalid", "no such", "unknown", "not found")):
            return line.strip()
    # Fall back to last non-empty line
    lines = [l.strip() for l in stderr.strip().splitlines() if l.strip()]
    return lines[-1] if lines else "(no output)"


def run_ffmpeg(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg command and return the result.

    Raises :class:`FFmpegError` with a parsed message on failure.
    """
    binary = _find_binary("ffmpeg")
    cmd = [binary, "-hide_banner", "-y", *args]
    log.debug("ffmpeg %s", " ".join(args))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and result.returncode != 0:
        raise FFmpegError(result.returncode, result.stderr)
    return result


def run_ffmpeg_piped(
    args: list[str],
    stdin_data: bytes,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    """Run an ffmpeg command with raw bytes piped to stdin.

    Raises :class:`FFmpegError` on failure.
    """
    binary = _find_binary("ffmpeg")
    cmd = [binary, "-hide_banner", "-y", *args]
    log.debug("ffmpeg %s", " ".join(args))
    result = subprocess.run(
        cmd,
        input=stdin_data,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace") if isinstance(result.stderr, bytes) else result.stderr
        raise FFmpegError(result.returncode, stderr_text)
    return result


def run_ffmpeg_streamed(
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.Popen[bytes]:
    """Start an ffmpeg process with stdin open for streaming writes.

    Returns the Popen object so the caller can write chunks to ``proc.stdin``
    and call ``proc.stdin.close()`` + ``proc.wait()`` when done.
    """
    binary = _find_binary("ffmpeg")
    cmd = [binary, "-hide_banner", "-y", *args]
    log.debug("ffmpeg %s", " ".join(args))
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def run_ffprobe(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run an ffprobe command and return the result."""
    binary = _find_binary("ffprobe")
    cmd = [binary, "-hide_banner", *args]
    log.debug("ffprobe %s", " ".join(args))
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
