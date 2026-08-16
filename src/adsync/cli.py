"""ADSync CLI — Typer-based entry point."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from adsync import __version__
from adsync.config import SyncConfig
from adsync.logging import setup_logging

app = typer.Typer(
    name="adsync",
    help="Synchronize audio description tracks with video files.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()


# ── Shared options ───────────────────────────────────────────────────────────


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"adsync {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", "-V", callback=_version_callback, is_eager=True),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """ADSync — Audio Description Synchronization."""
    from adsync.utils.subprocesses import check_dependencies

    try:
        check_dependencies()
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(2)

    setup_logging(verbose=verbose)


# ── sync ─────────────────────────────────────────────────────────────────────


@app.command()
def sync(
    video: Path = typer.Argument(..., help="Video file (e.g. episode.mkv)"),
    ad_audio: Path = typer.Argument(..., help="Audio description file (e.g. ad_track.m4a)"),
    output: Path = typer.Option(None, "-o", "--output", help="Output MKV path"),
    report: Optional[Path] = typer.Option(None, "--report", help="Write JSON report"),
    keep_temp: bool = typer.Option(False, "--keep-temp"),
    debug_dir: Optional[Path] = typer.Option(None, "--debug-dir"),
    language: str = typer.Option("eng", "--language"),
    ad_title: str = typer.Option("Audio Description", "--ad-title"),
    confidence_threshold: float = typer.Option(0.70, "--confidence-threshold"),
    max_stretch: float = typer.Option(0.01, "--max-stretch"),
    crossfade_ms: int = typer.Option(80, "--crossfade-ms"),
    analysis_sr: int = typer.Option(16000, "--analysis-sr"),
    mode: str = typer.Option("auto", "--mode", help="auto|offset|drift|piecewise|warp"),
    offset_adjust: float = typer.Option(0.0, "--offset-adjust", help="Manual offset tweak in seconds (positive = push AD later)"),
    codec: str = typer.Option("libopus", "--codec", help="Audio codec for AD track (libopus, aac, etc.)"),
    bitrate: str = typer.Option("96k", "--bitrate", help="Bitrate for AD track (e.g. 96k, 128k)"),
    warp_lambda_jump: float = typer.Option(2.0, "--warp-lambda-jump", help="Warp DP jump penalty weight"),
    warp_lambda_curve: float = typer.Option(5.0, "--warp-lambda-curve", help="Warp DP curvature penalty weight"),
    warp_lambda_speech: float = typer.Option(0.3, "--warp-lambda-speech", help="Warp speech bonus weight"),
    warp_candidates: int = typer.Option(5, "--warp-candidates", help="Max offset candidates per analysis window"),
    warp_search_radius: Optional[float] = typer.Option(None, "--warp-search-radius", help="Per-window search radius in seconds around the detected offset (default: auto from fingerprint spans, duration gap + offset scatter)"),
    no_fingerprint: bool = typer.Option(False, "--no-fingerprint", help="Skip landmark fingerprinting (global offset-span detection)"),
    no_speed_detect: bool = typer.Option(False, "--no-speed-detect", help="Skip automatic PAL-style speed detection/correction"),
    no_fp_anchor: bool = typer.Option(False, "--no-fp-anchor", help="Never anchor alignment windows on fingerprint matches"),
    no_multiband: bool = typer.Option(False, "--no-multiband", help="Correlate full-band only (skip per-band scoring)"),
) -> None:
    """Full sync pipeline — produces synced MKV output."""
    _validate_inputs(video, ad_audio)

    if output is None:
        output = video.with_suffix(".synced.mkv")

    config = SyncConfig(
        analysis_sr=analysis_sr,
        confidence_threshold=confidence_threshold,
        max_stretch=max_stretch,
        crossfade_ms=crossfade_ms,
        ad_language=language,
        ad_title=ad_title,
        mode=mode,
        offset_adjust=offset_adjust,
        output_codec=codec,
        output_bitrate=bitrate,
        warp_lambda_jump=warp_lambda_jump,
        warp_lambda_curve=warp_lambda_curve,
        warp_lambda_speech=warp_lambda_speech,
        warp_max_candidates=warp_candidates,
        warp_search_radius=warp_search_radius,
        fingerprint=not no_fingerprint,
        speed_detect=not no_speed_detect,
        fp_anchor=not no_fp_anchor,
        multiband=not no_multiband,
    )

    from adsync._pipeline import run_pipeline

    result = run_pipeline(
        video_path=video,
        ad_path=ad_audio,
        output_path=output,
        config=config,
        report_path=report,
        debug_dir=debug_dir,
        keep_temp=keep_temp,
        mux=True,
    )

    raise typer.Exit(0 if result.confidence >= config.confidence_threshold else 1)


# ── prep ─────────────────────────────────────────────────────────────────────


@app.command()
def prep(
    video: Path = typer.Argument(..., help="Source video (e.g. movie.mkv)"),
    output: Path = typer.Option(None, "-o", "--output", help="Output path (default: <name>.prepped.mkv)"),
    language: str = typer.Option("eng", "--language", "-l", help="Audio language to keep"),
    audio_index: Optional[int] = typer.Option(None, "--audio-index", help="Keep this exact audio stream index (overrides --language)"),
    codec: str = typer.Option("libopus", "--codec", help="Encoder when downmixing (libopus ~4x faster than aac)"),
    bitrate: Optional[str] = typer.Option(None, "--bitrate", help="Bitrate when downmixing (default: 192k opus / 256k aac)"),
    no_limiter: bool = typer.Option(False, "--no-limiter", help="Skip the true-peak limiter after downmixing"),
) -> None:
    """Prep a video for syncing: keep one language's audio (stereo), drop the rest.

    Video and subtitles are always stream-copied. The kept audio is
    stream-copied too when it is already stereo/mono; multichannel tracks are
    downmixed with a dialog-forward formula and a true-peak limiter.
    """
    _validate_inputs(video)

    from adsync.media.prep import prep_video

    state = {"last": 0.0}

    def _print_progress(done: float, total: float, speed: float) -> None:
        # Plain, newline-terminated updates (~every 10 s) instead of a live
        # bar — legible in logs and pleasant with screen readers.
        import time as _time
        now = _time.monotonic()
        if now - state["last"] < 10.0 or not total or done <= 0:
            return
        state["last"] = now
        pct = min(100.0, 100.0 * done / total)
        eta = (total - done) / speed if speed > 0 else 0.0
        console.print(
            f"  prep {pct:3.0f}%  ({_mmss(done)} / {_mmss(total)})  "
            f"{speed:.0f}x realtime  eta {_mmss(eta)}"
        )

    try:
        result = prep_video(
            video, output,
            language=language,
            audio_index=audio_index,
            codec=codec,
            bitrate=bitrate,
            limiter=not no_limiter,
            on_progress=_print_progress,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(2)

    console.print(
        f"[green]Prepped → {result.output_path}[/green]  "
        f"({result.elapsed_sec:.0f} s, audio={result.audio_codec})"
    )


def _mmss(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ── analyze ──────────────────────────────────────────────────────────────────


@app.command()
def analyze(
    video: Path = typer.Argument(..., help="Video file"),
    ad_audio: Path = typer.Argument(..., help="Audio description file"),
    report: Path = typer.Option("report.json", "--report"),
    mode: str = typer.Option("auto", "--mode"),
    analysis_sr: int = typer.Option(16000, "--analysis-sr"),
) -> None:
    """Run analysis only — no final mux."""
    _validate_inputs(video, ad_audio)

    config = SyncConfig(analysis_sr=analysis_sr, mode=mode)

    from adsync._pipeline import run_pipeline

    run_pipeline(
        video_path=video,
        ad_path=ad_audio,
        output_path=None,
        config=config,
        report_path=report,
        debug_dir=None,
        keep_temp=False,
        mux=False,
    )


# ── debug ────────────────────────────────────────────────────────────────────


@app.command()
def debug(
    video: Path = typer.Argument(..., help="Video file"),
    ad_audio: Path = typer.Argument(..., help="Audio description file"),
    workdir: Path = typer.Option("debug_out", "--workdir"),
    mode: str = typer.Option("auto", "--mode"),
    analysis_sr: int = typer.Option(16000, "--analysis-sr"),
) -> None:
    """Analysis + dump intermediates (WAVs, plots, anchors, segment maps)."""
    _validate_inputs(video, ad_audio)

    config = SyncConfig(analysis_sr=analysis_sr, mode=mode)

    from adsync._pipeline import run_pipeline

    run_pipeline(
        video_path=video,
        ad_path=ad_audio,
        output_path=None,
        config=config,
        report_path=workdir / "report.json",
        debug_dir=workdir,
        keep_temp=True,
        mux=False,
    )


# ── mux ──────────────────────────────────────────────────────────────────────


@app.command()
def mux(
    video: Path = typer.Argument(..., help="Video file"),
    ad_audio: Path = typer.Argument(..., help="Already-synced AD audio file"),
    output: Path = typer.Option(None, "-o", "--output"),
    language: str = typer.Option("eng", "--language"),
    ad_title: str = typer.Option("Audio Description", "--ad-title"),
) -> None:
    """Mux a pre-synced AD track into the video container."""
    _validate_inputs(video, ad_audio)

    if output is None:
        output = video.with_suffix(".synced.mkv")

    from adsync.media.mux import mux_ad_file

    mux_ad_file(video, ad_audio, output, language=language, title=ad_title)
    console.print(f"[green]Muxed → {output}[/green]")


# ── helpers ──────────────────────────────────────────────────────────────────


def _validate_inputs(*paths: Path) -> None:
    for p in paths:
        if not p.exists():
            console.print(f"[red]Error:[/red] File not found: {p}")
            raise typer.Exit(2)


if __name__ == "__main__":
    app()
