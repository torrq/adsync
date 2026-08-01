"""Runtime configuration and defaults."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SyncConfig(BaseModel):
    """All tuneable parameters for a sync run."""

    # analysis
    analysis_sr: int = Field(16000, description="Sample rate for analysis WAVs")
    mono: bool = Field(True, description="Convert to mono for analysis")
    highpass_hz: float = Field(80.0, description="High-pass filter cutoff")

    # features
    hop_length: int = Field(512, description="Hop length for feature extraction")
    n_fft: int = Field(2048, description="FFT window size")
    n_mels: int = Field(64, description="Number of mel bands")
    n_mfcc: int = Field(13, description="Number of MFCCs")

    # alignment
    confidence_threshold: float = Field(0.70, ge=0.0, le=1.0)
    max_stretch: float = Field(0.01, ge=0.0, le=0.1, description="Max stretch ratio per segment")
    crossfade_ms: int = Field(80, ge=0, le=500)
    anchor_window_sec: float = Field(8.0, description="Window size for anchor search")
    anchor_step_sec: float = Field(2.0, description="Step size for anchor search")

    # output
    ad_language: str = Field("eng", description="Language tag for AD track")
    ad_title: str = Field("Audio Description", description="Title for AD track")
    output_sr: int = Field(48000, description="Sample rate for final output audio")
    output_codec: str = Field("libopus", description="Audio codec for AD track (libopus or aac)")
    output_bitrate: str = Field("96k", description="Bitrate for output AD track")

    # adjustment
    offset_adjust: float = Field(0.0, description="Manual offset adjustment in seconds (positive = push AD later)")

    # warp mode
    warp_max_candidates: int = Field(5, ge=1, le=20, description="Max offset candidates per analysis window")
    warp_lambda_jump: float = Field(2.0, ge=0.0, description="DP penalty weight for offset jumps (per second)")
    warp_lambda_curve: float = Field(5.0, ge=0.0, description="DP penalty weight for curvature (second derivative)")
    warp_lambda_speech: float = Field(0.3, ge=0.0, description="DP bonus weight for speech-rich windows")
    warp_anchor_fraction: float = Field(0.6, ge=0.1, le=1.0, description="Fraction of decoded points used as PCHIP anchors")
    warp_discontinuity_threshold: float = Field(2.0, ge=0.5, description="Offset jump (seconds) to declare a timing discontinuity")
    warp_search_radius: float | None = Field(
        None,
        description="Per-window video-position search radius in seconds, centred on the "
        "global-offset hint. None = auto: scales with the video/AD duration gap and the "
        "measured offset scatter, so large edits stay inside the search range.",
    )
    multiband: bool = Field(
        True,
        description="Correlate in three frequency bands (per-band normalized, "
        "speech band down-weighted) so narration over quiet scenes cannot "
        "drown the correlation evidence.",
    )

    # fingerprinting
    fingerprint: bool = Field(
        True,
        description="Landmark-fingerprint the pair before alignment: locates offset "
        "spans and edit points globally (no search radius), flags unmatched AD "
        "regions, and steers warp mode's per-window search.",
    )
    speed_detect: bool = Field(
        True,
        description="When the fingerprint finds nothing (or a steep offset ramp), "
        "probe standard transfer ratios (PAL 25/24 and its inverse), refine the "
        "exact stretch from span offsets, and auto-correct the AD's speed and "
        "pitch before aligning.",
    )
    fp_anchor: bool = Field(
        True,
        description="When windowed correlation starves (different mixes of the same "
        "material) but fingerprint evidence is strong, anchor alignment windows "
        "directly on landmark matches.",
    )
    fp_anchor_min_coverage: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Correlation coverage below which fingerprint anchoring kicks in",
    )

    # mode
    mode: Literal["auto", "offset", "drift", "piecewise", "warp"] = Field("auto", description="auto|offset|drift|piecewise|warp")
