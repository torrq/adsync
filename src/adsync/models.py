"""Pydantic data models used throughout the pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator


# ── Media ────────────────────────────────────────────────────────────────────

class StreamInfo(BaseModel):
    index: int
    codec_type: str
    codec_name: str | None = None
    channels: int | None = None
    channel_layout: str | None = None
    sample_rate: int | None = None
    language: str | None = None
    title: str | None = None
    duration: float | None = None


class MediaInfo(BaseModel):
    path: str
    format_name: str | None = None
    duration: float | None = None
    video_streams: list[StreamInfo] = Field(default_factory=list)
    audio_streams: list[StreamInfo] = Field(default_factory=list)


# ── Features ─────────────────────────────────────────────────────────────────

class FeatureBundle(BaseModel):
    """Feature vectors computed from one audio source."""

    model_config = {"arbitrary_types_allowed": True}

    sr: int
    hop_length: int
    duration: float
    rms: NDArray[np.floating]
    onset: NDArray[np.floating]
    mel: NDArray[np.floating]
    mfcc: NDArray[np.floating]

    @field_validator("rms", "onset", "mel", "mfcc", mode="before")
    @classmethod
    def _coerce_to_ndarray(cls, v: Any) -> NDArray:
        if isinstance(v, np.ndarray):
            return v
        return np.asarray(v, dtype=np.float64)


# ── Alignment ────────────────────────────────────────────────────────────────

class Anchor(BaseModel):
    source_time: float
    target_time: float
    score: float
    window: float


class OffsetCandidate(BaseModel):
    """One candidate offset within a single analysis window."""
    offset_sec: float
    score: float
    peak_sharpness: float
    peak_ratio: float
    source: str = "corr"  # "corr" (cross-correlation) | "fingerprint"


class CandidateWindow(BaseModel):
    """All candidates for a single analysis window position."""
    source_center: float
    candidates: list[OffsetCandidate]
    speech_score: float
    energy: float


class FingerprintSpan(BaseModel):
    """A stretch of the AD that fingerprint matching placed at one offset."""
    ad_start: float
    ad_end: float
    offset: float
    matches: int


class WarpPoint(BaseModel):
    """A single point on the decoded warp path."""
    source_time: float
    target_time: float
    confidence: float
    is_anchor: bool = False


class WarpPath(BaseModel):
    """The full decoded warp path and fitted function metadata."""
    points: list[WarpPoint] = Field(default_factory=list)
    anchor_points: list[WarpPoint] = Field(default_factory=list)
    path_cost: float = 0.0
    mean_confidence: float = 0.0
    dropped_segments: int = 0        # excursion segments rejected by vetting
    bridged_sec: float = 0.0         # AD seconds coasting on flanking offsets
    n_segments: int = 0              # fitted PCHIP segments


class SegmentMap(BaseModel):
    src_start: float
    src_end: float
    dst_start: float
    dst_end: float
    offset: float
    stretch: float = 1.0
    confidence: float


# ── Report ───────────────────────────────────────────────────────────────────

class SyncReport(BaseModel):
    mode: str
    confidence: float
    global_offset: float | None = None
    drift_ppm: float | None = None
    anchors: list[Anchor] = Field(default_factory=list)
    segments: list[SegmentMap] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    output_path: str | None = None
    warp_path: WarpPath | None = None
    fingerprint_spans: list[FingerprintSpan] = Field(default_factory=list)
    fingerprint_unmatched: list[tuple[float, float]] = Field(default_factory=list)
    speed_stretch: float | None = None      # auto speed correction applied to the AD
    fp_anchor_windows: int = 0              # windows anchored on fingerprint matches
    fp_residual_p50_ms: float | None = None  # warp vs fingerprint matches, median
    fp_residual_p95_ms: float | None = None  # warp vs fingerprint matches, p95
