"""Monotone-chain filtering of fingerprint spans.

A valid AD→video playback map must be (near-)monotone in video time.
Repeated soundtrack content produces per-bucket majority votes that point
backwards (the Sinister credits song also plays 15 minutes earlier); those
buckets must lose to the globally consistent chain and surface as unmatched
instead of poisoning span hints.
"""

from __future__ import annotations

import numpy as np

from adsync.align.fingerprint import _offset_spans


def _synth_matches(regions: list[tuple[float, float, float, float]]):
    """Build (t_ad, t_vid) match arrays from (ad_lo, ad_hi, offset, per_sec) regions."""
    t_ad_parts = []
    t_vid_parts = []
    for ad_lo, ad_hi, offset, per_sec in regions:
        n = max(1, int((ad_hi - ad_lo) * per_sec))
        t_ad = np.linspace(ad_lo, ad_hi, n, endpoint=False)
        t_ad_parts.append(t_ad)
        t_vid_parts.append(t_ad + offset)
    t_ad = np.concatenate(t_ad_parts).astype(np.float32)
    t_vid = np.concatenate(t_vid_parts).astype(np.float32)
    order = np.argsort(t_ad)
    return t_ad[order], t_vid[order]


def _span_offsets(result) -> list[float]:
    return [round(s.offset, 1) for s in result.spans]


def test_backwards_credits_span_dropped() -> None:
    """A dense backwards vote at the tail (credits reprise) must be dropped."""
    t_ad, t_vid = _synth_matches([
        (0.0, 580.0, 46.0, 3.0),
        (580.0, 600.0, -200.0, 3.0),   # credits music matching an earlier scene
    ])
    result = _offset_spans(t_ad, t_vid, 600.0)

    assert all(abs(o - 46.0) < 1.0 for o in _span_offsets(result)), _span_offsets(result)
    # The dropped region must surface as unmatched (>= 2 buckets wide)
    assert any(lo <= 580.0 and hi >= 600.0 for lo, hi in result.unmatched), result.unmatched
    # ... and be reported as a dropped repeated-content range
    assert any(
        lo >= 570.0 and hi <= 610.0 and off < -100.0
        for lo, hi, off in result.dropped_ranges
    ), result.dropped_ranges


def test_small_regressing_span_dropped() -> None:
    """A light span regressing a few seconds against heavy neighbours is dropped."""
    t_ad, t_vid = _synth_matches([
        (0.0, 300.0, 46.5, 3.0),
        (300.0, 320.0, 19.0, 0.8),     # repeated cue, 27 s backwards, light support
        (320.0, 600.0, 46.5, 3.0),
    ])
    result = _offset_spans(t_ad, t_vid, 600.0)

    assert all(abs(o - 46.5) < 1.0 for o in _span_offsets(result)), _span_offsets(result)


def test_ad_extra_content_offset_drop_survives() -> None:
    """Legit decreasing offset (AD has extra content; video never regresses)."""
    t_ad, t_vid = _synth_matches([
        (0.0, 300.0, 40.0, 3.0),
        (305.0, 600.0, 35.0, 3.0),     # AD-only 5 s at 300–305; video continuous
    ])
    result = _offset_spans(t_ad, t_vid, 600.0)

    offsets = _span_offsets(result)
    assert any(abs(o - 40.0) < 1.0 for o in offsets), offsets
    assert any(abs(o - 35.0) < 1.0 for o in offsets), offsets


def test_forward_credits_jump_survives() -> None:
    """A real forward jump at the tail (extra video content) is kept."""
    t_ad, t_vid = _synth_matches([
        (0.0, 560.0, 46.0, 3.0),
        (570.0, 600.0, 71.0, 3.0),     # video has ~25 s extra before credits
    ])
    result = _offset_spans(t_ad, t_vid, 600.0)

    offsets = _span_offsets(result)
    assert any(abs(o - 46.0) < 1.0 for o in offsets), offsets
    assert any(abs(o - 71.0) < 1.0 for o in offsets), offsets


def test_inliers_exclude_dropped_regions() -> None:
    """Anchor-grade match pairs must not include matches from dropped buckets."""
    t_ad, t_vid = _synth_matches([
        (0.0, 580.0, 46.0, 3.0),
        (580.0, 600.0, -200.0, 3.0),
    ])
    result = _offset_spans(t_ad, t_vid, 600.0)

    assert result.match_t_ad is not None
    offsets = result.match_t_vid - result.match_t_ad
    assert float(np.min(offsets)) > 0.0, "backwards matches leaked into inliers"
