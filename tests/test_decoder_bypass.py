"""Decoder bypass of confidently-wrong candidates.

Repeated soundtrack music produces high-scoring correlation peaks at wrong
offsets.  When every strong candidate in a window contradicts the fingerprint
hint, the decoder must have a cheap pass-through route; short excursions with
agreeing flanks must be suppressed even without a fingerprint.  Real edits —
sustained evidence that agrees with the hint — must still be followed.
"""

from __future__ import annotations

import pytest

from adsync.align.warp_decode import decode_warp_path
from adsync.models import CandidateWindow, OffsetCandidate


def _win(center: float, cands: list[tuple[float, float]], speech: float = 0.5) -> CandidateWindow:
    return CandidateWindow(
        source_center=center,
        candidates=[
            OffsetCandidate(offset_sec=o, score=s, peak_sharpness=2.0, peak_ratio=2.0)
            for o, s in cands
        ],
        speech_score=speech,
        energy=1.0,
    )


def _decoy(i: int) -> float:
    """Incoherent spurious-peak offset — varies per window, never forms a line."""
    return 15.0 + (i * 7.919) % 30.0


def _blip_lattice() -> list[CandidateWindow]:
    """True offset +44 throughout; windows 10-12 only offer a wrong +36.6 peak."""
    lattice = []
    for i in range(30):
        t = 2.0 * i
        if 10 <= i <= 12:
            lattice.append(_win(t, [(36.6, 0.65)]))
        else:
            lattice.append(_win(t, [(44.0, 0.6), (_decoy(i), 0.35)]))
    return lattice


def test_hint_contradiction_gets_bypass() -> None:
    """With a fingerprint hint at +44, the wrong-peak windows must be bypassed."""
    points, _cost = decode_warp_path(_blip_lattice(), offset_hint_fn=lambda t: 44.0)
    for p in points:
        assert p.target_time - p.source_time == pytest.approx(44.0, abs=0.5), (
            f"excursion at t={p.source_time}: offset {p.target_time - p.source_time:+.2f}"
        )


def test_short_excursion_suppressed_without_hint() -> None:
    """Without any hint, a 3-window excursion between agreeing flanks is snapped."""
    points, _cost = decode_warp_path(_blip_lattice())
    for p in points:
        assert p.target_time - p.source_time == pytest.approx(44.0, abs=0.5), (
            f"excursion at t={p.source_time}: offset {p.target_time - p.source_time:+.2f}"
        )
    snapped = [p for p in points if 20.0 <= p.source_time <= 24.0]
    assert all(p.confidence <= 0.06 for p in snapped), "snapped points must be low-confidence"


def test_real_edit_still_followed() -> None:
    """A sustained offset step backed by evidence and hint must be taken."""
    lattice = []
    for i in range(60):
        t = 2.0 * i
        off = 44.0 if i < 30 else 50.0
        lattice.append(_win(t, [(off, 0.6), (_decoy(i), 0.3)]))
    hint = lambda t: 44.0 if t < 59.0 else 50.0
    points, _cost = decode_warp_path(lattice, offset_hint_fn=hint)
    for p in points:
        if abs(p.source_time - 59.0) <= 6.0:
            continue  # windows straddling the edit see two truths (harness convention)
        expected = 44.0 if p.source_time < 59.0 else 50.0
        assert p.target_time - p.source_time == pytest.approx(expected, abs=0.5), (
            f"t={p.source_time}: offset {p.target_time - p.source_time:+.2f}, expected {expected}"
        )


def test_real_tail_edit_preserved() -> None:
    """A large late jump (credits with extra video content) must survive."""
    lattice = []
    for i in range(70):
        t = 2.0 * i
        off = 46.0 if i < 55 else 71.0
        lattice.append(_win(t, [(off, 0.55)]))
    hint = lambda t: 46.0 if t < 109.0 else 71.0
    points, _cost = decode_warp_path(lattice, offset_hint_fn=hint)
    tail = [p for p in points if p.source_time >= 112.0]
    assert tail, "no tail points decoded"
    for p in tail:
        assert p.target_time - p.source_time == pytest.approx(71.0, abs=0.5), (
            f"tail flattened: t={p.source_time} offset {p.target_time - p.source_time:+.2f}"
        )
