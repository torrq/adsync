"""Viterbi / DP decoder — find the globally optimal offset path through the
candidate lattice.

Penalises offset jumps and curvature (second derivative of offset) so that
musically-confused intro regions are forced to follow the globally consistent
offset instead of chasing false local correlation peaks.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from adsync.models import CandidateWindow, WarpPoint

log = logging.getLogger("adsync")

# Sentinel for "no valid predecessor"
_INF = float("inf")

# Windows whose best candidate scores below this get synthetic pass-through
# alternatives appended, so a lone spurious peak is never a forced waypoint.
# Genuine matches score 0.5+; spurious peaks in quiet regions sit near the
# lattice floor (0.3).
_WEAK_EVIDENCE = 0.45

# A strong window still gets pass-through synthetics when every strong
# candidate sits further than this from the fingerprint hint: repeated
# soundtrack content scores high at physically wrong offsets, and without a
# bypass route the trellis is forced through it.  Real edits agree with
# their (monotone-filtered) span hint, so they keep their forced jump.
_HINT_CONTRA = 3.0

# Excursion runs up to this many windows with agreeing flanks are evidence
# noise, not edits: a real edit moves the offset permanently, so its flanks
# cannot agree.  Sized to cover a wrong-peak run plus the "shoulder" points
# the DP uses to ramp into it.  (Longer excursions → segment vetting.)
_MAX_SPUR_RUN = 6
_SPUR_FLANK = 5  # points per side for the robust flank estimate

# Per-window pull toward the fingerprint hint.  Sized to be decisive only
# where correlation evidence is absent: max penalty (weight × clamp = 0.2)
# is well under a genuine candidate's score (0.3–1.0), but inside an
# evidence gap it pins offset jumps to the fingerprint-located boundary
# instead of letting them drift to wherever the gap happens to end.
_HINT_WEIGHT = 0.02
_HINT_CLAMP = 10.0


def decode_warp_path(
    lattice: list[CandidateWindow],
    *,
    lambda_jump: float = 2.0,
    lambda_curve: float = 5.0,
    lambda_speech: float = 0.3,
    drift_hint_ppm: float | None = None,
    offset_hint: float | None = None,
    offset_hint_fn: Callable[[float], float] | None = None,
) -> list[WarpPoint]:
    """Run Viterbi over the candidate lattice and return the decoded path.

    Parameters
    ----------
    lattice : list[CandidateWindow]
        One entry per analysis window, each with 0..K offset candidates.
    lambda_jump : float
        Penalty per second of offset deviation from expected drift.
    lambda_curve : float
        Penalty per second of curvature (change in offset velocity).
    lambda_speech : float
        Bonus weight multiplied by speech_score * match_quality.
    drift_hint_ppm : float | None
        If available, the expected drift rate (from earlier pipeline stage).
        Used to compute expected offset change between windows so the jump
        penalty targets *deviations from drift*, not absolute jumps.
    offset_hint : float | None
        If available, a rough expected offset.  Biases the first window
        toward candidates near this value.
    offset_hint_fn : Callable | None
        Per-time expected offset (e.g. from fingerprint spans).  Empty
        windows get a synthetic candidate at this offset too, so quiet
        stretches follow the fingerprint instead of blindly carrying the
        previous window's best guess.

    Returns
    -------
    tuple[list[WarpPoint], float]
        One point per lattice window (including synthetic fill-ins for
        empty windows), and the total path cost.
    """
    if not lattice:
        return [], 0.0

    # ── Prepare the lattice: inject synthetic pass-through candidates ───
    # For windows with no candidates, create one that inherits the median
    # offset of the nearest non-empty window.
    prepared = _prepare_lattice(lattice, offset_hint, offset_hint_fn)

    N = len(prepared)  # number of time steps
    Ks = [len(w.candidates) for w in prepared]

    if all(k == 0 for k in Ks):
        log.warning("No candidates in any window — returning empty warp path")
        return [], 0.0

    # Drift expectation: offset change per second
    drift_per_sec = (drift_hint_ppm or 0.0) * 1e-6

    # Per-window expected offsets from the fingerprint (None when absent)
    hint_at: list[float | None] = [
        offset_hint_fn(w.source_center) if offset_hint_fn is not None else None
        for w in prepared
    ]

    def _reward(t: int, cand) -> float:
        r = -cand.score - lambda_speech * prepared[t].speech_score * cand.score
        if hint_at[t] is not None:
            r += _HINT_WEIGHT * min(abs(cand.offset_sec - hint_at[t]), _HINT_CLAMP)
        return r

    # ── Second-order Viterbi over (previous, current) candidate pairs ──
    # The curvature term compares this transition's offset delta with the
    # previous transition's, so the Markov state must be the candidate PAIR
    # (k_{t-1}, k_t).  A first-order DP that stores one best delta per node
    # merges prefixes whose deltas differ and can return a suboptimal path
    # (verified against brute force on evidence-gap trellises).
    offsets_by_t = [[c.offset_sec for c in w.candidates] for w in prepared]

    # t = 0: cost per single candidate
    cost0: list[float] = []
    for k, cand in enumerate(prepared[0].candidates):
        local = _reward(0, cand)
        if offset_hint is not None:
            local += 0.3 * abs(cand.offset_sec - offset_hint)
        cost0.append(local)

    if N == 1:
        best_k = int(np.argmin(cost0))
        path_indices = [best_k]
        total_cost = cost0[best_k]
    else:
        # t = 1: pair states (j at t-1, k at t); no curvature yet
        dt = prepared[1].source_center - prepared[0].source_center
        expected = drift_per_sec * dt
        pair_cost: list[list[float]] = [
            [
                cost0[j]
                + lambda_jump * abs((offsets_by_t[1][k] - offsets_by_t[0][j]) - expected)
                + _reward(1, prepared[1].candidates[k])
                for k in range(Ks[1])
            ]
            for j in range(Ks[0])
        ]
        # back2[t][j][k] = best predecessor candidate i at t-2 for state (j, k)
        back2: list[list[list[int]]] = [[[-1] * Ks[1] for _ in range(Ks[0])]]

        for t in range(2, N):
            K_pp, K_p, K_c = Ks[t - 2], Ks[t - 1], Ks[t]
            dt = prepared[t].source_center - prepared[t - 1].source_center
            expected = drift_per_sec * dt
            off_pp, off_p, off_c = offsets_by_t[t - 2], offsets_by_t[t - 1], offsets_by_t[t]

            new_cost = [[_INF] * K_c for _ in range(K_p)]
            new_back = [[0] * K_c for _ in range(K_p)]
            rewards = [_reward(t, prepared[t].candidates[k]) for k in range(K_c)]

            for j in range(K_p):
                col = pair_cost  # cost of states (i, j) is col[i][j]
                for k in range(K_c):
                    delta = off_c[k] - off_p[j]
                    base = lambda_jump * abs(delta - expected) + rewards[k]
                    best_total = _INF
                    best_i = 0
                    for i in range(K_pp):
                        prev_d = off_p[j] - off_pp[i]
                        total = col[i][j] + base + lambda_curve * abs(delta - prev_d)
                        if total < best_total:
                            best_total = total
                            best_i = i
                    new_cost[j][k] = best_total
                    new_back[j][k] = best_i

            pair_cost = new_cost
            back2.append(new_back)

        # ── Backtrack over pair states ─────────────────────────────────
        best_j, best_k, total_cost = 0, 0, _INF
        for j in range(len(pair_cost)):
            for k in range(len(pair_cost[j])):
                if pair_cost[j][k] < total_cost:
                    total_cost = pair_cost[j][k]
                    best_j, best_k = j, k

        path_indices = [0] * N
        path_indices[N - 1] = best_k
        path_indices[N - 2] = best_j
        for t in range(N - 1, 1, -1):
            i = back2[t - 1][path_indices[t - 1]][path_indices[t]]
            path_indices[t - 2] = i

    # ── Spur suppression ──────────────────────────────────────────────
    # Runs of up to _MAX_SPUR_RUN points jumping >2 s away from flanks that
    # agree with each other are evidence noise, not an edit — a real edit
    # moves the offset permanently, so its flanks cannot agree.  Flanks are
    # medians over _SPUR_FLANK points so the DP's ramp-in "shoulders" don't
    # hide the excursion.  Snap the run onto the flanks at low confidence.
    path_offsets = [
        prepared[t].candidates[path_indices[t]].offset_sec for t in range(N)
    ]
    spur = [False] * N
    t = 1
    while t < N - 1:
        left_flank = float(np.median(path_offsets[max(0, t - _SPUR_FLANK):t]))
        run_end = t
        while (run_end < N and run_end - t < _MAX_SPUR_RUN
               and abs(path_offsets[run_end] - left_flank) > 2.0):
            run_end += 1
        if run_end > t and run_end < N:
            right_flank = float(np.median(
                path_offsets[run_end:min(N, run_end + _SPUR_FLANK)]))
            if abs(right_flank - left_flank) <= 0.5:
                fill = 0.5 * (left_flank + right_flank)
                for i in range(t, run_end):
                    path_offsets[i] = fill
                    spur[i] = True
                t = run_end
        t += 1
    n_spurs = sum(spur)
    if n_spurs:
        log.info("Suppressed %d offset spur point(s) in the decoded path", n_spurs)

    # ── Convert to WarpPoint list ─────────────────────────────────────
    points: list[WarpPoint] = []
    for t in range(N):
        w = prepared[t]
        k = path_indices[t]
        cand = w.candidates[k]

        source_time = w.source_center
        target_time = source_time + path_offsets[t]
        confidence = 0.05 if spur[t] else cand.score * (0.5 + 0.5 * w.speech_score)

        points.append(WarpPoint(
            source_time=source_time,
            target_time=target_time,
            confidence=confidence,
        ))

    mean_conf = float(np.mean([p.confidence for p in points])) if points else 0.0
    log.info(
        "Warp decoder: %d points, path cost=%.2f, mean conf=%.3f",
        len(points), total_cost, mean_conf,
    )
    return points, total_cost


def _prepare_lattice(
    lattice: list[CandidateWindow],
    offset_hint: float | None,
    offset_hint_fn: Callable[[float], float] | None = None,
) -> list[CandidateWindow]:
    """Inject synthetic pass-through candidates so weak spots are bypassable.

    Empty windows get low-confidence candidates carrying the best offset of
    the nearest non-empty window on *each side* (plus the fingerprint hint
    when available), so the DP can bridge a quiet stretch coherently from
    either end.  Windows whose only evidence is weak get the same synthetic
    alternatives *appended*: without them, a lone spurious peak in a quiet
    region is mandatory — the trellis contains no other route — and it drags
    its offset across every synthetic window that follows.

    Strong windows get the synthetics too when **every** strong candidate
    contradicts the fingerprint hint by more than ``_HINT_CONTRA`` seconds:
    repeated soundtrack content scores high at physically wrong offsets,
    and a confidently wrong peak must never be the only route.  Windows
    whose evidence agrees with the hint stay as-is, so genuine edits keep
    their forced jump.
    """
    from adsync.models import OffsetCandidate

    # Find the median offset across all candidates for a fallback
    all_offsets: list[float] = []
    for w in lattice:
        for c in w.candidates:
            all_offsets.append(c.offset_sec)

    if not all_offsets:
        # Completely empty lattice — use hint or 0
        fallback_offset = offset_hint if offset_hint is not None else 0.0
    else:
        fallback_offset = float(np.median(all_offsets))

    # Best offset of the nearest non-empty window strictly before / after i
    n = len(lattice)
    fwd = np.full(n, fallback_offset)
    last = fallback_offset
    for i, w in enumerate(lattice):
        fwd[i] = last
        if w.candidates:
            last = w.candidates[0].offset_sec
    bwd = np.full(n, fallback_offset)
    nxt = fallback_offset
    for i in range(n - 1, -1, -1):
        bwd[i] = nxt
        if lattice[i].candidates:
            nxt = lattice[i].candidates[0].offset_sec

    # Local consensus for the no-fingerprint case: median best offset over a
    # one-sided neighbourhood each way (excluding self, wide enough that a
    # short wrong-peak run cannot dominate its own reference).
    best_off = np.full(n, np.nan)
    for i, w in enumerate(lattice):
        if w.candidates:
            best_off[i] = w.candidates[0].offset_sec
    _HOOD = 7

    def _side_medians(i: int) -> list[float]:
        meds: list[float] = []
        for lo, hi in ((max(0, i - _HOOD), i), (i + 1, min(n, i + 1 + _HOOD))):
            vals = best_off[lo:hi]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                meds.append(float(np.median(vals)))
        return meds

    prepared: list[CandidateWindow] = []
    for i, w in enumerate(lattice):
        strong = bool(w.candidates) and w.candidates[0].score >= _WEAK_EVIDENCE
        contradicted = False
        if strong:
            strong_offs = [c.offset_sec for c in w.candidates if c.score >= _WEAK_EVIDENCE]
            if offset_hint_fn is not None:
                # Fingerprint says this window belongs elsewhere.
                hint = float(offset_hint_fn(w.source_center))
                contradicted = all(abs(o - hint) > _HINT_CONTRA for o in strong_offs)
            else:
                # No fingerprint: a window at odds with the consensus on
                # *both* sides is a spur candidate; a real edit's windows
                # always agree with one side.
                refs = _side_medians(i)
                contradicted = bool(refs) and all(
                    abs(o - r) > _HINT_CONTRA for o in strong_offs for r in refs
                )
        if strong and not contradicted:
            prepared.append(w)
            continue

        offsets = [float(fwd[i]), float(bwd[i])]
        if offset_hint_fn is not None:
            offsets.append(float(offset_hint_fn(w.source_center)))
        existing = [c.offset_sec for c in w.candidates]
        unique: list[float] = []
        for o in offsets:
            if all(abs(o - u) > 0.05 for u in unique) and all(abs(o - e) > 0.05 for e in existing):
                unique.append(o)
        synthetic = [
            OffsetCandidate(
                offset_sec=o,
                score=0.1,  # low confidence
                peak_sharpness=1.0,
                peak_ratio=1.0,
            )
            for o in unique
        ]
        prepared.append(CandidateWindow(
            source_center=w.source_center,
            candidates=list(w.candidates) + synthetic,
            speech_score=w.speech_score,
            energy=w.energy,
        ))

    return prepared
