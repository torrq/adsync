# Repeated-Content Robustness & describealign-Inspired Upgrades

Date: 2026-08-02
Status: approved (user goal: beat describealign; Sinister confidence must rise above 88.7%)

## Problem

Sinister (2012) synced at 88.7% (medium) with audible damage. Root causes, confirmed
by diagnostic re-run (`Movies/Sinister/sinister.diag.report.json`):

1. **Fingerprint span votes are physically unconstrained.** Each 10 s bucket votes
   independently for its dominant offset. Repeated soundtrack music produced three
   false spans: `[40–60s]→+26.69`, `[3750–3770s]→+19.20`, `[6200–6230s]→−923.92`
   (end-credits song also plays at ~88 min). A playback map must be monotone in
   video time; nothing enforced that.
2. **Decoder bypass hole.** `_prepare_lattice` appends pass-through synthetics only
   when the window's best candidate scores < 0.45 (`_WEAK_EVIDENCE`). A confidently
   wrong repeated-music peak (score ≥ 0.45) leaves the trellis with no escape route:
   forced excursions at 790–794 s and 1810–1814 s (−7.4 s dips), chaos at
   3736–3832 s, −923.9 s for 30 s in the credits.
3. **Fit/render trust every segment.** `fit_warp_function` splits at jumps and fits
   each segment independently (monotonicity only enforced within segments);
   `_render_multi_segment` renders all segments additively. The −924 s segment put
   30 s of AD audio on top of the correct AD at ~88 min.
4. **Confidence is blind to these failures.** Smoothness is measured as the
   *fraction* of smooth transitions (25 disasters / 3184 ≈ 99% smooth); no
   monotonicity or misplacement term. 88.7% for a damaged output.

True structure of the pair (post ×1.045173 speed correction): plateaus at
+44.21 → +44.36 (~23 min) → +46.51 (~43 min) → +46.93 (~82 min) → +46.91, then a
real +70.9 credits tail; video has ~3 min extra credits beyond the AD.

## Design

### F1 — Monotone chain filter for fingerprint spans (`align/fingerprint.py`)

After `_offset_spans` builds spans, keep the maximum-weight subset that forms a
physically valid playback map: spans sorted by `ad_start` must have non-decreasing
video intervals (`b.ad_start + b.offset >= a.ad_end + a.offset − slack`,
slack ≈ 2 s for bucket quantization). Weight = match count. O(n²) DP (n ≤ ~100).
Dropped spans' buckets revert to unmatched; inlier match pairs (`match_t_ad/vid`)
are re-filtered to surviving spans. Log dropped spans explicitly ("span X→+26.69
conflicts with monotone map — likely repeated music; dropped").

Effects: hints/search-centering/radius sizing all become sane. Auto radius for
Sinister drops from ±90 s to ~±40 s.

### F2 — Universal pass-through synthetics (`align/warp_decode.py`)

`_prepare_lattice` appends fwd/bwd/hint synthetics (score 0.1) to **every** window,
not only empty/weak ones. A confident wrong peak then always competes against a
cheap coast route; jump+curvature penalties make short excursions lose while real
edits (sustained, evidence-backed) still win. Vectorize the inner Viterbi loops
(numpy over candidate axes) so the K-growth (5 → ~8) does not slow decode.

### F3 — Segment vetting and bridging (`align/warp_fit.py`)

After `_split_at_discontinuities`:

1. Score each raw segment: duration, summed evidence confidence.
2. Accept segments ≥ min duration (8 s) and ≥ min evidence; then enforce **global**
   video-time monotonicity across accepted segments (weighted LIS on segment video
   intervals, weight = evidence mass) — a segment whose video range regresses
   behind its accepted predecessor is rejected.
3. Rejected/gap AD ranges are **bridged**: extend the flanking accepted segments at
   their boundary offsets, meeting at the midpoint (or at a fingerprint span
   boundary when one falls inside the gap). Bridged ranges get confidence 0.05.
4. `warp_segment_ranges` must tile the AD timeline without overlap so the renderer
   physically cannot double-place audio.

### F4 — Confidence v2 + verification (`align/confidence.py`, report)

- New factors: monotonicity (violations found before vetting reduce score in
  proportion to their AD-time mass), bridged-time fraction, and **fingerprint
  residual QC**: evaluate the final warp at inlier match times; report p50/p95
  residual ms. Keep smoothness/coverage/density factors.
- Report: warp segment count (replaces the misleading "Segments 0"), plain-language
  edit map ("AD 00:00–23:30 plays 44.2 s late; 2.2 s of extra video at 42:52; …"),
  dropped-span notes, residual QC numbers.

### F5 — Multi-band correlation scoring (`align/candidate_lattice.py`)

describealign correlates three coarse frequency bands and multiplies match
probabilities; we adopt the idea properly: split the ~4 kHz correlation signal into
B = 4 log-spaced envelope bands (e.g. 0–300, 300–1200, 1200–2500, 2500–4000 Hz at
the downsampled rate), cross-correlate each band's envelope per window, and combine
per-band normalized scores by weighted geometric mean, down-weighting the
narration-dominant band so windows where the narrator talks over quiet scenes stop
starving. Gate: accuracy harness must hold (median ≤ 2.6 ms, max ≤ 35 ms) and
Sinister confidence/coverage must improve; otherwise ship F1–F4 only.

## Testing

- Unit tests per fix: synthetic span sets with planted repeated-content conflicts
  (F1); trellises with confident wrong peaks (F2); segment lists with excursions,
  overlaps, regressions (F3); confidence fixtures (F4); band-combine scoring (F5).
- `tools/accuracy_harness.py` before/after (regression gate).
- End-to-end: Sinister re-run — expect ~5–6 segments, no excursion beyond ±90 s,
  confidence > 88.7%, and clean audio at 13:10, 30:10, 62–64 min, 88 min, credits.

## Out of scope (recorded for later)

- describealign-style L1 piecewise-linear fit replacing PCHIP (only if residual
  wobble remains after F1–F5).
- describealign's "boost description volume" feature (separate feature request).
- Auto-retry fingerprint at PAL ratios on zero spans is already implemented
  (Step 4.6); no change.
