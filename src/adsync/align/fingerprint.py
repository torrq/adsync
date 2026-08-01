"""Audio landmark fingerprinting — global, radius-free alignment evidence.

Shazam-style constellation hashing: spectral peaks are paired into
(freq1, Δfreq, Δtime) landmark hashes, hashed positions in the AD are joined
against the video's hash table, and the matched (t_ad, t_video) pairs are
histogrammed into time-localized *offset spans* — "this stretch of the AD
sits at this offset in the video".

This answers alignment questions the windowed correlator cannot ask:
- offsets of any size, with no search radius at all
- edit points, located to ~a bucket width, before any DP runs
- **bad spots**: AD regions whose hashes match nowhere in the video
  (different edits, missing scenes, wrong source) — surfaced immediately

Landmarks are hashed from bed content (music/effects) and survive the
narration overlaid on top of it, level shifts, EQ, and lossy re-encodes.
All numpy; a 2 h pair fingerprints and matches in a few seconds.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter

log = logging.getLogger("adsync")

# ── Spectrogram / peak parameters ────────────────────────────────────────────
_N_FFT = 1024          # 64 ms window at 16 kHz → 15.6 Hz bins
_HOP = 256             # 16 ms hop → landmark time resolution
_FMIN_HZ = 100.0       # bed content below narration sibilance
_FMAX_HZ = 5000.0
_PEAK_NEIGHBORHOOD = (12, 9)   # (freq bins ~190 Hz, time frames ~144 ms)
_PEAKS_PER_SEC = 35            # density cap, magnitude-ranked per second
_FLOOR_PERCENTILE = 70.0       # peaks must clear this global dB percentile

# ── Landmark pairing ─────────────────────────────────────────────────────────
_PAIR_DT_MIN = 3       # frames (48 ms)
_PAIR_DT_MAX = 48      # frames (768 ms)
_PAIR_DF_MAX = 63      # bins (~980 Hz)
_PAIR_SHIFTS = 24      # successor candidates considered per anchor peak

# ── Matching / span extraction ───────────────────────────────────────────────
_MAX_HASH_OCCURRENCE = 100     # drop hashes this common in the video (hum, drones)
_BUCKET_SEC = 10.0             # AD-time bucket for offset localization
_OFFSET_RES = 0.05             # offset histogram resolution (seconds)
_MIN_BUCKET_MATCHES = 4        # dominant-offset support needed per bucket
_SPAN_MERGE_TOL = 0.25         # adjacent buckets merge if offsets agree this closely
_MIN_SPAN_SEC = 20.0           # ignore spans shorter than this
_MIN_SPAN_DENSITY = 0.8        # matches/sec for a span to count as "strong"
_INLIER_TOL = 0.75             # match kept as anchor evidence if this close to its bucket offset
_CHAIN_TOL = 0.5               # video-time regression allowed between chained buckets


@dataclass
class FingerprintSpan:
    """A stretch of the AD that sits at one offset in the video."""
    ad_start: float
    ad_end: float
    offset: float          # video_time − ad_time, positive = AD shifts later
    matches: int

    @property
    def density(self) -> float:
        dur = max(self.ad_end - self.ad_start, 1e-9)
        return self.matches / dur


@dataclass
class FingerprintResult:
    spans: list[FingerprintSpan] = field(default_factory=list)
    unmatched: list[tuple[float, float]] = field(default_factory=list)
    # (ad_start, ad_end, offset) ranges whose majority vote violated monotone
    # playback — repeated soundtrack content matched at impossible positions.
    dropped_ranges: list[tuple[float, float, float]] = field(default_factory=list)
    n_matches: int = 0
    # Inlier match pairs (sorted by AD time): raw evidence dense enough to
    # anchor alignment windows directly when correlation starves (remixed
    # soundtracks).  None when matching produced no supported spans.
    match_t_ad: NDArray[np.float32] | None = None
    match_t_vid: NDArray[np.float32] | None = None

    @property
    def strong(self) -> bool:
        """Enough evidence to steer the aligner (vs. merely inform the report).

        Judged on the whole: solid total match count, and most of the covered
        time backed by dense spans.  A single thin span in a quiet stretch
        must not disable steering — its neighbours carry the structure.
        """
        if not self.spans:
            return False
        covered = sum(s.ad_end - s.ad_start for s in self.spans)
        dense = sum(s.ad_end - s.ad_start for s in self.spans if s.density >= _MIN_SPAN_DENSITY)
        total_matches = sum(s.matches for s in self.spans)
        return total_matches >= 500 and covered > 0 and dense >= 0.6 * covered

    def offset_at(self, ad_time: float) -> float | None:
        """Offset of the span covering *ad_time*, or of the nearest span."""
        if not self.spans:
            return None
        for s in self.spans:
            if s.ad_start <= ad_time <= s.ad_end:
                return s.offset
        nearest = min(
            self.spans,
            key=lambda s: min(abs(ad_time - s.ad_start), abs(ad_time - s.ad_end)),
        )
        return nearest.offset


def fingerprint_align(
    y_vid: NDArray[np.floating],
    y_ad: NDArray[np.floating],
    sr: int,
) -> FingerprintResult:
    """Fingerprint both tracks, match hashes, and localize offset spans."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_vid = pool.submit(_landmarks, y_vid, sr)
        f_ad = pool.submit(_landmarks, y_ad, sr)
        vid_hashes, vid_times = f_vid.result()
        ad_hashes, ad_times = f_ad.result()

    t_ad, t_vid = _match(vid_hashes, vid_times, ad_hashes, ad_times)
    ad_duration = len(y_ad) / sr
    result = _offset_spans(t_ad, t_vid, ad_duration)

    if result.spans:
        summary = "; ".join(
            f"[{s.ad_start:.0f}–{s.ad_end:.0f}s]→{s.offset:+.2f}s ({s.matches} matches)"
            for s in result.spans[:6]
        )
        log.info(
            "Fingerprint: %d matches, %d span(s): %s%s",
            result.n_matches, len(result.spans), summary,
            " …" if len(result.spans) > 6 else "",
        )
    else:
        log.info("Fingerprint: %d matches — no reliable spans", result.n_matches)
    for lo, hi in result.unmatched:
        log.warning("Fingerprint: no matches for AD %.0f–%.0f s", lo, hi)
    return result


# ── Internals ────────────────────────────────────────────────────────────────


def _landmarks(
    y: NDArray[np.floating],
    sr: int,
) -> tuple[NDArray[np.uint32], NDArray[np.float32]]:
    """Constellation → packed landmark hashes with anchor times (seconds)."""
    S = np.abs(librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP))
    S_db = 20.0 * np.log10(S + 1e-10)
    del S

    bin_hz = sr / _N_FFT
    lo_bin = max(1, int(_FMIN_HZ / bin_hz))
    hi_bin = min(S_db.shape[0], int(_FMAX_HZ / bin_hz))
    S_db = S_db[lo_bin:hi_bin]

    local_max = S_db == maximum_filter(S_db, size=(2 * _PEAK_NEIGHBORHOOD[0] + 1,
                                                   2 * _PEAK_NEIGHBORHOOD[1] + 1))
    floor = np.percentile(S_db, _FLOOR_PERCENTILE)
    freq_idx, time_idx = np.nonzero(local_max & (S_db > floor))
    if len(time_idx) == 0:
        return np.empty(0, np.uint32), np.empty(0, np.float32)
    mags = S_db[freq_idx, time_idx]

    # Cap density: keep the strongest peaks per one-second bucket
    frames_per_sec = sr / _HOP
    bucket = (time_idx / frames_per_sec).astype(np.int64)
    order = np.lexsort((-mags, bucket))     # by bucket, magnitude descending
    rank_in_bucket = _running_rank(bucket[order])
    keep = order[rank_in_bucket < _PEAKS_PER_SEC]
    keep = keep[np.argsort(time_idx[keep], kind="stable")]
    t = time_idx[keep]
    f = freq_idx[keep]

    # Pair each anchor with nearby later peaks via index shifts
    hashes: list[NDArray[np.uint32]] = []
    times: list[NDArray[np.float32]] = []
    n = len(t)
    for k in range(1, min(_PAIR_SHIFTS, n - 1) + 1):
        dt = t[k:] - t[:-k]
        df = f[k:].astype(np.int64) - f[:-k].astype(np.int64)
        ok = (dt >= _PAIR_DT_MIN) & (dt <= _PAIR_DT_MAX) & (np.abs(df) <= _PAIR_DF_MAX)
        if not np.any(ok):
            continue
        f1 = f[:-k][ok].astype(np.uint32)
        h = (f1 << 13) | ((df[ok] + _PAIR_DF_MAX).astype(np.uint32) << 6) | dt[ok].astype(np.uint32)
        hashes.append(h)
        times.append((t[:-k][ok] / frames_per_sec).astype(np.float32))

    if not hashes:
        return np.empty(0, np.uint32), np.empty(0, np.float32)
    return np.concatenate(hashes), np.concatenate(times)


def _running_rank(sorted_groups: NDArray) -> NDArray[np.int64]:
    """Rank of each element within its (contiguous) group."""
    n = len(sorted_groups)
    if n == 0:
        return np.empty(0, np.int64)
    new_group = np.empty(n, dtype=bool)
    new_group[0] = True
    np.not_equal(sorted_groups[1:], sorted_groups[:-1], out=new_group[1:])
    idx = np.arange(n)
    group_start = np.maximum.accumulate(np.where(new_group, idx, 0))
    return idx - group_start


def _match(
    vid_hashes: NDArray[np.uint32],
    vid_times: NDArray[np.float32],
    ad_hashes: NDArray[np.uint32],
    ad_times: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Join AD hashes against the video hash table → (t_ad, t_vid) pairs."""
    if len(vid_hashes) == 0 or len(ad_hashes) == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)

    order = np.argsort(vid_hashes, kind="stable")
    vh = vid_hashes[order]
    vt = vid_times[order]

    lo = np.searchsorted(vh, ad_hashes, side="left")
    hi = np.searchsorted(vh, ad_hashes, side="right")
    counts = hi - lo
    keep = (counts > 0) & (counts <= _MAX_HASH_OCCURRENCE)
    if not np.any(keep):
        return np.empty(0, np.float32), np.empty(0, np.float32)

    c = counts[keep]
    starts = lo[keep]
    total = int(c.sum())
    # Ragged gather: absolute indices lo[i]..hi[i] for every kept AD hash
    flat = np.repeat(starts, c) + (np.arange(total) - np.repeat(np.cumsum(c) - c, c))
    t_vid = vt[flat]
    t_ad = np.repeat(ad_times[keep], c)
    return t_ad, t_vid


def _drop_nonmonotone_buckets(
    bucket_offset: NDArray[np.float64],
    bucket_matches: NDArray[np.int64],
) -> list[tuple[float, float, float]]:
    """Keep the maximum-weight monotone chain of supported buckets, in place.

    Nodes are supported buckets; a node maps its AD-time centre to
    ``centre + offset`` in the video.  A valid chain never lets video time
    regress by more than ``_CHAIN_TOL`` (bucket offsets carry ~50 ms jitter;
    genuine AD-only content makes video time *pause*, never rewind).  The
    kept chain maximises total match support, so isolated repeated-content
    votes lose to the surrounding structure regardless of their local count.
    """
    nodes = np.nonzero(~np.isnan(bucket_offset))[0]
    if len(nodes) <= 1:
        return []

    t_vid = (nodes + 0.5) * _BUCKET_SEC + bucket_offset[nodes]
    weights = bucket_matches[nodes].astype(np.float64)

    n = len(nodes)
    best = np.empty(n, dtype=np.float64)
    prev = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        compatible = np.nonzero(t_vid[:i] <= t_vid[i] + _CHAIN_TOL)[0]
        if len(compatible):
            j = compatible[np.argmax(best[compatible])]
            best[i] = best[j] + weights[i]
            prev[i] = j
        else:
            best[i] = weights[i]

    keep = np.zeros(n, dtype=bool)
    i = int(np.argmax(best))
    while i >= 0:
        keep[i] = True
        i = int(prev[i])

    dropped = nodes[~keep]
    if not len(dropped):
        return []
    ranges: list[tuple[int, int, float]] = []
    for b in dropped:
        if ranges and b == ranges[-1][1] + 1 and abs(bucket_offset[b] - ranges[-1][2]) < 1.0:
            ranges[-1] = (ranges[-1][0], int(b), ranges[-1][2])
        else:
            ranges.append((int(b), int(b), float(bucket_offset[b])))
    summary = "; ".join(
        f"[{lo * _BUCKET_SEC:.0f}–{(hi + 1) * _BUCKET_SEC:.0f}s]→{off:+.2f}s"
        for lo, hi, off in ranges[:6]
    )
    log.info(
        "Fingerprint: dropped %d bucket(s) voting against monotone playback "
        "(repeated content): %s%s",
        len(dropped), summary, " …" if len(ranges) > 6 else "",
    )
    bucket_offset[dropped] = np.nan
    bucket_matches[dropped] = 0
    return [
        (lo * _BUCKET_SEC, (hi + 1) * _BUCKET_SEC, off) for lo, hi, off in ranges
    ]


def _offset_spans(
    t_ad: NDArray[np.float32],
    t_vid: NDArray[np.float32],
    ad_duration: float,
) -> FingerprintResult:
    """Histogram matches into per-bucket dominant offsets, merge into spans."""
    result = FingerprintResult(n_matches=len(t_ad))
    n_buckets = max(1, int(np.ceil(ad_duration / _BUCKET_SEC)))
    if len(t_ad) == 0:
        result.unmatched = [(0.0, ad_duration)]
        return result

    offsets = (t_vid - t_ad).astype(np.float64)
    buckets = np.clip((t_ad / _BUCKET_SEC).astype(np.int64), 0, n_buckets - 1)
    obins = np.round(offsets / _OFFSET_RES).astype(np.int64)

    # Count (bucket, offset-bin) pairs; include ±1 bin so quantization-edge
    # matches vote together.  Offset bins are biased non-negative before
    # packing so negative offsets can't underflow into a neighbouring
    # bucket's key space.
    _BIAS = 1 << 21          # ±~29 h at 50 ms resolution
    _OBIN_SPACE = 1 << 22
    obins_b = np.clip(obins + _BIAS, 0, _OBIN_SPACE - 1)
    keys = np.concatenate([
        buckets * _OBIN_SPACE + np.clip(obins_b + d, 0, _OBIN_SPACE - 1)
        for d in (-1, 0, 1)
    ])
    uniq, counts = np.unique(keys, return_counts=True)
    u_bucket = uniq // _OBIN_SPACE
    # Dominant offset bin per bucket
    order = np.lexsort((-counts, u_bucket))
    first = _running_rank(u_bucket[order]) == 0
    dom = order[first]
    dom_bucket = u_bucket[dom]
    dom_count = counts[dom]
    dom_obin = (uniq[dom] % _OBIN_SPACE) - _BIAS

    bucket_offset = np.full(n_buckets, np.nan)
    bucket_matches = np.zeros(n_buckets, dtype=np.int64)
    good = dom_count >= _MIN_BUCKET_MATCHES
    bucket_offset[dom_bucket[good]] = dom_obin[good] * _OFFSET_RES
    bucket_matches[dom_bucket[good]] = dom_count[good]

    # Refine each supported bucket's offset with the median of its matches
    # near the dominant bin (the histogram bin is only ~50 ms wide).
    for b in np.nonzero(~np.isnan(bucket_offset))[0]:
        sel = (buckets == b) & (np.abs(offsets - bucket_offset[b]) <= 2 * _OFFSET_RES)
        if np.any(sel):
            bucket_offset[b] = float(np.median(offsets[sel]))

    # A playback map must be (near-)monotone in video time: repeated
    # soundtrack content can win a bucket's majority vote with a physically
    # impossible offset (the credits song also plays 15 minutes earlier).
    # Keep only the maximum-support chain of buckets whose video times
    # advance; the rest revert to unmatched.
    result.dropped_ranges = _drop_nonmonotone_buckets(bucket_offset, bucket_matches)

    # Merge adjacent supported buckets into spans
    spans: list[FingerprintSpan] = []
    cur: FingerprintSpan | None = None
    for b in range(n_buckets):
        if np.isnan(bucket_offset[b]):
            cur = None
            continue
        t0, t1 = b * _BUCKET_SEC, min((b + 1) * _BUCKET_SEC, ad_duration)
        if cur is not None and abs(bucket_offset[b] - cur.offset) <= _SPAN_MERGE_TOL:
            cur.ad_end = t1
            cur.matches += int(bucket_matches[b])
            cur.offset = cur.offset + (bucket_offset[b] - cur.offset) * 0.25
        else:
            cur = FingerprintSpan(ad_start=t0, ad_end=t1,
                                  offset=float(bucket_offset[b]),
                                  matches=int(bucket_matches[b]))
            spans.append(cur)

    spans = [s for s in spans if s.ad_end - s.ad_start >= _MIN_SPAN_SEC]
    result.spans = spans

    # Unmatched regions: runs of buckets not covered by any kept span
    covered = np.zeros(n_buckets, dtype=bool)
    for s in spans:
        covered[int(s.ad_start / _BUCKET_SEC): int(np.ceil(s.ad_end / _BUCKET_SEC))] = True

    # Keep the inlier match pairs: matches inside a covered bucket that agree
    # with that bucket's refined offset.  These are the anchor-grade evidence
    # consumers can window into when correlation has nothing to offer.
    if spans:
        bo = bucket_offset[buckets]
        inlier = covered[buckets] & ~np.isnan(bo) & (np.abs(offsets - bo) <= _INLIER_TOL)
        if np.any(inlier):
            order = np.argsort(t_ad[inlier], kind="stable")
            result.match_t_ad = t_ad[inlier][order].astype(np.float32)
            result.match_t_vid = t_vid[inlier][order].astype(np.float32)
    run_start: int | None = None
    for b in range(n_buckets + 1):
        gap = b < n_buckets and not covered[b]
        if gap and run_start is None:
            run_start = b
        elif not gap and run_start is not None:
            lo_t, hi_t = run_start * _BUCKET_SEC, min(b * _BUCKET_SEC, ad_duration)
            if hi_t - lo_t >= 2 * _BUCKET_SEC:
                result.unmatched.append((lo_t, hi_t))
            run_start = None

    return result
