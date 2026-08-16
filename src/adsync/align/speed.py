"""Global speed-shift detection — the PAL problem, solved once.

Audio description tracks captured from PAL DVDs run 25/24 fast with raised
pitch; vintage transfers often stack a few extra tenths of a percent of
transfer error on top.  Anything past ~1% defeats both landmark hashing
(frequency bins shift) and windowed correlation (in-window smear), so the
pair looks like a total mismatch when it is really the right movie at the
wrong speed.

Detection: when the plain fingerprint finds nothing (or a steep offset
ramp), re-fingerprint with the AD resampled at the standard transfer
ratios.  A decisive winner is then *refined*: span offsets are fitted with
a Theil-Sen-style weighted median of pairwise slopes — robust to genuine
edit steps, which a least-squares line would smear into the speed — and
the residual slope folds into the final stretch.  A verification pass at
the refined stretch must come back strong and flat.

Correction is a plain resample (tempo and pitch together): the inverse of
what the PAL transfer did, restoring the original cinema pitch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

from adsync.align.fingerprint import FingerprintResult, FingerprintSpan, fingerprint_align

log = logging.getLogger("adsync")

#: Standard transfer ratios to probe (stretch applied to the AD).
#: 25/24: AD sourced from PAL (plays fast) against film-speed video.
#: 24/25: film-speed AD against a PAL-sourced video.
STANDARD_RATIOS: tuple[float, ...] = (25.0 / 24.0, 24.0 / 25.0)

_MIN_MATCHES = 500          # span-backed matches for a candidate to be real
_DECISIVE_FACTOR = 3.0      # winner must beat baseline quality by this much
_TRIGGER_SLOPE = 0.003      # baseline offset ramp steeper than this = speed error
_VERIFY_SLOPE = 0.001       # residual slope allowed after refinement
_PAIR_MIN_DT = 60.0         # min AD-time separation for a slope pair


@dataclass
class SpeedDetection:
    """A measured global speed error and the evidence behind it."""
    stretch: float              # factor to apply to the AD (>1 = AD was fast)
    base_candidate: float       # standard ratio (or 1.0) that led to it
    slope: float                # residual slope measured at the base candidate
    fp: FingerprintResult       # fingerprint at the refined stretch

    @property
    def percent(self) -> float:
        return (self.stretch - 1.0) * 100.0


def stretch_fraction(stretch: float) -> tuple[int, int]:
    """Rational (up, down) approximation of *stretch* for exact resampling."""
    frac = Fraction(stretch).limit_denominator(100_000)
    return frac.numerator, frac.denominator


def fit_speed_slope(spans: list[FingerprintSpan], *, min_dt: float = _PAIR_MIN_DT) -> float:
    """Residual speed error (offset change per AD second) across spans.

    Weighted median of pairwise slopes between span midpoints (Theil-Sen
    flavour).  Pairs are distance-banded to [min_dt, 4×min_dt]: without the
    upper band, a mid-track edit step sits inside the *majority* of pairs
    and drags the median the way it drags a least-squares line; banded,
    step-straddling pairs are a small minority the median ignores.
    """
    if len(spans) < 2:
        return 0.0
    mids = np.array([(s.ad_start + s.ad_end) / 2.0 for s in spans])
    offs = np.array([s.offset for s in spans])
    wts = np.array([float(s.matches) for s in spans])

    i, j = np.triu_indices(len(spans), k=1)
    dt = mids[j] - mids[i]
    ok = (dt >= min_dt) & (dt <= 4.0 * min_dt)
    if not np.any(ok):
        ok = dt >= min_dt          # sparse spans: fall back to any long pair
        if not np.any(ok):
            return 0.0
    slopes = (offs[j][ok] - offs[i][ok]) / dt[ok]
    weights = np.minimum(wts[i][ok], wts[j][ok])

    order = np.argsort(slopes)
    cum = np.cumsum(weights[order])
    k = int(np.searchsorted(cum, 0.5 * cum[-1]))
    return float(slopes[order][min(k, len(order) - 1)])


def _slope_from_matches(fp: FingerprintResult) -> float | None:
    """Slope from raw inlier match pairs — sees ramps that span merging blends
    away (a slow drift keeps adjacent buckets within merge tolerance, so the
    span list can collapse to one long span with an averaged offset)."""
    if fp.match_t_ad is None or len(fp.match_t_ad) < 200:
        return None
    t = fp.match_t_ad.astype(np.float64)
    o = (fp.match_t_vid - fp.match_t_ad).astype(np.float64)
    if len(t) > 2000:
        idx = np.linspace(0, len(t) - 1, 2000).astype(np.int64)
        t, o = t[idx], o[idx]
    i, j = np.triu_indices(len(t), k=1)
    dt = t[j] - t[i]
    ok = (dt >= 30.0) & (dt <= 180.0)
    if int(ok.sum()) < 50:
        return None
    return float(np.median((o[j][ok] - o[i][ok]) / dt[ok]))


def _residual_slope(fp: FingerprintResult) -> float:
    """Best available residual-slope estimate for a fingerprint result."""
    slope = _slope_from_matches(fp)
    return slope if slope is not None else fit_speed_slope(fp.spans)


def _quality(fp: FingerprintResult) -> int:
    """Span-backed match count when the result is steering-grade, else 0."""
    return sum(s.matches for s in fp.spans) if fp.strong else 0


def _resampled(y_ad: np.ndarray, stretch: float) -> np.ndarray:
    up, down = stretch_fraction(stretch)
    return resample_poly(y_ad, up, down).astype(y_ad.dtype, copy=False)


def detect_speed(
    y_vid: np.ndarray,
    y_ad: np.ndarray,
    sr: int,
    *,
    baseline: FingerprintResult | None = None,
    candidates: tuple[float, ...] = STANDARD_RATIOS,
) -> SpeedDetection | None:
    """Detect and measure a global AD speed shift; None when speed is fine.

    *baseline* is the fingerprint of the unmodified pair (computed here if
    not supplied).  Detection is decisive by construction: a standard ratio
    must produce strong spans where the baseline produced ~none, or the
    baseline itself must show a steep offset ramp.
    """
    if baseline is None:
        baseline = fingerprint_align(y_vid, y_ad, sr)
    base_quality = _quality(baseline)

    # Case 1: fingerprint already locks on — speed is only wrong if the
    # offsets ramp (a shift small enough to survive hashing).
    if base_quality > 0:
        slope = _residual_slope(baseline)
        if abs(slope) < _TRIGGER_SLOPE:
            return None
        base_candidate, base_for_refine = 1.0, slope
    else:
        # Case 2: nothing matched at native speed — probe standard ratios.
        best_ratio, best_fp, best_quality = None, None, 0
        for ratio in candidates:
            fp_r = fingerprint_align(y_vid, _resampled(y_ad, ratio), sr)
            q = _quality(fp_r)
            log.info(
                "Speed probe ×%.6f: %d span-backed matches (%d spans)",
                ratio, q, len(fp_r.spans),
            )
            if q > best_quality:
                best_ratio, best_fp, best_quality = ratio, fp_r, q
        if (best_ratio is None
                or best_quality < _MIN_MATCHES
                or best_quality < _DECISIVE_FACTOR * max(base_quality, 1)):
            return None
        base_candidate = best_ratio
        base_for_refine = _residual_slope(best_fp)

    # Refine: fold the residual span-offset slope into the stretch.  The
    # slope is measured on the already-resampled timeline, so it composes
    # multiplicatively with the candidate ratio.
    stretch = base_candidate * (1.0 + base_for_refine)

    # Verify: the refined stretch must fingerprint strong and flat.
    fp_v = fingerprint_align(y_vid, _resampled(y_ad, stretch), sr)
    slope_v = _residual_slope(fp_v)
    if not fp_v.strong or abs(slope_v) > _VERIFY_SLOPE:
        log.warning(
            "Speed detection unverified (stretch ×%.6f: strong=%s, residual slope %.5f) "
            "— leaving the AD untouched",
            stretch, fp_v.strong, slope_v,
        )
        return None

    log.info(
        "Speed detected: AD runs ×%.6f (%+.2f%%) vs video "
        "(candidate ×%.6f, residual slope %+.5f, verified flat at %+.5f)",
        stretch, (stretch - 1.0) * 100.0, base_candidate, base_for_refine, slope_v,
    )
    return SpeedDetection(
        stretch=stretch,
        base_candidate=base_candidate,
        slope=base_for_refine,
        fp=fp_v,
    )
