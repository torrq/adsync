"""Small math / DSP utilities that don't belong in a larger module."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rms_normalize(y: NDArray[np.floating], target_db: float = -20.0) -> NDArray[np.floating]:
    """Normalize *y* so its RMS equals *target_db* dBFS."""
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-10:
        return y
    target_linear = 10 ** (target_db / 20.0)
    return y * (target_linear / rms)


def crossfade(a: NDArray[np.floating], b: NDArray[np.floating], n_samples: int) -> NDArray[np.floating]:
    """Overlap-add crossfade of length *n_samples* between *a* (end) and *b* (start)."""
    if n_samples <= 0:
        return np.concatenate([a, b])
    n_samples = min(n_samples, len(a), len(b))
    fade_out = np.linspace(1.0, 0.0, n_samples)
    fade_in = np.linspace(0.0, 1.0, n_samples)

    result = np.empty(len(a) + len(b) - n_samples, dtype=a.dtype)
    result[: len(a) - n_samples] = a[: -n_samples]
    result[len(a) - n_samples : len(a)] = a[-n_samples:] * fade_out + b[:n_samples] * fade_in
    result[len(a):] = b[n_samples:]
    return result


def catmull_rom_interp(
    src: NDArray[np.floating],
    pos: NDArray[np.float64],
    out: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Resample *src* at fractional sample positions *pos* into *out*.

    Catmull-Rom cubic over the last axis; edge samples replicate.  Linear
    interpolation loses ~3 dB at sr/4 for a constant half-sample offset —
    an audible dulling on a full film mix — while the cubic stays within
    ~1 dB there.  *out* must have *src*'s leading shape with last axis
    ``len(pos)``.
    """
    n = src.shape[-1]
    pos = np.clip(pos, 0.0, n - 1.0)
    idx = pos.astype(np.int64)
    np.clip(idx, 0, n - 2, out=idx)
    u = (pos - idx).astype(np.float32)
    p0 = src[..., np.maximum(idx - 1, 0)]
    p1 = src[..., idx]
    p2 = src[..., idx + 1]
    p3 = src[..., np.minimum(idx + 2, n - 1)]
    u2 = u * u
    u3 = u2 * u
    out[...] = 0.5 * (
        2.0 * p1
        + (p2 - p0) * u
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
        + (3.0 * (p1 - p2) + (p3 - p0)) * u3
    )
    return out


def linear_fit_offset(times: NDArray[np.floating], offsets: NDArray[np.floating]) -> tuple[float, float]:
    """Return (intercept, slope) for offset-vs-time via least-squares."""
    if len(times) < 2:
        return (float(offsets[0]) if len(offsets) else 0.0, 0.0)
    coeffs = np.polyfit(times, offsets, 1)  # slope, intercept
    return float(coeffs[1]), float(coeffs[0])
