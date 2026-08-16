"""Spot-check: measure the actual lag between the AD track and film audio
at given video timestamps in a muxed AD.mkv, by direct cross-correlation.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from scipy.signal import fftconvolve

from adsync.features.load import load_wav

SR = 16000
WIN = 10.0      # AD window length (s)
SEARCH = 25.0   # +/- search in film audio (s)

mkv = Path(sys.argv[1])
centers = [float(a) for a in sys.argv[2:]]

with tempfile.TemporaryDirectory() as td:
    film_p = Path(td) / "film.wav"
    ad_p = Path(td) / "ad.wav"
    for stream, out in (("a:0", film_p), ("a:1", ad_p)):
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(mkv),
             "-map", f"0:{stream}", "-ac", "1", "-ar", str(SR), str(out)],
            check=True,
        )
    film, _ = load_wav(film_p, sr=SR)
    ad, _ = load_wav(ad_p, sr=SR)

for c in centers:
    a0 = int((c - WIN / 2) * SR)
    a1 = int((c + WIN / 2) * SR)
    f0 = max(0, int((c - WIN / 2 - SEARCH) * SR))
    f1 = min(len(film), int((c + WIN / 2 + SEARCH) * SR))
    seg = ad[a0:a1].astype(np.float64)
    seg -= seg.mean()
    reg = film[f0:f1].astype(np.float64)
    reg -= reg.mean()
    corr = fftconvolve(reg, seg[::-1], mode="valid")
    seg_e = np.sqrt(np.sum(seg ** 2))
    cs = np.concatenate(([0.0], np.cumsum(reg ** 2)))
    n = len(seg)
    reg_e = np.sqrt(np.maximum(cs[n:] - cs[:-n], 1e-20))[: len(corr)]
    ncorr = corr / (seg_e * reg_e)
    best = int(np.argmax(ncorr))
    lag = (f0 + best - a0) / SR
    # Correlation at (and near) lag zero — is the film bed present under the
    # narration at the placed position?
    zero_idx = a0 - f0
    z0 = max(0, zero_idx - int(0.05 * SR))
    z1 = min(len(ncorr), zero_idx + int(0.05 * SR))
    corr_zero = float(np.max(ncorr[z0:z1])) if z1 > z0 else float("nan")
    print(
        f"video t={c:7.1f}s  best lag {lag * 1000:+8.1f} ms (corr {ncorr[best]:.3f})   "
        f"corr at lag 0: {corr_zero:.3f}"
    )
