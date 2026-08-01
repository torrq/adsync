"""Output-side QC: fingerprint the muxed AD track against the film audio.

Extracts both audio tracks from a finished AD.mkv and landmark-matches them.
If the render is correct, every span sits at offset ~0.000 s.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from adsync.align.fingerprint import fingerprint_align
from adsync.features.load import load_wav

mkv = Path(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    film = Path(td) / "film.wav"
    ad = Path(td) / "ad.wav"
    for stream, out in (("a:0", film), ("a:1", ad)):
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(mkv),
             "-map", f"0:{stream}", "-ac", "1", "-ar", "16000", str(out)],
            check=True,
        )
    y_film, sr = load_wav(film, sr=16000)
    y_ad, _ = load_wav(ad, sr=16000)
    fp = fingerprint_align(y_film, y_ad, sr)

print()
print(f"spans: {len(fp.spans)}   matches: {fp.n_matches}")
worst = 0.0
for s in fp.spans:
    worst = max(worst, abs(s.offset))
    flag = "  <-- OFF" if abs(s.offset) > 0.15 else ""
    print(f"  ad {s.ad_start:7.0f}-{s.ad_end:7.0f}  offset {s.offset:+8.3f} s  ({s.matches} m){flag}")
print(f"worst span offset: {worst * 1000:.0f} ms")
for lo, hi in fp.unmatched:
    print(f"  unmatched {lo:7.0f}-{hi:7.0f}  ({(hi - lo):.0f} s)")
