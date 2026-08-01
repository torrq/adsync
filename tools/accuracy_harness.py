"""Sync-accuracy harness: known edits in, measured placement error out.

Takes a content-matched (video-audio, AD) pair, applies edit scenarios with
exactly known time maps (cuts, insertions, offsets, clock drift), runs the
ADSync analysis pipeline on each, and scores every reported anchor against
ground truth.  The output is the number that backs any "sync accuracy" claim,
and the regression gate for alignment changes.

Usage:
    python tools/accuracy_harness.py --video excerpt.flac --ad ad_base.wav \
        --workdir out/ [--scenarios clean_offset cut20 ...]

The --video and --ad inputs must already be content-aligned (offset ~0), e.g.
a decoded excerpt of a synced pair.  Scenario edits are applied to the AD.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(REPO_SRC))

# Boundary exclusion: anchors whose analysis window straddles an edit point
# see two conflicting truths — skip them when scoring.
EDGE_EXCLUDE_SEC = 6.0


@dataclass
class Scenario:
    name: str
    description: str
    # ffmpeg -filter_complex applied to the base AD ("[0:a]...[out]")
    filter_complex: str
    # variant AD time -> true video time; return None where truth is ambiguous
    truth: Callable[[float], float | None]
    # expect the pipeline to succeed (confidence >= threshold)?
    expect_success: bool = True


def make_scenarios(dur: float) -> list[Scenario]:
    """Edit scenarios over a base pair of *dur* seconds, content-aligned at 0."""
    s: list[Scenario] = []

    s.append(Scenario(
        name="clean",
        description="No edit — measures the harness noise floor",
        filter_complex="[0:a]acopy[out]",
        truth=lambda t: t,
    ))

    s.append(Scenario(
        name="offset5",
        description="AD starts 5 s late (leading silence)",
        filter_complex="[0:a]adelay=5000:all=1[out]",
        truth=lambda t: t - 5.0 if t > 5.0 + 1.0 else None,
    ))

    cut_at, cut_len = dur * 0.33, 20.0
    s.append(Scenario(
        name="cut20",
        description=f"{cut_len:.0f} s removed from AD at {cut_at:.0f} s (ad-break style)",
        filter_complex=(
            f"[0:a]atrim=end={cut_at}[a1];"
            f"[0:a]atrim=start={cut_at + cut_len}[a2];"
            f"[a1][a2]concat=n=2:v=0:a=1[out]"
        ),
        truth=lambda t: t if t < cut_at else t + cut_len,
    ))

    cut2_at, cut2_len = dur * 0.7, 45.0
    s.append(Scenario(
        name="cut45",
        description=f"{cut2_len:.0f} s removed at {cut2_at:.0f} s (bigger than any fixed radius)",
        filter_complex=(
            f"[0:a]atrim=end={cut2_at}[a1];"
            f"[0:a]atrim=start={cut2_at + cut2_len}[a2];"
            f"[a1][a2]concat=n=2:v=0:a=1[out]"
        ),
        truth=lambda t: t if t < cut2_at else t + cut2_len,
    ))

    ca, cl_a = dur * 0.3, 10.0
    cb, cl_b = dur * 0.7, 25.0  # position in the base timeline

    def truth_twocuts(t: float) -> float:
        if t < ca:
            return t
        if t < cb - cl_a:
            return t + cl_a
        return t + cl_a + cl_b

    s.append(Scenario(
        name="twocuts",
        description=f"Two cuts: {cl_a:.0f} s at {ca:.0f} s and {cl_b:.0f} s at {cb:.0f} s",
        filter_complex=(
            f"[0:a]atrim=end={ca}[a1];"
            f"[0:a]atrim=start={ca + cl_a}:end={cb}[a2];"
            f"[0:a]atrim=start={cb + cl_b}[a3];"
            f"[a1][a2][a3]concat=n=3:v=0:a=1[out]"
        ),
        truth=truth_twocuts,
    ))

    ins_at, src_at, ins_len = dur * 0.5, dur * 0.17, 15.0

    def truth_insert(t: float) -> float | None:
        if t < ins_at:
            return t
        if t < ins_at + ins_len:
            return None  # duplicated recap content — either position is valid
        return t - ins_len

    s.append(Scenario(
        name="insert15",
        description=f"{ins_len:.0f} s of earlier content (recap) inserted at {ins_at:.0f} s",
        filter_complex=(
            f"[0:a]atrim=end={ins_at}[a1];"
            f"[0:a]atrim=start={src_at}:end={src_at + ins_len}[rec];"
            f"[0:a]atrim=start={ins_at}[a2];"
            f"[a1][rec][a2]concat=n=3:v=0:a=1[out]"
        ),
        truth=truth_insert,
    ))

    rate = 1.0002  # 200 ppm clock drift (pitch shifts with it, like real clocks)
    s.append(Scenario(
        name="drift200",
        description="200 ppm clock drift (asetrate)",
        filter_complex=f"[0:a]asetrate=44100*{rate},aresample=44100[out]",
        truth=lambda t: t * rate,
    ))

    pal = 25.0 / 24.0  # PAL transfer: AD plays 4.27% fast, pitch up
    s.append(Scenario(
        name="pal425",
        description="PAL-speed AD (25/24 fast, pitch up) — auto speed correction",
        filter_complex=f"[0:a]asetrate=44100*{pal},aresample=44100[out]",
        truth=lambda t: t * pal,
    ))

    late_at, late_len = dur * 0.88, 15.0
    s.append(Scenario(
        name="latecut",
        description=(
            f"{late_len:.0f} s removed at {late_at:.0f} s (88%) — most offset probes "
            "agree, so a global-offset shortcut silently breaks the tail"
        ),
        filter_complex=(
            f"[0:a]atrim=end={late_at}[a1];"
            f"[0:a]atrim=start={late_at + late_len}[a2];"
            f"[a1][a2]concat=n=2:v=0:a=1[out]"
        ),
        truth=lambda t: t if t < late_at else t + late_len,
    ))

    return s


@dataclass
class ScenarioResult:
    name: str
    mode: str = "?"
    confidence: float = 0.0
    flagged: bool = False
    n_anchors_scored: int = 0
    median_ms: float = float("nan")
    p95_ms: float = float("nan")
    max_ms: float = float("nan")
    pct_within_50ms: float = float("nan")
    elapsed_s: float = 0.0
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    global_offset: float = 0.0


def edit_points(sc: Scenario, dur: float, probe_step: float = 0.5) -> list[float]:
    """Locate truth discontinuities/undefined zones by probing the truth map."""
    pts: list[float] = []
    prev = sc.truth(0.0)
    t = probe_step
    while t < dur:
        cur = sc.truth(t)
        if (cur is None) != (prev is None):
            pts.append(t)
        elif cur is not None and prev is not None and abs((cur - prev) - probe_step) > 0.25:
            pts.append(t)
        prev = cur
        t += probe_step
    return pts


def run_scenario(
    sc: Scenario,
    video_path: Path,
    ad_base: Path,
    workdir: Path,
    dur: float,
    base_offset: float = 0.0,
) -> ScenarioResult:
    res = ScenarioResult(name=sc.name)
    variant = workdir / f"ad_{sc.name}.wav"
    report_path = workdir / f"report_{sc.name}.json"

    subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", str(ad_base),
         "-filter_complex", sc.filter_complex, "-map", "[out]", str(variant)],
        check=True,
    )

    from adsync._pipeline import run_pipeline
    from adsync.config import SyncConfig

    t0 = time.perf_counter()
    try:
        run_pipeline(
            video_path=video_path,
            ad_path=variant,
            output_path=None,
            config=SyncConfig(),
            report_path=report_path,
            debug_dir=None,
            keep_temp=False,
            mux=False,
        )
    except Exception as exc:  # scoring continues; the failure IS the result
        res.error = f"{type(exc).__name__}: {exc}"
        return res
    finally:
        res.elapsed_s = time.perf_counter() - t0

    rep = json.loads(report_path.read_text())
    res.mode = rep["mode"]
    res.confidence = rep["confidence"]
    res.flagged = rep["confidence"] < 0.70
    res.warnings = rep.get("warnings", [])
    res.global_offset = rep.get("global_offset") or 0.0
    if sc.name == "clean":
        # The base pair carries a small constant offset from how it was cut;
        # the clean run measures it and later scenarios score against it.
        base_offset = res.global_offset if abs(res.global_offset) < 0.5 else 0.0

    edits = edit_points(sc, dur)
    errors_ms: list[float] = []
    # Anchor conventions differ by mode: offset/drift anchors are
    # (source=video, target=AD); warp/piecewise anchors are
    # (source=AD, target=video).  truth() maps AD-variant time → video time.
    # When the pipeline auto-corrected speed, its anchors live on the
    # corrected timeline — map back to variant time before consulting truth.
    video_source = rep["mode"] in ("offset", "drift")
    stretch = rep.get("speed_stretch") or 1.0
    for a in rep.get("anchors", []):
        if video_source:
            ad_t, vid_t = a["target_time"], a["source_time"]
        else:
            ad_t, vid_t = a["source_time"], a["target_time"]
        variant_t = ad_t / stretch
        truth = sc.truth(variant_t)
        if truth is None:
            continue
        if any(abs(variant_t - e) < EDGE_EXCLUDE_SEC for e in edits):
            continue
        errors_ms.append(abs(vid_t - (truth + base_offset)) * 1000.0)

    if errors_ms:
        errors_ms.sort()
        n = len(errors_ms)
        res.n_anchors_scored = n
        res.median_ms = errors_ms[n // 2]
        res.p95_ms = errors_ms[min(n - 1, int(n * 0.95))]
        res.max_ms = errors_ms[-1]
        res.pct_within_50ms = 100.0 * sum(1 for e in errors_ms if e <= 50.0) / n
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, type=Path, help="Content-aligned video-audio file")
    ap.add_argument("--ad", required=True, type=Path, help="Content-aligned AD base audio (44.1 kHz)")
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--scenarios", nargs="*", help="Subset of scenario names")
    ap.add_argument("--json-out", type=Path, help="Write results JSON here")
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(args.video)],
        capture_output=True, text=True, check=True,
    )
    dur = float(json.loads(probe.stdout)["format"]["duration"])

    scenarios = make_scenarios(dur)
    if args.scenarios:
        wanted = set(args.scenarios)
        scenarios = [s for s in scenarios if s.name in wanted]

    # Clean runs first: it calibrates the base pair's inherent offset
    scenarios.sort(key=lambda s: s.name != "clean")

    results: list[ScenarioResult] = []
    base_offset = 0.0
    for sc in scenarios:
        print(f"── {sc.name}: {sc.description}")
        res = run_scenario(sc, args.video, args.ad, args.workdir, dur, base_offset)
        if sc.name == "clean" and not res.error and abs(res.global_offset) < 0.5:
            base_offset = res.global_offset
            print(f"   base-pair offset calibrated: {base_offset:+.4f} s")
        results.append(res)
        if res.error:
            print(f"   ERROR: {res.error}")
        else:
            print(
                f"   mode={res.mode}  conf={res.confidence:.3f}"
                f"{' [FLAGGED]' if res.flagged else ''}  "
                f"median={res.median_ms:.1f}ms  p95={res.p95_ms:.1f}ms  "
                f"max={res.max_ms:.1f}ms  <=50ms: {res.pct_within_50ms:.0f}%  "
                f"({res.n_anchors_scored} anchors, {res.elapsed_s:.0f}s)"
            )

    print("\n=== SUMMARY ===")
    print(f"{'scenario':<10} {'mode':<7} {'conf':>5} {'median':>8} {'p95':>8} {'<=50ms':>7} {'time':>6}")
    for r in results:
        if r.error:
            print(f"{r.name:<10} ERROR: {r.error}")
        else:
            print(
                f"{r.name:<10} {r.mode:<7} {r.confidence:>5.2f} "
                f"{r.median_ms:>7.1f}m {r.p95_ms:>7.1f}m {r.pct_within_50ms:>6.0f}% {r.elapsed_s:>5.0f}s"
            )

    if args.json_out:
        args.json_out.write_text(json.dumps([r.__dict__ for r in results], indent=2, default=str))
        print(f"\nresults → {args.json_out}")


if __name__ == "__main__":
    main()
