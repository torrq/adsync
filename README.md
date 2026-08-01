# ADSync

Sync an audio description track to your copy of a video, and mux them into one MKV. One command.

## Why this exists

First time I tried this was with an episode of a Netflix show and an AD track I'd pulled from [audiovault.net](https://audiovault.net). I had VLC open on one screen and Windows Media Player on the other. Play the video, start the AD a few seconds behind, listen, pause, rewind, nudge the AD a second earlier, play again. By the time the first ad break hit, the two were off in a way no single offset was going to fix. I gave up around minute twelve.

Doing that for one episode is a slow evening. Doing it for a season is a problem. I'm blind, so this isn't really an accessibility project for me. It's just how I watch TV.

The reason it's hard: whoever recorded the AD did it against their copy of the episode, and yours is almost never the same cut. Different intro length, different ad breaks, different frame rate, an inserted recap, the occasional missing scene. Drop the AD on top of a mismatched cut and by minute ten it's narrating the wrong thing.

ADSync takes the video and the unsynced AD, does the alignment in one pass, and hands you back an MKV with the AD embedded as a selectable audio stream. If it can't find a confident alignment it tells you instead of quietly producing a broken mix.

## Who this is for

- Blind and visually impaired viewers, and people building tools for us.
- Accessibility curators batch-aligning libraries of AD tracks.
- Archivists pairing older descriptive-audio recordings with modern releases.
- Anyone who has spent an evening in Audacity nudging timestamps and decided once was enough.

## The approach

The older, simpler way to do this is to chop the AD into fixed chunks, align each chunk independently, and stitch the results back together with crossfades. That works well when the two sources are close, a constant offset or a gentle drift. ADSync keeps a mode like that around (`piecewise`) for the easy cases and for comparison.

It gets harder when the cuts don't match: different ad breaks, an inserted recap, a trimmed scene. Independent per-chunk decisions can disagree with their neighbours, and once they do, the stitched output can skip or double back on itself. Warp mode takes a different trade-off:

- Build a top-K candidate lattice of plausible offsets at every analysis window.
- Run a Viterbi / DP decoder with explicit penalties on offset jumps and curvature, so the whole track gets solved as one piece instead of window-by-window.
- Fit a shape-preserving monotone PCHIP warp through the decoded points, so the time-map never runs backwards.
- Render the final audio from that continuous time-map, sample by sample. No chunk seams to glue.

Either the alignment holds end-to-end, or the confidence score flags it up front so you know to review.

## What you get

- One command in, one MKV out, with the AD embedded as a tagged, selectable track.
- Handles constant offset, linear clock drift, and discontinuous edits (ad-break changes, missing scenes, inserted recaps) in the same pipeline.
- Audio landmark fingerprinting (Shazam-style constellation hashes) maps which stretch of the AD sits at which offset **globally** — edits of any size are located before alignment runs, and AD regions that match nowhere in the video are called out as bad spots in the report.
- Global warp alignment: a Viterbi decoder walks the candidate lattice and fits a shape-preserving monotone warp, so the whole track is solved as one piece.
- Sub-sample accuracy via parabolic interpolation around cross-correlation peaks, roughly 1–3 ms.
- Streams PCM straight into FFmpeg, no huge temp files. Every run produces a confidence score and a warnings list.
- Debug mode that dumps feature CSVs, plots, and intermediate WAVs when you want to see what the aligner saw.

## How it works

ADSync tries four alignment strategies in order of complexity, and uses whichever one clears the confidence threshold:

| Mode | When it wins | What it does |
|---|---|---|
| `offset` | The AD is a clean shift of the video (same cut, different start time) | Single global offset via normalised cross-correlation on downsampled raw audio |
| `drift` | Both tracks match but sample rates or clocks differ slightly | Measures offset at several points, fits a weighted linear drift model |
| `warp` *(default fallback)* | Cuts differ — inserted scenes, missing recaps, shifted ad breaks | Builds a top-K candidate lattice per window, runs a DP decoder with jump/curvature penalties + speech-rich bonuses, then fits a shape-preserving PCHIP warp and renders the output from a continuous time-map |
| `piecewise` | The classic stitch-and-crossfade approach | Anchor search + piecewise map with crossfades. Kept around for the easy cases and for comparison |

Warp mode is the default when offset/drift aren't enough, since solving the time-map globally tends to hold together better than reconciling independent per-chunk decisions after the fact.

## Installation

Requirements:

- Python 3.10+
- FFmpeg and ffprobe on your `PATH` ([ffmpeg.org](https://ffmpeg.org))

Then:

```bash
git clone https://github.com/JohnnyTheCoder1/ADSync.git
cd ADSync
pip install -e .
```

For development / running tests:

```bash
pip install -e ".[dev]"
pytest
```

## Quick start

```bash
adsync sync episode.mkv ad_track.m4a -o episode.with-ad.mkv
```

That's it. Open the resulting MKV in VLC, MPV, Plex, or Jellyfin, pick the "Audio Description" track, press play.

## Commands

```bash
# Prep a source first (optional) — keep one language's audio as stereo, drop the rest.
# Video/subs are stream-copied; already-stereo tracks are copied too (remux speed).
# Multichannel tracks get a dialog-forward downmix + true-peak limiter (libopus,
# ~4x faster than the aac encoder; pass --codec aac if you need AAC).
adsync prep episode.mkv --language eng

# Full sync — produces the final MKV with AD embedded
adsync sync episode.mkv ad_track.m4a -o episode.synced.mkv

# Analysis only — writes a JSON report, no output file
adsync analyze episode.mkv ad_track.m4a --report report.json

# Debug — dumps feature CSVs, anchor plots, and intermediate WAVs
adsync debug episode.mkv ad_track.m4a --workdir debug_out

# Mux only — you already have a synced AD file, just drop it in the container
adsync mux episode.mkv already_synced_ad.m4a -o final.mkv
```

## Useful flags

| Flag | Default | Notes |
|---|---|---|
| `--mode` | `auto` | `auto`, `offset`, `drift`, `warp`, or `piecewise` |
| `--codec` | `libopus` | Encoder for the AD track (Opus is tiny and clean for speech) |
| `--bitrate` | `96k` | AD track bitrate |
| `--language` | `eng` | Language tag written into the MKV metadata |
| `--ad-title` | `Audio Description` | Title tag for the track |
| `--offset-adjust` | `0.0` | Manual nudge in seconds (positive = push AD later). Handy when sync is great but you want the narration to land slightly before/after |
| `--confidence-threshold` | `0.70` | Exit code 1 below this, so batch scripts can auto-flag risky runs |
| `--warp-lambda-jump` | `2.0` | Warp DP penalty for offset jumps |
| `--warp-lambda-curve` | `5.0` | Warp DP penalty for curvature |
| `--warp-lambda-speech` | `0.3` | Warp bonus for speech-rich windows |
| `--warp-candidates` | `5` | Max offset candidates per analysis window |
| `--warp-search-radius` | auto | Per-window search radius (s) around the detected offset; auto shrinks to fingerprint span hints, else scales with the duration gap and offset scatter |
| `--no-fingerprint` | off | Skip landmark fingerprinting (span detection, bad-spot warnings, per-window warp hints) |

Run `adsync <command> --help` for the full list.

## Tested on *From*

I've been running this end-to-end on episodes of *From* (the MGM+ horror/mystery series), pairing fan-contributed AD tracks with retail releases. Sync has held up on everything I've tried so far, including one episode where the AD and video had a real edit discontinuity mid-way through. Warp mode handled that as a single offset jump instead of trying to force a linear drift fit across it.

If you have the tracks, it works.

## Known artifacts

- None currently tracked. (The transient pitch drift during warped stretching was fixed by re-rendering stretch regions through a pitch-preserving WSOLA path — see `src/adsync/rebuild/warp_render.py`.)

## Measured accuracy

`tools/accuracy_harness.py` applies edits with exactly known time maps (cuts up to 45 s, insertions, offsets, 200 ppm clock drift) to real movie audio and scores every reported anchor against ground truth. Current numbers across all eight scenarios: **median placement error ≤ 2.6 ms, worst anchor ≤ 35 ms, 100% of anchors within 50 ms**. The harness doubles as the regression gate for alignment changes.

## Roadmap

- [x] Eliminate the transient pitch drift during warped stretching (WSOLA or phase-vocoder resampler)
- [x] Audio landmark fingerprinting: global offset-span detection, edit localization, and bad-spot (unmatched content) warnings
- [ ] Optional GPU-accelerated cross-correlation for faster runs on long files
- [ ] Precomputed AD offset database / cache
- [ ] Web UI for non-technical users
- [ ] Automatic detection of matching AD tracks from a library

## Contributing

Issues and PRs welcome. A few things that are especially useful:

- Bug reports with a JSON report attached (from `adsync analyze`).
- Hard cases. Tracks that come out wrong, even if all you can do is describe the source material rather than share it.
- Improvements to the warp decoder penalties, the confidence model, or the rendering path.

One change per PR if you can, makes review fast.

## License

MIT. See [LICENSE](LICENSE).

## Credits and a note on content

ADSync doesn't ship, download, or distribute any copyrighted video or audio. It's a local tool that runs against files you already have.

The AD tracks themselves mostly come from [audiovault.net](https://audiovault.net), which hosts a huge library of both real professionally-produced audio description (the tracks originally aired by networks and studios) and community-recorded ones. None of this workflow exists without them. If ADSync is useful to you, go support audiovault, and credit the people who recorded the tracks where you can.
