"""Build a candidate lattice — top-K offset hypotheses per analysis window.

Uses raw-audio cross-correlation (downsampled to ~4 kHz) for robustness
against repetitive musical content, with speech-likeness scoring per window.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, fftconvolve, find_peaks, sosfiltfilt

from adsync.models import CandidateWindow, FeatureBundle, OffsetCandidate

log = logging.getLogger("adsync")

# ── Multi-band correlation ───────────────────────────────────────────────────
# Narration mixed over the scene contaminates the speech band and dilutes a
# full-band correlation.  Splitting at these edges (Hz, at the ~4 kHz
# correlation rate) and normalizing per band keeps the clean bands' vote;
# the narration-dominant middle band gets the smallest weight.
_BAND_EDGES = (250.0, 1200.0)
_BAND_WEIGHTS = (0.4, 0.2, 0.4)   # low, mid (speech), high


def build_candidate_lattice(
    video_features: FeatureBundle,
    ad_features: FeatureBundle,
    *,
    y_vid: NDArray | None = None,
    y_ad: NDArray | None = None,
    audio_sr: int = 16000,
    window_sec: float = 8.0,
    step_sec: float = 2.0,
    search_radius_sec: float = 30.0,
    offset_hint: float = 0.0,
    offset_hint_fn: Callable[[float], float] | None = None,
    min_score: float = 0.3,
    max_candidates: int = 5,
    multiband: bool = True,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[CandidateWindow]:
    """Scan the AD track in windows and keep top-K offset candidates per window.

    When *y_vid* and *y_ad* are provided, uses raw-audio cross-correlation
    (downsampled to ~4 kHz) instead of onset features.  This is substantially
    more robust in music-heavy regions where onset patterns repeat.

    *offset_hint* (seconds, positive = AD shifts later) centres each window's
    video search region on the prevailing global offset, so ``search_radius_sec``
    buys room for *edits around that offset* rather than being spent reaching
    the offset itself.  *offset_hint_fn*, when given, overrides it per window:
    called with the window's AD-time centre, it returns that window's expected
    offset (e.g. from fingerprint spans), so even huge edits need only a small
    radius.

    Returns a list of :class:`CandidateWindow`, one per analysis position.
    """
    # Decide signal source: raw audio (preferred) or onset features
    if y_vid is not None and y_ad is not None:
        ds_factor = max(1, audio_sr // 4000)
        ds_sr = audio_sr / ds_factor

        def _ds(y: NDArray) -> NDArray:
            n = len(y) - len(y) % ds_factor
            return np.mean(y[:n].reshape(-1, ds_factor), axis=1)

        vid = _ds(y_vid).astype(np.float64)
        ad = _ds(y_ad).astype(np.float64)
        sec_per_sample = 1.0 / ds_sr
    else:
        vid = video_features.onset.astype(np.float64)
        ad = ad_features.onset.astype(np.float64)
        sec_per_sample = video_features.hop_length / video_features.sr
        multiband = False  # onset features are not a waveform

    # Band decomposition: two zero-phase lowpass copies per track; the three
    # bands are derived per window by subtraction (low = L0, mid = L1 − L0,
    # high = full − L1).  Zero-phase filtering keeps peak positions aligned
    # across bands so the combined correlation stays sub-sample sharp.
    vid_lp: list[NDArray] = []
    ad_lp: list[NDArray] = []
    if multiband:
        nyq = 0.5 / sec_per_sample
        for edge in _BAND_EDGES:
            sos = butter(4, edge / nyq, btype="low", output="sos")
            vid_lp.append(sosfiltfilt(sos, vid).astype(np.float32))
            ad_lp.append(sosfiltfilt(sos, ad).astype(np.float32))

    win_samples = int(window_sec / sec_per_sample)
    step_samples = int(step_sec / sec_per_sample)
    search_samples = int(search_radius_sec / sec_per_sample)
    hint_samples = int(round(offset_hint / sec_per_sample))
    min_peak_dist = max(1, win_samples // 4)

    # Precompute mel spectrogram info for speech scoring
    mel = ad_features.mel  # (n_mels, n_frames), dB re track max
    mel_hop_sec = ad_features.hop_length / ad_features.sr
    n_mels = mel.shape[0]
    # Narration band: ~100-4000 Hz.  Mel bins are not linearly spaced in Hz,
    # so map through the same mel frequencies the spectrogram was built with.
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=ad_features.sr / 2.0)
    narration_lo = int(np.searchsorted(mel_freqs, 100.0))
    narration_hi = max(narration_lo + 1, int(np.searchsorted(mel_freqs, 4000.0)))

    ad_positions = list(range(0, len(ad) - win_samples, step_samples))
    total_windows = len(ad_positions)

    def _one_window(ad_start: int) -> CandidateWindow:
        ad_seg = ad[ad_start: ad_start + win_samples].copy()
        ad_seg -= np.mean(ad_seg)
        ad_energy = np.sqrt(np.sum(ad_seg ** 2))
        if ad_energy < 1e-10:
            # Silent window — no candidates
            source_center = (ad_start + win_samples / 2) * sec_per_sample
            return CandidateWindow(
                source_center=source_center,
                candidates=[],
                speech_score=0.0,
                energy=0.0,
            )

        ad_center_sec = (ad_start + win_samples / 2) * sec_per_sample

        # Search region in video, centred on the expected offset for this
        # window (fingerprint-informed when available, global otherwise)
        if offset_hint_fn is not None:
            h = int(round(offset_hint_fn(ad_center_sec) / sec_per_sample))
        else:
            h = hint_samples
        nominal_vid_start = ad_start + h
        search_start = max(0, nominal_vid_start - search_samples)
        search_end = min(len(vid), nominal_vid_start + search_samples + win_samples)

        if search_end - search_start < win_samples:
            return CandidateWindow(
                source_center=ad_center_sec,
                candidates=[],
                speech_score=0.0,
                energy=float(ad_energy),
            )

        def _norm_corr_of(v_reg_in: NDArray, a_seg_in: NDArray) -> NDArray | None:
            """Normalized sliding correlation of one (pre-sliced) band pair."""
            a_seg = np.asarray(a_seg_in, np.float64)
            a_seg = a_seg - np.mean(a_seg)
            a_energy = np.sqrt(np.sum(a_seg ** 2))
            if a_energy < 1e-10:
                return None
            v_reg = np.asarray(v_reg_in, np.float64)
            v_reg = v_reg - np.mean(v_reg)
            raw = fftconvolve(v_reg, a_seg[::-1], mode="valid")
            if len(raw) == 0:
                return None
            v_sq = v_reg ** 2
            cs_b = np.empty(len(v_sq) + 1, dtype=np.float64)
            cs_b[0] = 0.0
            np.cumsum(v_sq, out=cs_b[1:])
            v_norms_b = np.sqrt(np.maximum(
                cs_b[win_samples: win_samples + len(raw)] - cs_b[:len(raw)], 1e-20,
            ))
            return raw / (a_energy * v_norms_b)

        v_full = vid[search_start:search_end]
        a_full = ad[ad_start: ad_start + win_samples]
        if vid_lp:
            # Weighted per-band normalized correlations:
            # low = L0, mid = L1 − L0 (speech), high = full − L1.
            v_l0 = vid_lp[0][search_start:search_end]
            v_l1 = vid_lp[1][search_start:search_end]
            a_l0 = ad_lp[0][ad_start: ad_start + win_samples]
            a_l1 = ad_lp[1][ad_start: ad_start + win_samples]
            pairs = (
                (v_l0, a_l0),
                (v_l1 - v_l0, a_l1 - a_l0),
                (v_full - v_l1, a_full - a_l1),
            )
            weights = _BAND_WEIGHTS
        else:
            pairs = ((v_full, a_full),)
            weights = (1.0,)

        norm_corr = None
        weight_sum = 0.0
        for (v_sig, a_sig), w_b in zip(pairs, weights):
            band = _norm_corr_of(v_sig, a_sig)
            if band is None:
                continue
            if norm_corr is None:
                norm_corr = w_b * band
            else:
                norm_corr += w_b * band
            weight_sum += w_b
        if norm_corr is None:
            return CandidateWindow(
                source_center=ad_center_sec,
                candidates=[],
                speech_score=0.0,
                energy=float(ad_energy),
            )
        norm_corr /= weight_sum
        n_pos = len(norm_corr)

        mean_corr = float(np.mean(np.abs(norm_corr)))
        peak_indices, _ = find_peaks(norm_corr, distance=min_peak_dist)

        if len(peak_indices) == 0:
            best_idx = int(np.argmax(norm_corr))
            if norm_corr[best_idx] >= min_score:
                peak_indices = np.array([best_idx])
            else:
                return CandidateWindow(
                    source_center=ad_center_sec,
                    candidates=[],
                    speech_score=0.0,
                    energy=float(ad_energy),
                )

        peak_scores = norm_corr[peak_indices]
        top_order = np.argsort(-peak_scores)[:max_candidates]
        top_indices = peak_indices[top_order]
        top_scores = peak_scores[top_order]

        mask = top_scores >= min_score
        top_indices = top_indices[mask]
        top_scores = top_scores[mask]

        if len(top_indices) == 0:
            return CandidateWindow(
                source_center=ad_center_sec,
                candidates=[],
                speech_score=0.0,
                energy=float(ad_energy),
            )

        # Build candidates with parabolic refinement and quality metrics
        candidates: list[OffsetCandidate] = []
        best_peak_score = float(top_scores[0])
        second_best = float(top_scores[1]) if len(top_scores) > 1 else 0.0

        for rank, (pidx, pscore) in enumerate(zip(top_indices, top_scores)):
            pidx = int(pidx)
            score = float(pscore)

            # Sub-sample parabolic interpolation
            sub = 0.0
            if 0 < pidx < len(norm_corr) - 1:
                y_prev = float(norm_corr[pidx - 1])
                y_peak = float(norm_corr[pidx])
                y_next = float(norm_corr[pidx + 1])
                denom = y_prev - 2.0 * y_peak + y_next
                if abs(denom) > 1e-12:
                    sub = 0.5 * (y_prev - y_next) / denom
                    score = float(y_peak - 0.25 * (y_prev - y_next) * sub)

            # Convert to offset in seconds
            vid_match_start = search_start + pidx + sub
            vid_center_sec = (vid_match_start + win_samples / 2) * sec_per_sample
            offset_sec = vid_center_sec - ad_center_sec

            # Quality metrics
            sharpness = score / max(mean_corr, 1e-10)
            if rank == 0:
                ratio = score / max(second_best, 1e-10)
                ratio = min(ratio, 10.0)
            else:
                ratio = score / max(best_peak_score, 1e-10)

            candidates.append(OffsetCandidate(
                offset_sec=offset_sec,
                score=score,
                peak_sharpness=sharpness,
                peak_ratio=ratio if rank == 0 else ratio,
            ))

        # Speech-likeness score from mel spectrogram
        speech_score = _compute_speech_score(
            mel, mel_hop_sec, ad_center_sec, window_sec,
            narration_lo, narration_hi,
        )

        return CandidateWindow(
            source_center=ad_center_sec,
            candidates=candidates,
            speech_score=speech_score,
            energy=float(ad_energy),
        )

    # Windows are independent, and numpy/scipy release the GIL for the heavy
    # array work (the FFTs dominate), so a thread pool spreads the scan across
    # every core without copying the waveforms.  Windows are batched into
    # chunks so scheduling overhead stays negligible.  Results keep scan order.
    def _chunk(idx_start: int, idx_end: int) -> list[CandidateWindow]:
        return [_one_window(ad_positions[i]) for i in range(idx_start, idx_end)]

    slots: list[list[CandidateWindow]] = []
    if total_windows:
        n_workers = min(32, os.cpu_count() or 1, total_windows)
        chunk_len = max(1, min(64, total_windows // (n_workers * 4) or 1))
        bounds = list(range(0, total_windows, chunk_len))
        slots = [[] for _ in bounds]
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_chunk, b, min(b + chunk_len, total_windows)): i
                for i, b in enumerate(bounds)
            }
            n_done = 0
            for future in as_completed(futures):
                result = future.result()
                slots[futures[future]] = result
                n_done += len(result)
                if on_progress is not None:
                    on_progress(n_done, total_windows)

    lattice: list[CandidateWindow] = [w for chunk in slots for w in chunk]

    log.info(
        "Candidate lattice: %d windows, %d with candidates (avg %.1f candidates/window)",
        len(lattice),
        sum(1 for w in lattice if w.candidates),
        np.mean([len(w.candidates) for w in lattice if w.candidates]) if any(w.candidates for w in lattice) else 0,
    )
    return lattice


def augment_lattice_with_fingerprint(
    lattice: list[CandidateWindow],
    fp,
    *,
    half_window: float = 4.0,
    min_matches: int = 6,
    max_score: float = 0.8,
) -> int:
    """Append fingerprint-derived offset candidates to a starved lattice.

    When two sources carry different *mixes* of the same material (remastered
    release vs. original-mix AD bed), dense correlation features starve while
    sparse landmark matches still lock on.  Each window whose ±*half_window*
    neighbourhood holds at least *min_matches* inlier match pairs gets a
    candidate at their median offset, scored by local match density — above
    the decoder's synthetic bridges (0.1) and weak spurious peaks, below a
    strong genuine correlation (capped at *max_score*).

    Returns the number of windows that received a fingerprint candidate.
    """
    if fp is None or fp.match_t_ad is None or len(fp.match_t_ad) == 0:
        return 0

    t_ad = fp.match_t_ad
    offs = (fp.match_t_vid - fp.match_t_ad).astype(np.float64)
    centers = np.array([w.source_center for w in lattice])
    lo = np.searchsorted(t_ad, centers - half_window, side="left")
    hi = np.searchsorted(t_ad, centers + half_window, side="right")

    raw = np.full(len(lattice), np.nan)
    counts = hi - lo
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a >= min_matches:
            raw[i] = float(np.median(offs[a:b]))

    # Median-smooth over 5 neighbouring windows: knocks down landmark-grid
    # jitter (~tens of ms) without rounding off genuine edit steps, which
    # survive a running median.
    smoothed = raw.copy()
    for i in np.nonzero(~np.isnan(raw))[0]:
        hood = raw[max(0, i - 2): i + 3]
        hood = hood[~np.isnan(hood)]
        if len(hood) >= 3:
            smoothed[i] = float(np.median(hood))

    n_augmented = 0
    for i, w in enumerate(lattice):
        if np.isnan(smoothed[i]):
            continue
        score = min(max_score, 0.4 + 0.02 * int(counts[i]))
        w.candidates.append(OffsetCandidate(
            offset_sec=float(smoothed[i]),
            score=score,
            peak_sharpness=1.0,
            peak_ratio=1.0,
            source="fingerprint",
        ))
        w.candidates.sort(key=lambda c: -c.score)
        n_augmented += 1

    if n_augmented:
        log.info(
            "Fingerprint anchors: %d of %d windows anchored from %d landmark matches",
            n_augmented, len(lattice), len(t_ad),
        )
    return n_augmented


def _compute_speech_score(
    mel: NDArray,
    mel_hop_sec: float,
    center_sec: float,
    window_sec: float,
    narration_lo: int,
    narration_hi: int,
) -> float:
    """Estimate speech-likeness of a window from mel spectrogram.

    Combines spectral flatness (low = tonal/speech) with narration-band
    energy ratio (speech concentrates 100-4000 Hz).
    """
    n_frames = mel.shape[1]
    frame_start = max(0, int((center_sec - window_sec / 2) / mel_hop_sec))
    frame_end = min(n_frames, int((center_sec + window_sec / 2) / mel_hop_sec))

    if frame_end <= frame_start:
        return 0.0

    window_mel = mel[:, frame_start:frame_end]

    # extract_basic stores mel in dB (power_to_db, ref=track max, so values
    # are <= 0).  Flatness and the band ratio both need linear power; the
    # relative scale from the dB reference cancels out of both.
    power = np.power(10.0, 0.1 * window_mel)

    # Spectral flatness per frame: geometric_mean / arithmetic_mean
    log_mean = np.mean(np.log(power + 1e-10), axis=0)
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(power, axis=0)
    flatness = np.mean(geo_mean / np.maximum(arith_mean, 1e-10))
    flatness = float(np.clip(flatness, 0.0, 1.0))

    # Narration-band energy ratio
    total_energy = np.sum(power)
    if total_energy < 1e-10:
        return 0.0
    narration_energy = np.sum(power[narration_lo:narration_hi, :])
    narration_ratio = float(narration_energy / total_energy)

    # Combine: low flatness = speech-like, high narration ratio = speech-like
    speech_score = 0.5 * (1.0 - flatness) + 0.5 * narration_ratio
    return float(np.clip(speech_score, 0.0, 1.0))
