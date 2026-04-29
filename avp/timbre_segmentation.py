"""
Timbre-based temporal segmentation for AAVP.

Computes interior boundary times on a video's audio track using MFCC features
plus agglomerative clustering, mirroring VideoTree/AM-Tree's recipe.  These
boundaries seed the preplan ASR pass (one transcript per anchor) and double
as candidate windows for in-region speech extraction.

Heavy dependencies (``librosa``, ``scipy``) are imported lazily so a pure-
visual install can still load the rest of the package.  All functions return
empty lists on failure rather than raising; callers degrade gracefully.
"""
from __future__ import annotations

import math
from typing import List


def compute_timbre_boundaries(
    video_path: str,
    duration_sec: float,
    *,
    anchor_interval_sec: float = 15.0,
    n_mfcc: int = 20,
    sample_rate: int = 22050,
    debug: bool = False,
) -> List[float]:
    """Return interior MFCC-agglomerative boundaries in seconds.

    Replicates ``VideoTree/AM-Tree/data_extraction/visual.find_dynamic_boundaries``
    while fixing its truncated ``librosa.get_duration`` call: we always compute
    duration from the loaded waveform after ``librosa.load``.

    Args:
        video_path:           Source video (any container readable by ffmpeg/audioread).
        duration_sec:         Caller-provided duration hint.  Used to size
                              ``n_segments`` even before the audio is decoded;
                              the actual decoded duration overrides it once
                              available.
        anchor_interval_sec:  Target spacing between boundaries.  ``n_segments
                              = max(2, ceil(duration / anchor_interval_sec))``.
        n_mfcc:               Number of MFCC coefficients (20 matches VideoTree).
        sample_rate:          Resample rate for ``librosa.load``.  22 050 Hz is
                              a memory/quality compromise — MFCC segmentation
                              is robust to mild downsampling.
        debug:                Emit progress/error logs.

    Returns:
        Sorted list of interior boundary times (seconds) with the trivial
        ``t = 0`` boundary dropped.  Returns ``[]`` on any failure (missing
        deps, decode error, silent track, agglomerative failure).
    """
    if duration_sec is None or duration_sec <= 0:
        if debug:
            print(f"[timbre] Invalid duration {duration_sec}, skipping boundary computation")
        return []

    try:
        import numpy as np
        import librosa
        import scipy.stats
    except Exception as exc:
        if debug:
            print(f"[timbre] Missing librosa/scipy/numpy ({exc}); returning empty boundaries")
        return []

    try:
        y, sr = librosa.load(video_path, sr=sample_rate, mono=True)
    except Exception as exc:
        if debug:
            print(f"[timbre] librosa.load failed for {video_path}: {exc}")
        return []

    if y is None or len(y) == 0:
        if debug:
            print(f"[timbre] {video_path} has no audio samples — no boundaries produced")
        return []

    decoded_duration = float(librosa.get_duration(y=y, sr=sr))
    effective_duration = decoded_duration if decoded_duration > 0 else float(duration_sec)

    n_segments = max(2, int(math.ceil(effective_duration / max(anchor_interval_sec, 1.0))))

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    except Exception as exc:
        if debug:
            print(f"[timbre] mfcc failed: {exc}")
        return []

    if mfcc.shape[1] < n_segments:
        # Track too short to produce that many homogeneous regions; fall back
        # to evenly spaced anchors over the decoded duration.
        if debug:
            print(
                f"[timbre] track too short ({mfcc.shape[1]} frames < {n_segments} segments) "
                f"— using uniform fallback"
            )
        step = effective_duration / float(n_segments)
        return [round(step * i, 3) for i in range(1, n_segments)]

    try:
        mfcc_scaled = scipy.stats.zscore(mfcc, axis=1)
        # zscore on a constant row produces NaN — replace with zeros so
        # agglomerative clustering does not blow up on silent tracks.
        mfcc_scaled = np.nan_to_num(mfcc_scaled, copy=False)
    except Exception as exc:
        if debug:
            print(f"[timbre] zscore failed: {exc}")
        return []

    try:
        boundaries = librosa.segment.agglomerative(mfcc_scaled, n_segments)
    except Exception as exc:
        if debug:
            print(f"[timbre] agglomerative failed: {exc}")
        return []

    try:
        boundary_times = librosa.frames_to_time(boundaries, sr=sr)
    except Exception as exc:
        if debug:
            print(f"[timbre] frames_to_time failed: {exc}")
        return []

    interior = [
        float(t)
        for t in boundary_times
        if t > 1e-3 and t < effective_duration - 1e-3
    ]
    interior = sorted(set(round(t, 3) for t in interior))

    if debug:
        print(
            f"[timbre] {video_path}: duration={effective_duration:.1f}s, "
            f"n_segments={n_segments}, boundaries={len(interior)}"
        )
    return interior
