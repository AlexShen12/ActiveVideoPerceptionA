"""
Disk-cached preplan timbre anchors + ASR transcripts.

Combines :mod:`timbre_segmentation` (MFCC agglomerative boundaries),
:func:`audio_utils.extract_audio_snippet` (ffmpeg WAV windows) and
:mod:`local_asr` (faster-whisper transcripts) into a single cache-aware
``preplan_anchors`` call.  The resulting list seeds both the planner soft-
evidence block and per-round regional speech enrichment, which means we
never run the heavy ``librosa.load`` + MFCC + Whisper pipeline twice for
the same video.

Cache key combines absolute path, file size, mtime, and the parameters
that affect the output (anchor interval, window length, whisper model).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .audio_utils import extract_audio_snippet
from .local_asr import transcribe_wav
from .timbre_segmentation import compute_timbre_boundaries


# Cache schema version.  Bump when the on-disk shape changes so old cache
# files are ignored automatically.
_CACHE_VERSION = 1


@dataclass
class TimbreAnchor:
    """Single preplan anchor: a timbre boundary with a cached ASR transcript.

    Fields:
        center_sec:       Boundary time in original-video seconds.
        window_start_sec: Actual ASR window start (clamped to ≥ 0).
        window_end_sec:   Actual ASR window end (clamped to ≤ duration).
        transcript:       faster-whisper transcript; empty string when no
                          speech was detected or ASR was unavailable.
    """
    center_sec: float
    window_start_sec: float
    window_end_sec: float
    transcript: str = ""


# ---------------------------------------------------------------------------
# Cache key + IO
# ---------------------------------------------------------------------------

def _cache_key(video_path: str, anchor_interval_sec: float, timbre_window_sec: float, whisper_model: str) -> str:
    """Stable hash combining path identity and parameters that affect output."""
    try:
        st = os.stat(video_path)
        size = st.st_size
        mtime = int(st.st_mtime)
    except OSError:
        size = -1
        mtime = -1
    payload = "|".join([
        os.path.abspath(video_path),
        str(size),
        str(mtime),
        f"int={anchor_interval_sec:.3f}",
        f"win={timbre_window_sec:.3f}",
        f"wm={whisper_model}",
        f"v={_CACHE_VERSION}",
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cache_file(cache_dir: str, key: str) -> Path:
    return Path(cache_dir) / f"{key}.json"


def load_cache(
    cache_dir: str,
    video_path: str,
    *,
    anchor_interval_sec: float,
    timbre_window_sec: float,
    whisper_model: str,
) -> Optional[List[TimbreAnchor]]:
    """Return cached anchors for *video_path*, or None on miss/parse failure."""
    if not cache_dir:
        return None
    key = _cache_key(video_path, anchor_interval_sec, timbre_window_sec, whisper_model)
    path = _cache_file(cache_dir, key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or data.get("version") != _CACHE_VERSION:
        return None
    items = data.get("anchors")
    if not isinstance(items, list):
        return None
    out: List[TimbreAnchor] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            out.append(TimbreAnchor(
                center_sec=float(it["center_sec"]),
                window_start_sec=float(it["window_start_sec"]),
                window_end_sec=float(it["window_end_sec"]),
                transcript=str(it.get("transcript", "")),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def save_cache(
    cache_dir: str,
    video_path: str,
    anchors: List[TimbreAnchor],
    *,
    anchor_interval_sec: float,
    timbre_window_sec: float,
    whisper_model: str,
) -> None:
    """Persist anchors to ``<cache_dir>/<key>.json``; silently no-op on errors."""
    if not cache_dir:
        return
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        return
    key = _cache_key(video_path, anchor_interval_sec, timbre_window_sec, whisper_model)
    payload: Dict[str, Any] = {
        "version": _CACHE_VERSION,
        "video_path": os.path.abspath(video_path),
        "anchor_interval_sec": anchor_interval_sec,
        "timbre_window_sec": timbre_window_sec,
        "whisper_model": whisper_model,
        "anchors": [asdict(a) for a in anchors],
    }
    try:
        _cache_file(cache_dir, key).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preplan_anchors(
    video_path: str,
    duration_sec: float,
    *,
    cache_dir: str,
    anchor_interval_sec: float = 15.0,
    timbre_window_sec: float = 5.0,
    sample_rate: int = 16000,
    whisper_model: str = "base",
    whisper_device: str = "auto",
    whisper_compute_type: str = "default",
    temp_dir: Optional[str] = None,
    debug: bool = False,
) -> List[TimbreAnchor]:
    """Return cached or freshly computed preplan anchors for *video_path*.

    On cache hit no librosa/MFCC/Whisper work is performed; on miss we run
    the full pipeline (timbre boundaries → ffmpeg windows → Whisper) and
    persist the result before returning.

    Args:
        video_path:            Path to the source video.
        duration_sec:          Caller's duration hint.
        cache_dir:             Cache directory (created on demand).  Use an
                               empty string to disable caching entirely.
        anchor_interval_sec:   Target boundary spacing.
        timbre_window_sec:     Total ASR window per anchor (the window
                               half-width passed to ffmpeg is half of this).
        sample_rate:           ffmpeg WAV sample rate (Whisper resamples
                               internally; 16 kHz keeps disk usage low).
        whisper_model:         faster-whisper model id (``"base"`` etc.).
        whisper_device:        ``"auto"`` | ``"cpu"`` | ``"cuda"``.
        whisper_compute_type:  faster-whisper compute type.
        temp_dir:              ffmpeg WAV scratch dir; defaults beside the
                               video as in :func:`extract_audio_snippet`.
        debug:                 Emit progress/error logs.
    """
    cached = load_cache(
        cache_dir,
        video_path,
        anchor_interval_sec=anchor_interval_sec,
        timbre_window_sec=timbre_window_sec,
        whisper_model=whisper_model,
    )
    if cached is not None:
        if debug:
            print(f"[timbre-cache] hit {video_path}: {len(cached)} anchor(s)")
        return cached

    if debug:
        print(f"[timbre-cache] miss {video_path}: computing anchors")

    boundaries = compute_timbre_boundaries(
        video_path,
        duration_sec=duration_sec,
        anchor_interval_sec=anchor_interval_sec,
        debug=debug,
    )
    if not boundaries:
        # Persist an empty list so subsequent runs don't re-enter librosa for
        # videos that produced no usable anchors (silent / unreadable audio).
        save_cache(
            cache_dir,
            video_path,
            [],
            anchor_interval_sec=anchor_interval_sec,
            timbre_window_sec=timbre_window_sec,
            whisper_model=whisper_model,
        )
        return []

    half_width = max(timbre_window_sec / 2.0, 0.1)
    anchors: List[TimbreAnchor] = []
    for center in boundaries:
        start = max(0.0, center - half_width)
        end = min(duration_sec, center + half_width) if duration_sec > 0 else center + half_width
        if end <= start:
            continue
        wav_path = extract_audio_snippet(
            video_path,
            center,
            half_width,
            sample_rate=sample_rate,
            temp_dir=temp_dir,
            debug=debug,
        )
        transcript = ""
        if wav_path:
            transcript = transcribe_wav(
                wav_path,
                model=whisper_model,
                device=whisper_device,
                compute_type=whisper_compute_type,
                debug=debug,
            )
        anchors.append(TimbreAnchor(
            center_sec=float(center),
            window_start_sec=float(start),
            window_end_sec=float(end),
            transcript=transcript,
        ))

    save_cache(
        cache_dir,
        video_path,
        anchors,
        anchor_interval_sec=anchor_interval_sec,
        timbre_window_sec=timbre_window_sec,
        whisper_model=whisper_model,
    )
    if debug:
        print(f"[timbre-cache] saved {len(anchors)} anchor(s) for {video_path}")
    return anchors


def filter_anchors_in_regions(
    anchors: List[TimbreAnchor],
    regions: List,
) -> List[TimbreAnchor]:
    """Return the subset of *anchors* whose centre falls in any region.

    ``regions`` accepts ``[(start, end), ...]`` or ``[[start, end], ...]``.
    Empty input or empty regions returns ``[]``.
    """
    if not anchors or not regions:
        return []
    spans: List[tuple] = []
    for r in regions:
        try:
            s = float(r[0])
            e = float(r[1])
        except (TypeError, ValueError, IndexError):
            continue
        if e > s:
            spans.append((s, e))
    if not spans:
        return []
    out: List[TimbreAnchor] = []
    for a in anchors:
        c = a.center_sec
        if any(s <= c <= e for s, e in spans):
            out.append(a)
    return out
