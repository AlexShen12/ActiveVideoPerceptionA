"""
Audio Utilities for Active Audio-Visual Perception (AAVP)
=========================================================

Local audio extraction via ffmpeg, parallel to video_utils.py.

Design principles:
- All extraction uses the same ffmpeg subprocess approach as create_video_clip.
- No ASR or sound classification here — only raw WAV extraction.
  Gemini handles all semantic interpretation (transcription, acoustic tagging).
- WAV at 16 kHz mono: ~64 KB per 2 s snippet — negligible disk cost.
- Every function returns None on failure rather than raising, so the caller
  can degrade gracefully (skip audio enrichment, keep visual evidence).
"""
from __future__ import annotations

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional


# ======================================================
# ffmpeg availability
# ======================================================

def check_ffmpeg_audio_support() -> bool:
    """Return True if ffmpeg is available and can decode audio (aac/pcm codecs).

    Runs ``ffmpeg -codecs`` and checks for known audio codec markers.
    Also verifies the binary is on PATH via shutil.which so we never
    attempt extraction when ffmpeg is absent.
    """
    if not shutil.which("ffmpeg"):
        return False
    try:
        result = subprocess.run(
            ["ffmpeg", "-codecs", "-hide_banner"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout + result.stderr
        # pcm_s16le is the standard PCM codec we request; aac is common input
        return "pcm_s16le" in output or "aac" in output or "DEA" in output
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


# ======================================================
# Internal helpers
# ======================================================

def _ensure_dir(temp_dir: Optional[str], video_path: str) -> Optional[str]:
    """Resolve and create a temp directory, returning the path or None on failure."""
    if temp_dir is None:
        temp_dir = str(Path(video_path).parent / "temp_audio")
    try:
        os.makedirs(temp_dir, exist_ok=True)
        if not os.access(temp_dir, os.W_OK):
            return None
        return temp_dir
    except OSError:
        return None


def _format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for ffmpeg -ss / -to arguments."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _run_ffmpeg_audio(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    sample_rate: int,
    debug: bool,
) -> bool:
    """Execute ffmpeg to extract a mono PCM WAV from *video_path* in [start_sec, end_sec].

    Returns True on success, False on any failure (missing file, codec error, etc.).
    Uses ``-ss`` before ``-i`` for fast seek, ``-vn`` to drop video,
    ``-acodec pcm_s16le`` for guaranteed Gemini-compatible WAV output.
    """
    if not os.path.exists(video_path):
        if debug:
            print(f"❌ [audio_utils] Video file not found: {video_path}")
        return False

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", _format_time(start_sec),
        "-to", _format_time(end_sec),
        "-i", video_path,
        "-vn",                    # strip video stream
        "-acodec", "pcm_s16le",   # 16-bit PCM — universally readable
        "-ar", str(sample_rate),  # target sample rate
        "-ac", "1",               # mono
        "-y",                     # overwrite without prompt
        output_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            if debug:
                print(
                    f"❌ [audio_utils] ffmpeg failed (code {result.returncode}) "
                    f"for {video_path} [{start_sec:.1f}–{end_sec:.1f}s]:\n"
                    f"  {result.stderr.strip()}"
                )
            return False

        # Confirm output was actually written and is non-empty
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            if debug:
                print(
                    f"⚠️  [audio_utils] ffmpeg exited 0 but output is missing/empty: "
                    f"{output_path}"
                )
            return False

        if debug:
            size_kb = os.path.getsize(output_path) / 1024
            print(
                f"✅ [audio_utils] Extracted audio [{start_sec:.1f}–{end_sec:.1f}s] "
                f"→ {output_path} ({size_kb:.1f} KB)"
            )
        return True

    except subprocess.TimeoutExpired:
        if debug:
            print(f"⏱️  [audio_utils] ffmpeg timed out for {video_path}")
        return False
    except (FileNotFoundError, OSError) as exc:
        if debug:
            print(f"❌ [audio_utils] Could not run ffmpeg: {exc}")
        return False


# ======================================================
# Public extraction API
# ======================================================

def extract_audio_snippet(
    video_path: str,
    center_sec: float,
    half_width_sec: float = 2.5,
    output_format: str = "wav",
    sample_rate: int = 16000,
    temp_dir: Optional[str] = None,
    debug: bool = False,
) -> Optional[str]:
    """Extract a short audio clip centred at *center_sec*.

    The window is ``[center_sec - half_width_sec, center_sec + half_width_sec]``,
    clamped so it never starts before 0 s (no upper-bound clamp — the caller
    should pass a valid duration if clamping the end is also desired).

    Output filename: ``{stem}_audio_{center_sec:.1f}s.wav``

    Args:
        video_path:      Path to the source video file.
        center_sec:      Midpoint of the desired audio window in seconds.
        half_width_sec:  Half-width of the window (default 2.5 s → 5 s total).
        output_format:   Container format, always ``"wav"`` for Gemini compatibility.
        sample_rate:     PCM sample rate in Hz (16 000 Hz is the Gemini minimum).
        temp_dir:        Directory for the output file; created if absent.
        debug:           Print progress/error messages when True.

    Returns:
        Absolute path to the extracted WAV file, or ``None`` on failure.
    """
    resolved_dir = _ensure_dir(temp_dir, video_path)
    if resolved_dir is None:
        if debug:
            print(f"❌ [audio_utils] Could not create temp dir for {video_path}")
        return None

    start_sec = max(0.0, center_sec - half_width_sec)
    end_sec = center_sec + half_width_sec  # caller clamps to duration if needed

    stem = Path(video_path).stem
    filename = f"{stem}_audio_{center_sec:.1f}s.{output_format}"
    output_path = os.path.join(resolved_dir, filename)

    # Re-use existing file if it is already present and non-empty
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        if debug:
            print(f"ℹ️  [audio_utils] Reusing cached snippet: {output_path}")
        return output_path

    success = _run_ffmpeg_audio(
        video_path, start_sec, end_sec, output_path, sample_rate, debug
    )
    return output_path if success else None


def extract_audio_region(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_format: str = "wav",
    sample_rate: int = 16000,
    temp_dir: Optional[str] = None,
    debug: bool = False,
) -> Optional[str]:
    """Extract audio for an explicit ``[start_sec, end_sec]`` region.

    Functionally identical to :func:`extract_audio_snippet` but takes explicit
    boundaries instead of a centre ± half-width.  Used when the reflector
    requests audio over a targeted ``zoom_region``.

    Output filename: ``{stem}_audio_region_{start_sec:.1f}s_{end_sec:.1f}s.wav``

    Args:
        video_path:    Path to the source video file.
        start_sec:     Region start in seconds (clamped to ≥ 0).
        end_sec:       Region end in seconds.
        output_format: Container format (``"wav"``).
        sample_rate:   PCM sample rate in Hz.
        temp_dir:      Directory for the output file; created if absent.
        debug:         Print progress/error messages when True.

    Returns:
        Absolute path to the extracted WAV file, or ``None`` on failure.
    """
    resolved_dir = _ensure_dir(temp_dir, video_path)
    if resolved_dir is None:
        if debug:
            print(f"❌ [audio_utils] Could not create temp dir for {video_path}")
        return None

    start_sec = max(0.0, start_sec)
    if end_sec <= start_sec:
        if debug:
            print(
                f"⚠️  [audio_utils] Skipping region with non-positive duration: "
                f"[{start_sec:.1f}, {end_sec:.1f}]"
            )
        return None

    stem = Path(video_path).stem
    filename = f"{stem}_audio_region_{start_sec:.1f}s_{end_sec:.1f}s.{output_format}"
    output_path = os.path.join(resolved_dir, filename)

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        if debug:
            print(f"ℹ️  [audio_utils] Reusing cached region: {output_path}")
        return output_path

    success = _run_ffmpeg_audio(
        video_path, start_sec, end_sec, output_path, sample_rate, debug
    )
    return output_path if success else None


# ======================================================
# Gap probe generation
# ======================================================

def generate_gap_probes(
    evidence_timestamps: list[tuple[float, float]],
    duration_sec: float,
    max_probes: int = 5,
    min_gap_sec: float = 10.0,
) -> list[float]:
    """Generate sparse probe times in uncovered gaps between evidence timestamps.

    Identifies the longest gaps (≥ *min_gap_sec*) between consecutive evidence
    intervals, places one probe at each gap midpoint, and returns up to
    *max_probes* times sorted ascending.  This catches off-screen narration or
    brief sounds that produced no visual ``key_evidence`` in the initial pass.

    The timeline is treated as::

        [0, first_start] + gaps between consecutive intervals + [last_end, duration_sec]

    Args:
        evidence_timestamps: List of ``(timestamp_start, timestamp_end)`` from
                             ``key_evidence``.  Overlapping intervals are merged
                             before gap analysis.
        duration_sec:        Total video duration in seconds.
        max_probes:          Maximum number of gap probes to return.
        min_gap_sec:         Minimum gap length (seconds) to be considered for probing.

    Returns:
        Sorted list of probe midpoint times (seconds), at most *max_probes* long.
        Returns empty list if no qualifying gaps are found.
    """
    if not evidence_timestamps or duration_sec <= 0:
        # No evidence or zero-length video — probe the video midpoint as a fallback
        if duration_sec > 0:
            return [duration_sec / 2.0]
        return []

    # Clamp and sort intervals
    intervals = sorted(
        [(max(0.0, s), min(duration_sec, e)) for s, e in evidence_timestamps
         if e > s],
        key=lambda x: x[0],
    )

    # Merge overlapping/adjacent intervals
    merged: list[tuple[float, float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Build gap list: prefix gap + between-interval gaps + suffix gap
    gaps: list[tuple[float, float]] = []

    # Gap before first interval
    if merged[0][0] > 0:
        gaps.append((0.0, merged[0][0]))

    # Gaps between consecutive intervals
    for i in range(len(merged) - 1):
        gap_start = merged[i][1]
        gap_end = merged[i + 1][0]
        if gap_end > gap_start:
            gaps.append((gap_start, gap_end))

    # Gap after last interval
    if merged[-1][1] < duration_sec:
        gaps.append((merged[-1][1], duration_sec))

    # Filter by minimum gap size and sort largest first
    qualifying = [
        (g_start, g_end)
        for g_start, g_end in gaps
        if (g_end - g_start) >= min_gap_sec
    ]
    qualifying.sort(key=lambda g: g[1] - g[0], reverse=True)

    # Pick midpoints from the largest gaps up to max_probes
    probes = [
        (g_start + g_end) / 2.0
        for g_start, g_end in qualifying[:max_probes]
    ]
    return sorted(probes)


# ======================================================
# Cleanup
# ======================================================

def cleanup_audio_artifacts(
    paths: list[str],
    debug: bool = False,
) -> None:
    """Remove temporary WAV files created during audio enrichment.

    Silently skips files that no longer exist.  Call this after the Gemini
    audio enrichment call completes so temp files do not accumulate across
    rounds.

    Args:
        paths: List of absolute file paths to delete.
        debug: Print a message for each deleted/skipped file when True.
    """
    for path in paths:
        try:
            if os.path.exists(path):
                os.remove(path)
                if debug:
                    print(f"🗑️  [audio_utils] Removed: {path}")
        except OSError as exc:
            if debug:
                print(f"⚠️  [audio_utils] Could not remove {path}: {exc}")
