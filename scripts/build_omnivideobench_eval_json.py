#!/usr/bin/env python3
"""
Build an OmniVideoBench evaluation JSON compatible with avp.eval_dataset.

OmniVideoBench's canonical annotation is a nested JSON list:

    [
      {
        "video": "video_10",
        "video_type": "Cartoon",
        "duration": "04:23",
        "questions": [
          {
            "question": "...",
            "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
            "correct_option": "B",
            "question_type": "causal reasoning",
            "audio_type": "Sound",
            ...
          }
        ]
      },
      ...
    ]

The HuggingFace snapshot may also ship a data.parquet file in a row-per-question
layout. Both are supported: pass --input for JSON, --parquet for parquet.

Selection order (``--length-bucket`` and ``--max-videos``)
──────────────────────────────────────────────────────────
**Bucket filter is applied to the full annotation first**, then ``--max-videos N``
keeps the first *N* **top-level videos** in file order *within the filtered set*.
So e.g. ``--length-bucket ultralong --max-videos 30`` means: among *all* clips
classified as ultralong by duration, take the *first 30* such videos in list order.
It does **not** mean: take the first 30 videos in the file, then keep only
ultralongs (that would be a different subset).

Output format
─────────────
One flat JSON array with one record per QA pair, shaped exactly as
avp.eval_dataset expects:

  {
    "video_id":    "video_10",
    "question_id": "video_10__0",       # synthetic if not in source
    "question":    "...",
    "options":     ["A. ...", ...],
    "answer":      "B",                 # single letter
    "solution":    "<answer>B</answer>",
    "path":        "/abs/path/video_10.mp4",
    "duration":    263.0,               # seconds (float); omitted if unparseable
    "question_type": "...",
    "audio_type":    "..."
  }

Usage
─────
  # From nested JSON
  python scripts/build_omnivideobench_eval_json.py \\
      --input  /data/omnivideobench/data.json \\
      --video-root /data/omnivideobench/videos \\
      --output eval_omnivideo_with_paths.json \\
      --max-videos 30

  # From HF parquet
  python scripts/build_omnivideobench_eval_json.py \\
      --parquet /data/omnivideobench/data.parquet \\
      --video-root /data/omnivideobench/videos \\
      --output eval_omnivideo_with_paths.json \\
      --max-videos 30

  # From HuggingFace Hub directly (requires HF_TOKEN / huggingface-cli login)
  python scripts/build_omnivideobench_eval_json.py \\
      --use-datasets \\
      --video-root /data/omnivideobench/videos \\
      --output eval_omnivideo_with_paths.json \\
      --max-videos 30

  # First 30 *Ultralong* videos only (>10 min) — per OmniVideoBench paper (Fig. 3)
  python scripts/build_omnivideobench_eval_json.py \\
      --input  /data/omnivideobench/data.json \\
      --video-root /data/omnivideobench/videos \\
      --output eval_omnivideo_with_paths.json \\
      --length-bucket ultralong \\
      --max-videos 30
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


# ── Duration helpers ──────────────────────────────────────────────────────────

def parse_duration(s: str | None) -> float | None:
    """Parse 'MM:SS' or 'HH:MM:SS' duration string to seconds. Returns None on failure."""
    if s is None:
        return None
    # Direct numeric (seconds): int, float, numpy scalars, etc.
    if not isinstance(s, str) and not isinstance(s, bool):
        try:
            x = float(s)
            if math.isnan(x) or math.isinf(x):
                return None
            return x
        except (TypeError, ValueError):
            return None
    s = str(s).strip()
    if not s:
        return None
    parts = s.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, TypeError):
        pass
    # Try interpreting as plain numeric seconds
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# OmniVideoBench paper (Fig. 3): duration categories (seconds)
#   Short    : < 1 min
#   Medium   : 1–5 min
#   Long     : 5–10 min
#   Ultrlong : > 10 min


def classify_omnivideo_duration_sec(sec: float) -> str:
    """Map duration in seconds to OmniVideoBench's official bucket name."""
    if sec < 60.0:
        return "short"
    if sec < 300.0:  # < 5 min
        return "medium"
    if sec <= 600.0:  # 5 min through 10:00
        return "long"
    return "ultralong"


def in_length_bucket(sec: float | None, length_bucket: str) -> bool:
    """If length_bucket is 'all', accept any row with a known duration; else match bucket."""
    if length_bucket == "all":
        return True
    if sec is None:
        return False
    return classify_omnivideo_duration_sec(sec) == length_bucket


# Video filename extensions we treat as "already has an extension" (no --ext append).
_KNOWN_VIDEO_SUFFIXES: tuple[str, ...] = (
    ".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv"
)


def resolve_video_path(video_root: Path, video_field: str, ext: str) -> Path:
    """Map annotation `video` id / relpath to a local file path.

    HuggingFace parquet often stores values like ``videos/video_25.mp4`` (relative
    to the dataset root) while the user points ``--video-root`` at the
    ``.../videos`` directory. Naive join would create ``.../videos/videos/...``.

    The same data may also use ``video_25.mp4``; appending ``--ext`` would yield
    ``video_25.mp4.mp4``. We only append *ext* when the relative path has no
    known video extension on its last component.
    """
    v = str(video_field).strip().replace("\\", "/")
    if not v or v in (".", ".."):
        return video_root / f"_invalid_{video_field!s}{ext}"

    try:
        root_r = video_root.resolve()
    except OSError:
        root_r = video_root

    # Drop a redundant leading "videos/" when the user already chdir'd into that folder.
    if root_r.name.lower() == "videos":
        pfx = "videos/"
        if v[: len(pfx)].lower() == pfx:
            v = v[len(pfx) :]

    # Last path component: if it already looks like a video file, do not add ext.
    rel = Path(v)
    last = rel.name.lower()
    if any(last.endswith(sfx) for sfx in _KNOWN_VIDEO_SUFFIXES):
        return video_root / v
    ex = ext if ext.startswith(".") else f".{ext}"
    return video_root / f"{v}{ex}"


# ── Record builder ────────────────────────────────────────────────────────────

def make_record(
    video_id: str,
    question_idx: int,
    question: str,
    options: list[str],
    correct_option: str,
    video_path: Path,
    duration_str: str | None = None,
    question_type: str = "",
    audio_type: str = "",
    question_id: str | None = None,
) -> dict:
    answer = str(correct_option).strip().upper()
    rec: dict = {
        "video_id": video_id,
        "question_id": question_id or f"{video_id}__{question_idx}",
        "question": question,
        "options": list(options),
        "answer": answer,
        "solution": f"<answer>{answer}</answer>",
        "path": str(video_path.resolve()),
    }
    dur = parse_duration(duration_str)
    if dur is not None:
        rec["duration"] = dur
    if question_type:
        rec["question_type"] = question_type
    if audio_type:
        rec["audio_type"] = audio_type
    return rec


# ── JSON path ─────────────────────────────────────────────────────────────────

def load_from_json(
    input_path: Path,
    video_root: Path,
    ext: str,
    max_videos: int | None,
    keep_missing: bool,
    length_bucket: str = "all",
) -> tuple[list[dict], int]:
    with open(input_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        sys.exit(f"Expected a JSON array at the top level, got {type(data).__name__}")

    n_before = len(data)
    if length_bucket != "all":
        filtered: list[dict] = []
        dropped = 0
        for entry in data:
            dur_s = parse_duration(entry.get("duration"))
            if in_length_bucket(dur_s, length_bucket):
                filtered.append(entry)
            else:
                dropped += 1
        data = filtered
        print(
            f"Length-bucket {length_bucket!r}: {len(data)} of {n_before} top-level video "
            f"entries (dropped {dropped} outside bucket)"
        )
        if not data:
            print(
                "No videos left after length-bucket filter. "
                "Check annotations or use --length-bucket all.",
                file=sys.stderr,
            )

    if max_videos and max_videos > 0:
        data = data[:max_videos]
        print(f"Sliced to first {max_videos} video entries in bucket order → {len(data)} entries")

    records: list[dict] = []
    skipped = 0

    for entry in data:
        video_id = str(entry.get("video", "unknown"))
        duration_str = str(entry.get("duration", "")) or None
        video_path = resolve_video_path(video_root, video_id, ext)

        if not video_path.is_file():
            skipped += 1
            if not keep_missing:
                print(f"  SKIP (missing): {video_path}", file=sys.stderr)
                continue
            print(f"  WARNING (missing): {video_path}", file=sys.stderr)

        questions = entry.get("questions", [])
        for q_idx, qa in enumerate(questions):
            question = str(qa.get("question", ""))
            options = qa.get("options", [])
            if hasattr(options, "tolist"):
                options = options.tolist()
            else:
                options = list(options)
            correct_option = str(qa.get("correct_option", ""))
            question_type = str(qa.get("question_type", ""))
            audio_type = str(qa.get("audio_type", ""))

            records.append(make_record(
                video_id=video_id,
                question_idx=q_idx,
                question=question,
                options=options,
                correct_option=correct_option,
                video_path=video_path,
                duration_str=duration_str,
                question_type=question_type,
                audio_type=audio_type,
            ))

    return records, skipped


# ── Parquet path ──────────────────────────────────────────────────────────────

def _row_duration_sec(row: object) -> float | None:
    """Get duration in seconds from a pandas Series row (best-effort)."""
    import pandas as pd

    if not isinstance(row, pd.Series):
        return None
    for key in ("duration", "video_duration", "duration_sec", "length"):
        if key not in row.index:
            continue
        s = row[key]
        if s is None or (isinstance(s, float) and math.isnan(s)):
            continue
        if isinstance(s, str) and not s.strip():
            continue
        p = parse_duration(s)
        if p is not None:
            return p
    return None


def load_from_parquet(
    parquet_path: Path,
    video_root: Path,
    ext: str,
    max_videos: int | None,
    keep_missing: bool,
    length_bucket: str = "all",
) -> tuple[list[dict], int]:
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required for parquet mode: pip install pandas pyarrow")

    df = pd.read_parquet(str(parquet_path))
    print(f"Loaded {len(df)} rows from parquet.")
    print(f"Columns: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"First row sample:\n{df.iloc[0].to_dict()}\n")

    # Detect the video column — OmniVideoBench uses "video"; fall back to "video_id".
    video_col = "video" if "video" in df.columns else "video_id"

    n_before = len(df)
    if length_bucket != "all":
        mask: list[bool] = []
        for i in range(len(df)):
            sec = _row_duration_sec(df.iloc[i])
            mask.append(in_length_bucket(sec, length_bucket))
        df = df.loc[mask].reset_index(drop=True)  # type: ignore[assignment]
        print(
            f"Length-bucket {length_bucket!r}: {len(df)} of {n_before} rows "
            f"(dropped {n_before - len(df)} outside bucket)"
        )
        if len(df) == 0:
            print(
                "No rows left after length-bucket filter. "
                "Check parquet columns or use --length-bucket all.",
                file=sys.stderr,
            )

    if max_videos and max_videos > 0:
        unique_vids: list[str] = []
        seen: set[str] = set()
        for v in df[video_col].astype(str):
            if v not in seen:
                seen.add(v)
                unique_vids.append(v)
        keep_ids = set(unique_vids[:max_videos])
        df = df[df[video_col].astype(str).isin(keep_ids)].reset_index(drop=True)
        print(f"Filtered to first {max_videos} unique videos in bucket order → {len(df)} rows")

    # Detect answer column: prefer "correct_option", then "answer".
    answer_col = (
        "correct_option" if "correct_option" in df.columns
        else "answer" if "answer" in df.columns
        else None
    )
    if answer_col is None:
        sys.exit(
            "Cannot find an answer column (tried 'correct_option', 'answer'). "
            "Run with --dump-columns to inspect the parquet schema."
        )

    records: list[dict] = []
    skipped = 0

    for _, row in df.iterrows():
        video_id = str(row[video_col])
        video_path = resolve_video_path(video_root, video_id, ext)

        if not video_path.is_file():
            skipped += 1
            if not keep_missing:
                print(f"  SKIP (missing): {video_path}", file=sys.stderr)
                continue
            print(f"  WARNING (missing): {video_path}", file=sys.stderr)

        question = str(row.get("question", ""))
        opts = row.get("options", [])
        if hasattr(opts, "tolist"):
            opts = opts.tolist()
        else:
            opts = list(opts) if opts is not None else []

        correct_option = str(row.get(answer_col, "")).strip()
        duration_str = str(row.get("duration", "")) or None
        question_type = str(row.get("question_type", ""))
        audio_type = str(row.get("audio_type", ""))
        question_id = str(row.get("question_id", "")) or None

        records.append(make_record(
            video_id=video_id,
            question_idx=len(records),
            question=question,
            options=opts,
            correct_option=correct_option,
            video_path=video_path,
            duration_str=duration_str,
            question_type=question_type,
            audio_type=audio_type,
            question_id=question_id,
        ))

    return records, skipped


# ── HuggingFace datasets path ─────────────────────────────────────────────────

def load_from_hub(
    video_root: Path,
    ext: str,
    max_videos: int | None,
    keep_missing: bool,
    length_bucket: str = "all",
) -> tuple[list[dict], int]:
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("datasets is required for --use-datasets mode: pip install datasets")

    print("Loading NJU-LINK/OmniVideoBench from HuggingFace Hub...")
    ds = load_dataset("NJU-LINK/OmniVideoBench", split="train", trust_remote_code=True)
    print(f"Dataset features: {ds.features}")

    # Convert to list of dicts and delegate to the JSON logic by writing a temp file
    import tempfile, json as _json, os
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    rows = [row for row in ds]
    _json.dump(rows, tmp)
    tmp.close()
    try:
        records, skipped = load_from_json(
            input_path=Path(tmp.name),
            video_root=video_root,
            ext=ext,
            max_videos=max_videos,
            keep_missing=keep_missing,
            length_bucket=length_bucket,
        )
    finally:
        os.unlink(tmp.name)
    return records, skipped


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert OmniVideoBench annotations to AVP eval JSON with local video paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input (mutually exclusive)
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--input", "-i",
        metavar="PATH",
        help="Path to the OmniVideoBench nested JSON annotation file (data.json)",
    )
    src.add_argument(
        "--parquet",
        metavar="PATH",
        help="Path to data.parquet (HF row-per-question layout)",
    )
    src.add_argument(
        "--use-datasets",
        action="store_true",
        help="Download directly from HuggingFace Hub via the datasets library",
    )

    ap.add_argument(
        "--video-root",
        required=True,
        metavar="DIR",
        help="Directory containing video_*.mp4 files",
    )
    ap.add_argument(
        "--output", "-o",
        required=True,
        metavar="PATH",
        help="Output JSON path for avp.eval_dataset",
    )
    ap.add_argument(
        "--ext",
        default=".mp4",
        help="Video file extension (default: .mp4)",
    )
    ap.add_argument(
        "--max-videos",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Only include questions for the first N top-level video entries "
            "after optional --length-bucket filter, in file order (e.g. 30)"
        ),
    )
    ap.add_argument(
        "--length-bucket",
        default="all",
        choices=["all", "short", "medium", "long", "ultralong"],
        help=(
            "Filter to OmniVideoBench duration category (paper Fig. 3, by clip length in seconds): "
            "short <1 min, medium 1-5 min, long 5-10 min, ultrlong >10 min. "
            "Default: all (no filter). Use 'ultralong' for the >10 min split from the paper."
        ),
    )
    ap.add_argument(
        "--keep-missing",
        action="store_true",
        help="Include records whose video file does not exist on disk (path still set)",
    )
    ap.add_argument(
        "--dump-columns",
        action="store_true",
        help="Print parquet columns and a sample row then exit (useful for debugging schema)",
    )
    args = ap.parse_args()

    # Validate exactly one source
    if not (args.input or args.parquet or args.use_datasets):
        ap.error("Provide one of: --input, --parquet, or --use-datasets")

    video_root = Path(args.video_root)
    if not video_root.is_dir():
        sys.exit(f"video-root directory not found: {video_root}")

    # Parquet dump-only mode
    if args.dump_columns and args.parquet:
        try:
            import pandas as pd
        except ImportError:
            sys.exit("pandas is required: pip install pandas pyarrow")
        df = pd.read_parquet(args.parquet)
        print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
        print(f"\nFirst row:\n{df.iloc[0].to_dict()}")
        return

    # Dispatch
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_file():
            sys.exit(f"Input JSON not found: {input_path}")
        records, skipped = load_from_json(
            input_path=input_path,
            video_root=video_root,
            ext=args.ext,
            max_videos=args.max_videos,
            keep_missing=args.keep_missing,
            length_bucket=args.length_bucket,
        )
    elif args.parquet:
        parquet_path = Path(args.parquet)
        if not parquet_path.is_file():
            sys.exit(f"Parquet file not found: {parquet_path}")
        records, skipped = load_from_parquet(
            parquet_path=parquet_path,
            video_root=video_root,
            ext=args.ext,
            max_videos=args.max_videos,
            keep_missing=args.keep_missing,
            length_bucket=args.length_bucket,
        )
    else:
        records, skipped = load_from_hub(
            video_root=video_root,
            ext=args.ext,
            max_videos=args.max_videos,
            keep_missing=args.keep_missing,
            length_bucket=args.length_bucket,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nWrote {len(records)} records → {out_path}")
    if skipped:
        print(
            f"Skipped {skipped} rows with missing video files "
            "(pass --keep-missing to include them)"
        )


if __name__ == "__main__":
    main()
