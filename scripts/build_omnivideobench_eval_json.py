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
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── Duration helpers ──────────────────────────────────────────────────────────

def parse_duration(s: str | None) -> float | None:
    """Parse 'MM:SS' or 'HH:MM:SS' duration string to seconds. Returns None on failure."""
    if not s:
        return None
    parts = str(s).strip().split(":")
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
) -> tuple[list[dict], int]:
    with open(input_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        sys.exit(f"Expected a JSON array at the top level, got {type(data).__name__}")

    if max_videos and max_videos > 0:
        data = data[:max_videos]
        print(f"Sliced to first {max_videos} video entries → {len(data)} entries")

    records: list[dict] = []
    skipped = 0

    for entry in data:
        video_id = str(entry.get("video", "unknown"))
        duration_str = str(entry.get("duration", "")) or None
        video_path = video_root / f"{video_id}{ext}"

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

def load_from_parquet(
    parquet_path: Path,
    video_root: Path,
    ext: str,
    max_videos: int | None,
    keep_missing: bool,
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

    if max_videos and max_videos > 0:
        unique_vids = list(dict.fromkeys(str(v) for v in df[video_col]))
        keep_ids = set(unique_vids[:max_videos])
        df = df[df[video_col].astype(str).isin(keep_ids)].reset_index(drop=True)
        print(f"Filtered to first {max_videos} unique videos → {len(df)} rows")

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
        video_path = video_root / f"{video_id}{ext}"

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
            "Only include questions for the first N unique video entries "
            "(e.g. --max-videos 30)"
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
        )
    else:
        records, skipped = load_from_hub(
            video_root=video_root,
            ext=args.ext,
            max_videos=args.max_videos,
            keep_missing=args.keep_missing,
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
