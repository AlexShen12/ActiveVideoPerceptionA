#!/usr/bin/env python3
"""
Build a VideoMME evaluation JSON from a HuggingFace parquet file.

Each output record includes a resolved local video path so that
avp.eval_dataset can find the file on disk.  Records with missing
video files are skipped (or optionally kept with a warning).

Usage:
    python scripts/build_videomme_eval_json.py \
        --parquet test-00000-of-00001.parquet \
        --video-root /path/to/videomme/videos \
        --output eval_videomme_with_paths.json \
        [--ext .mp4] \
        [--key-field videoID] \
        [--keep-missing] \
        [--merge-duration avp/eval_anno/eval_videomme.json]

Video naming
------------
By default, each video file is expected at:

    <video-root>/<videoID><ext>

where <videoID> comes from the `videoID` column of the parquet.  If your
unzip layout uses a different filename (e.g. the `video_id` column), pass
--key-field video_id.

If the parquet only has parquet metadata with no `videoID` column but you
have folders named by video_id (e.g. "001/"), pass --key-field video_id.

Duration
--------
The parquet `duration` column contains string labels ("short", "medium",
"long") rather than numeric seconds.  By default this column is omitted from
the output JSON to avoid a crash in eval_dataset (which formats it as float).

Pass --merge-duration <path-to-eval_videomme.json> to look up numeric
durations from the bundled annotation file by matching on `question_id`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_reference_durations(ref_path: str) -> dict[str, float]:
    """Return {question_id: duration_sec} from the bundled eval_videomme.json."""
    with open(ref_path) as f:
        data = json.load(f)
    result: dict[str, float] = {}
    for entry in data:
        qid = entry.get("question_id")
        dur = entry.get("duration")
        if qid and dur is not None:
            try:
                result[qid] = float(dur)
            except (TypeError, ValueError):
                pass
    return result


def build_path(video_root: Path, key: str, ext: str) -> Path:
    return video_root / f"{key}{ext}"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert VideoMME parquet to AVP eval JSON with local video paths"
    )
    ap.add_argument("--parquet", required=True, help="Path to .parquet file")
    ap.add_argument(
        "--video-root",
        required=True,
        help="Directory containing extracted VideoMME videos",
    )
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument(
        "--ext",
        default=".mp4",
        help="Video file extension (default: .mp4)",
    )
    ap.add_argument(
        "--key-field",
        default="videoID",
        choices=["videoID", "video_id"],
        help="Parquet column used to build the filename (default: videoID)",
    )
    ap.add_argument(
        "--keep-missing",
        action="store_true",
        help="Include rows whose video file does not exist (path still set)",
    )
    ap.add_argument(
        "--merge-duration",
        default=None,
        metavar="REF_JSON",
        help="Optional: path to bundled eval_videomme.json to add numeric durations",
    )
    ap.add_argument(
        "--max-videos",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Only include questions for the first N unique videos "
            "(e.g. --max-videos 100 → ~300 rows for VideoMME's 3 questions/video). "
            "Videos are taken in the order they appear in the parquet."
        ),
    )
    args = ap.parse_args()

    # ── imports ──────────────────────────────────────────────────────────────
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required: pip install pandas pyarrow")

    parquet_path = Path(args.parquet)
    if not parquet_path.is_file():
        sys.exit(f"Parquet file not found: {parquet_path}")

    video_root = Path(args.video_root)
    if not video_root.is_dir():
        sys.exit(f"video-root directory not found: {video_root}")

    # ── load optional duration reference ─────────────────────────────────────
    ref_durations: dict[str, float] = {}
    if args.merge_duration:
        ref_durations = load_reference_durations(args.merge_duration)
        print(f"Loaded {len(ref_durations)} reference durations from {args.merge_duration}")

    # ── read parquet ──────────────────────────────────────────────────────────
    df = pd.read_parquet(str(parquet_path))
    print(f"Loaded {len(df)} rows from {parquet_path}")
    print(f"Columns: {df.columns.tolist()}")

    required = {"question", "options", "answer", args.key_field}
    missing_cols = required - set(df.columns)
    if missing_cols:
        sys.exit(f"Missing required columns in parquet: {missing_cols}")

    # ── optionally restrict to first N unique videos ──────────────────────────
    if args.max_videos is not None and args.max_videos > 0:
        unique_vids = list(dict.fromkeys(str(r) for r in df["video_id"]))
        keep = set(unique_vids[: args.max_videos])
        df = df[df["video_id"].astype(str).isin(keep)].reset_index(drop=True)
        print(f"Filtered to first {args.max_videos} videos → {len(df)} rows")

    # ── build records ─────────────────────────────────────────────────────────
    records = []
    skipped = 0
    warned_missing = 0

    for _, row in df.iterrows():
        key = str(row[args.key_field])
        video_path = build_path(video_root, key, args.ext)

        if not video_path.is_file():
            warned_missing += 1
            if not args.keep_missing:
                continue
            print(f"  WARNING: video not found — {video_path}", file=sys.stderr)

        # options: parquet stores as numpy array; eval expects a plain list
        opts = row["options"]
        if hasattr(opts, "tolist"):
            opts = opts.tolist()
        else:
            opts = list(opts)

        answer = str(row["answer"]).strip()

        rec: dict = {
            "video_id": str(row.get("video_id", key)),
            "videoID": str(row.get("videoID", key)),
            "question_id": str(row.get("question_id", "")),
            "task_type": str(row.get("task_type", "")),
            "domain": str(row.get("domain", "")),
            "sub_category": str(row.get("sub_category", "")),
            "question": str(row["question"]),
            "options": opts,
            "answer": answer,
            "solution": f"<answer>{answer}</answer>",
            "path": str(video_path.resolve()),
        }

        # Merge numeric duration when available; skip the string label
        qid = rec["question_id"]
        if qid in ref_durations:
            rec["duration"] = ref_durations[qid]
        # No else: intentionally omit duration rather than pass a non-float string

        records.append(rec)

    if not args.keep_missing:
        skipped = warned_missing

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nWrote {len(records)} records → {out_path}")
    if skipped:
        print(
            f"Skipped {skipped} rows with missing video files "
            f"(pass --keep-missing to include them)"
        )


if __name__ == "__main__":
    main()
