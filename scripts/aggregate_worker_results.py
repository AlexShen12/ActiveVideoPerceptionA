#!/usr/bin/env python3
"""Aggregate accuracy from worker_*/results.jsonl under a run output directory.

Example:
  python scripts/aggregate_worker_results.py /path/to/avp_out_videomme
  python scripts/aggregate_worker_results.py /path/to/avp_out_videomme --write-merge
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

WORKER_DIR_RE = re.compile(r"^worker_(\d+)$")


def discover_workers(run_dir: Path) -> list[tuple[int, Path]]:
    workers: list[tuple[int, Path]] = []
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {run_dir}")
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = WORKER_DIR_RE.match(p.name)
        if m:
            workers.append((int(m.group(1)), p))
    workers.sort(key=lambda x: x[0])
    return workers


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad_lines = 0
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
                print(f"warning: {path}:{line_no}: invalid JSON", file=sys.stderr)
    return rows, bad_lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_dir",
        type=Path,
        help="Output directory containing worker_N subfolders (e.g. avp_out_videomme)",
    )
    ap.add_argument(
        "--write-merge",
        action="store_true",
        help="Write merged results.jsonl and summary.json at run_dir root",
    )
    args = ap.parse_args()
    run_dir = args.run_dir.resolve()

    workers = discover_workers(run_dir)
    if not workers:
        print(f"error: no worker_* directories under {run_dir}", file=sys.stderr)
        return 1

    merged: list[dict[str, Any]] = []
    per_worker: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    dupes = 0

    for wid, wdir in workers:
        results_path = wdir / "results.jsonl"
        if not results_path.is_file():
            print(f"warning: missing {results_path}", file=sys.stderr)
            per_worker.append(
                {
                    "worker_id": wid,
                    "path": str(wdir),
                    "samples": 0,
                    "correct": 0,
                    "accuracy": None,
                    "skipped": True,
                }
            )
            continue

        rows, bad_lines = load_jsonl(results_path)
        c = sum(1 for r in rows if r.get("correct") is True)
        n = len(rows)
        acc = (c / n) if n else None
        for r in rows:
            vid = str(r.get("video_id", ""))
            q = str(r.get("question", ""))
            key = (vid, q)
            if key in seen_keys:
                dupes += 1
            seen_keys.add(key)

        merged.extend(rows)
        per_worker.append(
            {
                "worker_id": wid,
                "path": str(wdir),
                "results_file": str(results_path),
                "samples": n,
                "correct": c,
                "accuracy": acc,
                "bad_json_lines": bad_lines,
                "skipped": False,
            }
        )

    total_correct = sum(1 for r in merged if r.get("correct") is True)
    total_n = len(merged)
    overall = (total_correct / total_n) if total_n else 0.0

    summary = {
        "run_dir": str(run_dir),
        "total_samples": total_n,
        "total_correct": total_correct,
        "overall_accuracy": overall,
        "duplicate_video_question_rows": dupes,
        "workers": per_worker,
        "aggregated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    print(json.dumps(summary, indent=2))
    print(
        f"\nOverall: {total_correct}/{total_n} = {100.0 * overall:.4f}%",
        file=sys.stderr,
    )
    if dupes:
        print(
            f"warning: {dupes} duplicate (video_id, question) rows across workers — "
            "chunks may overlap or a merge was run twice.",
            file=sys.stderr,
        )

    if args.write_merge:
        out_results = run_dir / "results.jsonl"
        out_summary = run_dir / "summary.json"
        with out_results.open("w") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with out_summary.open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {out_results}", file=sys.stderr)
        print(f"Wrote {out_summary}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
