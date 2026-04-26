#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# run_videomme_eval.sh
#
# End-to-end VideoMME evaluation with AAVP audio mode.
#
# Usage:
#
#   cp .env.example .env   # then set GEMINI_API_KEY (and optional Vertex vars)
#
#   export VIDEO_ROOT=/path/to/videomme/videos
#   bash scripts/run_videomme_eval.sh
#
# Credentials are read from ${REPO_ROOT}/.env by default. Override with:
#   export ENV_FILE=/path/to/custom.env
#
# The script:
#   1. Builds eval_videomme_with_paths.json from the parquet + video root.
#   2. Runs avp.eval_parallel on the full JSON.
#
# All paths are resolved relative to the repo root (ActiveVideoPerceptionA).
# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Repo root ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Secrets & env overrides from .env ─────────────────────────────────────────
# shellcheck source=dotenv.sh
source "${SCRIPT_DIR}/dotenv.sh"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
    dotenv_load "${ENV_FILE}"
    echo "(loaded ${ENV_FILE})"
else
    echo "NOTE: No .env at ${ENV_FILE} — set GEMINI_API_KEY in the shell or add a .env file." >&2
fi

# ── Required: video files ────────────────────────────────────────────────────
# Directory containing extracted VideoMME .mp4 files.
# File naming convention: <videoID>.mp4  (e.g. fFjv93ACGo8.mp4)
VIDEO_ROOT="${VIDEO_ROOT:-/path/to/videomme/videos}"

# ── Parquet source ───────────────────────────────────────────────────────────
PARQUET_FILE="${PARQUET_FILE:-${REPO_ROOT}/test-00000-of-00001.parquet}"

# ── Output JSON (built in step 1) ────────────────────────────────────────────
ANN_OUT="${ANN_OUT:-${REPO_ROOT}/eval_videomme_with_paths.json}"

# ── Config file (copy + edit config.example.json; see note below) ────────────
# Must have audio_enabled=true and Vertex/API credentials set.
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/avp/config.json}"

# ── Eval output directory ────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/avp/out_videomme_aavp}"

# ── Parallelism & timing ─────────────────────────────────────────────────────
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_TURNS="${MAX_TURNS:-3}"
TIMEOUT="${TIMEOUT:-2000}"   # seconds per sample (audio adds ffmpeg + extra API call)
# Seconds to sleep between samples within each worker (reduces API call rate).
SLEEP_BETWEEN_SAMPLES="${SLEEP_BETWEEN_SAMPLES:-0}"

# ── Optional: limit evaluation to first N unique videos ──────────────────────
# Set MAX_VIDEOS=100 to run a 100-video pilot (≈300 questions for VideoMME).
# Leave unset or 0 to evaluate the full dataset.
MAX_VIDEOS="${MAX_VIDEOS:-0}"

# ── Optional: reference annotation for numeric durations ─────────────────────
MERGE_DURATION_ARG=""
REF_JSON="${REPO_ROOT}/avp/eval_anno/eval_videomme.json"
if [[ -f "${REF_JSON}" ]]; then
    MERGE_DURATION_ARG="--merge-duration ${REF_JSON}"
fi

# ────────────────────────────────────────────────────────────────────────────
echo "=========================================="
echo " VideoMME AAVP Evaluation"
echo "=========================================="
echo "  REPO_ROOT   : ${REPO_ROOT}"
echo "  VIDEO_ROOT  : ${VIDEO_ROOT}"
echo "  PARQUET     : ${PARQUET_FILE}"
echo "  ANN_OUT     : ${ANN_OUT}"
echo "  CONFIG_FILE : ${CONFIG_FILE}"
echo "  OUT_DIR     : ${OUT_DIR}"
echo "  NUM_WORKERS : ${NUM_WORKERS}"
echo "  MAX_TURNS   : ${MAX_TURNS}"
echo "  TIMEOUT     : ${TIMEOUT}s"
echo "  MAX_VIDEOS  : ${MAX_VIDEOS:-all}"
echo "=========================================="

# ── Guard: video root must exist ─────────────────────────────────────────────
if [[ ! -d "${VIDEO_ROOT}" ]]; then
    echo "ERROR: VIDEO_ROOT does not exist: ${VIDEO_ROOT}" >&2
    echo "  Set VIDEO_ROOT to the directory containing extracted .mp4 files." >&2
    exit 1
fi

# ── Guard: config must exist ─────────────────────────────────────────────────
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: CONFIG_FILE not found: ${CONFIG_FILE}" >&2
    echo "  Copy avp/config.example.json, set your credentials, and enable audio:" >&2
    echo '    "audio_enabled": true' >&2
    exit 1
fi

# ── Step 1: Build eval JSON ───────────────────────────────────────────────────
echo ""
echo "── Step 1: Building eval JSON from parquet ──"

MAX_VIDEOS_ARG=""
if [[ "${MAX_VIDEOS:-0}" -gt 0 ]]; then
    MAX_VIDEOS_ARG="--max-videos ${MAX_VIDEOS}"
fi

python scripts/build_videomme_eval_json.py \
    --parquet  "${PARQUET_FILE}" \
    --video-root "${VIDEO_ROOT}" \
    --output   "${ANN_OUT}" \
    ${MERGE_DURATION_ARG} \
    ${MAX_VIDEOS_ARG}

echo "Annotation JSON ready: ${ANN_OUT}"

# ── Step 2: Run parallel evaluation ──────────────────────────────────────────
echo ""
echo "── Step 2: Running avp.eval_parallel ──"
SLEEP_ARG=""
if [[ "${SLEEP_BETWEEN_SAMPLES:-0}" -gt 0 ]]; then
    SLEEP_ARG="--sleep-between-samples ${SLEEP_BETWEEN_SAMPLES}"
fi

python -m avp.eval_parallel \
    --ann         "${ANN_OUT}" \
    --out         "${OUT_DIR}" \
    --config      "${CONFIG_FILE}" \
    --max-turns   "${MAX_TURNS}" \
    --num-workers "${NUM_WORKERS}" \
    --timeout     "${TIMEOUT}" \
    ${SLEEP_ARG}

echo ""
echo "=========================================="
echo " Evaluation complete."
echo " Results : ${OUT_DIR}/results.jsonl"
echo " Summary : ${OUT_DIR}/summary.json"
echo "=========================================="
