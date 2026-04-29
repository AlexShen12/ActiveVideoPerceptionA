#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_omnivideobench_eval.sh  (ActiveVideoPerceptionA — audio + video)
#
# End-to-end OmniVideoBench evaluation with AAVP (audio-visual pipeline).
#
# Usage:
#
#   cp .env.example .env   # set GEMINI_API_KEY (and optional Vertex vars)
#
#   export OMNIVIDEO_INPUT=/path/to/data.json   # or data.parquet
#   export OMNIVIDEO_VIDEO_ROOT=/path/to/videos
#   bash scripts/run_omnivideobench_eval.sh
#
# Credentials are read from ${REPO_ROOT}/.env by default. Override with:
#   export ENV_FILE=/path/to/custom.env
#
# Audio note:
#   OmniVideoBench is designed to require both audio and visual reasoning.
#   Set audio_enabled: true in avp/config.json (see avp/config.aavp.json for a
#   template) so the AAVP Observer actually processes the audio track.
#
# The script:
#   1. Builds eval_omnivideo_with_paths.json from the annotation + video root.
#   2. Runs avp.eval_parallel on that JSON.
#
# All paths are resolved relative to the repo root (ActiveVideoPerceptionA/).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Repo root ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Secrets & env overrides from .env ────────────────────────────────────────
# shellcheck source=dotenv.sh
source "${SCRIPT_DIR}/dotenv.sh"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
    dotenv_load "${ENV_FILE}"
    echo "(loaded ${ENV_FILE})"
else
    echo "NOTE: No .env at ${ENV_FILE} — set GEMINI_API_KEY in the shell or add a .env file." >&2
fi

# ── Required: OmniVideoBench annotation file ─────────────────────────────────
# Either a nested JSON (data.json) or a row-per-question parquet (data.parquet).
OMNIVIDEO_INPUT="${OMNIVIDEO_INPUT:-}"
if [[ -z "${OMNIVIDEO_INPUT}" ]]; then
    echo "ERROR: OMNIVIDEO_INPUT is not set." >&2
    echo "  export OMNIVIDEO_INPUT=/path/to/data.json  (or data.parquet)" >&2
    echo "  Run scripts/install_omnivideobench.sh first if you haven't downloaded the dataset." >&2
    exit 1
fi

# ── Required: video files ────────────────────────────────────────────────────
OMNIVIDEO_VIDEO_ROOT="${OMNIVIDEO_VIDEO_ROOT:-}"
if [[ -z "${OMNIVIDEO_VIDEO_ROOT}" ]]; then
    echo "ERROR: OMNIVIDEO_VIDEO_ROOT is not set." >&2
    echo "  export OMNIVIDEO_VIDEO_ROOT=/path/to/directory/containing/video_*.mp4" >&2
    exit 1
fi

# ── Output JSON (built in step 1) ────────────────────────────────────────────
ANN_OUT="${ANN_OUT:-${REPO_ROOT}/eval_omnivideo_with_paths.json}"

# ── Config file (must have audio_enabled: true) ───────────────────────────────
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/avp/config.json}"

# ── Eval output directory ─────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/avp/out_omnivideo_aavp}"

# ── Parallelism & timing ──────────────────────────────────────────────────────
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_TURNS="${MAX_TURNS:-3}"
TIMEOUT="${TIMEOUT:-2000}"   # seconds per sample; audio adds ~20-60s overhead

# Pause between samples within each worker (seconds).
# 5s gives ~24 req/min across 2 workers — safe for Flash's 10 RPM free tier.
# Set to 0 on paid quota.
SLEEP_BETWEEN_SAMPLES="${SLEEP_BETWEEN_SAMPLES:-5}"

# ── Video limit ───────────────────────────────────────────────────────────────
# Default: first 30 videos (each may have 1–3 questions).
MAX_VIDEOS="${MAX_VIDEOS:-30}"

# ── Duration bucket (OmniVideoBench paper Fig. 3) ─────────────────────────────
# all | short | medium | long | ultralong
# "ultralong" = clips with duration *strictly* over 10 minutes — then take the
# first MAX_VIDEOS of those in file order.  Default: all (no duration filter).
OMNIVIDEO_LENGTH_BUCKET="${OMNIVIDEO_LENGTH_BUCKET:-all}"

# ─────────────────────────────────────────────────────────────────────────────
echo "=========================================="
echo " OmniVideoBench AAVP Evaluation"
echo "=========================================="
echo "  REPO_ROOT            : ${REPO_ROOT}"
echo "  OMNIVIDEO_INPUT      : ${OMNIVIDEO_INPUT}"
echo "  OMNIVIDEO_VIDEO_ROOT : ${OMNIVIDEO_VIDEO_ROOT}"
echo "  ANN_OUT              : ${ANN_OUT}"
echo "  CONFIG_FILE          : ${CONFIG_FILE}"
echo "  OUT_DIR              : ${OUT_DIR}"
echo "  NUM_WORKERS          : ${NUM_WORKERS}"
echo "  MAX_TURNS            : ${MAX_TURNS}"
echo "  TIMEOUT              : ${TIMEOUT}s"
echo "  MAX_VIDEOS           : ${MAX_VIDEOS:-all}"
echo "  LENGTH_BUCKET        : ${OMNIVIDEO_LENGTH_BUCKET}"
echo "  SLEEP/SAMPLE         : ${SLEEP_BETWEEN_SAMPLES}s"
echo "=========================================="

# ── Guard: video root must exist ─────────────────────────────────────────────
if [[ ! -d "${OMNIVIDEO_VIDEO_ROOT}" ]]; then
    echo "ERROR: OMNIVIDEO_VIDEO_ROOT does not exist: ${OMNIVIDEO_VIDEO_ROOT}" >&2
    exit 1
fi

# ── Guard: config must exist and note about audio ────────────────────────────
if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: CONFIG_FILE not found: ${CONFIG_FILE}" >&2
    echo "  Copy avp/config.aavp.json to avp/config.json and set your credentials." >&2
    echo "  Ensure audio_enabled: true is set for the audio-visual pipeline." >&2
    exit 1
fi

# ── Step 1: Build eval JSON ───────────────────────────────────────────────────
echo ""
echo "── Step 1: Building eval JSON from OmniVideoBench annotations ──"

MAX_VIDEOS_ARG=""
if [[ "${MAX_VIDEOS:-0}" -gt 0 ]]; then
    MAX_VIDEOS_ARG="--max-videos ${MAX_VIDEOS}"
fi

LENGTH_BUCKET_ARG=(--length-bucket "${OMNIVIDEO_LENGTH_BUCKET}")

# Choose --input or --parquet based on file extension.
INPUT_ARG=""
case "${OMNIVIDEO_INPUT}" in
    *.parquet) INPUT_ARG="--parquet ${OMNIVIDEO_INPUT}" ;;
    *)         INPUT_ARG="--input ${OMNIVIDEO_INPUT}" ;;
esac

python scripts/build_omnivideobench_eval_json.py \
    ${INPUT_ARG} \
    --video-root "${OMNIVIDEO_VIDEO_ROOT}" \
    --output     "${ANN_OUT}" \
    "${LENGTH_BUCKET_ARG[@]}" \
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
