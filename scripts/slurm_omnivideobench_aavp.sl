#!/bin/bash

# ── Job metadata ─────────────────────────────────────────────────────────────
#SBATCH --job-name=avp_omnivideobench_aavp
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --time=3-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # 2 eval workers + 14 headroom for ffmpeg/audio
#SBATCH --mem=64g
#SBATCH --gres=gpu:1

#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access

# ── Email notifications ───────────────────────────────────────────────────────
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alshen@unc.edu

# ────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Modules (Longleaf uses "module add", not "module load") ──────────────────
module purge
module add anaconda/2024.02      # provides conda + base Python

# ffmpeg is available via conda-forge (installed in the avp environment).
# If your site has a site-wide ffmpeg module, add it here instead:
# module add ffmpeg

# ── Conda environment ─────────────────────────────────────────────────────────
# Non-interactive batch jobs need conda.sh before "conda activate".
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
# Activate the avp conda environment created per the project README:
#   conda create -n avp python=3.10 -y && conda activate avp
#   conda install -c conda-forge ffmpeg && pip install -r requirements.txt
conda activate avp

# Prefer this env's python/pip over module paths (avoids wrong interpreter on compute nodes).
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# A non-empty PYTHONPATH can pick up a stray local "google" package and break
# "from google import genai" with: ImportError ... (unknown location)
unset PYTHONPATH

# Ensure required packages are present for this interpreter.
python -m pip install -q --no-cache-dir "google-genai>=0.4.0" "google-auth>=2.0.0"

# Ensure HF packages are available (needed by build_omnivideobench_eval_json.py --parquet).
python -m pip install -q --no-cache-dir \
    "huggingface_hub>=0.22.0" "datasets>=2.18.0" "pandas>=1.5.0" "pyarrow>=12.0.0"

# Under sbatch the script is copied to /var/spool/slurmd/.../slurm_script — use
# SLURM_SUBMIT_DIR (cwd where you ran sbatch; should be the repo root).
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
SCRIPT_DIR="${REPO_ROOT}/scripts"

cd "${REPO_ROOT}"

source "${SLURM_SUBMIT_DIR:-${PWD}}/scripts/dotenv.sh"
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
    dotenv_load "${ENV_FILE}"
    echo "(loaded credentials from ${ENV_FILE})"
else
    echo "WARNING: No .env at ${ENV_FILE} — set ENV_FILE or create .env with GEMINI_API_KEY." >&2
fi

# ── Paths ─────────────────────────────────────────────────────────────────────
# /work is Longleaf's high-throughput scratch — use it for video I/O and output.
# Replace <onyen> with your UNC ONYEN.

# OmniVideoBench annotation file (nested JSON or row-per-question parquet).
# Download via: bash scripts/install_omnivideobench.sh
export OMNIVIDEO_INPUT="${OMNIVIDEO_INPUT:-/users/a/l/alshen/omnivideobench_data/data.json}"

# Directory containing video_*.mp4 files.
export OMNIVIDEO_VIDEO_ROOT="${OMNIVIDEO_VIDEO_ROOT:-/users/a/l/alshen/omnivideobench_data/videos}"

# Config JSON with credentials and audio_enabled: true (see avp/config.aavp.json).
# IMPORTANT: OmniVideoBench is designed for audio-visual reasoning. Copy
# avp/config.aavp.json → avp/config.json and set audio_enabled: true.
export CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/avp/config.json}"

# Annotation JSON built from step 1 of the eval run.
export ANN_OUT="${ANN_OUT:-${REPO_ROOT}/eval_omnivideo_with_paths.json}"

# Output directory: use /work for large result files.
export OUT_DIR="${OUT_DIR:-/users/a/l/alshen/AVPA/ActiveVideoPerceptionA/avpa_out_omnivideo}"

# ── Parallelism & rate control ────────────────────────────────────────────────
# 2 workers is the safe default for Gemini 2.5 Flash on a free/trial API key.
# Raise to 4–8 once you've confirmed sufficient quota headroom.
export NUM_WORKERS="${NUM_WORKERS:-2}"

# Lower MAX_TURNS to reduce API calls per sample (each turn = 1–2 Gemini calls).
export MAX_TURNS="${MAX_TURNS:-3}"

export TIMEOUT="${TIMEOUT:-2000}"    # seconds per sample; audio adds ~20-60s overhead

# Pause between samples within each worker (seconds).
# 5s gives ~24 req/min across 2 workers — safe for Flash's 10 RPM free tier.
# Set to 0 on paid quota.
export SLEEP_BETWEEN_SAMPLES="${SLEEP_BETWEEN_SAMPLES:-5}"

# Limit to the first N unique videos.
# Each OmniVideoBench video typically has 1–3 QA pairs.
# Set to 0 to evaluate the full dataset.
export MAX_VIDEOS="${MAX_VIDEOS:-30}"

# Paper duration bucket (Fig. 3): all | short | medium | long | ultralong
# "ultralong" = only clips *strictly* longer than 10 min, then first MAX_VIDEOS
# in annotation order.  Default all. Example:
#   export OMNIVIDEO_LENGTH_BUCKET=ultralong
export OMNIVIDEO_LENGTH_BUCKET="${OMNIVIDEO_LENGTH_BUCKET:-all}"

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUT_DIR}"

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "============================================"
echo " Job ${SLURM_JOB_ID} on $(hostname)"
echo " Started: $(date)"
echo " Node: ${SLURMD_NODENAME:-unknown}"
echo "============================================"
echo " REPO_ROOT            : ${REPO_ROOT}"
echo " OMNIVIDEO_INPUT      : ${OMNIVIDEO_INPUT}"
echo " OMNIVIDEO_VIDEO_ROOT : ${OMNIVIDEO_VIDEO_ROOT}"
echo " CONFIG_FILE          : ${CONFIG_FILE}"
echo " ANN_OUT              : ${ANN_OUT}"
echo " OUT_DIR              : ${OUT_DIR}"
echo " NUM_WORKERS          : ${NUM_WORKERS}"
echo " MAX_TURNS            : ${MAX_TURNS}"
echo " TIMEOUT              : ${TIMEOUT}s per sample"
echo " MAX_VIDEOS           : ${MAX_VIDEOS}"
echo " SLEEP/SAMPLE         : ${SLEEP_BETWEEN_SAMPLES}s"
echo "============================================"
python --version
ffmpeg -version 2>&1 | head -1
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
bash scripts/run_omnivideobench_eval.sh

echo ""
echo "============================================"
echo " Job finished: $(date)"
echo " Results: ${OUT_DIR}/summary.json"
echo "============================================"
