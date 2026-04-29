#!/bin/bash

# ── Job metadata ─────────────────────────────────────────────────────────────
#SBATCH --job-name=avp_videomme_aavp
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --time=3-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # 14 eval workers + 2 headroom for ffmpeg
#SBATCH --mem=64g       
#SBATCH --gres=gpu:1          # ~6 GB per worker; lower to 64g if needed

#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alshen@email.unc.edu

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
# conda create -n avp python=3.10 -y && conda activate avp
# conda install -c conda-forge ffmpeg && pip install -r requirements.txt
conda activate avp

# Prefer this env's python/pip over module paths (avoids wrong interpreter on compute nodes).
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# A non-empty PYTHONPATH can pick up a stray local "google" package and break
# "from google import genai" with: ImportError ... (unknown location)
unset PYTHONPATH

# Ensure google-genai is present for *this* interpreter (login vs compute mismatch).
python -m pip install -q --no-cache-dir "google-genai>=0.4.0" "google-auth>=2.0.0"

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

# Directory containing extracted VideoMME .mp4 files.
export VIDEO_ROOT="${VIDEO_ROOT:-/users/a/l/alshen/AVPA/ActiveVideoPerceptionA/videomme_data/data}"

# Config JSON — default to bundled AAVP template (audio_enabled: true).
# Override CONFIG_FILE before sbatch if you use a merged file.
export CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/avp/config.aavp.json}"

# Annotation JSON built from the parquet (produced in step 1 of the eval run).
export ANN_OUT="${ANN_OUT:-${REPO_ROOT}/eval_videomme_with_paths.json}"

# Output directory: use /work for large result files.
export OUT_DIR="${OUT_DIR:-/users/a/l/alshen/AVPA/ActiveVideoPerceptionA/avpa_out}"

# ── Parallelism & rate control ────────────────────────────────────────────────
# 2 workers is the safe default for Gemini 2.5 Flash on a free/trial API key.
# Raise to 4–8 once you've confirmed you have enough quota headroom.
export NUM_WORKERS="${NUM_WORKERS:-2}"

# Lower MAX_TURNS to reduce API calls per sample (each turn = 1–2 Gemini calls).
export MAX_TURNS="${MAX_TURNS:-3}"

export TIMEOUT="${TIMEOUT:-2000}"    # seconds per sample; audio adds ~20-60s overhead

# Pause between samples within each worker (seconds).  5 s gives ~24 req/min
# across 2 workers, well within Flash's 10 RPM free tier; set to 0 for paid quota.
export SLEEP_BETWEEN_SAMPLES="${SLEEP_BETWEEN_SAMPLES:-5}"

# Limit to the first N unique videos (300 questions for VideoMME).
# Set to 0 to evaluate the full dataset.
export MAX_VIDEOS="${MAX_VIDEOS:-100}"

# ── Setup ─────────────────────────────────────────────────────────────────────
# SLURM creates the log file but NOT the parent directory.
mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OUT_DIR}"

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "============================================"
echo " Job ${SLURM_JOB_ID} on $(hostname)"
echo " Started: $(date)"
echo " Node: ${SLURMD_NODENAME:-unknown}"
echo "============================================"
echo " REPO_ROOT   : ${REPO_ROOT}"
echo " VIDEO_ROOT  : ${VIDEO_ROOT}"
echo " CONFIG_FILE : ${CONFIG_FILE}"
echo " ANN_OUT     : ${ANN_OUT}"
echo " OUT_DIR     : ${OUT_DIR}"
echo " NUM_WORKERS : ${NUM_WORKERS}"
echo " MAX_TURNS   : ${MAX_TURNS}"
echo " TIMEOUT     : ${TIMEOUT}s per sample"
echo " MAX_VIDEOS  : ${MAX_VIDEOS}"
echo " SLEEP/SAMPLE: ${SLEEP_BETWEEN_SAMPLES}s"
echo "============================================"
python --version
ffmpeg -version 2>&1 | head -1
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
bash scripts/run_videomme_eval.sh

echo ""
echo "============================================"
echo " Job finished: $(date)"
echo " Results: ${OUT_DIR}/summary.json"
echo "============================================"
