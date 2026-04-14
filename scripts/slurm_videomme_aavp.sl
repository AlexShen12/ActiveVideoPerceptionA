#!/bin/bash

# ── Job metadata ─────────────────────────────────────────────────────────────
#SBATCH --job-name=avp_videomme_aavp
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

#SBATCH --time=3-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16       # 14 eval workers + 2 headroom for ffmpeg
#SBATCH --mem=64g                 # ~6 GB per worker; lower to 64g if needed

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
# Activate the avp conda environment created per the project README:
#   conda create -n avp python=3.10 -y && conda activate avp
#   conda install -c conda-forge ffmpeg && pip install -r requirements.txt
conda activate avp

# ── Repo root ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Credentials from .env ─────────────────────────────────────────────────────
# Place a .env in the repo root (never commit it; see .env.example), e.g.:
#   GEMINI_API_KEY=...
# Optional: GOOGLE_APPLICATION_CREDENTIALS, VERTEX_PROJECT, VERTEX_LOCATION
# Override path: sbatch --export=ALL,ENV_FILE=/nas/longleaf/home/$USER/.secrets/avp.env ...
# shellcheck source=dotenv.sh
source "${SCRIPT_DIR}/dotenv.sh"
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
export VIDEO_ROOT="${VIDEO_ROOT:-/work/users/a/l/<onyen>/videomme/videos}"

# Config JSON with credentials and audio_enabled=true (see avp/config.aavp.json).
export CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/avp/config.json}"

# Annotation JSON built from the parquet (produced in step 1 of the eval run).
export ANN_OUT="${ANN_OUT:-${REPO_ROOT}/eval_videomme_with_paths.json}"

# Output directory: use /work for large result files.
export OUT_DIR="${OUT_DIR:-/work/users/<f>/<i>/<onyen>/avp_out/videomme_aavp}"

# ── Parallelism ───────────────────────────────────────────────────────────────
# Reserve 2 cores for ffmpeg subprocesses; each eval worker gets one core.
_CORES="${SLURM_CPUS_PER_TASK:-16}"
export NUM_WORKERS="${NUM_WORKERS:-$(( _CORES > 2 ? _CORES - 2 : _CORES ))}"
export MAX_TURNS="${MAX_TURNS:-3}"
export TIMEOUT="${TIMEOUT:-2000}"    # seconds per sample; audio adds ~20-60s overhead

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
echo " NUM_WORKERS : ${NUM_WORKERS}  (of ${_CORES} allocated cores)"
echo " MAX_TURNS   : ${MAX_TURNS}"
echo " TIMEOUT     : ${TIMEOUT}s per sample"
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
