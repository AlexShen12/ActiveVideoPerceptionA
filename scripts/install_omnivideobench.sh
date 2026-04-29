#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# install_omnivideobench.sh
#
# Downloads the OmniVideoBench dataset from HuggingFace into a local directory
# and installs the Python packages required to read/use it.
#
# Prerequisites
# ─────────────────────────────────────────────────────────────────────────────
#   1. Accept the dataset license on HuggingFace and obtain an API token:
#        https://huggingface.co/datasets/NJU-LINK/OmniVideoBench
#
#   2. Authenticate — pick ONE method:
#        export HF_TOKEN="hf_..."           # ← preferred for batch/Slurm jobs
#        huggingface-cli login              # ← interactive, writes ~/.huggingface/token
#
#   3. The dataset snapshot contains large video files (~tens of GB).
#      Make sure OMNIVIDEO_LOCAL_DIR points to a scratch filesystem with enough
#      space (e.g. Longleaf /work or /proj).
#
# Usage
# ─────────────────────────────────────────────────────────────────────────────
#   # Defaults (downloads to ./omnivideobench_data next to this script's repo root)
#   bash scripts/install_omnivideobench.sh
#
#   # Custom destination
#   OMNIVIDEO_LOCAL_DIR=/work/<onyen>/omnivideobench bash scripts/install_omnivideobench.sh
#
# After this script completes, the following env vars are printed for you to
# export when running the evaluation:
#
#   OMNIVIDEO_INPUT    — path to the annotation JSON (data.json or derived)
#   OMNIVIDEO_VIDEO_ROOT — directory containing video_*.mp4 files
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Destination ───────────────────────────────────────────────────────────────
OMNIVIDEO_LOCAL_DIR="${OMNIVIDEO_LOCAL_DIR:-${REPO_ROOT}/omnivideobench_data}"

HF_REPO="NJU-LINK/OmniVideoBench"

echo "========================================================"
echo " OmniVideoBench installer"
echo "========================================================"
echo " HF repo       : ${HF_REPO}"
echo " Local dir     : ${OMNIVIDEO_LOCAL_DIR}"
echo "========================================================"

# ── Python deps ──────────────────────────────────────────────────────────────
echo ""
echo "── Installing Python dependencies ──"
pip install -q -r "${SCRIPT_DIR}/requirements-omnivideobench.txt"
echo "   Dependencies installed."

# ── Token check ──────────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
    # Fall back to the token stored by huggingface-cli login
    CACHED_TOKEN="${HOME}/.huggingface/token"
    if [[ -f "${CACHED_TOKEN}" ]]; then
        export HF_TOKEN="$(cat "${CACHED_TOKEN}")"
        echo "   Using cached HuggingFace token from ${CACHED_TOKEN}."
    else
        echo ""
        echo "ERROR: No HuggingFace token found." >&2
        echo "  Either:" >&2
        echo "    export HF_TOKEN=hf_..." >&2
        echo "    or run: huggingface-cli login" >&2
        echo "  Then accept the dataset license at:" >&2
        echo "    https://huggingface.co/datasets/${HF_REPO}" >&2
        exit 1
    fi
fi

# ── Download ──────────────────────────────────────────────────────────────────
mkdir -p "${OMNIVIDEO_LOCAL_DIR}"

echo ""
echo "── Downloading dataset snapshot (this may take a long time for the videos) ──"

# Enable hf_transfer for faster multi-part downloads if available.
python - <<PYEOF
import os, sys

# hf_transfer speeds up large file downloads when installed
try:
    import hf_transfer  # noqa: F401
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    print("   hf_transfer enabled for faster download.")
except ImportError:
    pass

from huggingface_hub import snapshot_download

token = os.environ.get("HF_TOKEN")
local_dir = "${OMNIVIDEO_LOCAL_DIR}"

repo_id = "${HF_REPO}"
print(f"   Downloading {repr(repo_id)} → {local_dir}")
print("   This may take a while for large video files...")
sys.stdout.flush()

snapshot_download(
    repo_id="${HF_REPO}",
    repo_type="dataset",
    local_dir=local_dir,
    token=token,
    # Resume partial downloads automatically
    local_dir_use_symlinks=False,
)

print("   Snapshot downloaded.")
PYEOF

# ── Detect video root and annotation files ────────────────────────────────────
echo ""
echo "── Detecting data layout ──"

# Find the directory that contains .mp4 files (one level deep search).
VIDEO_ROOT=""
while IFS= read -r candidate; do
    dir="$(dirname "${candidate}")"
    if [[ -n "${dir}" && "${dir}" != "." ]]; then
        VIDEO_ROOT="${dir}"
        break
    fi
done < <(find "${OMNIVIDEO_LOCAL_DIR}" -maxdepth 4 -name "*.mp4" | head -1)

if [[ -z "${VIDEO_ROOT}" ]]; then
    VIDEO_ROOT="${OMNIVIDEO_LOCAL_DIR}"
    echo "   WARNING: No .mp4 files found under ${OMNIVIDEO_LOCAL_DIR}."
    echo "   The dataset may not include video files in the snapshot, or they may"
    echo "   live in a subdirectory not yet extracted. Set OMNIVIDEO_VIDEO_ROOT manually."
else
    echo "   Videos detected at: ${VIDEO_ROOT}"
fi

# Find the primary annotation file — prefer data.json, fall back to data.parquet.
ANNOTATION_FILE=""
for candidate in \
    "${OMNIVIDEO_LOCAL_DIR}/data.json" \
    "${OMNIVIDEO_LOCAL_DIR}/data/data.json" \
    "${OMNIVIDEO_LOCAL_DIR}/data.parquet" \
    "${OMNIVIDEO_LOCAL_DIR}/data/data.parquet" \
; do
    if [[ -f "${candidate}" ]]; then
        ANNOTATION_FILE="${candidate}"
        break
    fi
done

if [[ -z "${ANNOTATION_FILE}" ]]; then
    # Last-resort: find any .json or .parquet
    ANNOTATION_FILE="$(find "${OMNIVIDEO_LOCAL_DIR}" -maxdepth 3 \( -name "*.json" -o -name "*.parquet" \) | head -1)"
fi

echo "   Annotation file: ${ANNOTATION_FILE:-NOT FOUND}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo " Download complete."
echo ""
echo " Next steps — export these before running evaluation:"
echo ""
echo "   export OMNIVIDEO_INPUT=\"${ANNOTATION_FILE}\""
echo "   export OMNIVIDEO_VIDEO_ROOT=\"${VIDEO_ROOT}\""
echo ""
echo " Then run (from the repo root, ActiveVideoPerceptionA/):"
echo "   bash scripts/run_omnivideobench_eval.sh"
echo ""
echo " Or submit the Slurm job:"
echo "   sbatch scripts/slurm_omnivideobench_aavp.sl"
echo ""
echo " NOTE: OmniVideoBench is audio-visual. run_omnivideobench_eval.sh defaults to"
echo " avp/config.aavp.json (audio_enabled: true); merge your API credentials there,"
echo " or export CONFIG_FILE=... if you use another JSON."
echo "========================================================"
