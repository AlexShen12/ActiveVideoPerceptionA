
## Table of Contents
- [Highlights](#highlights)
- [Setup](#setup)
- [Evaluation](#evaluation)
- [VideoMME](#videomme)
- [OmniVideoBench](#omnivideobench)
- [Citation](#citation)

---

## Highlights


**Audio-Extended Active Video Perception (AVPA / AAVP)** builds on **AVP**: it still treats video as an interactive environment and gathers compact, query-relevant evidence through **active** perception, and adds a **time-aligned speech prior** (pre-observation ASR) that **informs planning**, **pairs with each visual pass**, and supports **multimodal reflection**.

**Key ideas:**
- Treat long videos as **interactive environments**
- Iteratively **plan → observe → reflect** using **vision and speech**
- **Adaptively** choose **where** and **how** the VLM watches each round
- Check **cross-modal** sufficiency for **grounding** and **faithful** answers

AVPA extends AVP toward **audio-visual** and long-form video QA without replacing **iterative visual** observation.


## Setup

### 1. Create Conda Environment
Create and activate a fresh conda environment with the required Python version:

```bash
conda create -n avp python=3.10 -y
conda activate avp
```

### 2. Install System Dependencies

```bash
conda install -c conda-forge ffmpeg
ffmpeg -version
pip install -r requirements.txt
```

### 3. Credentials and config

Merge your Gemini credentials into a config JSON (start from `avp/config.example.json` for video-only AVP, or `avp/config.aavp.json` for **AVPA / AAVP** with audio).

**Vertex AI:** Set `project` and `location` in the config.

**Google AI Studio:** Set the `GEMINI_API_KEY` environment variable or `"api_key"` in the config.

The benchmark driver scripts load a repo-root `.env` if present (see `scripts/dotenv.sh`). You can export the same variables in your shell instead.

### 4. Optional: annotation format for custom benchmarks

For ad-hoc JSON annotations, see `avp/eval_anno/` and point your run script at the file you prepared.

---
## Evaluation

### Generic annotation JSON

Set these in `avp/parrelel_run.sh` before running:

- **ANNOTATION_FILE** – Path to your annotation JSON
- **OUTPUT_DIR** – Directory where results will be written
- **CONFIG_FILE** – Path to your config JSON (e.g. `config.example.json`)

Optional (with defaults):

- **LIMIT** – Max number of samples (omit for no limit)
- **MAX_TURNS** – Max plan–execute cycles per sample (default: 3)
- **NUM_WORKERS** – Number of parallel workers (default: 4)
- **TIMEOUT** – Timeout per sample in seconds, prevent API failure (omit for no timeout, suggested to use)

Example:

```bash
bash avp/parrelel_run.sh
```

---

## VideoMME

These steps run **AVPA (AAVP)** on [VideoMME](https://arxiv.org/abs/2401.09115) using `scripts/run_videomme_eval.sh`: build a path-resolved eval JSON from the official **parquet** split, then launch `python -m avp.eval_parallel`.

### 1. Data

1. Download the **VideoMME** annotation parquet and video files from the benchmark authors (e.g. [Hugging Face — `lmms-lab/Video-MME`](https://huggingface.co/datasets/lmms-lab/Video-MME)).
2. Extract videos so each file is named `<videoID>.mp4` (default column `videoID`) directly under one directory, e.g. `/path/to/videomme/videos/fFjv93ACGo8.mp4`.

### 2. Python extras

Parquet IO needs **pandas** and **pyarrow**:

```bash
pip install pandas pyarrow
```

### 3. Config

Use `avp/config.aavp.json` (default for the script) or a copy with your credentials and `"audio_enabled": true` for the full audio–visual pipeline.

### 4. Run

```bash
export VIDEO_ROOT=/path/to/videomme/videos
export PARQUET_FILE=/path/to/test-00000-of-00001.parquet   # if not using repo-root default
export CONFIG_FILE="${PWD}/avp/config.aavp.json"
bash scripts/run_videomme_eval.sh
```

**Useful environment overrides** (see `scripts/run_videomme_eval.sh`):

| Variable | Role |
| -------- | ---- |
| `VIDEO_ROOT` | Directory of `<videoID>.mp4` files (**required**) |
| `PARQUET_FILE` | Input parquet (default: `${REPO_ROOT}/test-00000-of-00001.parquet`) |
| `ANN_OUT` | Built JSON path (default: `eval_videomme_with_paths.json`) |
| `OUT_DIR` | Worker output (default: `avp/out_videomme_aavp`) |
| `NUM_WORKERS`, `MAX_TURNS`, `TIMEOUT` | Parallelism and per-sample limits |
| `MAX_VIDEOS` | Pilot on first *N* unique videos (`0` = full set) |
| `SLEEP_BETWEEN_SAMPLES` | Throttle API rate between items |

If `avp/eval_anno/eval_videomme.json` is present, durations are merged automatically. Results: `${OUT_DIR}/results.jsonl` and `summary.json`.

---

## OmniVideoBench

These steps run **AVPA (AAVP)** on [OmniVideoBench](https://huggingface.co/datasets/NJU-LINK/OmniVideoBench) (audio–visual QA). The eval script expects **`audio_enabled: true`** in your config.

### 1. Download the dataset

Accept the dataset license on Hugging Face and authenticate (`export HF_TOKEN=hf_...` or `huggingface-cli login`), then:

```bash
pip install -r scripts/requirements-omnivideobench.txt
bash scripts/install_omnivideobench.sh
```

By default files go to `./omnivideobench_data` (override with `OMNIVIDEO_LOCAL_DIR`). The installer prints suggested `OMNIVIDEO_INPUT` and `OMNIVIDEO_VIDEO_ROOT` paths when it finishes.

### 2. Config

Use `avp/config.aavp.json` or equivalent with **`"audio_enabled": true`** (the driver verifies this unless `VERIFY_OMNI_AUDIO=0`).

### 3. Run

```bash
export OMNIVIDEO_INPUT=/path/to/data.json        # or data.parquet
export OMNIVIDEO_VIDEO_ROOT=/path/to/videos      # directory with benchmark videos
export CONFIG_FILE="${PWD}/avp/config.aavp.json"
bash scripts/run_omnivideobench_eval.sh
```

**Useful environment overrides** (see `scripts/run_omnivideobench_eval.sh`):

| Variable | Role |
| -------- | ---- |
| `OMNIVIDEO_INPUT` | Nested JSON or parquet annotations (**required**) |
| `OMNIVIDEO_VIDEO_ROOT` | Video directory (**required**) |
| `ANN_OUT` | Built JSON (default: `eval_omnivideo_with_paths.json`) |
| `OUT_DIR` | Worker output (default: `avp/out_omnivideo_aavp`) |
| `MAX_VIDEOS` | Default `30` unique videos; set **`MAX_VIDEOS=0`** for the full benchmark (no cap) |
| `OMNIVIDEO_LENGTH_BUCKET` | `all` \| `short` \| `medium` \| `long` \| `ultralong` (paper duration bins) |
| `NUM_WORKERS`, `MAX_TURNS`, `TIMEOUT`, `SLEEP_BETWEEN_SAMPLES` | Parallelism and limits |

**Cluster jobs:** See `scripts/slurm_videomme_aavp.sl` and `scripts/slurm_omnivideobench_aavp.sl` as templates; adjust paths and partitions for your site.

**Aggregate sharded workers:** `python scripts/aggregate_worker_results.py <OUT_DIR> --write-merge`

---

## Citation

If you find the previous work useful, please cite:

```bibtex
@misc{wang2025activevideoperceptioniterative,
      title={Active Video Perception: Iterative Evidence Seeking for Agentic Long Video Understanding}, 
      author={Ziyang Wang and Honglu Zhou and Shijie Wang and Junnan Li and Caiming Xiong and Silvio Savarese and Mohit Bansal and Michael S. Ryoo and Juan Carlos Niebles},
      year={2025},
      eprint={2512.05774},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.05774}, 
}