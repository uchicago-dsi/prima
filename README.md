# prima
Polygenic Risk and Imaging Multimodal Assessment

## Pipeline Overview

The end-to-end workflow for training/inference is:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA ACQUISITION                                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. scrape_ibroker.py  → scrapes iBroker for study metadata                  │
│ 2. export.py          → requests exports from iBroker based on metadata     │
│ 3. sync_local.py      → syncs exported DICOMs to GPFS (from HIRO share)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA SOURCES (after acquisition)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Disk DICOMs     /gpfs/.../ChiMEC/MG/  → actual imaging data              │
│ 2. Phenotype CSV   Phenotype_ChiMEC_*.csv → labels (case/control, datedx)   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PREPROCESSING (preprocess.py)                                       │
│   Scans disk DICOMs → extracts metadata from DICOM headers                  │
│   Outputs: sot/views.parquet, sot/exams.parquet, out/manifest.parquet       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: EMIT CSV (preprocess.py emit-csv or automatic)                      │
│   Joins exams with phenotype CSV → creates Mirai-compatible manifest        │
│   Output: out/mirai_manifest.csv (only exams with labels)                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: INFERENCE (run_mirai_sharded.py)                                    │
│   Runs Mirai model on manifest → predictions                                │
│   Output: out/validation_output.csv                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Important**: The preprocessing pipeline works directly on disk DICOMs. Historical data
that exists on disk but is missing from iBroker (e.g., from earlier studies) will still
be processed and can be used for training, as long as patients have phenotype labels.

---

## Installation

### Main environment

```bash
# install micromamba if needed
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -r ~/micromamba 
source ~/.bashrc
micromamba config append channels conda-forge

# create + activate
micromamba create -y -f env.yaml
eval "$(micromamba shell hook -s bash)"
micromamba activate prima

# install torch/vision matching cu111 wheels (from the official index)
python -m pip install \
  torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html  # from "previous versions"

# your repo + mirai fork (zarr patch)
git clone --recursive git@github.com:uchicago-dsi/prima.git
cd prima
git submodule add git@github.com:annawoodard/Mirai vendor/mirai
git submodule update --init --recursive

# install your code + mirai
pip install -e .
pip install -r requirements.txt
pip install -r requirements-dev.txt  # optional: linting + notebook extras
pip install -e vendor/mirai

# fix GLIBCXX/libstdc++ compatibility issue for lifelines/pandas
# add conda/micromamba lib to LD_LIBRARY_PATH (required for survival analysis)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# add to ~/.bashrc or ~/.bash_profile to make permanent:
# echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Selenium environment (for export.py and scrape_ibroker.py)

The main `prima` environment has dependency conflicts with selenium/geckodriver. Use a separate environment:

```bash
micromamba create -y -f environment-selenium.yml
micromamba activate selenium-ff
pip install pandas requests lxml tqdm
```

### Development

Runtime dependencies live in `requirements.txt`; install `requirements-dev.txt` for tooling (Ruff, notebook helpers).

Before sending changes:
```bash
ruff format .
ruff check --fix .
```

---

## Data Acquisition

Data acquisition involves scraping metadata from iBroker, requesting exports, and syncing files to GPFS. These steps use the `selenium-ff` environment.

### scrape_ibroker.py

Scrapes the iBroker web interface to build a catalog of all imaging studies for ChiMEC patients. For each patient, it extracts both "Exported" studies (already requested) and "Available" studies (not yet exported), collecting metadata like study date, description, modality, accession number, and export status.

```bash
micromamba activate selenium-ff
export IBROKER_USERNAME=your_username
export IBROKER_PASSWORD=your_password
python scrape_ibroker.py
```

Output: `data/imaging_metadata.csv` — the complete catalog of studies in iBroker.

The script:
- Uses Selenium for login, then switches to fast HTTP requests for scraping
- Resumes from where it left off if interrupted
- Records which studies are already on disk

### export.py

Uses the metadata from `scrape_ibroker.py` to request exports from iBroker. It identifies studies that need to be downloaded (matching modality, not already exported, not already on disk) and submits export requests via the web interface.

```bash
micromamba activate selenium-ff
export IBROKER_USERNAME=your_username
export IBROKER_PASSWORD=your_password
python export.py --auto-confirm
```

Common options:
```bash
# check status without exporting anything
python export.py --status-only

# run continuously with 1-hour waits between cycles
python export.py --auto-confirm --loop-wait 1h

# export a different modality
python export.py --auto-confirm --modality CT

# include patients without genotyping data
python export.py --auto-confirm --no-genotyping-filter

# limit batch size per cycle
python export.py --auto-confirm --batch-size 50
```

The script:
- Merges patient phenotype data, study keys, and iBroker metadata
- Filters to target modality (default: MG) and excludes already-exported or on-disk studies
- Submits export requests in batches, saving progress to `data/export_state.csv`
- Optionally audits pending exports to update status (`--refresh-export-status`)

### HIRO → ChiMEC Data Transfer

After exports are requested, files appear on a HIRO CIFS share. Use `sync_local.py` to sync them to GPFS.

#### Mount the HIRO share

```bash
sudo mount -t cifs //cifs01uchadccd.uchad.uchospitals.edu/radiology/HIRO /mnt/uchad_samba \
  -o credentials=/home/annawoodard/creds,vers=3.0,noperm,uid=$(id -u),gid=$(id -g),file_mode=0660,dir_mode=0770
```

Credentials file format:
```
username=your_username
password=your_password
domain=UCHAD
```

#### Run sync_local.py

Configure source/destination paths at the top of `sync_local.py`:
```python
SRC_ROOT = Path("/mnt/uchad_samba/16352A/")
DST_ROOT = Path("/gpfs/data/huo-lab/Image/ChiMEC/MG")
```

```bash
# dry run first (no files moved or transferred)
python sync_local.py --dry-run --no-auto-restart

# real transfer with immediate source deletion (default behavior)
python sync_local.py

# queue for deletion instead of immediate delete
python sync_local.py --no-immediate-delete

# single sync pass (default auto-restarts every 2 minutes)
python sync_local.py --no-auto-restart
```

The script:
- Waits for exam stability (no file changes within 10 minutes) before transferring
- Uses rsync internally for efficient copying with progress tracking
- Deletes source files after successful transfer (or queues them to `_synced_and_queued_for_deletion/`)
- Cleans up empty patient directories older than 1 hour
- Auto-restarts every 2 minutes by default to pick up new exports
- Handles SIGINT/SIGTERM gracefully for clean shutdown

Logs are written to `sync.log` and stdout.

---

## Preprocessing

### Step 1: Run preprocessing

Scan disk DICOMs, extract metadata, select full-quad exams (L-CC, L-MLO, R-CC, R-MLO), and write Zarr cache:

```bash
# full preprocessing (discovery + zarr cache)
python preprocess.py preprocess \
  --raw /gpfs/data/huo-lab/Image/ChiMEC/MG \
  --workers 32

# summary only (skip zarr cache, useful for quick analysis)
python preprocess.py preprocess \
  --raw /gpfs/data/huo-lab/Image/ChiMEC/MG \
  --workers 32 \
  --summary
```

Outputs are written to `{raw}/sot/` and `{raw}/out/` by default:
- `sot/views.parquet` — individual DICOM views with metadata
- `sot/exams.parquet` — exam-level aggregated metadata
- `sot/dicom_tags.parquet` — all DICOM tags (wide format)
- `out/manifest.parquet` — Zarr URIs for each view
- `out/mirai_manifest.csv` — Mirai-compatible CSV with labels (auto-generated if --labels provided)

### Step 2: Generate Mirai CSV (if not done in step 1)

```bash
python preprocess.py emit-csv \
  --raw /gpfs/data/huo-lab/Image/ChiMEC/MG \
  --labels /gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/Phenotype_ChiMEC_2025Oct4.csv
```

### Incremental processing

To process only new exams (append to existing SoT tables):

```bash
python preprocess.py preprocess \
  --raw /gpfs/data/huo-lab/Image/ChiMEC/MG \
  --workers 32 \
  --incremental
```

### Monitoring and checkpoints

```bash
# monitor progress in a separate terminal
python preprocess.py monitor --interval 30

# list checkpoints
python preprocess.py checkpoint list

# show detailed status
python preprocess.py checkpoint status

# clean old checkpoints (>7 days)
python preprocess.py checkpoint clean --max-age-days 7
```

---

## Running Mirai

**Important**: Before running mirai, ensure LD_LIBRARY_PATH is set:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### GPU compatibility

```bash
# check your GPU compute capability
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Common GPUs and required PyTorch:
- **A100** (compute 8.0): needs CUDA 11+ with sm_80 → use `cu118`
- **V100** (compute 7.0): needs CUDA 10+ with sm_70 → use `cu102` or higher
- **P100** (compute 6.0): needs CUDA 9+ with sm_60 → use `cu102` or higher

```bash
# install/reinstall PyTorch with correct CUDA version (A100 example)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# verify GPU is detected
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# check compiled architectures if you get "no kernel image" errors
python -c "import torch; print(f'Compiled for: {torch.cuda.get_arch_list()}')"
```

### Single GPU

```bash
python scripts/main.py \
  --model_name mirai_full \
  --img_encoder_snapshot "/gpfs/data/huo-lab/Image/annawoodard/.mirai/snapshots/mgh_mammo_MIRAI_Base_May20_2019.p" \
  --transformer_snapshot "/gpfs/data/huo-lab/Image/annawoodard/.mirai/snapshots/mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020.p" \
  --calibrator_snapshot "/gpfs/data/huo-lab/Image/annawoodard/.mirai/snapshots/calibrators/MIRAI_FULL_PRED_RF.callibrator.p" \
  --batch_size 8 \
  --num_workers 4 \
  --dataset csv_mammo_risk_all_full_future \
  --img_mean 7047.99 \
  --img_size 1664 2048 \
  --img_std 12005.5 \
  --metadata_path /gpfs/data/huo-lab/Image/ChiMEC/MG/out/mirai_manifest.csv \
  --test \
  --prediction_save_path /gpfs/data/huo-lab/Image/ChiMEC/MG/out/validation_output.csv \
  --cuda
```

### Multi-GPU

```bash
python scripts/main.py \
  --model_name mirai_full \
  --img_encoder_snapshot "/gpfs/data/huo-lab/Image/annawoodard/.mirai/snapshots/mgh_mammo_MIRAI_Base_May20_2019.p" \
  --transformer_snapshot "/gpfs/data/huo-lab/Image/annawoodard/.mirai/snapshots/mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020.p" \
  --calibrator_snapshot "/gpfs/data/huo-lab/Image/annawoodard/.mirai/snapshots/calibrators/MIRAI_FULL_PRED_RF.callibrator.p" \
  --batch_size 128 \
  --num_workers 16 \
  --dataset csv_mammo_risk_all_full_future \
  --img_mean 7047.99 \
  --img_size 1664 2048 \
  --img_std 12005.5 \
  --metadata_path /gpfs/data/huo-lab/Image/ChiMEC/MG/out/mirai_manifest.csv \
  --test \
  --prediction_save_path /gpfs/data/huo-lab/Image/ChiMEC/MG/out/validation_output.csv \
  --cuda \
  --num_gpus 4 \
  --data_parallel
```

### Sharded parallel processing (recommended for large datasets)

```bash
python run_mirai_sharded.py \
  --max_samples_per_shard 500 \
  --metadata_path /gpfs/data/huo-lab/Image/ChiMEC/MG/out/mirai_manifest.csv \
  --prediction_save_path /gpfs/data/huo-lab/Image/ChiMEC/MG/out/validation_output.csv \
  --model_name mirai_full \
  --img_encoder_snapshot "/gpfs/data/huo-lab/Image/annawoodard/.mirai/snapshots/mgh_mammo_MIRAI_Base_May20_2019.p" \
  --transformer_snapshot "/gpfs/data/huo-lab/Image/annawoodard/.mirai/snapshots/mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020.p" \
  --calibrator_snapshot "/gpfs/data/huo-lab/Image/annawoodard/.mirai/snapshots/calibrators/Mirai_calibrator_mar12_2022.p" \
  --batch_size 8 \
  --num_workers 4 \
  --dataset csv_mammo_risk_all_full_future \
  --img_mean 7047.99 \
  --img_size 1664 2048 \
  --img_std 12005.5 \
  --test \
  --cuda
```

The sharded script will:
- Split the CSV into chunks (use `--num_shards N` or `--max_samples_per_shard N`)
- Submit one GPU job per chunk via SLURM
- Automatically collate results when jobs complete
- Save recovery metadata for resuming if the main process dies

Recovery and debugging:
```bash
# recover from a crashed run
python run_mirai_sharded.py --recover /path/to/mirai_shards/job_metadata.json

# debug with a small sample
python run_mirai_sharded.py --debug_max_samples 100 --num_shards 2 ...
```

### Performance tuning

Monitor GPU utilization:
```bash
watch -n 1 nvidia-smi
```

If process seems hung or slow:
```bash
# LOW GPU MEMORY (<1GB) = stuck in data loading, not GPU compute
# Try with NO multiprocessing first:
--num_workers 0 --batch_size 16

# If that works, gradually increase:
--num_workers 2 --batch_size 32
--num_workers 4 --batch_size 64

# add CUDA_LAUNCH_BLOCKING=1 to see exactly where it hangs
CUDA_LAUNCH_BLOCKING=1 python scripts/main.py ... --cuda
```

Common bottlenecks:
- **GPU memory <1GB, stuck at 0%**: data loading hang — use `--num_workers 0`
- **First batch very slow**: normal with large images (1664x2048) — can take 2-5 minutes initially
- **Low GPU utilization** (<50%): increase `--batch_size`
- **High GPU memory**: decrease `--batch_size`
- **Network filesystem**: multiprocessing can cause deadlocks — start with `--num_workers 0`

**Notes**:
- The LD_LIBRARY_PATH setting is required because pandas (a lifelines dependency) needs GLIBCXX_3.4.29
- Ignore "Restarting run from scratch" message during inference — it's always printed even when only running `--test`

---

## Analysis

### analyze_mirai.py

Compute per-horizon AUC and survival metrics (Uno's C-index, time-dependent AUC, integrated Brier score).

```bash
# basic usage (expects validation_output.csv and mirai_manifest.csv in same directory)
python analyze_mirai.py --out-dir /gpfs/data/huo-lab/Image/ChiMEC/MG/out

# custom output path
python analyze_mirai.py \
  --out-dir /gpfs/data/huo-lab/Image/ChiMEC/MG/out \
  --out /path/to/custom_summary.json

# override prediction column mapping
python analyze_mirai.py \
  --out-dir /gpfs/data/huo-lab/Image/ChiMEC/MG/out \
  --map 1:risk_1year 5:risk_5year

# filter to a specific split
python analyze_mirai.py \
  --out-dir /gpfs/data/huo-lab/Image/ChiMEC/MG/out \
  --split test

# k-fold cross-validation for IPCW sensitivity
python analyze_mirai.py \
  --out-dir /gpfs/data/huo-lab/Image/ChiMEC/MG/out \
  --kfold 5
```

K-fold CV splits at the patient level to avoid leakage and assesses IPCW sensitivity when all data is "test".

### analyze_metadata.py

Analyze imaging metadata and generate summary plots.

```bash
# basic usage
python analyze_metadata.py --modality MG

# dump screening patients (scans ≥3 months before diagnosis)
python analyze_metadata.py --modality MG --dump-screening-patients
```

Available modalities: CR, DX, MG, US, CT, MR, NM, PT, XA, RF, ES, XC, PX, RG

Creates plots in `plots/` including download status, scans per patient distributions, modality distributions, and time-to-diagnosis analysis.

---

## Quality Control

### qc_gallery.py

Interactively review and mark mammogram exams for quality control. Generates combined 4-view figures (L CC, L MLO, R CC, R MLO) with an HTML gallery.

#### Recommended workflow: Prioritize low-performing exams

```bash
# prioritize worst-performing exams for efficient QC
python qc_gallery.py --serve --max-exams 100 --prioritize-errors \
  --pred-csv /gpfs/data/huo-lab/Image/ChiMEC/MG/out/validation_output.csv \
  --meta-csv /gpfs/data/huo-lab/Image/ChiMEC/MG/out/mirai_manifest.csv
```

This will:
1. Compute prediction error scores for each exam
2. Show you the worst-performing exams first (missed cancers, false alarms)
3. Weight misclassified cases 2x so you see them immediately

For remote access, set up SSH port forwarding:
```bash
ssh -L 5000:localhost:5000 user@remote-host
```
Then open `http://localhost:5000/` in your browser.

#### QC workflow: Iterative pattern discovery (for large datasets)

For large datasets (e.g., 18k exams), use an iterative approach:

**Batch 1: Initial QC (100 worst-performing exams)**
```bash
# QC first batch prioritized by prediction error
python qc_gallery.py --serve --max-exams 100 --prioritize-errors \
  --pred-csv /gpfs/data/huo-lab/Image/ChiMEC/MG/out/validation_output.csv \
  --meta-csv /gpfs/data/huo-lab/Image/ChiMEC/MG/out/mirai_manifest.csv
```

**Analyze patterns:**
```bash
# After QC'ing 20-50 exams, analyze what makes bad exams bad
python analyze_qc_patterns.py --qc-file data/qc_status.json

# This will print:
# - DICOM tags that differ between bad and good exams
# - Suggested filters (e.g., "Exclude if PositionerPrimaryAngle < 0")
# - Known problematic patterns (implants, scanned film, positioning issues)
# - Device models with high failure rates
```

**Test filter hypotheses (streamlined workflow):**
```bash
# Step 1: Test filters and generate exam lists
python test_positioning_filters.py --max-samples 20

# Output shows:
# - How many exams each filter catches
# - Saves exam IDs to data/filter_tests/*.txt

# Step 2: QC specific filter (interactive menu - easiest!)
./qc_filter.sh
# Select filter from menu → loads ONLY those exams → fast!

# Or QC specific filter directly:
python qc_gallery.py --serve --exam-list data/filter_tests/scanned_film_exams.txt
python qc_gallery.py --serve --exam-list data/filter_tests/gems_ffdm_tc1_exams.txt

# Step 3: After QC, analyze which filters worked
python analyze_qc_patterns.py
```

**Batch 2+: Apply validated filters and repeat**

After validating a filter catches only bad exams:

1. **Add to auto-exclude config** (one line edit):
```bash
# Edit data/qc_auto_exclude.json
{
  "filters": ["gems_ffdm_tc1"]  # add validated filter names here
}
```

2. **Restart server** - those exams automatically excluded:
```bash
python qc_gallery.py --serve --max-exams 100 --prioritize-errors \
  --pred-csv ... --meta-csv ...
# GEMS exams now auto-skipped, marked as "auto_excluded"
```

3. **View cutflow** (click "📊 Cutflow" button in browser):
- See how many exams each filter excludes
- See filter effectiveness (precision/recall)
- Validate filters before adding to config

**Warning**: Only add filters with >95% precision! Check cutflow first.

With server running, you can also dynamically test filters:
- Use dropdown to preview a filter
- Click "Apply Filter" to see those exams
- Manually QC the batch
- If validated → add to `qc_auto_exclude.json`

**Standard QC controls:**
1. Mark exams using keyboard shortcuts: `G` = Good, `R` = Needs review, `B` = Bad, Arrow keys = Navigate
2. QC status auto-saves to `data/qc_status.json`
3. Re-run the script to continue — by default, exams marked "good" or "bad" are automatically skipped
4. Apply QC filter in downstream analysis with `--qc-file` argument (see below)

**Default behavior**: Future QC runs skip exams marked as "good" (already approved) or "bad" (already rejected), showing only "review" and unmarked exams.

#### Alternative: Random sampling (if no predictions yet)

```bash
# server mode (recommended for remote work)
python qc_gallery.py --serve --max-exams 100 --random
```

#### Common options

```bash
# custom port
python qc_gallery.py --serve --port 8080

# filter to specific patient
python qc_gallery.py --serve --patient 12345

# re-visit exams previously marked as "bad"
python qc_gallery.py --serve --max-exams 100 --qc-skip-status good

# only show completely unmarked exams (skip all QC'd exams)
python qc_gallery.py --serve --max-exams 100 --qc-skip-status good bad review

# use different prediction horizon for error scoring (default: 5 years)
python qc_gallery.py --serve --prioritize-errors --horizon 3 \
  --pred-csv ... --meta-csv ...

# custom paths
python qc_gallery.py \
  --serve \
  --views /path/to/views.parquet \
  --raw /gpfs/data/huo-lab/Image/ChiMEC/MG \
  --output qc_output \
  --qc-file /path/to/my_qc_status.json
```

#### Local mode (without server)

```bash
python qc_gallery.py --max-exams 10
# then open qc_output/gallery.html
```

### Applying QC filters to analysis

After QC, exclude failed exams from downstream analysis by passing `--qc-file`:

```bash
# analyze_mirai.py with QC filtering
python analyze_mirai.py \
  --out-dir /gpfs/data/huo-lab/Image/ChiMEC/MG/out \
  --qc-file data/qc_status.json
```

**QC filter behavior**:

In future QC runs (default: `--qc-skip-status good bad`):
- Exams marked `"good"` or `"bad"` are skipped (already decided)
- Exams marked `"review"` or unmarked are shown (need attention)
- To re-visit "bad" exams: `--qc-skip-status good`

In downstream analysis (`analyze_mirai.py --qc-file`):
- Exams marked `"bad"` or `"review"` are excluded from metrics
- Exams marked `"good"` or unmarked are included
- Original `validation_output.csv` remains unchanged
- QC decisions stored separately in `qc_status.json`

This design allows you to:
- Efficiently QC by not showing already-decided exams
- Re-run QC with different criteria
- Audit what was excluded
- Easily revert filtering decisions
