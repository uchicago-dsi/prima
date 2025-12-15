# prima
Polygenic Risk and Imaging Multimodal Assessment

# chimec list
/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/List_ChiMEC_priority_2025July30.csv

Micromamba install:
```
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -r ~/micromamba 
source ~/.bashrc
micromamba config append channels conda-forge
```

Installation recipe:
```bash
# create + activate
micromamba create -y -f env.yaml
eval "$(micromamba shell hook -s bash)"
micromamba activate prima

# install torch/vision matching cu111 wheels (from the official index)
python -m pip install \
  torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html  # from “previous versions”

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

Runtime dependencies live in `requirements.txt`; install `requirements-dev.txt` to pick up
tooling such as Ruff and optional notebook helpers used during development.

Before sending changes, run the project formatters:

```bash
ruff format .
ruff check --fix .
```





## HIRO → ChiMEC Data Transfer Instructions (macOS client → GPFS)

This document describes the tested steps for moving data from a HIRO CIFS share to our ChiMEC GPFS storage.

Goal: one-way, resumable, safe copy of files. Files are deleted from HIRO after successful transfer; nothing is ever deleted from ChiMEC.

---
### 1. Mount the HIRO share on macOS

- Open Finder → Go → Connect to Server...
- Enter the server string (use your UCHAD username):
  `smb://UCHAD;annawoodard@cifs01uchadccd.uchad.uchospitals.edu/radiology/HIRO/16352A`
- Authenticate with your UCHAD password.
- Finder mounts the share under `/Volumes/16352a` (the exact name may vary if you remount multiple times). This mounted path is the source for rsync.

---
### 2. Mount the HIRO share on Linux

To mount the HIRO share on a Linux system:

```bash
sudo mount -t cifs //cifs01uchadccd.uchad.uchospitals.edu/radiology/HIRO /mnt/uchad_samba \
  -o credentials=/home/annawoodard/creds,vers=3.0,noperm,uid=$(id -u),gid=$(id -g),file_mode=0660,dir_mode=0770
```

This mounts the share at `/mnt/uchad_samba`. Ensure the credentials file exists and contains your UCHAD username and password in the format:
```
username=your_username
password=your_password
domain=UCHAD
```

This requires special access granted from the admins, e.g.:
```
# cat /etc/sudoers.d/annawoodard.conf
annawoodard ALL=(root) NOPASSWD: /usr/sbin/mount.cifs
annawoodard ALL=(root) NOPASSWD: /usr/sbin/umount
```
---
### 3. Install a modern rsync on macOS

The Apple-provided rsync is outdated. Use **Homebrew** to install a modern version.

```bash
brew install rsync
```

This installs modern rsync as `/opt/homebrew/bin/rsync` (Apple Silicon) or `/usr/local/bin/rsync` (Intel).

---
### 4. Set source and destination paths

```bash
SRC="/Volumes/16352a/"  # mounted HIRO subdir; note trailing slash
DST="annawoodard@cri-ksysappdsp3.cri.uchicago.edu:/gpfs/data/huo-lab/Image/ChiMEC/"
```

---
### 5. Dry run (safe preview)

This command performs a dry run, showing what will be copied without actually moving any files.

```bash
/opt/homebrew/bin/rsync -aHvn --itemize-changes --append-verify --partial \
  --remove-source-files \
  "$SRC" "$DST"
```
- `-n`: dry run, nothing is copied or removed
- `--itemize-changes`: shows a detailed list of what would happen
- `--remove-source-files`: is inert in a dry-run but will delete files from HIRO after a successful copy in a real run

Check that the file mapping is correct (`/Volumes/16352a/...` → `/gpfs/.../ChiMEC/...`).

---
### 6. Real transfer (resumable, safe delete-on-success)

This command performs the actual data transfer and can be stopped and restarted as needed.

```bash
LOG="$HOME/hiro2chimec-$(date +%F).log"
caffeinate -dims /opt/homebrew/bin/rsync \
  -aH --info=progress2 --human-readable \
  --append-verify --partial --remove-source-files \
  --log-file="$LOG" \
  "$SRC" "$DST"
```
- `caffeinate`: prevents the Mac from sleeping during the transfer
- `--append-verify`: enables robust resume on partial files and ensures checksum verification
- `--partial`: keeps partially copied files to allow for a restart
- `--remove-source-files`: deletes files from HIRO only after a successful copy to the destination
- `--log-file`: records all transfers to a log file

---
### 7. Cleanup (optional)

Directories on HIRO will remain after the files are moved. They can be pruned manually later if desired, but this is not required.

---
### Performance note

Over VPN, throughput was measured at ~3 MB/s (approximately 4 days per terabyte).

### check how many exams have transferred

```bash
grep '<f++++++++++' hiro2chimec-2025-08-21.log  | awk '{print $5}' | cut -d/ -f1-2 | sort -u | wc -l
```

### running mirai

**Important**: Before running mirai, ensure LD_LIBRARY_PATH is set to use conda's libstdc++:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Then run:

**GPU Compatibility Check**: Ensure PyTorch is compiled for your GPU architecture:

```bash
# check your GPU compute capability (on compute node with GPU access)
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

Common GPUs and required PyTorch:
- **A100** (compute 8.0): needs CUDA 11+ with sm_80 → use `cu118`
- **V100** (compute 7.0): needs CUDA 10+ with sm_70 → use `cu102` or higher
- **P100** (compute 6.0): needs CUDA 9+ with sm_60 → use `cu102` or higher

Install PyTorch for your setup (A100 example with CUDA 11.8):
```bash
# install/reinstall PyTorch with correct CUDA version
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# verify GPU is detected and supported
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

If you get `RuntimeError: CUDA error: no kernel image is available for execution on the device`, your PyTorch doesn't support your GPU. Check compiled architectures and reinstall:
```bash
python -c "import torch; print(f'Compiled for: {torch.cuda.get_arch_list()}')"
# should include sm_XX matching your GPU's compute capability (e.g., sm_80 for A100)
```

For single GPU:
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

**Performance monitoring and tuning**:

Monitor GPU utilization in another terminal:
```bash
watch -n 1 nvidia-smi
```

If process seems hung or slow (check `nvidia-smi`):
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
- **GPU memory <1GB, stuck at 0%**: data loading hang - use `--num_workers 0` to disable multiprocessing
- **First batch very slow**: normal with large images (1664x2048) - can take 2-5 minutes per batch initially
- **Low GPU utilization** (<50%): increase `--batch_size` 
- **High GPU memory**: decrease `--batch_size`
- **Network filesystem**: multiprocessing (`--num_workers`) can cause deadlocks - start with `0`

For multi-GPU inference, add `--num_gpus N` and `--data_parallel`:
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

For parallel processing across multiple GPU jobs (recommended for large datasets):
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
  --cuda \
  --debug_max_samples 20
```

The sharded script will:
- Split the CSV into chunks (use `--num_shards N` or `--max_samples_per_shard N`)
- Submit one GPU job per chunk via SLURM
- Automatically collate results when jobs complete
- Save recovery metadata for resuming if the main process dies

To recover from a crashed run:
```bash
python run_mirai_sharded.py --recover /path/to/mirai_shards/job_metadata.json
```

For debugging with a small sample:
```bash
python run_mirai_sharded.py --debug_max_samples 100 --num_shards 2 ...
```

**Checkpointing workaround** (manual alternative): Mirai doesn't save incremental results. To avoid losing work, split your metadata CSV into chunks and process separately:
```bash
# split manifest into chunks of 1000 patients each
split -l 1000 -d --additional-suffix=.csv mirai_manifest.csv chunk_

# process each chunk (keep header row)
head -1 mirai_manifest.csv > chunk_00_with_header.csv
tail -n +2 chunk_00.csv >> chunk_00_with_header.csv

python scripts/main.py ... \
  --metadata_path chunk_00_with_header.csv \
  --prediction_save_path output_00.csv

# merge results
head -1 output_00.csv > final_output.csv
tail -n +2 -q output_*.csv >> final_output.csv
```

**Notes**:
- The LD_LIBRARY_PATH setting is required because pandas (a lifelines dependency) needs GLIBCXX_3.4.29, which is provided by conda/micromamba's libstdc++ but not by the system library. Setting this before Python starts ensures the correct library is used.
- Ignore "Restarting run from scratch" message during inference - it's always printed even when only running `--test` (not training).

### analyzing mirai output

After running Mirai inference, use `analyze_mirai.py` to compute per-horizon AUC and survival metrics (Uno's C-index, time-dependent AUC, integrated Brier score).

The script expects `validation_output.csv` and `mirai_manifest.csv` in the same directory. Basic usage:
```bash
python analyze_mirai.py \
  --out-dir /gpfs/data/huo-lab/Image/ChiMEC/MG/out
```

This will write results to `{out_dir}/summary.json` by default. To specify a different output path:
```bash
python analyze_mirai.py \
  --out-dir /gpfs/data/huo-lab/Image/ChiMEC/MG/out \
  --out /path/to/custom_summary.json
```

The script auto-detects prediction columns (looks for names containing 'risk', 'pred', or 'prob' with horizon numbers). To override:
```bash
python analyze_mirai.py \
  --out-dir /gpfs/data/huo-lab/Image/ChiMEC/MG/out \
  --map 1:risk_1year 5:risk_5year
```

Filter to a specific split (if your metadata has a `split_group` column):
```bash
python analyze_mirai.py \
  --out-dir /gpfs/data/huo-lab/Image/ChiMEC/MG/out \
  --split test
```

**k-fold cross-validation for IPCW sensitivity**: When all your data is "test" (no separate training set), use k-fold CV to assess how sensitive IPCW-based metrics are to the censoring distribution estimate:
```bash
python analyze_mirai.py \
  --out-dir /gpfs/data/huo-lab/Image/ChiMEC/MG/out \
  --kfold 5
```

This splits at the patient level (not exam level) to avoid leakage. For each fold, the training fold estimates the IPCW censoring distribution, and the test fold evaluates metrics using that estimate. Results include mean ± std across folds.

**Output**: The script prints:
- Per-horizon summary: cases, controls, excluded, prevalence, AUC
- Survival metrics: time-dependent AUC, Uno's C-index, integrated Brier score
- With `--kfold`: aggregated statistics across folds

JSON output includes detailed per-fold results when using k-fold CV.

### analyze_metadata.py

Use `analyze_metadata.py` to analyze imaging metadata and generate summary plots for a specified modality. The script performs ChiMEC patient data analysis, generates distribution plots, and tracks download status.

**Basic usage**:
```bash
python analyze_metadata.py --modality MG
```

This analyzes mammography (MG) data and generates plots in the `plots/` directory, including:
- Download status by category
- Scans per patient distributions
- Modality distributions
- Time-to-diagnosis analysis
- Cases vs controls comparisons

**Available modalities**: CR, DX, MG, US, CT, MR, NM, PT, XA, RF, ES, XC, PX, RG

**Dump screening patients**:
```bash
python analyze_metadata.py --modality MG --dump-screening-patients
```

This generates a CSV of patients with screening scans at least 3 months before diagnosis, useful for identifying screening cohorts.

The script creates both "all" and "genotyped" versions of all plots, comparing the full dataset against the genotyped subset.

### QC gallery

Use `qc_gallery.py` to interactively review and mark mammogram exams for quality control. The tool generates combined 4-view figures (L CC, L MLO, R CC, R MLO) and provides an HTML gallery with keyboard navigation and QC marking.

**Basic usage** (server mode, recommended for remote work):
```bash
python qc_gallery.py --serve --max-exams 100 --random
```

This will:
1. Generate combined 4-view figures for each exam
2. Create an interactive HTML gallery
3. Start an HTTP server on port 5000

**Accessing the gallery remotely**:

If running on a remote server, set up SSH port forwarding:
```bash
# In a separate terminal on your local machine
ssh -L 5000:localhost:5000 user@remote-host
```

Then open `http://localhost:5000/` in your browser.

**QC workflow**:

1. Mark exams using keyboard shortcuts:
   - `G` = Good
   - `R` = Needs review
   - `B` = Bad
   - Arrow keys = Navigate between exams

2. QC status auto-saves to the server after each click (saved to `data/qc_status.json` by default)

3. To continue QC on remaining exams, re-run the script - exams already marked "good" will be automatically skipped

**Common options**:

Limit number of exams and sample randomly:
```bash
python qc_gallery.py --serve --max-exams 50 --random
```

Use a custom port:
```bash
python qc_gallery.py --serve --port 8080
```

Filter to a specific patient:
```bash
python qc_gallery.py --serve --patient 12345
```

Use a custom QC file location:
```bash
python qc_gallery.py --serve --qc-file /path/to/my_qc_status.json
```

Specify custom input/output paths:
```bash
python qc_gallery.py \
  --serve \
  --views /path/to/views.parquet \
  --raw /gpfs/data/huo-lab/Image/ChiMEC/MG \
  --output qc_output
```

**Local mode** (without server):

If you prefer not to use the server, you can generate the gallery and open it directly:
```bash
python qc_gallery.py --max-exams 10
```

Then open `qc_output/gallery.html` in your browser. Note: in local mode, QC data downloads as `qc_status.json` on each click - you'll need to manually move it to your desired location.

**QC file format**:

The QC status file is a JSON mapping exam IDs to status:
```json
{
  "exam_id_1": "good",
  "exam_id_2": "review",
  "exam_id_3": "bad"
}
```

Valid statuses are: `"good"`, `"review"`, `"bad"`. Exams marked as "good" are automatically skipped on subsequent runs.