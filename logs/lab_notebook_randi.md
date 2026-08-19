# Prima Randi Lab Notebook

Canonical Prima notebook for work run from Randi. Append experiment, Slurm, QC, and debugging entries here instead of creating thread- or campaign-specific notebooks.

## Current Summary

- Randi Prima checkout in use on `cri22in002`: `/gpfs/data/huo-lab/Image/annawoodard/prima`.
- Live Slurm policy checked on 2026-06-25: H200 141GB GPUs are available on `catherineq` and `siweiq`; use explicit `--qos=opportunistic` for idle lab hardware and expect possible requeue/preemption. Do not reuse DSI `general`/`interactive` assumptions.
- Actual MG raw root is `/gpfs/data/huo-lab/Image/ChiMEC/MG`; SoT tables exist at `/gpfs/data/huo-lab/Image/ChiMEC/MG/sot/views.parquet` and `exams.parquet`.
- QC export cache at `/gpfs/data/huo-lab/Image/annawoodard/prima/qc_export` now covers all 7,840 eligible post-auto-filter exams. The eight previously blocked montages were regenerated from archived DICOM members after the lineage fix.
- Production `sot/views.parquet` and `qc_export/views_for_qc.parquet` use durable `source_archive_relpath` + `source_archive_member` locators with SOP UID and SHA-256 identity; neither table contains the removed transient `dicom_path` field.
- Current SoT/QC tables retain exactly one canonical image per `(exam, laterality, view)`; selection is deterministic rather than random, but fallback to an alternate image is impossible until preprocessing persists the pre-selection candidate inventory and view-level QC.
- Qwen model environment is staged separately from `prima` and `hfdp` at `/gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-qwen` with Python 3.11, torch 2.5.1 CUDA 12.4, and transformers 5.12.1. Do not modify `hfdp`; it was only inspected for package-version comparison.
- Active Qwen blocker: the base `Qwen3.5-397B-A17B-FP8` checkpoint is downloading slowly from Hugging Face into `/gpfs/data/huo-lab/Image/annawoodard/models/Qwen3.5-397B-A17B-FP8`. Compute nodes could not reach Hugging Face, so the download is running detached from the login node.
- Faster checkpoint staging path found on 2026-06-25: `cri-datamover.cri.uchicago.edu` cannot connect to `huggingface.co` directly, but it can pull signed `us.aws.cdn.hf.co` URLs. Current downloader mints signed URLs on the login node and transfers bytes from datamover into the shared GPFS model directory.

## 2026-05-15

### Notebook Initialized (2026-05-15 14:37 CT)

#### Question

Where should Prima work on Randi be recorded so it does not split into thread-specific notebooks?

#### Action

Created a repo-local Randi notebook under `/home/annawoodard/prima/logs/lab_notebook_randi.md`.

#### Evidence

- DSI notebook: `/home/annawoodard/prima/logs/lab_notebook_dsi.md`.
- Randi notebook: `/home/annawoodard/prima/logs/lab_notebook_randi.md`.
- Repo guidance: `/home/annawoodard/prima/AGENTS.md`.

#### Result

Prima now has one repo-local notebook path for DSI and one repo-local notebook path for Randi.

#### Decision Impact

Future Prima work should append to the environment-specific notebook under `logs/` rather than creating a new notebook for each conversation or campaign.

## 2026-06-25

### Randi Actual-Data Setup And Cache Verification (2026-06-25 10:34 CDT)

#### Question

Can the vertical-line auto-QC workflow move from the DSI copied cache packet to Randi actual data, and what blocks the first H200 smoke job?

#### Action

Pulled the latest `main` history into `/gpfs/data/huo-lab/Image/annawoodard/prima`, preserving the two local commits on top of `origin/main`. Read the updated Slurm guidance and checked live Randi partitions/QoS. Validated the QC/Qwen Python entrypoints in the `prima` env, installed missing launcher/model packages, discovered actual-data paths, and ran a real-data QC cache refresh:

```bash
/gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima/bin/python qc/qc_gallery.py \
  --preprocess-all \
  --raw /gpfs/data/huo-lab/Image/ChiMEC/MG \
  --views /gpfs/data/huo-lab/Image/ChiMEC/MG/sot/views.parquet \
  --output /gpfs/data/huo-lab/Image/annawoodard/prima/qc_export
```

#### Evidence

- Checkout: `/gpfs/data/huo-lab/Image/annawoodard/prima`, branch `main`, clean except local branch is ahead of `origin/main` by two commits.
- Live H200 partitions: `catherineq` and `siweiq`, each with `gpu:nvidia_h200-141gb:4`; visible QoS includes `opportunistic`, `catherine_priority`, and `siwei_priority`.
- Python env after package install: Python 3.8.20, torch 2.0.1+cu118, transformers 4.46.3, accelerate 1.0.1, safetensors 0.5.3, submitit 1.5.4.
- CLI checks passed: `py_compile` for `auto_annotate_qc.py`, `submit_auto_qc.py`, `qc/qc_gallery.py`, and `scripts/build_qwen35_fp8_repair.py`; `submit_auto_qc.py --help`; `qc/qc_gallery.py --help`.
- SoT counts: `views.parquet` has 42,904 rows across 10,726 exams and 1,830 patients; `exams.parquet` has 10,726 exams.
- Auto-filters excluded 2,886 exams total: scanned film, GE Senographe FFDM TC1, and duplicate SOP UID filters.
- Eligible post-filter exams: 7,840. Cached eligible montages: 7,832. Missing eligible montages: 8. The 8 missing eligible exams have 32 referenced view files, and none of those referenced DICOM files are present.
- Refreshed artifacts: `/gpfs/data/huo-lab/Image/annawoodard/prima/qc_export/gallery.html` and `/gpfs/data/huo-lab/Image/annawoodard/prima/data/qc_state.json`.

#### Result

Randi actual-data SoT and montage cache are usable for a human-reviewable batch from the 7,832 cached eligible exams. The cache refresh did not generate new montages because the only uncached eligible exams reference missing source DICOMs. `submit_auto_qc.py` is now importable, but the model side is not ready: no local Qwen3.5 397B FP8 checkpoint was found in the obvious model roots, and the current Python 3.8 env only receives `transformers==4.46.3`, which does not expose `qwen3_5_moe`.

#### Decision Impact

Next discriminative step is to stage or identify the Randi-local Qwen3.5-397B-A17B-FP8 checkpoint and a compatible model-serving env, likely Python 3.11 with a newer/source Transformers build and an H200-capable torch. Once that exists, build the PRIMA repair wrapper and submit the first smoke job with explicit Randi flags such as `--partition catherineq --qos opportunistic --gpuspec nvidia_h200-141gb --ngpus 4`, plus `PRIMA_QWEN35_FP8_FORCE_BF16_EXPERTS=1`.

### Qwen Model Env And Checkpoint Staging (2026-06-25 15:22 CDT)

#### Question

Can Randi run the Qwen3.5-397B FP8 vertical-line detector stack without destabilizing the existing `hfdp` environment, and can the base checkpoint be staged locally?

#### Action

Built a separate model environment instead of modifying `hfdp`:

```bash
micromamba create -y --offline -n prima-qwen python=3.11 pip pytorch=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -c conda-forge
PYTHONNOUSERSITE=1 /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-qwen/bin/python -s -m pip install \
  pandas pyarrow tqdm submitit accelerate safetensors einops pillow huggingface_hub "transformers>=5.5.0"
```

Verified the env with `PYTHONNOUSERSITE=1` and `python -s`: torch 2.5.1, CUDA 12.4 build, transformers 5.12.1, accelerate 1.14.0, safetensors 0.8.0, Qwen3.5 MoE modules, and `AutoModelForImageTextToText` imports. Repo CLI checks under `prima-qwen` passed for `py_compile`, `submit_auto_qc.py --help`, and `scripts/build_qwen35_fp8_repair.py --help`.

Started checkpoint staging:

```bash
/gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-qwen/bin/hf download \
  Qwen/Qwen3.5-397B-A17B-FP8 \
  --local-dir /gpfs/data/huo-lab/Image/annawoodard/models/Qwen3.5-397B-A17B-FP8
```

The public repo has 107 files, including 94 safetensor shards totaling about 406,198,638,888 bytes. No HF token is configured, so all requests are anonymous.

The first Slurm download attempt on `tier1q` failed quickly:

```text
JobID=12283267
State=FAILED
ExitCode=1:0
Reason in stderr: remote repo cannot be accessed, [Errno 111] Connection refused
```

This indicates the CPU compute nodes do not have working outbound Hugging Face access. Switched to a detached login-node sequential downloader:

```bash
logs/qwen35_fp8_download_login.sh
```

Current active download:

- PID file: `/gpfs/data/huo-lab/Image/annawoodard/prima/logs/slurm/qwen35_fp8_download_login.pid`
- Log: `/gpfs/data/huo-lab/Image/annawoodard/prima/logs/slurm/qwen35_fp8_download_login_20260625_151551.log`
- Base checkpoint path: `/gpfs/data/huo-lab/Image/annawoodard/models/Qwen3.5-397B-A17B-FP8`
- Strategy: download non-safetensor metadata first, then download the 94 shards sequentially for visible completion and cleaner resume behavior.

Prepared but did not submit the repair wrapper job:

```bash
sbatch logs/qwen35_fp8_repair.sbatch
```

#### Evidence

- `prima-qwen` env path: `/gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-qwen`.
- Metadata/tokenizer files are present in the checkpoint directory: `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `preprocessor_config.json`, and prior config files.
- At the time of this entry, shard `model.safetensors-00001-of-00094.safetensors` was actively downloading. The Xet log showed about 468 MB transferred for the shard, but no final shard file had materialized yet because Xet writes the final `.safetensors` after the shard completes.
- Direct anonymous HTTP range probe from the login node was slow, about 1 MiB in 23.95 s, confirming that a token or alternate staging source would materially reduce wall time.

#### Result

The model-serving environment is ready and isolated. The base checkpoint is not complete yet; download is running detached from the login node and is expected to be slow without an HF token. The repair-wrapper build cannot start until all 94 safetensor shards validate.

#### Decision Impact

Monitor the downloader with:

```bash
ps -f -p "$(cat logs/slurm/qwen35_fp8_download_login.pid)"
tail -n 80 logs/slurm/qwen35_fp8_download_login_20260625_151551.log
find /gpfs/data/huo-lab/Image/annawoodard/models/Qwen3.5-397B-A17B-FP8 -maxdepth 1 -name '*.safetensors' | wc -l
```

After the checkpoint validates, submit `logs/qwen35_fp8_repair.sbatch` on CPU. After the repaired wrapper exists, use H200 `catherineq` or `siweiq` with explicit `--qos=opportunistic` for a 1-3 exam smoke inference.

### Datamover Checkpoint Download Switch (2026-06-25 16:22 CDT)

#### Question

Can `cri-datamover.cri.uchicago.edu` stage the Qwen checkpoint faster than the login-node anonymous Hugging Face download?

#### Action

Logged into `cri-datamover.cri.uchicago.edu` non-interactively. It resolves to `hawkhurst.cri.uchicago.edu`, sees the same GPFS model and repo paths, and can run the shared `prima-qwen` env. Direct `curl https://huggingface.co` from datamover fails with TLS connection reset, but datamover can reach `us.aws.cdn.hf.co` and `cas-server.xethub.hf.co`.

Generated signed Hugging Face CDN URLs from the login node and tested datamover range download against the signed CDN URL. The probe downloaded 1 MiB at about 1.7 MB/s; a full resumed shard transfer then ran much faster.

Stopped the slow login-node `hf download`, verified the 768 MB `.incomplete` file matched the first 1 MiB of the signed CDN shard, moved it to:

```text
/gpfs/data/huo-lab/Image/annawoodard/models/Qwen3.5-397B-A17B-FP8/model.safetensors-00001-of-00094.safetensors.part
```

Started the datamover downloader:

```bash
logs/qwen35_fp8_download_datamover.sh
```

#### Evidence

- Datamover host: `hawkhurst.cri.uchicago.edu`.
- Datamover direct HF front-door failure: `curl: (35) OpenSSL SSL_connect: Connection reset by peer in connection to huggingface.co:443`.
- Datamover CDN test: signed `us.aws.cdn.hf.co` URL range download returned HTTP 206 and about 1.7 MB/s.
- Active datamover controller PID file: `/gpfs/data/huo-lab/Image/annawoodard/prima/logs/slurm/qwen35_fp8_download_datamover.pid`.
- Active datamover log: `/gpfs/data/huo-lab/Image/annawoodard/prima/logs/slurm/qwen35_fp8_download_datamover_20260625_161704.log`.
- Shard 1 completed with exact expected size:

```text
model.safetensors-00001-of-00094.safetensors
expected_size=4295778960
remote curl_size_download=3490981320
remote curl_speed_download=44209223
remote curl_time_total=78.965058
```

#### Result

The datamover strategy works. Shard 1 completed and shard 2 started. The observed resumed shard throughput was about 44 MB/s, much faster than the previous login-node/Xet path.

#### Decision Impact

Let the datamover downloader continue. If throughput stays near shard-1 speed, the 406 GB checkpoint should finish in a few hours rather than multiple days. After validation, submit `logs/qwen35_fp8_repair.sbatch`.

### Datamover Download Restart Status (2026-06-25 17:39 CDT)

#### Question

Is the Qwen checkpoint download still progressing after the signed-URL resolver failure and foreground debug run?

#### Action

Stopped the foreground `bash -x` debug run after it completed shard 8 and began shard 9, then relaunched the datamover downloader detached with normal logging:

```bash
nohup setsid env QWEN35_DOWNLOAD_ATTEMPTS_PER_FILE=40 QWEN35_DATAMOVER_HOST=cri-datamover.cri.uchicago.edu \
  bash logs/qwen35_fp8_download_datamover.sh \
  > logs/slurm/qwen35_fp8_download_datamover_20260625_173736.log 2>&1 < /dev/null &
```

#### Evidence

- Active controller PID: `2724284`.
- Active log: `logs/slurm/qwen35_fp8_download_datamover_20260625_173736.log`.
- Durable checkpoint state after restart: 10 complete safetensor shards, 42,957,785,504 bytes.
- Shard 9 completed exactly after the clean restart:

```text
remote_complete target=/gpfs/data/huo-lab/Image/annawoodard/models/Qwen3.5-397B-A17B-FP8/model.safetensors-00009-of-00094.safetensors size=4295778960
```

- Shard 10 completed exactly at 68.8 MB/s, and shard 11 was actively downloading with a growing partial file; observed partial size at the status check was 2,660,589,568 bytes.

#### Result

The downloader is healthy again. One interrupted shard-9 partial became oversized, the script treated it as incomplete, and the clean run restarted that shard from byte 0 rather than accepting a bad file.

#### Decision Impact

Continue monitoring the datamover log and completed-shard count. If the run exits again, relaunching the same script is safe because exact-size completed shards are skipped and partial files are retried. Submit `logs/qwen35_fp8_repair.sbatch` only after all 94 safetensor shards validate.

### Qwen Checkpoint Complete and Repair Submitted (2026-06-25 20:06 CDT)

#### Question

Did the datamover checkpoint download finish cleanly, and can the PRIMA repair wrapper build start?

#### Action

Validated the downloaded base checkpoint against `model.safetensors.index.json`, checked representative safetensor headers, and inspected Slurm policy before submitting the repair wrapper build. Slurm dry-run rejected the original repair request with `--mem=120G` on `tier1q`; `--mem=100G` passed, so the repair sbatch was changed accordingly.

Submitted:

```bash
sbatch --parsable logs/qwen35_fp8_repair.sbatch
```

#### Evidence

- Base checkpoint: `/gpfs/data/huo-lab/Image/annawoodard/models/Qwen3.5-397B-A17B-FP8`.
- Download log: `logs/slurm/qwen35_fp8_download_datamover_20260625_173736.log`.
- Download result: 94 / 94 safetensor shards, 406,151,669,464 bytes, no partial files.
- Validation result: index expected 94 shards, actual 94 shards, no missing or extra shards; representative safetensor files opened successfully.
- Repair sbatch: `logs/qwen35_fp8_repair.sbatch`.
- Repair output path: `/gpfs/data/huo-lab/Image/annawoodard/models/Qwen3.5-397B-A17B-FP8-prima-repair`.
- Repair job ID: `12291859`.
- Initial Slurm state: `PENDING (Priority)` on `tier1q`.
- Expected Slurm logs:

```text
logs/slurm/qwen35_fp8_repair_12291859.out
logs/slurm/qwen35_fp8_repair_12291859.err
```

#### Result

The base checkpoint download is complete and validated. The repair-wrapper build is submitted but has not yet started at the time of this entry.

#### Decision Impact

Monitor job `12291859`. Once it finishes, verify `repair_manifest.json` and per-layer cache files under the repair output path, then run a 1-3 exam H200 smoke inference with `--qos=opportunistic` on `catherineq` or `siweiq`.

Follow-up at 20:07 CDT: job `12291859` is running on `cri22cn006`; 2 / 60 layer cache files have been written under the repair cache, each about 8.59 GB. Preliminary ETA from the first two layers is about 45-60 minutes for the full repair cache, assuming the cadence holds.

Follow-up at 21:34 CDT: repair job `12291859` completed successfully with `ExitCode=0:0` after 57:26. The output directory now has `repair_manifest.json` and 60 / 60 cache files under `/gpfs/data/huo-lab/Image/annawoodard/models/Qwen3.5-397B-A17B-FP8-prima-repair/cache` totaling 515,427,550,560 bytes. Validation opened representative cache files and confirmed manifest format `prima_qwen35_fp8_repair_manifest_v1`, `force_bf16_experts=true`, `repair_layer_spec=all`, and `dequant_down_proj_spec=all`. Next step is a 1-3 exam H200 smoke inference using the repaired wrapper.

### Randi H200 One-Exam Smoke Submitted (2026-06-25 21:40 CDT)

#### Question

Can the repaired Qwen3.5 FP8 wrapper start on Randi H200 hardware against one actual-data cached montage?

#### Action

Created one-exam smoke list:

```text
qc_redo/debug_exam_lists/randi_h200_smoke_1exam.txt
```

Initial submission through `submit_auto_qc.py` created job `12292803`, but it failed before Python started because submitit's generated script called `srun`, and `srun` on the allocated node could not read `/etc/slurm/slurm.conf`:

```text
srun: error: s_p_parse_file: cannot stat file /etc/slurm/slurm.conf: Permission denied
srun: fatal: Unable to process configuration file
```

This was a scheduler-wrapper failure, not a model/runtime failure. Created and submitted a direct `sbatch` script that runs `auto_annotate_qc.py` without nested `srun`:

```bash
sbatch logs/qwen397b_vertical_line_randi_h200_smoke_1exam.sbatch
```

#### Evidence

- Direct smoke job ID: `12292820`.
- Partition/QoS: `siweiq` / `opportunistic`.
- GPU request: `gres/gpu:nvidia_h200-141gb:4`.
- Node: `cri22cn418`.
- Slurm logs:

```text
logs/slurm/qwen397b_vertical_line_randi_h200_smoke_1exam_12292820.out
logs/slurm/qwen397b_vertical_line_randi_h200_smoke_1exam_12292820.err
```

- Run file target: `qc_redo/auto_qc_runs/randi_h200_smoke/qwen397b_vertical_line_randi_h200_smoke_1exam.json`.
- At 21:40 CDT, job `12292820` was running, saw `CUDA_VISIBLE_DEVICES=0,1,2,3`, and had resolved the repaired wrapper to the base checkpoint and repair cache.

#### Result

The one-exam smoke is running. No model inference result exists yet because the run file is written only after model load and exam scoring finish.

#### Decision Impact

Monitor job `12292820`. If it completes, inspect the run JSON and debug dump, then promote to a 3-exam smoke or the planned 100-300 exam review batch. If it fails, triage runtime logs from the direct sbatch job rather than the obsolete submitit `srun` failure.

### Auto-QC Pilot/Checkpointing Added and H200 Smoke Resubmitted (2026-06-25 23:30 CDT)

#### Question

Can we avoid wasting the ~80 minute Qwen warmup on tiny opportunistic jobs, and can the repaired model actually score after the missing runtime dependency is fixed?

#### Action

Measured the completed one-exam direct smoke attempt `12292994`. It reached `model ready` after loading base weights and all 60 repaired FP8 cache layers, then failed at first exam because the isolated `prima-qwen` model env was missing `qwen_vl_utils`:

```text
ModuleNotFoundError: No module named 'qwen_vl_utils'
```

Installed the missing package only into the isolated model env:

```bash
PYTHONNOUSERSITE=1 /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-qwen/bin/python -m pip install qwen-vl-utils
```

Implemented minimal opportunistic-safe auto-QC infrastructure:

- `prima/auto_qc.py`: atomic run-file writes and optional top-level `probe_tag` preservation.
- `auto_annotate_qc.py`: per-exam checkpoint/resume, compatibility checks before resuming a run file, SIGTERM/SIGINT shutdown flag, filesystem pilot queue with `pending/`, `running/`, `done/`, and `failed/`, stale running-task reclaim, and task status sidecars.
- `submit_auto_qc.py`: `--no-srun` option after the first submitit attempt showed nested `srun` fails on the allocated node.

Validated:

```bash
PYTHONNOUSERSITE=1 /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-qwen/bin/python -m py_compile auto_annotate_qc.py submit_auto_qc.py prima/auto_qc.py
PYTHONNOUSERSITE=1 /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima/bin/python -m ruff format auto_annotate_qc.py submit_auto_qc.py prima/auto_qc.py
PYTHONNOUSERSITE=1 /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima/bin/python -m ruff check --fix auto_annotate_qc.py submit_auto_qc.py prima/auto_qc.py
```

Also ran a non-GPU helper sanity check covering pilot queue creation, atomic task claim, task status sidecar write, run-file save, and `probe_tag` reload.

Created pilot smoke queue and sbatch:

```text
qc_redo/pilot_queues/randi_h200_smoke/pending/randi_h200_pilot_smoke_1exam.json
logs/qwen397b_vertical_line_randi_h200_pilot_smoke_1exam.sbatch
```

Submitted:

```bash
sbatch --parsable logs/qwen397b_vertical_line_randi_h200_pilot_smoke_1exam.sbatch
```

#### Evidence

- Failed direct smoke: job `12292994`, `FAILED ExitCode=1:0`, elapsed `01:20:29`, log `logs/slurm/qwen397b_vertical_line_randi_h200_smoke_1exam_12292994.err`.
- Warmup from that job: base weights loaded by `22:29:41`, repair cache layer 59 loaded and model ready at `23:20:10`; first useful scoring had not started before the missing-package failure.
- New pilot smoke: job `12293567`, partition/QoS `siweiq` / `opportunistic`, node `cri22cn418`, started at `2026-06-25T23:28:25-05:00`.
- New pilot run file: `qc_redo/auto_qc_runs/randi_h200_smoke/qwen397b_vertical_line_randi_h200_pilot_smoke_1exam.json`.
- New pilot debug dir: `qc_redo/auto_qc_debug/randi_h200_pilot_smoke_1exam`.
- At the first check, job `12293567` was running and the task remained in `pending/`, which is expected because the pilot claims work only after the model is resident.

#### Result

The code path is now resumable and pilot-capable, and the missing Qwen utility dependency is installed in `prima-qwen`. The live pilot smoke `12293567` is warming the model; no inference result exists yet.

#### Decision Impact

Do not launch small non-pilot opportunistic batches. If `12293567` reaches first exam scoring and writes a valid run JSON/debug dump, promote to a larger pilot queue for the 100-300 exam human-review batch. If it fails, triage `logs/slurm/qwen397b_vertical_line_randi_h200_pilot_smoke_1exam_12293567.err`; the next most likely failure point is actual generation/vision preprocessing, not model loading or missing `qwen_vl_utils`.

### H200 Pilot Smoke Succeeds with Torch 2.12 and DeepGEMM Disabled (2026-06-26 00:55 CDT)

#### Question

Can the repaired Qwen3.5 FP8 wrapper run one actual-data montage end-to-end in the isolated `prima-qwen` env on opportunistic H200s?

#### Action

Kept all model-stack changes isolated to:

```text
/gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-qwen
```

The separate `hfdp` project/env was not modified.

Updated `prima-qwen` to a maintained stack that has the FP8 APIs required by Transformers 5.12:

```text
torch 2.12.1+cu130
torchvision 0.27.1+cu130
triton 3.7.1
kernels 0.12.3
qwen-vl-utils installed
```

Added a runtime-only Transformers hub-kernel patch in `auto_annotate_qc.py` so compute nodes use cached kernel revisions rather than resolving `version=2` through the Hugging Face refs API:

```text
finegrained-fp8@061130fedf845f320c56de4425f7404f6512c87e
deep-gemm@9590415046fa95a187af7ea03391d4782047170d
```

Set the pilot sbatch environment to:

```bash
HF_HUB_OFFLINE=1
PRIMA_QWEN35_FP8_FORCE_BF16_EXPERTS=1
TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
```

#### Evidence

Intermediate failures:

- `12293567`: model ready, then failed on first generation with `torch.cuda.get_device_properties()` called without a device argument.
- `12293740`: fixed the no-device call, then failed because `kernels` was missing.
- `12293855`: `kernels==0.15.2` broke Transformers import because its `LayerRepository` API required `revision` or `version`.
- `12293873`: `kernels==0.12.3` restored import, but compute-node Hub access failed with `Connection refused`.
- `12293980`: pinned cached kernels and upgraded Torch; model ready, then DeepGEMM failed at first matmul with `Permission denied [/usr/local/cuda/bin/nvcc]`.

Successful smoke:

- Job: `12293994`.
- Scheduler result: `COMPLETED`, `ExitCode=0:0`, elapsed `00:11:31`, node `cri22cn418`.
- Script: `logs/qwen397b_vertical_line_randi_h200_pilot_smoke_1exam.sbatch`.
- Run file: `qc_redo/auto_qc_runs/randi_h200_smoke/qwen397b_vertical_line_randi_h200_pilot_smoke_1exam.json`.
- Queue task: `qc_redo/pilot_queues/randi_h200_smoke/done/randi_h200_pilot_smoke_1exam_retry4.json`.
- Debug dump: `qc_redo/auto_qc_debug/randi_h200_pilot_smoke_1exam/1.2.826.0.1.3680043.8.829.2.2.2.1.4223.500.json`.

Timing from `12293994`:

- Script start: `00:33:36`.
- Model load start: `00:33:45`.
- Repair layer 0 loaded: `00:38:38`.
- Repair layer 59/model ready: `00:43:13`.
- Task claimed: `00:43:18`.
- One-exam generation/checkpoint: `105.93s`.
- Task done/run JSON written: `00:45:04`.

Saved model response for exam `1.2.826.0.1.3680043.8.829.2.2.2.1.4223.500`:

```text
EVIDENCE: none
ANSWER: NO
CONFIDENCE: high
REVIEW: NO
```

Parsed result: no `vertical line (detector artifact)` suggestion.

#### Result

The repaired Qwen3.5 FP8 wrapper can now score actual-data cached montages end-to-end on Randi H200s with the isolated `prima-qwen` env. The practical warmup for the working configuration was about 9.5-10 minutes, followed by about 106 seconds for the first one-exam generation with the finegrained FP8 fallback.

#### Decision Impact

Use the pilot queue design for the review batch. Do not use DeepGEMM linear on these nodes unless `/usr/local/cuda/bin/nvcc` access is fixed or DeepGEMM is precompiled in a way that avoids that path. The next discriminative step is a small multi-exam pilot queue, then the planned 100-300 exam positive-enriched plus random-negative human-review batch if the multi-exam smoke remains stable.

### Resident Pilot Multi-Task Smoke Succeeds (2026-06-26 01:10 CDT)

#### Question

Can one opportunistic H200 allocation keep Qwen resident and process multiple pilot queue tasks without reloading the model?

#### Action

Created a separate multi-task smoke queue with two task files, two exams each:

```text
qc_redo/pilot_queues/randi_h200_multitask_smoke/pending/randi_h200_multitask_smoke_task01.json
qc_redo/pilot_queues/randi_h200_multitask_smoke/pending/randi_h200_multitask_smoke_task02.json
```

Submitted:

```bash
sbatch --parsable logs/qwen397b_vertical_line_randi_h200_pilot_smoke_multitask.sbatch
```

The script used the same working env/config as the one-exam success:

```bash
HF_HUB_OFFLINE=1
PRIMA_QWEN35_FP8_FORCE_BF16_EXPERTS=1
TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
```

#### Evidence

- Job: `12294049`.
- Scheduler result: `COMPLETED`, `ExitCode=0:0`, elapsed `00:12:33`, node `cri22cn418`.
- Run file: `qc_redo/auto_qc_runs/randi_h200_multitask_smoke/qwen397b_vertical_line_randi_h200_pilot_smoke_multitask.json`.
- Task status:
  - `task01` done at `01:02:12`.
  - `task02` done at `01:03:23`.
- Debug directory: `qc_redo/auto_qc_debug/randi_h200_multitask_smoke`.

Timing:

- Script start: `00:50:42`.
- Repair layer 0 loaded: `00:55:56`.
- Repair layer 59/model ready: `01:00:29`.
- Task01 claimed: `01:00:35`; 2 exams completed in `01:37`.
- Task02 claimed immediately at `01:02:12`; 2 exams completed in `01:10`.
- No second model load occurred between tasks.

Saved results:

```text
1.2.826.0.1.3680043.8.829.2.2.2.1.29994.400: no suggestions
1.2.840.113619.2.182.1602174323676128.1450373588.11611965: no suggestions
2.16.840.1.114151.2915498749809018983562502043350063692880241209: no suggestions
2.16.840.1.114151.4.886.41655.4068.8536816: no suggestions
```

#### Result

The resident pilot design works as intended: after the ~9.8 minute warmup, the same process claimed a second queued task and scored it without reloading Qwen.

#### Decision Impact

For the human-review batch, submit a small number of long-lived pilot workers with many small queue task files rather than one Slurm job per task. Keep task sizes modest so opportunistic preemption wastes only the current task, while per-exam checkpointing preserves completed exams in the run JSON.

### Actual-Data Review Queue Started (2026-06-26 08:01 CDT)

#### Question

Can we start a positive-enriched actual-data inference queue now so human review can proceed as model results arrive?

#### Action

Added `scripts/build_vertical_line_review_queue.py` to build a reproducible 240-exam queue from the actual-data-backed cached montage export:

```bash
PYTHONNOUSERSITE=1 /gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima-qwen/bin/python scripts/build_vertical_line_review_queue.py \
  --views qc_export/views_for_qc.parquet \
  --export-dir qc_export \
  --out-dir qc_redo/review_batches/vertical_line_actual_data_review \
  --queue-dir qc_redo/pilot_queues/vertical_line_actual_data_review \
  --total-exams 240 \
  --heuristic-exams 140 \
  --task-size 5 \
  --seed 20260626 \
  --exclude-exam-list qc_redo/review_batches/vertical_line_actual_data_review/exclude_smoke_exam_ids.txt
```

The smoke-test exam IDs were excluded. The queue builder scored 7,827 cached montages and selected:

- 140 heuristic-enriched exams.
- 100 random-negative-enrichment exams.
- 48 tasks of 5 exams each.

Created H200 opportunistic pilot scripts:

```text
logs/qwen397b_vertical_line_actual_review_pilot_siweiq.sbatch
logs/qwen397b_vertical_line_actual_review_pilot_catherineq.sbatch
```

Both use:

```bash
HF_HUB_OFFLINE=1
PRIMA_QWEN35_FP8_FORCE_BF16_EXPERTS=1
TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
```

Submitted:

```bash
sbatch --parsable logs/qwen397b_vertical_line_actual_review_pilot_siweiq.sbatch
sbatch --parsable logs/qwen397b_vertical_line_actual_review_pilot_catherineq.sbatch
```

#### Evidence

- Batch root: `qc_redo/review_batches/vertical_line_actual_data_review`.
- Queue root: `qc_redo/pilot_queues/vertical_line_actual_data_review`.
- Run file: `qc_redo/auto_qc_runs/vertical_line_actual_data_review/qwen397b_vertical_line_actual_data_review.json`.
- Debug dir: `qc_redo/auto_qc_debug/vertical_line_actual_data_review`.
- `siweiq` job: `12294521`, started on `cri22cn418`.
- `catherineq` job: `12294522`, initially pending with reason `Resources`.
- At submission, the queue had `pending 48`; `siweiq` was in model-load warmup.

#### Result

Inference is started with the resident pilot design. Expected first task claim is after the known ~9.5-10 minute warmup. With one H200 worker, the 240-exam queue is expected to take a few hours after warmup; a second H200 worker will reduce this if `catherineq` becomes available.

#### Decision Impact

Monitor task movement under `qc_redo/pilot_queues/vertical_line_actual_data_review/{running,done,failed}` and the shared run JSON. Human review can begin once the first tasks complete and the review gallery points at the run file and batch manifest.

#### First Live Checkpoint

At `2026-06-26 08:10:36 CDT`, job `12294521` reported `model ready`. It started the pilot at `08:10:41` and claimed `task_0001`.

`task_0001` completed at `08:14:03` with:

```json
{
  "scored": 5,
  "skipped_existing": 0,
  "interrupted": false
}
```

The first task produced 0 vertical-line positives. By `08:15 CDT`, `task_0002` was running and the shared run JSON had 7 scored exams, 0 positive exams. Clean queue counts then were:

```text
pending 46
running 1
done 1
failed 0
```

Dedicated human review QC state file:

```text
qc_redo/review_batches/vertical_line_actual_data_review/qc_state_vertical_line_actual_data_review.json
```

### Actual-Data Review Queue Completed (2026-06-27)

#### Question

Did the 240-exam actual-data vertical-line review queue finish cleanly, and what should the next agent do first?

#### Action

Checked Slurm accounting, queue directories, shared auto-QC run JSON, and the human-review QC state on `2026-07-10`.

#### Evidence

Slurm accounting:

```text
12294521 qwen397b_vline_review_siw siweiq     COMPLETED ExitCode=0:0 Elapsed=02:41:57 Start=2026-06-26T08:01:01 End=2026-06-26T10:42:58 Node=cri22cn418
12294522 qwen397b_vline_review_cat catherineq COMPLETED ExitCode=0:0 Elapsed=01:18:29 Start=2026-06-26T23:49:00 End=2026-06-27T01:07:29 Node=cri22cn417
```

Slurm logs:

```text
logs/slurm/qwen397b_vertical_line_actual_review_siweiq_12294521.out
logs/slurm/qwen397b_vertical_line_actual_review_siweiq_12294521.err
logs/slurm/qwen397b_vertical_line_actual_review_catherineq_12294522.out
logs/slurm/qwen397b_vertical_line_actual_review_catherineq_12294522.err
```

Terminal queue counts:

```text
pending 0
running 0
done 48
failed 0
```

Run file:

```text
qc_redo/auto_qc_runs/vertical_line_actual_data_review/qwen397b_vertical_line_actual_data_review.json
```

Run summary:

```text
run_id: 2026-06-26T130102+0000_Qwen3.5-397B-A17B-FP8-prima-repair
model: Qwen3.5-397B-A17B-FP8-prima-repair
probe_tag: vertical line (detector artifact)
scored_exams: 240
positive_exams: 13
```

Positive exam IDs:

```text
1.2.826.0.1.3680043.8.829.2.2.2.1.28084.300
2.16.840.1.114151.1647600044392085289904072242927255158290191016
2.16.840.1.114151.1976139852882831678302369073516359450130201029
2.16.840.1.114151.254921799333471188989637772138308173330211026
2.16.840.1.114151.2867159520634596470693698205547951816210220428
2.16.840.1.114151.3223932336907247163893937230426355317840240102
2.16.840.1.114151.3359343871087068672067392237092969119250180517
2.16.840.1.114151.4.231.41021.3677.6431087
2.16.840.1.114151.4.231.42129.4596.8576028
2.16.840.1.114151.4.231.42225.6145.8792576
2.16.840.1.114151.462829609239406492196060964695032725010170511
2.16.840.1.114151.588878425608523128672406379005865671760231221
2.16.840.1.114151.860317488921187770485453327821119421970512
```

Batch manifest:

```text
qc_redo/review_batches/vertical_line_actual_data_review/manifest.csv
```

Manifest summary:

```text
rows: 240
heuristic_top: 140
random_negative_enrichment: 100
```

Human-review state:

```text
qc_redo/review_batches/vertical_line_actual_data_review/qc_state_vertical_line_actual_data_review.json
```

As of `2026-07-10`, the human-review state had 0 entries.

#### Result

The actual-data review inference queue completed successfully. The model produced 13 suggested vertical-line positives out of 240 selected exams, but no human review has been recorded in the dedicated QC state file yet.

#### Decision Impact

The next discriminative step is human review of the 240-exam batch, then evaluation of the reviewed model positives and negatives. A compact restart handoff is available at:

```text
logs/vertical_line_actual_data_handoff.md
```

### Exhaustive Image-Outlier Scoring Submitted (2026-07-10 11:27 CDT)

#### Question

Can the existing image/header outlier analysis be run exhaustively over the archive-aware mammography cache while automated QC continues separately?

#### Action

Added the fail-fast Slurm launcher `logs/image_outliers_exhaustive.sbatch` and submitted it with:

```bash
sbatch --parsable logs/image_outliers_exhaustive.sbatch
```

The scorer is NumPy/Zarr/PIL work parallelized with `ProcessPoolExecutor`; it has no GPU code path. The job therefore uses routine nonpreemptible CPU capacity rather than reserving an idle GPU:

```text
partition: tier1q
qos: normal
account: huo-lab
cpus: 16
memory: 64G
walltime: 06:00:00
```

#### Evidence

- Job: `12747725`.
- Initial state: `RUNNING` on `cri22cn035` at `2026-07-10 11:26:22 CDT`.
- Inputs: archive-aware manifest, Zarr cache, views table, and DICOM-tag table under `/gpfs/data/huo-lab/Image/ChiMEC/MG/`.
- Manifest rows announced by the scorer: `82,384` cached views.
- Output root: `data/qc_image_outliers_exhaustive`.
- Logs: `logs/slurm/image_outliers_exhaustive_12747725.out` and `logs/slurm/image_outliers_exhaustive_12747725.err`.
- Startup stderr was empty.
- The prior 5,000-view artifacts under `data/qc_image_outliers` are preserved unchanged.

#### Result

Job `12747725` completed successfully with `ExitCode=0:0` after `00:10:34`. It scored all `82,384` manifest views across `20,596` exams with zero load errors and wrote all expected artifacts under `data/qc_image_outliers_exhaustive`:

```text
README.md
image_outlier_scores.csv
exam_outlier_scores.csv
review_exam_ids.txt
top_outlier_contact_sheet.png
```

Integrity checks found zero duplicate view keys, zero duplicate exam keys, zero non-finite outlier scores, and exactly four scored views for every exam. Output files were written mode `0600`; the score tables remain ignored local data because they contain identifiers.

Aggregate score summaries:

```text
image outlier score: median 2.240, p95 11.923, p99 14.073, max 20.000
exam outlier score:  median 3.197, p95 13.421, p99 15.558, max 20.000
exam triage score:   median 3.225, p95 18.510, p99 20.487, max 23.454
```

All top-80 triage images carried at least one deterministic rule flag. Their leading statistical features were horizontal-line score (`38`), median pixel intensity (`30`), high-pixel fraction (`9`), and relative X-ray exposure (`3`). Visual inspection of `top_outlier_contact_sheet.png` agreed with the ranking: the highest rows are dominated by burned-in/marked-up film-like images and several GEMS-processing device/paddle views.

#### Decision Impact

Exhaustive scoring is complete and ready to link to the corrected Mirai predictions. Because the operational triage tail is entirely dominated by known rule flags, performance analysis should keep the continuous `outlier_score` separate from `triage_score` and report deterministic flag categories explicitly rather than interpreting the combined ranking as an independent QC signal.

### Native vLLM Auto-QC Migration Started (2026-07-10)

#### Question

Can Prima replace the model-specific Qwen3.5 Transformers repair path with a
maintained open-weight serving stack while preserving the existing montage,
queue, run-file, and human-review workflow?

#### Action

Audited private repository `dsi-clinic/local-llms` at commit
`e39d4c1909b0957376f589f04d59803eb778ede4`. It is an EHR text-extraction
application, not an installable inference library. Adapted its useful managed
vLLM/OpenAI service boundary into Prima instead of importing the application.

Implemented:

- Pinned vLLM/OpenAI environment in `env-vllm.yaml`.
- Pinned model registry in `qc/auto_qc_models.json` for the existing upstream
  Qwen3.5-397B FP8 checkpoint (4 H200) and Qwen3.5-27B FP8 fallback (1 H200).
- Loopback-only managed server, health check, proxy isolation, request/access
  log suppression, process-group teardown, exact runtime preflight, visible-GPU
  validation, and complete-shard/revision validation in `prima/vllm_server.py`.
- Multimodal base64 OpenAI requests, strict structured output, mode-specific
  prompts, fail-closed parsing, sanitized API errors, and generation-setting
  resume compatibility in `auto_annotate_qc.py` and `prima/auto_qc.py`.
- Reproducible download and migration-smoke input builders under `scripts/`.
- Operator documentation in `docs/auto_qc_vllm.md`.

Created a deterministic two-exam smoke input containing one prior
model-positive and one prior model-negative montage without logging identifiers:
`qc_redo/review_batches/vllm_migration_smoke/exam_ids.txt`. These are agreement
probes, not human ground truth.

#### Evidence

- Focused tests: `11 passed` in `tests/test_vllm_auto_qc.py`.
- Targeted Ruff checks, compilation, and `git diff --check`: clean.
- Existing Qwen3.5-397B snapshot resolves to pinned revision
  `ea5b4f81096f3901c91dea97f81324302495781d`; every indexed weight shard is
  present and non-empty.
- The completed 240-exam baseline used `marker_classifier`,
  `confidence_specificity`, and zero few-shot examples; the migration smoke
  freezes those same prompt settings.
- Dedicated `prima-vllm` environment installation and the pinned 27B snapshot
  download are in progress. `/tmp` is `noexec`, so installation uses executable
  scratch `TMPDIR`; Hugging Face Xet is disabled after a reproducible stall.

#### Result

The non-GPU migration path is implemented and validated. No accuracy claim is
made, and no full scan has been launched. The vLLM path uses upstream FP8
weights and native kernels, whereas the prior baseline used repaired BF16
experts, so agreement is diagnostic rather than ground truth.

#### Decision Impact

Run the 27B model first on the two-exam smoke to validate transport, parsing,
checkpointing, and teardown cheaply. Then run the original 397B FP8 checkpoint
on four H200s over the same inputs. Apply at most one targeted runtime fix for a
clear startup failure; if four-H200 397B serving remains infeasible, pivot to
27B rather than adding another checkpoint repair. Do not expand to the 240-exam
batch until a reviewed balanced panel can estimate sensitivity and false
positive behavior.

### Mirai Pre-QC Outlier Robustness Campaign Submitted (2026-07-10)

#### Question

How much do outcome-blind image/header outlier scores change corrected Mirai
performance before manual or automated QC is available, after separating known
rule flags, acquisition context, repeated exams, and label-definition effects?

#### Action

Implemented a config-driven, aggregate-only analysis in
`analysis/analyze_mirai_outlier_robustness.py`, configured by
`configs/mirai_outlier_robustness.yaml`. It uses exhaustive scores from
`data/qc_image_outliers_exhaustive`, exact calendar-date labels with a true
six-month washout, frozen global and rule-unflagged score tails,
patient-weighted exam-level AUC, case-mix-dependent Brier sensitivity, patient
bootstrap, and an exact-stratified patient-cluster permutation null. It also
reports acquisition, time-to-cancer, repeat-exam, label-source, legacy-label,
and removed-tail-composition diagnostics.

Two read-only reviews caught and motivated fixes before the definitive launch:

- exam-wise pseudo-deletion was replaced by whole patient-tail-mask permutation
  so affected patients, complete patient loss, repeated-exam clustering, and
  row-level matching strata remain fixed;
- label reclassification now uses the outer union of exact and legacy cohorts;
  plots use actual eligible-exam removal fractions and literal confidence
  interval endpoints; baseline primary AUCs receive patient-bootstrap CIs;
  tail acquisition/flag composition is persisted; context minima are applied
  after horizon eligibility; and absolute calibration is explicitly not claimed.

The first submission, job `12759090`, was canceled while still pending after
the final audit found the three reporting issues above. No output directory was
created. The corrected full run was then submitted with:

```bash
sbatch --parsable logs/mirai_outlier_robustness.sbatch
```

#### Evidence

- Corrected tier1 job `12760765` was canceled while still pending behind the
  CPU backlog. Final job `12761173` started at `2026-07-10 14:07:41 CDT` on
  spare CPU capacity of `cri22cn501` without requesting a GPU.
- Resources: `sxmq`, `priority` QoS, one CPU, 16 GB, six hours.
- Full inferential budget: 2,000 patient-bootstrap replicates, 1,000 matched
  permutations, and 500 quintile-bootstrap replicates.
- Exact-current-source smoke output: `/tmp/prima_mirai_outlier_smoke8`.
- Static checks: targeted Ruff, Python compilation, and launcher syntax passed.
- Definitive aggregate output root:
  `/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix/outlier_robustness`.

#### Preliminary Audit Findings

- Predictions match exhaustive scores for 20,405/20,412 exams (99.966%).
- The exact six-month 1-year cohort has 155 case exams in 152 patients, making
  its confidence interval essential.
- Diagnosis-source conflicts occur in 675 exam rows across 91 patients, but an
  alternate precedence changes only one 1-year and 25 five-year case statuses.
- At the primary global top-5% threshold, 99.4%/99.3% of removed 1-year/5-year
  exams are rule-flagged and 95.5%/95.2% are GE exams.
- Exact matching forces 81.3% of global top-5% removed exams at one year and
  70.4% at five years; this is only a partial conditional diagnostic. The
  rule-unflagged analysis is more exchangeable (42.6%/35.6% forced).

#### Result

Job `12761173` completed successfully with `ExitCode=0:0` after `00:11:23`;
peak batch RSS was about 204 MB. It wrote 22 expected aggregate artifacts. All
22 tail analyses retained all 2,000 requested patient-bootstrap replicates and
all 1,000 requested matched permutations; every condition also produced 1,000
unique null draws. Aggregate-only/finite-value checks, threshold nesting,
tail-composition sums, post-eligibility context minimums, restrictive file
permissions, provenance, and visual inspection of all three figures passed.

Primary exact-date baseline results after the true six-calendar-month washout:

```text
1y AUC 0.5932 (95% patient-bootstrap CI 0.5388-0.6455); 155 case exams / 152 patients
5y AUC 0.6209 (95% patient-bootstrap CI 0.6005-0.6413); 2,046 case exams / 974 patients
```

Primary top-5% exclusion results:

```text
global 1y:    delta AUC +0.0033 (95% CI -0.0036 to +0.0127)
global 5y:    delta AUC -0.0008 (95% CI -0.0035 to +0.0020)
unflagged 1y: delta AUC +0.0089 (95% CI -0.0012 to +0.0218)
unflagged 5y: delta AUC +0.0007 (95% CI -0.0045 to +0.0064)
```

Every primary exclusion-effect interval includes zero. The unflagged top-10%
1-year sensitivity has a positive delta (`+0.0148`, CI `+0.0030` to `+0.0304`),
but its observed change lies inside the context-matched permutation envelope
(`+0.0126` to `+0.0195`; empirical percentile `0.280`). This is evidence that
deleting that case-mix/context pattern can change AUC, not that high statistical
outlier score specifically harms Mirai.

The dominant result is confounding, not QC failure. Global top-5% exclusions
are about 99% deterministic-rule-flagged, 95% GE, and two-thirds from 2001-2010.
Across all kept exams, GE has median outlier score `10.92` and 100% rule-flagging
versus Hologic median `2.11` and 2.0% flagging; 2001-2010 median is `10.04`
versus roughly `2.2-2.3` after 2015. First repeated exams are similarly enriched
for older acquisition (`median 8.50`, 50.4% flagged) relative to later exams
(`median 2.57`, 23.1% flagged). The score is therefore largely an acquisition
and deterministic-header signature in the global tail.

The label definition changes more than outlier exclusion. Exact prediagnostic
AUC is `0.7305` at one year and `0.6991` at five years, dropping to `0.5932` and
`0.6209` after the true six-month washout. At one year, the exact and legacy TTC
proxy use disjoint case sets: all 155 exact 6-12-month cases are legacy-ineligible,
while 630 exact controls are legacy cases because integer flooring shifts the
interval. Thus the legacy `years_to_cancer >= 0.5` result is not a faithful
six-month analysis. Diagnosis-source precedence is a smaller uncertainty: an
alternate precedence changes only one 1-year and 25 five-year case statuses.

#### Sensemaking

The nearest reference is the Omoleye paper's true-washout AUC of about `0.64`
at one year and `0.63` at five years. The widened exact-date cohort's five-year
estimate agrees closely; the one-year estimate is lower but its interval reaches
the reference value and is based on only 152 independent case patients. A real
outlier-induced failure would predict consistently positive exclusion effects
that exceed matched deletions across thresholds and horizons. That pattern is
absent. Wide 1-year quintile intervals and non-monotonic 5-year quintiles also
argue against reading a causal score-performance gradient from the plots.

#### Decision Impact

Do not use statistical outlier exclusion as a substitute for QC, and do not
drop the raw top-score tail by default while automated QC is pending. The current
evidence says corrected Mirai discrimination is not materially fragile to the
prespecified top-5% score tail. Preserve the frozen thresholds and tail tables
as a pre-QC benchmark, make exact calendar-date labeling the primary evaluator,
and treat the integer-label results as legacy reproduction only. When automated
QC arrives, cross-tabulate its labels against deterministic flags, manufacturer,
era, and the frozen global/unflagged score tails, then repeat the same AUC-delta
analysis without tuning cutoffs to outcomes. Absolute calibration remains out of
scope until defensible population-sampling weights are available.

### 2026-07-10 — Mirai longitudinal consistency and prediagnostic change

#### Context and question

The SABCS submission needs a scientifically defensible result before manual QC
is available. The expanded data are from the same high-risk cohort as the prior
Omoleye/Woodard/Huo validation, not a new target population. The existing
trajectory script aligned cases to diagnosis and controls to their last exam;
both groups rose toward their anchor, so that analysis could not establish a
cancer-specific trajectory and used the wrong diagnosis-date precedence for the
current evaluator.

The new question was whether Mirai estimates show within-patient longitudinal
consistency across approximately annual mammograms, and whether prediagnostic
change exceeds ordinary aging/acquisition drift. This is explicitly not a QC
replacement and is called longitudinal consistency rather than test-retest
reliability because real biological change is expected between annual exams.

#### Implementation and design

Added:

- `analysis/analyze_mirai_longitudinal_stability.py`
- `configs/mirai_longitudinal_stability.yaml`

The shared exact-date loader in
`analysis/analyze_mirai_outlier_robustness.py` now also retains
`age_at_exam_years` in memory for matching. The longitudinal producer:

- uses corrected expanded predictions from `out_all_modefix`;
- uses diagnosis precedence `datedx`, then `date_diagnosis`, then
  `datedx_new`, with no floored-label fallback;
- forms consecutive pairs before applying the current implant/scanned-film
  exclusions, then requires both endpoints to survive;
- defines annual intervals as 0.5-1.5 years;
- uses phenotype controls with at least one exact year of follow-up after the
  later exam for the primary stability analysis;
- compares same exact model, model change within manufacturer, and manufacturer
  change;
- matches future-case same-model pairs after a true six-calendar-month washout
  to up to five control pairs, exact on device model and within calipers of five
  years for age, two years for calendar time, and 0.25 years for interval;
- uses 2,000 patient-cluster bootstrap replicates;
- persists aggregate-only tables and figures; pair and match identifiers remain
  in memory.

Exact full command:

```bash
eval "$(micromamba shell hook -s bash)"
micromamba activate prima
python analysis/analyze_mirai_longitudinal_stability.py configs/mirai_longitudinal_stability.yaml
```

Targeted smoke runs, `ruff`, `py_compile`, aggregate-only/finite-value checks,
file-permission checks, and visual inspection of all three figures passed.

#### Result

The source loader found 20,405 predicted exams with acquisition/outlier context
in 3,674 patients. The current policy retains 18,630 exams in 3,599 patients.
There were 11,836 annual consecutive pairs in 2,453 patients. The primary
control analysis contained 4,906 pairs in 1,140 patients, including 2,923
same-model pairs in 974 patients.

Acquisition-conditional consistency was strong and showed a stepwise gradient:

```text
                                  1-year risk                  5-year risk
same model:                       rho 0.7919 (0.7666-0.8146)   rho 0.8228 (0.8021-0.8408)
same manufacturer, model change: rho 0.6473 (0.5989-0.6907)   rho 0.6904 (0.6477-0.7279)
manufacturer change:             rho 0.1753 (0.0837-0.2634)   rho 0.1757 (0.0823-0.2646)
```

Two-or-more-quintile shifts occurred in 12.8%/9.6% of same-model
1-year/5-year pairs versus 43.4%/45.1% after manufacturer change. Median
absolute logit changes were 0.117/0.134 on the same model versus 0.329/0.362
after manufacturer change. Same-model correlations remained strong with five
years of control follow-up (0.805/0.836), rule-unflagged pairs (0.781/0.808),
modern Hologic pairs (0.736/0.756), and one pair per patient (0.762/0.797).

The transition effect also survived a paired within-patient check. Among 433
controls contributing both same-model and manufacturer-change intervals, the
mean absolute logit change was 0.351 larger for 1-year risk (95% CI
0.284-0.422) and 0.274 larger for 5-year risk (0.234-0.316) during the
manufacturer-change interval.

Matching retained 860/945 same-model future-case pairs (91.0%) overall. In the
0.5-2-year window it retained 230 pairs in 213 patients, with five controls per
case typically and mean absolute differences of 0.96 years in age, 0.24 years
in calendar time, and 0.038 years in inter-exam interval. Median excess
annualized logit-risk change versus matched controls was positive:

```text
1-year risk: +0.051 (95% matched-set patient-bootstrap CI +0.011 to +0.137)
5-year risk: +0.068 (95% matched-set patient-bootstrap CI +0.036 to +0.101)
```

The 2-5-year and >=5-year windows centered near zero. The corresponding mean
effects in the final two years were +0.101 (CI -0.018 to +0.215) and +0.053
(-0.013 to +0.117), so the trajectory finding is distribution-sensitive. A
strict within-case comparison on the same exact device model at both distant
and imminent intervals was positive at all horizons but contained only 19
patients and is secondary. Rule-unflagged matched sensitivities retained the
direction of the final-two-year median shift.

#### Sensemaking and skeptical review

Stable patient rank on unchanged acquisition is plausible because Mirai
contains persistent anatomy/risk factors, while annual aging produces modest
true change. If the acquisition transition were irrelevant, model and
manufacturer changes should not sharply reduce correlation or inflate absolute
change, especially within the same patients; both falsifiers failed. The large
manufacturer effect is nevertheless observational, dominated by GE/Hologic
transitions, and is not a randomized device experiment.

If cancer-specific acceleration were strong and homogeneous, both median and
mean matched effects would be positive with intervals excluding zero and the
effect would persist broadly across the final two years. Only the robust median
and the smaller 0.5-1-year subgroup clearly support change; mean intervals cross
zero. Therefore the data support an acquisition-conditional consistency claim
and only an exploratory prediagnostic-change claim.

#### Decision impact

Lead the SABCS abstract with the acquisition-context result: serial Mirai
estimates are reasonably consistent on an unchanged model but are not directly
comparable across manufacturer changes. Use prediagnostic change as a secondary,
explicitly exploratory observation. Do not call the analysis QC, QC-clean,
population calibration, clinical readiness, or validation in a new population.

A 2,386-character submission draft (excluding whitespace, including title and
AI disclosure) is at
`docs/abstracts/sabcs_2026_mirai_longitudinal_draft.md`.

Primary artifact directory:

`/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix/longitudinal_stability`

### 2026-07-10 — Native vLLM auto-QC migration completed

#### Goal and design

Restarted the vertical-line auto-QC work by auditing the private
`dsi-clinic/local-llms` repository at commit
`e39d4c1909b0957376f589f04d59803eb778ede4`. The repository is a text-only EHR
application rather than a reusable multimodal package. Prima adopted its useful
boundary—a managed local `vllm serve` process plus an OpenAI-compatible client—
while retaining Prima's montage, queue, checkpoint, and gallery workflow.

The original `Qwen/Qwen3.5-397B-A17B-FP8` checkpoint remained the first test.
The prespecified fallback was the official Apache-2.0
`Qwen/Qwen3.5-27B-FP8`, not another repaired Transformers wrapper. The runtime
smoke used two existing actual-data montages: one prior-model positive and one
prior-model negative. This tests transport and regression behavior only; these
are not human labels.

#### Implementation

Added a pinned, isolated `prima-vllm` environment and native server/client path:

- vLLM 0.24.0, OpenAI 2.45.0, Torch 2.11.0+cu130, NumPy 2.3.5, and NVIDIA
  CUDA `nvcc` 13.0.88;
- immutable model revisions and complete-shard validation in
  `qc/auto_qc_models.json`;
- loopback-only ephemeral serving, disabled request/access logs, direct health
  checks, structured JSON output, sanitized failures, and reliable teardown;
- executable `TMPDIR`, complete CUDA-toolkit, tensor-parallel, and GPU-count
  preflights;
- Qwen image profiling capped at 2,097,152 pixels, above the measured smoke
  montage area;
- for 27B on H200, native DeepGEMM FP8 linear kernels, native vLLM sampling,
  and Triton/FLA GDN prefill. Optional FlashInfer FP8 GEMM/sampling paths are
  disabled because their packaged JIT paths failed on the cluster image.

The CUDA toolchain required several cluster-specific corrections discovered by
actual GPU startup: `/tmp` is `noexec`; `/usr/local/cuda` is not readable by
jobs; NVIDIA's conda toolkit stores headers under a target directory and the
`cicc` stage under `nvvm/bin`. The managed server now points `CUDA_HOME` at the
conda toolkit root, prepends `nvvm/bin`, and keeps FlashInfer workspace files in
executable scratch. Login-node automatic Torch selection had also installed a
CPU build initially; the environment now installs the explicit `cu130` backend.

#### GPU evidence and decision

The 397B checkpoint is natively compatible with vLLM tensor parallelism on four
H200s. Job `12761187` reached TP initialization, but GPFS loading took 282.3
seconds for the first of 94 shards. Prefetch in job `12761456` reduced the first
shard to about one second, but device loading still reached only shard 4 after
6m23s, projecting to hours. Both jobs were stopped. This is an operational
rejection, not a model-compatibility failure.

The 27B path loaded 28.54 GiB on one H200. Startup failures were reproduced and
fixed rather than hidden: no-exec runtime code (`12762079`), an excessive
checkpoint-default vision profile (`12762167`, `12762218`), FlashInfer FP8 JIT
failure (`12762247`), inaccessible system CUDA (`12762271`), and conda CUDA
layout issues (`12762289`, `12762753`, `12763469`). Final job `12764322`
completed with exit code 0. It loaded weights in 46.39 seconds on a cold page
cache, completed CUDA-graph profiling, warmed DeepGEMM, became healthy, returned
HTTP 200 for both image requests, wrote two structured exam records, and tore
down the allocation. Both records parsed successfully; one contained the target
tag and one did not. Compared internally by exam key to the prior repaired 397B
run, the prior positive stayed positive and the prior negative stayed negative
(2/2 agreement). No identifiers or raw responses were copied into this log.

After adding model-scoped environment switches to frozen run compatibility
metadata, the smoke artifact was regenerated end to end. Final reproducibility
job `12765660` completed with exit code 0 in 3m52s, again returned two HTTP 200
responses and the same positive/negative pattern, and recorded both the server
arguments and environment settings in `inference_settings`.

Primary smoke artifact:

`qc_redo/auto_qc_runs/vllm_migration_smoke/qwen35_27b_fp8_vllm_zero_shot.json`

#### Sensemaking and next decision

The migration is technically successful: native open-source Qwen 27B now serves
multimodal QC through the adopted local-LLM lifecycle without the old
checkpoint-repair loader. Two-case agreement only falsifies a gross migration
regression. It cannot establish sensitivity or specificity because the old
model is not ground truth and the dedicated review state still has no human
labels.

Do not launch a cohort-wide QC campaign yet. The next decision-changing
experiment is a human-reviewed balanced panel containing all 13 prior
vertical-line positives plus matched predicted negatives, scored under the
frozen 27B prompt. Use those labels to estimate whether 27B has enough recall
for triage and whether prompt/model changes are warranted.

### 2026-07-10 — `local-llms` dependency re-audit

#### Question and action

Re-audited `dsi-clinic/local-llms` after the request to import it as a
dependency rather than reproduce its implementation. Fetched `origin/main` in
the existing audit checkout and confirmed that both local and remote remain at
commit `e39d4c1909b0957376f589f04d59803eb778ede4`. Inspected the complete tree,
`pipeline/vllm_launch.py`, `scripts/serve.py`, `scripts/run_pipeline.py`, and
the serving environment.

#### Evidence and result

The repository currently has no `pyproject.toml`, `setup.py`, or `setup.cfg`,
so it cannot be installed as a PEP 508 Git dependency. Its importable-looking
top-level package is generically named `pipeline`, and `pipeline/__init__.py`
mutates `sys.path` to add the checkout root. The only serving logic exposed in
a module is the 45-line `pipeline/vllm_launch.py`: a local-directory existence
check and a basic `vllm serve` argument builder. The health poll, subprocess
lifecycle, and OpenAI client are embedded in `scripts/run_pipeline.py`, not a
reusable API. Its model registry is text-model-specific and lacks Prima's
immutable revision, complete-weight, tensor-parallel, environment, loopback,
CUDA-toolkit, and executable-scratch requirements.

Prima already imports the maintained dependencies that perform the substantive
work: vLLM for model serving and the OpenAI Python client for requests. A direct
`local-llms` import is therefore not currently possible. Adding its checkout to
`sys.path`, vendoring the module, or importing the generic `pipeline` package
would create an unpinned/path-dependent integration and was rejected.

#### Decision impact

Keep the current thin Prima integration rather than introduce a fake
dependency that is less reproducible and less safe. The next dependency step,
if desired, belongs upstream: turn `local-llms` into an installable,
non-application-named package; move the managed server lifecycle into its
public API; parameterize cluster/model policy; then pin Prima to an immutable
Git revision. Until that exists, importing it would remove only a few lines of
command assembly while weakening the validated QC serving path. So what: no
Prima runtime code changed in this re-audit, and the native vLLM smoke result
remains valid.

### 2026-07-10 — Cross-validated incremental value of Mirai score change

#### Question and decision rule

Test whether a two-exam Mirai trajectory should enter the SABCS abstract as an
incrementally predictive result. The nearest baseline was the latest
horizon-specific Mirai score. The candidate added annualized within-patient
logit-score change. The 2-year horizon was primary because the preceding
matched analysis localized the possible change signal to the final two years.

The positive claim required all of the following: a patient-bootstrap delta-AUC
interval excluding zero; positive direction after age/calendar/device context
adjustment; positive direction in rule-unflagged same-model exams; at least 80%
positive patient-level CV partitions; and a gain exceeding the manufacturer-
change technical negative control.

#### Action and implementation

Added:

- `analysis/analyze_mirai_incremental_trajectory.py`
- `configs/mirai_incremental_trajectory.yaml`

The analysis used exact-date outcomes and the true six-calendar-month washout.
Primary pairs were 0.5-1.5 years apart on the same exact acquisition model with
both endpoints surviving current implant/scanned-film exclusions. Five-fold
cross-validation was grouped by patient and repeated across 10 fixed partitions.
Each patient had equal total training/evaluation weight. The primary
two-feature combination used unpenalized logistic regression; the context
sensitivity added age, calendar year, interval, and exact model indicators with
a fixed L2 penalty. Rule-unflagged same-model pairs and manufacturer-change
pairs were prespecified sensitivities.

A smoke run exposed that pooling probabilities from separately fitted folds can
distort AUC through fold-specific offsets, especially with only 46 one-year
case pairs. Before the full run, the evaluator was corrected to compute paired
AUCs within held-out folds and macro-average them. Its 2,000-replicate interval
resamples patient clusters within the prespecified folds. This was a methods
correction before trusting or reporting the full result, not an outcome-tuned
choice. The unpenalized primary model also ruled out arbitrary shrinkage of the
correlated latest-score/change terms.

Exact full command:

```bash
eval "$(micromamba shell hook -s bash)"
micromamba activate prima
python analysis/analyze_mirai_incremental_trajectory.py configs/mirai_incremental_trajectory.yaml
```

#### Evidence and result

The primary 2-year cohort contained 3,642 same-model pairs in 1,378 patients,
including 258 case pairs in 239 patients.

```text
latest score fold-macro AUC:          0.6111 (95% CI 0.5702-0.6509)
latest score + annualized change AUC: 0.6109 (95% CI 0.5700-0.6503)
paired delta AUC:                    -0.0002 (95% CI -0.0083 to +0.0083)
```

Only 1/10 patient-level partitions had a positive macro delta; the median
partition delta was -0.0008. The context-adjusted delta was -0.0003 (CI -0.0052
to +0.0048). The rule-unflagged delta was -0.0077 (-0.0170 to +0.0017), and the
manufacturer-change negative-control delta was -0.0011 (-0.0306 to +0.0312).
The primary bootstrap retained all 2,000 replicates. Five-year gains were also
near zero in the full same-model cohort (+0.0007, CI -0.0052 to +0.0067); the
one-year combination was unstable and worse with only 46 case pairs and is not
a basis for a claim.

All five prespecified decision gates were evaluated; the overall
incremental-value criterion failed. Aggregate-only/finite-value checks, `ruff`,
`py_compile`, restrictive output permissions, and visual inspection of the AUC
gain/ROC figure passed.

Primary artifact directory:

`/gpfs/data/huo-lab/Image/ChiMEC/MG/out_all_modefix/trajectory_incremental_value`

#### Sensemaking and decision impact

The null incremental result is compatible with the preceding trajectory
analysis: a robust median score shift near diagnosis can occur without reliably
reordering patients once the latest mammogram score is known. The latest score
may already absorb the visible disease-related signal, while change is
heterogeneous relative to ordinary within-patient variation. A simple
two-exam linear change feature therefore does not add demonstrated
discrimination in this cohort.

Do not claim incremental predictive value. Keep acquisition-conditional
longitudinal consistency as the abstract's primary result, retain the near-
diagnosis median shift as exploratory, and add the negative cross-validated
result as an important boundary. The SABCS draft was updated accordingly and is
now 2,746 characters excluding whitespace:

`docs/abstracts/sabcs_2026_mirai_longitudinal_draft.md`

### 2026-07-10 — Installable `local-llms` package prepared for review

#### Goal and action

Converted the reusable serving layer in `dsi-clinic/local-llms` into an
installable dependency. Created the stable checkout
`/gpfs/data/huo-lab/Image/annawoodard/local-llms`, branched current upstream
`main` (`004d546`) as `package-reusable-vllm-api`, and committed the work as
`b729e8ef32e2074f9b774902c45facb37cc4b8de` (`Package reusable vLLM server
lifecycle`). The branch remains local and unpushed for user review.

#### Evidence and result

Added `pyproject.toml` for distribution `dsi-local-llms` and a public
`local_llms` package containing validated offline path resolution, server
configuration, argv/environment construction, loopback port selection, health
checks, early-exit log diagnostics, and full process-group teardown. Moved the
upstream EHR runner and standalone server to that same API; the wheel includes
only `local_llms`, not the application-specific `pipeline` namespace.

Validation passed: seven pytest unit tests, Ruff, compile checks, both upstream
CLI import/dry-run paths, a clean Python 3.11 wheel install, and a pinned
`git+file` install from commit `b729e8e`. The install check confirmed version
`0.1.0`, successful `import local_llms`, and no installed `pipeline` module.
The working tree is clean. No push, PR, Prima dependency edit, or GPU job was
performed.

#### Decision impact

The upstream blocker is resolved locally: after user review and an upstream
push/merge, Prima can pin the Git commit and replace its copied portable server
lifecycle while retaining Prima-specific model snapshot, CUDA, executable
scratch, and QC policy checks. So what: review `b729e8e` first; do not modify
Prima's runtime dependency until the commit is reachable from GitHub.

### 2026-07-10 — Prima validated against the unpushed `local-llms` build

#### Goal and action

Tested whether Prima could consume upstream commit
`b729e8ef32e2074f9b774902c45facb37cc4b8de` before it is pushed. Built a local
wheel with the Python 3.11 `prima-vllm` environment, recorded provenance at
`/scratch/annawoodard/builds/dsi-local-llms/b729e8ef32e2074f9b774902c45facb37cc4b8de/README.md`,
and installed the wheel into `prima-vllm`. SHA-256 is
`e2966201ab59b4a9e9016903462d0291d7d6552814ada109a3d96921b8afa806`.

Refactored Prima's uncommitted vLLM migration code so
`prima/vllm_server.py` retains model snapshot and cluster/CUDA policy but
delegates command construction, port selection, health checks, early-exit
diagnostics, and process-group teardown to the installed `local_llms` package.
Frozen inference metadata now includes `dsi-local-llms`; the runtime preflight
requires version `0.1.0`. Updated the focused tests and local-build instructions
in `docs/auto_qc_vllm.md` and `env-vllm.yaml`.

#### Evidence and result

Ruff and compile checks passed. The base Python 3.8 Prima environment passed 14
tests with the dependency-only adapter test skipped; `prima-vllm` executed all
15 tests successfully. Runtime validation resolved the vLLM executable and all
three pinned distributions (`dsi-local-llms 0.1.0`, `openai 2.45.0`, `vllm
0.24.0`).

GPU smoke job `12772550` ran on one opportunistic H200 and completed `0:0` in
4m09s. The dependency-managed server used the expected loopback and frozen
kernel/model arguments, loaded 28.54 GiB, completed CUDA graph capture, produced
exactly two structured records with zero runtime failures, and released the
allocation. Output is
`qc_redo/auto_qc_runs/vllm_migration_smoke/qwen35_27b_fp8_local_llms_zero_shot.json`;
its runtime metadata contains all three pinned versions. It used the same two
exam keys as the prior native-vLLM smoke and matched both classifications (2/2)
without printing identifiers.

#### Decision impact

The local package is sufficient to build and validate Prima before any push;
the copied portable server lifecycle is no longer needed. The remaining
engineering step is reproducibility after review: push/merge upstream, replace
the temporary checksummed-wheel instruction with an immutable Git dependency,
and reinstall from that reachable commit. So what: no further GPU migration
work is needed before upstream review; scientific validation still requires the
human-reviewed positive/negative panel.

### 2026-07-11 — Independent 240-exam vertical-line inference completed

#### Goal and design

Abandoned the earlier repaired-397B model's 13 suggestions at the user's
request. They are not labels, inclusion criteria, or decision evidence. Used
the existing independently constructed 240-exam panel solely as 140
heuristic-ranked montages plus 100 seeded random montages from 7,827 cached
exams. Kept the dedicated human QC state empty and did not expose model
suggestions in the blinded-review instructions.

Submitted the frozen Qwen3.5-27B-FP8 marker-classifier configuration through
`submit_auto_qc.py` on `siweiq --qos=opportunistic`, one H200, four-hour
ceiling. Input list and sampling manifest are under
`qc_redo/review_batches/vertical_line_actual_data_review/`; output is
`qc_redo/auto_qc_runs/vertical_line_actual_data_review/qwen35_27b_vertical_line_actual_data_review.json`.

#### Evidence and result

Slurm job `12810570` completed `0:0` in 5m59s. vLLM loaded 28.54 GiB, became
healthy, scored all 240 unique manifest exams, checkpointed 240 structured
records, and recorded zero request failures. Output integrity checks confirmed
only the allowed target tag and pinned runtime versions
`dsi-local-llms==0.1.0`, `openai==2.45.0`, and `vllm==0.24.0`. The human state
remains at zero records.

The model flagged 136/140 heuristic-top exams (97.1%) and 0/100 random exams.
All 136 suggestions were emitted as high confidence. This near-perfect
agreement with the scalar selection stratum is not accuracy evidence and may
represent useful recognition, heuristic-correlated overcalling, or both. No
sensitivity, specificity, or exclusion claim is justified without blinded
human labels.

Successful teardown logged vLLM 0.24 `EngineDeadError` and a leaked-semaphore
warning after the shutdown marker. A parent-first shutdown change was committed
upstream (`d4f2d74`, package 0.1.1), built, installed, and passed eight upstream
plus 15 Prima tests. Two-exam retry job `12811425` completed `0:0` in 2m00s
with two records and the same one-flag pattern, but reproduced the messages
after application shutdown. This falsified simultaneous process-group
signalling as the cause. The feature branch now ends at documentation commit
`31718a4`; it remains unpushed. Treat the signature as benign only after
explicit shutdown, complete outputs, application shutdown completion, and zero
exit; never hide the same error during startup or inference.

#### Decision impact

Compute is complete and no Slurm job remains. The next decision-changing step
is blinded human review of all 240 exams using the empty dedicated QC state and
without `--auto-run-file`. Exact instructions are recorded in
`qc_redo/review_batches/vertical_line_actual_data_review/README.md`. Only after
all labels exist should the 27B run be loaded to calculate TP/FP/FN/TN and
inspect every disagreement. So what: do not run the full dataset or tune the
prompt from the 136 flags alone; obtain independent labels first.

### 2026-07-11 — View-level QC fallback requirement identified

#### Question and evidence

Visually inspected a user-specified four-view montage without recording its
identifier. Three panels appeared conventional; the selected R MLO contained a
large bright rectangular/curved field-edge structure in the background beside
the breast. The structure did not resemble the target narrow detector seam
crossing breast tissue. The most defensible current label is target-negative
for `vertical line (detector artifact)`, with a separate nonstandard-view issue
such as positioning/compression paddle or collimation hardware in field.

Checked both `qc_export/views_for_qc.parquet` and the authoritative
`/gpfs/data/huo-lab/Image/ChiMEC/MG/sot/views.parquet`. The inspected exam has
exactly four rows—one L/R × CC/MLO candidate—so no replacement view is
available in current metadata. Dataset-wide, the SoT contains 42,904 canonical
candidate rows in 42,904 `(exam, laterality, view)` groups: zero duplicate
groups and zero full-quad exams with retained alternatives.

Code inspection showed that selection is not random. `prima/view_selection.py`
uses a deterministic key based on magnification proximity to 1, presentation
intent, pixel spacing, and path; `pipelines/preprocess.py` and
`qc/build_qc_header_sidecars.py` then drop all but the first row per canonical
slot. Therefore any pre-selection alternatives are discarded before the SoT
and cannot be recovered by the current QC consumer.

#### Sensemaking and decision impact

Observation and mechanism agree: exam-level montage QC cannot support safe
fallback because it neither localizes the failed slot nor retains alternate
source images. The required future flow is candidate inventory → view-level QC
→ deterministic first passing candidate per L/R × CC/MLO slot → exam pass only
when all four slots have a passing candidate. Preserve selected/rejected SOP
lineage and failure reasons; never substitute a different laterality, view, or
special magnification image. The falsifier would be a pre-selection source
table showing additional candidate rows that current SoT counts missed; none is
present in the inspected authoritative table. So what: before full-dataset
automatic exclusion, update preprocessing to persist all candidates and update
the QC schema from exam-only tags to view/SOP-level decisions.

### 2026-07-11 — Archived DICOM lineage defect diagnosed

#### Question

Why could a selected montage not be opened through its `views.parquet`
`dicom_path`, and can the selected image still be mapped to its exact original
DICOM without exposing identifiers in logs?

#### Action and evidence

Traced the selected SoT rows through `pipelines/preprocess.py` and the raw MG
tree. The worker records the physical `.dcm` path used during processing, but a
successful run then archives the enclosing exam directory as `.tar.zst` and
removes the raw directory. Archive materializations are likewise temporary.
Consequently, all 42,904 current SoT `dicom_path` values are nonexistent even
though every row maps deterministically to one of 10,722 existing exam
archives. The selected exam's archive exists and its four expected member
paths are present. Streaming the four members from the archive reproduced all
four stored SHA-256 values exactly.

Relevant producer locations are `_process_exam_dir()` (records
`str(p.resolve())`), `archive_exam_dir()` (creates the archive and removes the
directory), and the final `archive_processed_raw_exams()` call in
`pipelines/preprocess.py`.

#### Result and decision impact

The original DICOM bytes are not missing; the SoT stores a transient physical
path instead of a durable archive/member locator. This also revises the prior
interpretation of the eight uncached eligible montages: their downstream reader
cannot resolve archived sources. The SoT producer must persist an explicit
archive path plus archive-member path (and continue retaining SOP UID and
SHA-256), while QC consumers must use one fail-fast source reader. Regenerate
the affected SoT/QC outputs after that schema change rather than supporting
mixed schemas. So what: do not treat a dead `dicom_path` as evidence that a
DICOM was lost; fix source lineage before implementing alternate-view QC.

### 2026-07-11 — Durable DICOM lineage rebuild launched

#### Goal and implementation

Replaced the transient `dicom_path` contract with
`source_archive_relpath` + `source_archive_member`, retaining SOP Instance UID
and SHA-256 as semantic and byte-level identity checks. Added the centralized
resolver in `prima/dicom_source.py`, updated preprocessing and every Python
consumer, replaced synthetic cached-view rows with source-linked rows, and
added `ops/audit_dicom_lineage.py`. QC sidecars now subset authoritative
`views.parquet` and `dicom_tags.parquet` rather than reimplementing DICOM header
extraction.

Targeted checks passed for archived materialization, unpacked-file resolution,
four SHA-256 matches, producer output with no transient column, QC montage
generation, SoT-derived sidecars, cached-view lineage, and identifier-free QC
logs. Ruff and compile checks passed on the changed Python entrypoints.

#### Rebuild campaign

Started a metadata-only, all-available rebuild from 33,690 on-disk exam
records into `/scratch/annawoodard/prima_dicom_lineage_rebuild`; provenance and
the exact command are in that directory's `README.md`. Tier1 shard jobs are
`12820060`–`12820075`, dependency merge is `12820076`, and the deterministic
100-member SOP/SHA audit is `12821341`. Production SoT and QC sidecars remain
untouched until the staged outputs pass identity/count/source validation.

#### Decision impact

The original DICOM mapping is now an explicit schema invariant rather than an
inference from a dead path. So what: promote only after the dependency audit
passes, then rebuild `qc_export/views_for_qc.parquet` from the promoted SoT and
run the same audit against production before resuming alternate-view QC work.

#### Completion, sensemaking, and production state

The first all-available rebuild was technically valid but not cohort-valid: it
produced 82,428 views / 20,607 exams rather than the production 42,904 views /
10,726 exams. Its 100-member audit passed, but comparison showed 24 production
identities absent and eight overlapping identities sourced from different
bytes/accessions. The mechanism was the fingerprint's latest-directory choice
for duplicate patient/study UID records. This result was not promoted.

Rebuilt again from an explicit allowlist of the 10,722 archive records
referenced by production. Jobs `12823062`–`12823077`, merge `12823078`, and
audit `12823103` all completed `0:0`, reproducing 42,904 views, 10,726 exams,
and 42,848 unique SOP tag rows. The current ranking policy selected a different
SOP in 24 slots across 19 exams; because this was a lineage-only change, those
policy changes were not silently accepted. The selection-preservation artifact
restored all production identities, verified all 24 replaced members by SOP
and SHA-256, and had zero shared-metadata differences. Final audit job
`12824658` passed 100 additional member hashes.

Promoted the audited table atomically to
`/gpfs/data/huo-lab/Image/ChiMEC/MG/sot/views.parquet` and rebuilt the QC
sidecars from authoritative SoT metadata. Job `12825067` then recovered the
eight previously uncached montages from archives with zero errors. Final
production state is 42,904 SoT rows / 10,726 exams and 31,360 QC rows / 7,840
fully cached eligible exams. A production member read passed SOP and SHA-256;
the cached-gallery smoke passed without identifiers in logs. All 38 campaign
jobs completed `0:0` and none remain active.

Pre-promotion byte-identical metadata backups are under
`/scratch/annawoodard/prima_dicom_lineage_rebuild/production_backup`; full
commands, job IDs, staging artifacts, the explicit allowlist, and the
selection-preservation script are documented in the rebuild `README.md`. So
what: durable source lineage is complete, and alternate-view QC can now be
implemented against explicit candidate SOPs without changing the current
selected quad implicitly.

### 2026-07-11 — Vertical-line QC restarted at view level

#### Question and design reset

The live exam-level review state contained 62 records (26 good, 36 annotated),
even though the gallery header showed `42/214`. The fraction was navigation
position within a filtered gallery, while remaining/count summaries came from
different universes. More importantly, an exam tag cannot localize the failed
source view and therefore cannot support same-slot fallback. The 62 labels are
retained only as historical work; none were converted into view labels.

The replacement experiment asks whether Qwen3.5-27B detects a vertical detector
seam in one mammogram rather than reproducing an exam-montage enrichment proxy.
The frozen panel has 160 independent views: 80 heuristic-enriched and 80 random,
one view per exam. Human labels are binary (`pass` or `vertical_line`), start
empty, and remain disconnected from model suggestions. The continuation gate
is sensitivity and specificity of at least 0.90 with no repeated missed
morphology; passing this pilot does not authorize deployment-grade automatic
exclusion.

#### Implementation and evidence

Added a SHA-keyed view QC schema and loopback-only reviewer in
`prima/view_qc.py` and `qc/view_qc_gallery.py`. Its header explicitly separates
`position`, `reviewed/total`, and `remaining`; the browser receives no patient,
exam, accession, stratum, or model-suggestion fields. The old exam gallery was
also relabeled so navigation position cannot be read as completed QC.

`qc/build_view_qc_pilot.py` scored all 31,360 canonical views / 7,840 cached
eligible exams, sampled the frozen 160-view panel, then reconstructed every
review PNG from its archived DICOM with SOP and SHA verification. All 160 are
nondegenerate and at or below the strict two-million-pixel bound. Batch
provenance and exact review/evaluation commands are in
`qc_redo/review_batches/vertical_line_view_review/README.md`.

Added the separate view-model schema and runner in `prima/view_auto_qc.py`,
`qc/run_view_auto_qc.py`, and `submit_view_auto_qc.py`. The single-view prompt
removes the montage-specific cross-view cue. Two-view smoke job `12830154`
completed `0:0`; full frozen inference job `12830180` completed `0:0` in 3m15s
with 160/160 HTTP 200 responses, exact manifest coverage, and 160 prompt/debug
provenance records. The human state remains 0/160. Model outputs must not be
loaded during blinded review.

Preprocessing now emits every eligible source as `view_candidates.parquet`,
with deterministic `selection_rank` and one `is_selected` row per exact L/R
CC/MLO slot. Shard jobs `12830075`–`12830090` and merge `12830091` all completed
`0:0`, yielding 59,617 candidate rows, 42,904 slots, and 10,726 exams. There are
14,344 slots with alternates across 5,664 exams. The current policy disagreed
with production in the same 24 known slots, so `ops/align_view_candidates.py`
promoted the authoritative production SOP/SHA to rank 1 and retained all other
candidates. Audit job `12830268` passed 200/200 deterministic archive-member SOP
and SHA checks. The byte-identical aligned file was atomically installed as
`/gpfs/data/huo-lab/Image/ChiMEC/MG/sot/view_candidates.parquet`; no existing
production table was replaced. Full provenance is under
`/scratch/annawoodard/prima_view_candidates_rebuild`.

Exact-slot fallback logic is implemented in `prima/view_fallback.py` and
`qc/select_qc_views.py`: only explicitly passing candidates may be selected,
and another laterality or projection can never substitute. Unreviewed or
exhausted slots remain unresolved. Focused validation passed 24 tests with one
general-environment dependency skip; the serving-only subset passed 18/18 in
`prima-vllm`.

#### Decision impact

The engineering path is no longer blocked: original-source mapping, candidate
inventory, individual-view rendering, frozen model inference, evaluation, and
same-slot fallback are all explicit. The decision-changing blocker is now the
fresh 160-view blinded human review. Do not tune the prompt or launch cohort-wide
candidate inference before those labels are complete. So what: run the new
view gallery, finish 160 binary labels, then execute the evaluator and inspect
every disagreement before deciding whether to scale or pivot.

### 2026-07-12 — Blinded view gate passed; full candidate campaign launched

#### Question and evidence

All 160 view labels were saved: 70 `vertical_line` and 90 `pass`, including the
last manifest item. The apparent inability to advance was not lost state; the
gallery stayed on item 160 because no explicit completion state existed. The
gallery now displays `COMPLETE`, explains that all labels are saved, and marks
the final navigation button `End reached`.

The frozen Qwen3.5-27B run agreed with all 160 human labels: TP=70, FN=0, FP=0,
TN=90. Overall sensitivity and specificity are both 1.0; Wilson 95% lower bounds
are 0.948 and 0.959. The heuristic arm was 69 positive / 11 negative and the
random arm was 1 positive / 79 negative. Evaluation artifacts are under
`qc_redo/evaluations/vertical_line_view_review/qwen35_27b_vertical_line_view_pilot`.
The model run predates review, all state records have `source=human`, and the
browser endpoint contains no model output, so label leakage is not the leading
explanation. The strongest remaining weakness is sampling, especially subtle
mid-distribution morphology; therefore the result clears the frozen research
gate but is not treated as deployment proof.

#### Full candidate action and current state

Prepared a restricted 59,617-view campaign under
`/scratch/annawoodard/prima_view_auto_qc/vertical_line_candidates`. The source
table SHA is frozen in `campaign.json`; 16 CPU shards verify SOP and SHA before
atomic owner-only rendering, and dependent validation requires exact image
coverage. Initial jobs `12831495`–`12831511` failed before rendering because
Submitit 1.5.4 plugin discovery raises `KeyError('submitit')` under the live
Python 3.8 environment. Pinned maintained Submitit 1.5.3, switched the launcher
to its direct Slurm executor, and resubmitted jobs `12831519`–`12831534` with
validator `12831535`. The second attempt is healthy: all 16 jobs are running,
image count is advancing, and no traceback is present.

Submitted inference jobs `12831544`–`12831559` with dependency
`afterok:12831535`: four one-H200 lanes on `siweiq`, four on `catherineq`, and
eight on `zhoulabq`, all opportunistic and checkpointed per view. After exact
coverage, merge the 16 run files, apply same-slot rank fallback without changing
production, and build a blinded targeted audit of replacements, exhausted slots,
and random pass controls.

One source in render shard 10 then passed archive-member SOP/SHA validation but
failed pixel decoding because stored pixel bytes disagreed with the declared
geometry. This is a source QC failure, not a model target or a scheduler error.
The renderer now records expected pixel-decode failures explicitly; validation
excludes them from the inference manifest, and fallback treats them as
deterministic non-passes without labeling them as vertical seams. The 15 healthy
jobs continued. Cancelled obsolete validator `12831535` and never-started GPU
jobs `12831544`–`12831559`; resubmitted only shard 10 as restart-safe job
`12831735`. A representative full-campaign PNG was visually intact and mode
`0600`. The new validator and GPU campaign will be submitted after all 16 render
markers exist.

#### Decision impact

The perfect pilot is credible enough to scale the frozen classifier, but the
claim remains narrow: it detects the sampled rendered seam morphology. So what:
finish the full candidate campaign, inspect operational coverage and fallback
counts, then require the targeted decision audit before promoting any selected
view table into production.

#### Campaign completion

All 16 render markers were written and every render job completed `0:0`.
Accounting was exact: 59,617 candidates = 58,868 newly rendered + 748 reused by
the restarted shard + one deterministic pixel-decode failure. Validator job
`12832093` then opened and decoded every installed PNG, completed `0:0`, and
wrote a 59,616-row inference manifest plus 16 equal 3,726-row shards. The full
candidate manifest and browser-safe inference manifest each have exact unique
SHA-keyed coverage and mode `0600`.

Final H200 inference jobs `12832351`–`12832366` all completed `0:0`. Opportunistic
preemption requeued several `zhoulabq` shards, which resumed from per-view
checkpoints without lost coverage. The merged run exactly covers all 59,616
model inputs. Together with the deterministic render rejection, fallback
decisions cover all 42,904 exact slots. Outputs are restricted under
`/scratch/annawoodard/prima_view_auto_qc/vertical_line_candidates`; no production
SoT or downstream manifest was changed.

The targeted audit is ready under
`/scratch/annawoodard/prima_view_auto_qc/vertical_line_fallback_audit`: 177
individual views from 125 decision groups, fresh state 0/177, exact hidden-model
coverage, and browser fields limited to view hash/laterality/projection/order.
It samples accepted fallbacks, exhausted decisions, and model-pass controls but
does not expose those strata in the gallery. The audit evaluator scores both
view agreement and slot-level decision validity. Representative pilot controls
supported the intended mechanism: the random-stratum positive showed a clear
full-height detector band, while a high-heuristic negative showed anatomy/edge
structure without that band.

The reusable scaling workflow is committed locally as `eef05e1` on
`feature/durable-dicom-lineage`; nothing is pushed. Focused validation passed
31 tests with one expected skip. So what: complete the blinded 177-view decision
audit before promoting or replacing any production selected-view table.

### 2026-07-12 — Fallback audit exposed prompt mismatch; revised 27B confirmation frozen

#### Question and completed audit

The 177-view targeted audit finished with 118 human passes and 59 human vertical
seams. Under the original policy, Qwen3.5-27B produced TP=56, FN=3, FP=21,
TN=97: sensitivity 0.949 and specificity 0.822. Strictly, 18/50 sampled
alternate-pass groups and 5/25 sampled no-pass groups disagreed with the human
candidate sequence; all 50 original-pass controls were correct. The operational
failure modes were more informative than the aggregate count: 15 alternate
groups replaced a human-pass original with another human-pass candidate, three
selected a human seam, and five no-pass groups unnecessarily exhausted human
passes.

All 56 true positives were high-confidence, whereas all 16 medium-confidence
suggestions were false positives. Reinterpreting the frozen run with a
high-confidence reject threshold preserved TP=56/FN=3 while improving to
FP=5/TN=113. In the full 42,904-slot dry run this recovered 162 exhausted slots,
changed 188 selections/resolutions, and increased complete four-slot exams from
10,357 to 10,483. These outputs are restricted under
`/scratch/annawoodard/prima_view_auto_qc/vertical_line_candidates/policy_high`;
production tables were not changed.

Visual review supported a concrete mechanism. The three misses were prominent
full-height detector-panel boundaries, often with repeated bright curved bands,
not subtle lines. Many false positives were surgical scar markers, normal frame
edges, or compression hardware. The prior prompt described only a narrow,
low-contrast seam and discouraged bar-edge calls, so the misses were consistent
with a target-definition mismatch rather than insufficient model capacity.

#### Minimal prompt experiment

Added the `detector_boundary_v2` prompt variant, which explicitly treats both a
narrow gray detector seam and a broad full-height detector-panel boundary as
positive while excluding scar wires, anatomy, text, crop/frame borders, and
normal film edges. Also made the model reject-confidence threshold explicit in
evaluation and fallback selection. Job `12833649` ran the revised prompt over
the already labeled 177-view development set on one opportunistic H200 and
completed `0:0`. With the predeclared high-confidence rule it scored TP=59,
FN=0, FP=2, TN=116: sensitivity 1.000 (Wilson lower 0.939), specificity 0.983
(Wilson lower 0.940), and NPV 1.000 (Wilson lower 0.968). Every one of the 125
sampled slot decisions was operationally safe. The two remaining false positives
were conservative boundary/hardware calls whose available fallback still passed.
Because this prompt was written after inspecting the audit errors, this result is
development evidence rather than an independent promotion gate. The 397B arm
was not launched because the one-variable 27B prompt change resolved the live
hypothesis.

#### Independent confirmation and next gate

The audit builder now excludes all exact-slot groups from prior manifests and
includes every candidate in each sampled group. The evaluator now reports both
`decision_safe` (no selected seam or false exhaustion) and `decision_exact`
(no unnecessary predecessor rejection), requires an explicit confidence
threshold, and verifies complete contiguous candidate sequences.

A fresh confirmation is frozen at
`/scratch/annawoodard/prima_view_auto_qc/vertical_line_fallback_confirmation_full`:
210 views covering all candidates in 125 sampled slots (50 alternate-pass
stratum, 25 no-pass stratum, 50 original-pass controls). It has zero view and
slot-group overlap with either the 160-view pilot or the 177-view development
audit. Hidden revised-model job `12833719` completed `0:0` on an opportunistic
H200 with exact 210/210 coverage. The human state remains 0/210, and model
outputs/strata are not exposed by the gallery.

The frozen promotion gate is zero unsafe slot decisions, view sensitivity at
least 0.95, view specificity at least 0.90, and no repeated missed morphology.
If it passes, rerun the revised 27B prompt over all 59,616 rendered candidates,
generate a high-confidence promotion candidate and impact report, then rerun
Mirai. If it fails, inspect every failure before deciding between another prompt
revision, a deterministic hybrid, or the 397B model. So what: the engineering
and GPU work are complete; only the independent 210-view blinded confirmation
remains before cohort-wide revised inference.

Implementation was committed locally as `3cbcf97` (`Refine vertical-line
fallback validation`) on `feature/durable-dicom-lineage`; nothing was pushed.
Repository formatting and linting passed, and the maintained Prima suite passed
35 tests with one expected skip. A repository-wide pytest collection additionally
reached the read-only Mirai vendor tests and failed only on their absent calibrator
snapshot and a release URL returning 404; neither failure exercises this change.

The reviewer now defaults permanently to port 8767 and displays the image at a
maximum of 92% of the available review area. The live confirmation server was
restarted on the same port with all persisted labels intact, so the existing SSH
tunnel and browser URL remain valid. This UI preference is committed locally as
`d278a45` (`Stabilize view QC gallery display`); the maintained suite passed 36
tests with one expected skip.

## 2026-07-14 — LLM-only Mirai QC residual challenge

### Question

Can the retained deterministic-eligibility plus six-component Qwen3.5-27B
baseline recover its unequivocal residual tubing, exposure, and rotation
failures through small LLM/VLM changes without repeating the failed embedding
pivot or sacrificing usable-view specificity and fallback safety?

### Action

Routed this new work to the Randi notebook after reloading the repository rules;
the immediately preceding residual-target chronology remains in
`logs/lab_notebook_dsi.md`. Reconstructed the six adjudicated disagreements and
matched hard controls from the source-linked 202-view development audit, then
visually inspected the relevant target views and same-slot alternatives. The
inspection identified two separable model failures: the rotation v1 prompt
described normal side-edge CC geometry ambiguously, while the faint catheter was
visible in source pixels but missed at the single full-view representation.

Committed `c525670` (`Add targeted LLM QC diagnostics`) with a rotation-v2 prompt
and a generic multiscale view input that preserves the canonical full view while
adding four overlapping enlarged foreground bands in a separate model-input
column. A real-view smoke initially failed on an older Pillow API, was corrected
to the repository-compatible resize API, and then produced a valid 1600x1200
grayscale composite in which the catheter is visibly retained in two detail
bands. No automated tests were created or run; scoped ruff, import compilation,
and real-artifact inspection were used.

Submitted CPU builder job `12891491` from
`qc_redo/auto_qc_development/mirai_input_llm_challenge/build_multiscale.sbatch`
to generate the deterministic 202-view multiscale bank. Existing `gpu_session`
allocations were observed but not entered or claimed; this run uses a separate
allocation.

Job `12891491` completed `0:0`, but visual inspection of a hard negative showed
that low nonzero detector background could dilute its enlarged panels. Tightened
the generic foreground rule to use an adaptive background threshold and
substantial row/column support, committed the correction as `fd3439c` (`Focus
multiscale QC details`), removed only the stale derived multiscale outputs, and
resubmitted the same build as job `12891500`. The corrected real-view smoke
keeps both the faint catheter positive and the short marker-wire negative clear.
Job `12891500` then completed `0:0` in `00:01:38`. The corrected bank has exact
202/202 unique model paths, all 1600x1200 grayscale, source-preserving canonical
columns, output-manifest SHA-256
`1744339087d48549a2cb1221ffd71feb02c818cbb320c764d7a14baeb7e4ad40`,
and ordered image-bank SHA-256
`609c38ebbd1344b3afd7c5ff3c259b9f6c58dd937bc9430df2919d641f1c251d`.
Froze `protocol.json` after this validation and mechanically verified all 14
recorded source, prompt, reference, run, code, manifest, and image-bank digests
against commit `fd3439c` before model submission.

Submitted the two independent opportunistic typed-H200 jobs through the
repository-native Submitit launcher with `--no-srun`: rotation-v2 job `12891526`
and unchanged-prompt multiscale-tubing job `12891527`. Their run files are under
`qc_redo/auto_qc_development/mirai_input_llm_challenge/components/`; Submitit
logs are under the corresponding `qwen35_27b_mirai_*_20260714_1136*` run roots
in `/ess/scratch/scratch1/annawoodard/prima_view_auto_qc/submitit_runs/`.

### Evidence

- retained primary baseline: TP 87, FN 6, FP 3, TN 106
- definite development positives: tubing order 24, exposure order 46, rotation
  order 177
- explicit hard controls: upright same-slot order 9; prior rotation false
  positives 37, 75, and 157; usable marker/device control 165
- reference-ambiguous orders 34 and 138 are not required positive recoveries
- challenge provenance and frozen decision rules:
  `qc_redo/auto_qc_development/mirai_input_llm_challenge/README.md`
- next check: reconcile job `12891500`, validate exact 202-view output coverage
  and image-bank provenance, then freeze all hashes before H200 inference

### Decision Impact

Run only two one-variable Qwen arms next: rotation-v2 on canonical full views and
unchanged tubing-v1 on multiscale views. Keep the already successful
exposure-v1 recovery as a development candidate delta. A failed new arm rejects
only that delta; the retained modular baseline remains active until a combined
candidate passes a new patient/exam-disjoint whole-exam gate.

### First challenge result and frozen follow-up

Jobs `12891526` and `12891527` completed `0:0` with exact 202/202 run and
debug-record coverage under the pinned Qwen3.5-27B FP8 and vLLM runtime. The
rotation-v2 prompt on the unchanged canonical view eliminated the protected
normal-view positives but produced no positive calls and still missed required
rotation order 177. This falsifies only the rotation-v2 single-view prompt
delta; it does not falsify the retained modular baseline or same-exam visual
comparison.

The unchanged tubing-v1 prompt on the multiscale input recovered required order
24 and improved the primary whole-exam result from TP 87/FN 6 to TP 88/FN 5,
with one fewer unsafe selected slot and one fewer unsafe accepted exam. It also
called protected usable scar-marker order 165 positive, so it fails the exact
component challenge and is not promoted as-is. Because order 165 was already a
baseline false positive, the aggregate FP count did not increase; that
aggregate improvement does not override the prespecified morphology control.
This falsifies only the unchanged tubing-v1 prompt on multiscale inputs, while
supporting the representation hypothesis that the faint catheter becomes
visible at higher effective scale.

Before further inference, froze `followup_protocol.json` at commit `7abc497`
and mechanically verified its commit and branch; baseline, reference, prior-run,
and prior-score hashes; all 202 context and multiscale rows; both ordered image
banks; and both prompt file and normalized-text hashes. The two next
one-variable falsifiers are:

- rotation-v2 with the already frozen target-dominant same-exam context input;
  it must recover order 177 while keeping orders 9, 37, 75, and 157 negative;
- tubing-v2 on the unchanged multiscale bank; it must keep order 24 positive
  while keeping orders 34 and 165 negative.

The retained deterministic-plus-six-component Qwen baseline remains active.
Neither follow-up may be combined unless it passes both its exact component
challenge and the monotonic whole-exam safety comparison.

Submitted both frozen follow-ups as independent opportunistic typed-H200 jobs
through the repository Submitit launcher with `--no-srun`: rotation-v2 with
same-exam context is job `12892010`, and tubing-v2 with the unchanged multiscale
bank is job `12892011`. The eight live shared `gpu_session` allocations were
observed but not entered or claimed. Next check: reconcile both jobs in Slurm,
then require exact 202/202 run and debug artifacts before scoring either arm.

Both jobs completed `0:0` in under four minutes and passed the artifact audit:
exact 202/202 run and debug coverage, matching canonical records, pinned input,
prompt, model, revision, and runtime hashes, parsed responses, and thinking
disabled. Rotation-v2 with passive same-exam context still produced zero
positives, missed order 177, and left the whole-exam result unchanged. Its raw
decision relied only on the posterior chest wall being at the right image edge.
Because the unchanged prompt still said `Target view only` and never instructed
the model to compare reference panels, this falsifies passive addition of
same-exam context, not explicit comparative orientation reasoning.

Tubing-v2 passed its exact preregistered order-24/34/165 challenge and preserved
the order-24 recovery, but introduced false positive order 142. The primary
result became TP 88/FN 5/FP 4/TN 105, with unsafe selected slots improving from
6 to 5 and unsafe accepted exams from 4 to 3, but false-exhausted slots worsening
from 2 to 3. Visual review confirmed order 142 is a clean usable LMLO with
branching, tapering normal breast vessels rather than manufactured tubing. Its
slot's first candidate, order 118, contains a large implanted device; order 142
is the only usable fallback, so rejecting both causes the new false exhaustion.
Order 165 remains a correctly protected scar-marker negative. Additional
target-specific review found that orders 41 and 128 are also non-tubing device
or vascular controls, while orders 18, 24, 104, and 126 show convincing
procedural tubing. This falsifies tubing-v2 as a directly promotable additive
component but leaves a specific vessel-versus-tube prompt correction open.

The next smallest one-variable checks are therefore a context-aware rotation
prompt on the unchanged context bank and a tubing-v3 morphology prompt on the
unchanged multiscale bank. The rotation prompt must actively compare upright
MLO superior/inferior landmarks rather than using side-edge attachment alone.
The tubing prompt must require a manufactured-device cue and treat branching,
tapering, variable-caliber breast vessels as negative even when enlarged edges
look parallel. The retained baseline remains active throughout.

Committed the two prompt-only refinements as `623cbc0` (`Refine visual QC
prompts`) and froze `refinement_protocol.json` before inference. Mechanical
validation matched the commit and branch; all baseline, prior-run, prior-score,
input-manifest, provenance, ordered image-bank, prompt-file, and normalized
prompt hashes. The rotation challenge remains order 177 positive with orders 9,
37, 75, and 157 protected negative. The expanded tubing challenge requires
orders 18, 24, 104, and 126 positive and orders 34, 41, 118, 128, 142, and 165
negative, preventing a narrow order-142 fix from hiding loss of known tubing or
new confusion with vessels, implants, pacemaker hardware, and marker wires.

Submitted the frozen refinements as independent opportunistic typed-H200 jobs
with `--no-srun`: context-aware rotation is job `12892273`, and vessel-aware
tubing is job `12892274`. Existing shared persistent `gpu_session` allocations
were observed but not entered or claimed. Next check: require clean Slurm
completion and exact 202/202 hashed run/debug coverage before scoring.

Jobs `12892273` and `12892274` completed `0:0` and passed exact 202/202
run/debug, prompt, model, runtime, and input-bank validation. Neither prompt
refinement passed. Context-aware rotation emitted eight high-confidence
positives but still called order 177 normal, reintroduced protected false
positive order 37, and produced a false exhaustion. Its one recovered baseline
failure was order 46, whose actual failure is exposure rather than rotation.
The raw explanations explicitly inverted the desired distinction: order 37 was
called 90 degrees rotated relative to references, while order 177 was said to
match upright references. This falsifies further near-duplicate Qwen3.5-27B
rotation prompt wording on the current context representation, not model scale
or the broader VLM family.

The vessel-aware tubing-v3 prompt produced zero positives. It protected every
vessel/device/marker negative but missed all four adjudicated tubing positives,
including residual order 24. Together, v2 and v3 bracket the 27B zero-shot
prompt boundary: v2 recovers the faint tube but confuses a branching vessel;
v3 removes that confusion by suppressing the target entirely. This falsifies
another zero-shot wording pass, not contrastive visual examples or a second
observer/judge stage. The retained deterministic-plus-six-component baseline
remains active, with neither failed refinement combined.

Next mechanism checks: change only model scale to the installed Qwen3.5-397B
FP8 on the five-case frozen rotation challenge before paying for the full
202-view run; for tubing, use a small alternating positive/negative multiscale
exemplar bank and keep its scored views disjoint from the examples. Promote
neither mechanism without its exact target challenge, then require a new
patient/exam-disjoint whole-exam audit for any final candidate.

Added an explicit-review-order mode to the canonical view-subset producer and
committed it as `aad9dd3` (`Support explicit view challenges`). A real workflow
run produced the five-view rotation challenge beside the frozen context
manifest, preserving source review orders 9, 37, 75, 157, and 177 and all
canonical/model paths. Froze `rotation_scale_protocol.json` before inference
and mechanically verified the code, registry, Qwen3.5-397B-A17B FP8 revision,
TP=4 requirement, source and challenge manifests, ordered image bank, prompt,
and nearest 27B run/score hashes. Only model scale changes. A failure on these
five views stops the 397B rescue before a full-panel run; a pass advances the
unchanged arm to all 202 views.

Submitted the frozen scale-only rotation challenge as independent four-H200
opportunistic job `12892941` with `--no-srun`. Existing shared persistent
allocations were observed but not entered or claimed. Next check: reconcile
Slurm and require exactly five linked, parsed debug records with the registered
397B revision before scoring the challenge.

Built the fixed four-shot tubing mechanism experiment with the canonical
few-shot producer and committed its multiscale-input support as `38a9b54`
(`Preserve multiscale few-shot inputs`). The alternating bank is orders 104
(tube present), 128 (branching vessel absent), 126 (tube present), and 165
(scar-marker wire absent). Its matched evaluation set contains the other 198
development views; required challenge order 24 and protected vessel controls
118 and 142 share no audit exam with any exemplar. Froze
`tubing_four_shot_protocol.json` before inference. A corrected mechanical audit
passed all 30 checks: branch/commit and owner hashes, registered 27B revision,
source/provenance/manifest hashes, exact example order and labels, disjoint view
partition, byte-identical canonical and multiscale copies, loader validation,
ordered image-bank hashes, prompt identity with zero-shot v2, nearest-reference
hashes, and absence of run/score outputs. The only changed mechanism is the
four fixed examples. Failure rejects this bank, not few-shot prompting in
general; a pass freezes the bank for a new patient/exam-disjoint audit rather
than treating this 198-view development run as a promotion gate.

Submitted the frozen tubing four-shot development run as independent
opportunistic typed-H200 job `12893046` through the repository Submitit
launcher with `--no-srun`. No shared persistent allocation was entered. Next
check: reconcile Slurm, require exact 198/198 run and debug coverage with the
frozen exemplar/prompt/model/input provenance, then score order 24 positive and
orders 118 and 142 negative.

The scale-only Qwen3.5-397B-A17B FP8 rotation challenge completed `0:0` after
16m34s. It passed all 25 run/debug provenance checks with exact 5/5 coverage,
the registered model revision and four-GPU configuration, pinned vLLM/OpenAI
runtime, frozen context inputs and prompt, parsed responses, and thinking
disabled. The preregistered score passed exactly: order 177 was the sole
positive, while protected controls 9, 37, 75, and 157 remained negative. This
supports model scale for this frozen prompt/input pair and, per the prespecified
rule, advances the unchanged arm to the full 202-view development panel.

Froze `rotation_scale_full_protocol.json` before full-panel inference and added
stable submit/score wrappers. The continuation changes only the 27B model to
the registered 397B model for the rotation component; it retains the full
same-exam context bank, prompt, deterministic decoding, high-confidence
threshold, modular baseline, and whole-exam delta rules. A 29-check mechanical
audit verified the source owners, successful challenge artifacts, baseline
metrics, exact 202-view input bank, model/prompt hashes, absence of outputs, and
end-to-end scoring contract. The arm must still recover order 177, protect all
four controls, introduce at most one false positive, and worsen none of unsafe
selected slots, false exhausted slots, or unsafe accepted exams. A pass only
freezes a candidate for a new patient/exam-disjoint audit; it is not itself a
promotion result.

Submitted the frozen full-panel 397B rotation continuation as independent
opportunistic four-H200 job `12893774` through Submitit with `--no-srun`. No
shared persistent allocation was entered. Next check: require clean Slurm
completion plus exact 202/202 run/debug provenance before scoring the component,
primary and adjudicated view metrics, fallback safety, and incremental delta.

Tubing four-shot job `12893046` completed `0:0` in 5m20s and passed exact
198/198 artifact validation, including the registered 27B revision, pinned
runtime, evaluation input hash, fixed four-example manifest/hash/order/labels,
four examples in every debug request, parsed responses, and thinking disabled.
It failed the preregistered example-exam-disjoint challenge: order 24 changed
from the zero-shot v2 true positive to a high-confidence negative, order 142
remained a high-confidence vessel false positive, and order 148 became an
additional positive. The model described 142 as parallel-walled curvilinear
tubing and 148 as a thin curving parallel-walled line, while reporting no
evidence at all for 24. Thus this fixed four-example bank neither preserved the
sensitive zero-shot cue nor taught the needed vessel distinction. This rejects
the bank, not few-shot prompting generally, other VLMs, or the retained modular
baseline. No whole-exam promotion score is valid because the four exemplar
views were intentionally removed from the 198-view development run.

Prepared the next one-variable tubing mechanism check: Qwen3.5-397B-A17B FP8
with the unchanged zero-shot v2 prompt and multiscale representation. The
explicit ten-view challenge contains adjudicated tubing positives 18, 24, 104,
and 126 and protected vessel/device/marker controls 34, 41, 118, 128, 142, and
165. Froze `tubing_scale_protocol.json` before inference. A 26-check audit
verified the source code, nearest 27B reference, failed four-shot artifacts,
exact source-linked challenge rows and ordered image bank, model/prompt hashes,
complete positive/negative partition, absent outputs, and independent
`--no-srun` launch contract. Only model scale changes. Failure stops this scale
rescue before a full-panel run but does not reject observer/judge designs,
other open-source VLMs, or the retained baseline.

Submitted the frozen scale-only tubing challenge as independent opportunistic
four-H200 job `12894593` through Submitit with `--no-srun`. No shared persistent
allocation was entered. Next check: require clean Slurm completion and exact
10/10 run/debug provenance before scoring all four positives and six protected
negatives.

Tubing scale job `12894593` completed `0:0` in 3m21s and passed all 29
artifact checks with exact 10/10 coverage. The 397B model failed the frozen
challenge decisively: it detected positives 18, 104, and 126 but still missed
the faint tube at 24, and it falsely flagged protected controls 41, 118, 128,
and 142. Only controls 34 and 165 remained negative. Therefore model scale is
not the rescue for tubing under the v2 prompt and multiscale representation,
and this branch stops before a full 202-view run. This failure does not affect
the independent rotation-scale result, observer/judge mechanisms, other
open-source VLMs, or the retained deterministic-plus-modular baseline.

The next tubing mechanism is a decomposed candidate-and-veto system rather
than another direct binary prompt. The retained 27B v2 detector marks only
orders 18, 24, 118, 126, and 142 at high confidence, so a new conservative
observer runs on exactly those five candidates and asks only whether the
strongest line is an obvious vessel/wire/hardware mimic. A generic derived-run
owner now enforces that the veto manifest covers every sensitive-detector
positive and retains a candidate only when no high-confidence veto is present.
Committed this reusable infrastructure and prompt as `34b2747` (`Add VLM veto
gating`).

Froze `tubing_veto_protocol.json` before inference. The exact observer challenge
requires vetoing vessel control 142 while protecting true tubes 18, 24, and
126; device-positive order 118 is deliberately operationally neutral because
the retained baseline already excludes it. If the observer passes, the frozen
score path builds the gated full run and requires recovery of order 24, zero
introduced false positives, and no worse fallback-safety readout. A 30-check
audit verified code, baseline, candidate-run hash and exact five positives,
veto manifest and ordered bank, registered 27B model, prompt contract, absent
outputs, and fail-closed end-to-end scoring.

Submitted the frozen tubing mimic-veto observer as independent opportunistic
one-H200 job `12896048` through Submitit with `--no-srun`. No shared persistent
allocation was entered. Next check: require clean Slurm completion and exact
5/5 run/debug provenance before the fail-closed component and whole-exam score.

Tubing mimic-veto job `12896048` completed `0:0` in 1m34s and passed exact 5/5
artifact validation. The observer challenge passed perfectly: order 142 was the
only high-confidence veto, while true tubes 18, 24, and 126 and operationally
neutral device view 118 were not vetoed. The generic gate therefore retained
candidate positives 18, 24, 118, and 126 and removed only 142.

The first downstream score attempt stopped after building the correct gated and
combined runs because `tubing_veto_protocol.json` omitted the evaluator's
descriptive `reference` field. This was protocol plumbing, not a model result;
no thresholds or decisions changed. Added the required field, preserved the
completed artifacts, and resumed from evaluation rather than deleting or
re-running inference. The primary incremental result passes: TP 88/FN 5/FP
3/TN 106, sensitivity 0.9462, specificity 0.9725, one recovered baseline false
negative (required order 24), zero introduced false positives, unsafe selected
slots 6 to 5, false exhausted slots unchanged at 2, and unsafe accepted exams
4 to 3. The adjudicated read is TP 90/FN 4/FP 1/TN 107, sensitivity 0.9574,
specificity 0.9907, unsafe selected slots 4, false exhausted slots 1, and unsafe
accepted exams 2. This arm is retained for the combined development candidate;
it does not supersede the active baseline by itself and still fails the global
zero-unsafe whole-exam gate.

Full-panel rotation-scale job `12893774` completed `0:0` in 26m52s; about 23
minutes were GPFS weight loading and only about two minutes were inference. It
passed all 32 artifact checks but failed both the exact and incremental gates.
The 397B run missed required rotated order 177, falsely flagged protected order
157, emitted 22 component positives, recovered zero baseline false negatives,
introduced nine false positives, and increased false exhausted slots from 2 to
9. Primary counts were TP 87/FN 6/FP 12/TN 97; adjudicated counts were TP 89/FN
5/FP 10/TN 98. This arm is rejected and will not be combined.

More importantly, exact challenge views were not reproducible between the
five-view and full runs despite byte-identical images, identical view IDs,
model revision, prompt, decoding settings, and runtime. Order 177 flipped from
high-confidence positive in the challenge to high-confidence negative in the
full run; order 157 flipped from high-confidence negative to high-confidence
positive. The other three challenge views remained negative. Thus the tiny
397B challenge pass was not trustworthy evidence, and temperature-zero FP8 MoE
inference is not operationally deterministic enough here to rescue rotation.
This falsifies the 397B scale rescue for the frozen rotation prompt/input pair,
not the retained baseline or other VLM families.

### Independent-family rotation challenger frozen (2026-07-14 14:31 CDT)

Question: can a non-Qwen open-source VLM recover the residual sideways-MLO
failure reproducibly, without repeating the misleading five-view-only 397B
pass?

Selected the official MIT-licensed `zai-org/GLM-4.6V-FP8` checkpoint at pinned
revision `33172e26eb88482cf3d0a36fced01d05454734ec`. Its
`Glm4vMoeForConditionalGeneration` architecture is supported by the installed
vLLM 0.24.0 runtime; the isolated `prima-vllm` environment also has OpenAI
2.45.0 and Transformers 5.13.0. The checkpoint contains 41 weight shards and
110,004,730,904 weight bytes. The GLM chat template was inspected locally and
honors `enable_thinking=false` through its `/nothink` path.

The initial anonymous eight-worker Hub download on the login node stalled for
30 minutes with zero process read/write bytes. Stopped that path and extended
the canonical `scripts/download_auto_qc_model.py` owner with a pinned,
resumable datamover mode. A real 1 MiB range probe through
`cri-datamover.cri.uchicago.edu` reached about 16.6 MB/s. The full eight-worker
transfer then completed and `prima.vllm_server.resolve_model_path` verified the
local snapshot at
`/gpfs/data/huo-lab/Image/annawoodard/models/GLM-4.6V-FP8`. The model registry
now contains a one-H200 eager-serving configuration; this is still a runtime
hypothesis until the real smoke proves it fits. The documented `/net/projects2`
Prima prefix did not exist on this host, so direct checks used the active
`/gpfs/data/huo-lab/Image/annawoodard/micromamba/envs/prima` prefix discovered
from `micromamba env list`.

Added a generic exact-challenge scorer that requires identical operational
decisions across independent server starts and validates the frozen manifest,
model revision, serving arguments, runtime versions, prompt hash, and full
coverage before scoring. This is the prevention mechanism for the 397B
near-miss: a tiny pass is now only a runtime/directional smoke, never promotion
evidence.

Froze
`qc_redo/auto_qc_development/mirai_input_llm_challenge/rotation_glm46v_protocol.json`
(SHA-256
`f15add4ef79aca58d290af54ff3eec9108d07ac8005abd07bdae8fe50280eb6c`)
before inference. Seventeen mechanical checks passed. The minimal decision
matrix is: one five-view one-H200 smoke; if and only if it marks order 177 at
high confidence and protects orders 9, 37, 75, and 157, run two independent
full 202-view server starts. Both full runs must pass the exact component and
incremental whole-exam gates and agree on every high-confidence decision. A
pass retains one identical decision vector for the combined development
candidate; a failure rejects only this GLM rotation rescue. The deterministic
DICOM plus frozen modular Qwen baseline and the successful tubing veto remain
active throughout.

Next check: inspect the shared H200 broker, then launch the smoke only through
`shared_h200_pool.sh run --project prima`; never attach a raw `srun --jobid`
step and never duplicate the task in opportunistic Slurm.

Smoke launch update: the broker rejected request
`prima-prima_glm46v_rotation_smoke-536264aa1b` before compute because each
shared holder has only 4 CPUs and 16 GB host memory, below the requested 16
CPUs and unsafe for staging 110 GB of weights. The request is recorded as
blocked, not queued. Following the broker's profile-derived fallback, submitted
the unchanged, restartable smoke exactly once through the repository
`submit_view_auto_qc.py` launcher with `--no-srun`, typed one-H200 GRES,
256 GB host memory, and opportunistic `zhoulabq,catherineq,siweiq`. The first
headnode attempt failed before queueing because direct environment Python did
not expose the sibling `vllm` executable on `PATH`; added the environment-bin
prefix and retried after `validate_vllm_runtime` resolved the pinned executable.
Job `12902114` is running on `zhoulabq`/`opportunistic`; Submitit logs are under
`/ess/scratch/scratch1/annawoodard/prima_view_auto_qc/submitit_runs/glm46v_mirai_rotation_v3_context_smoke_20260714_143518/submitit_logs`.
The managed server started at 14:35:30 CDT. Next check: server readiness or a
specific load failure, followed by exact 5/5 run/debug artifact validation.

### GLM smoke falsified direct rotation prompt; orientation-choice reset (2026-07-14 15:09 CDT)

Job `12902114` completed `0:0` in 3m13s on one opportunistic H200. The pinned
106B GLM-4.6V-FP8 checkpoint loaded in 102.45 GiB, and the run passed exact 5/5
API and debug-artifact validation. It nevertheless failed the frozen directional
gate: all five views were high-confidence negative, so required order 177 was
missed while controls 9, 37, 75, and 157 remained protected. The raw answer for
177 explicitly asserted that the view was upright. Per protocol, no full-panel
GLM repeats were launched. This rejects only GLM-4.6V with the frozen direct
context prompt and representation.

Raw-pixel reinspection corrected an important description error in the prior
work. Order 177 is not approximately 90 degrees sideways. It is approximately
180 degrees inverted relative to its same-exam, same-slot usable LMLO fallback.
The source still declares MLO at a 36.1-degree positioner angle, and a blinded
downsample comparison with that fallback reaches correlation 0.93 only after a
180-degree target rotation. The original human label remains a clear visual
exclusion; what changes is the agent-assigned mechanism wording. Repeated prompts
that emphasized sideways anatomy were therefore a low-yield loop.

Applied the rut-breaker reset by changing the visual decision structure rather
than rewording another direct classifier. Added a deterministic target-preserving
panel that crops substantial breast foreground and shows A=current display plus
B/C/D=90/180/270-degree clockwise rotations. The new prompt asks whether A is
plausibly upright and requires another candidate to unmistakably restore whole
anatomy. The five-view real-input bank contains exact canonical lineage and has
ordered panel SHA-256
`7342bf7bea758bf0614c03e07d6c6d9dc8d4773511d65eb2699ae0f3eef78d84`.
Visual inspection shows C as the clear upright LMLO for order 177 and A as the
clear upright LMLO for control order 9. Next check: commit the reusable builder
and prompt, freeze their exact hashes, then run only the five-view GLM smoke.

Committed and pushed the generic GLM download/repeatability tooling as
`c378a4e` and the orientation-choice builder/prompt as `f36d32f` on
`feature/durable-dicom-lineage`. Froze the five-view protocol at
`qc_redo/auto_qc_development/mirai_input_llm_challenge/rotation_orientation_choice_smoke_protocol.json`
with SHA-256
`3297394e9c19f137d291099930045cb56e6b58980e6ec665e78a473512591a48`;
ten code, model, prompt, manifest, provenance, and absent-output checks passed.

The required shared-H200 broker status showed eight idle holders and no queued
claims or external job steps, but every holder has only 4 CPUs and 16 GiB host
memory. That is unsuitable for the 110 GB checkpoint, so no shared task was
claimed or entered. Submitted the restartable smoke exactly once through the
broker profile's typed-H200 opportunistic route using the repository Submitit
launcher with `--no-srun`, 16 CPUs, and 256 GiB RAM. Job `12906562` is the only
copy. Logs are under
`/ess/scratch/scratch1/annawoodard/prima_view_auto_qc/submitit_runs/glm46v_mirai_rotation_v4_choice_smoke_20260714_151552/submitit_logs`.
Next check: classify scheduler/runtime health, then require exact 5/5 artifacts
and the frozen order-177/C-mechanism plus four-control decision gate.

Job `12906562` completed `0:0` in 2m37s with five successful API calls. Thirteen
artifact checks passed: exact 5/5 canonical and debug coverage, unique response
IDs, frozen model/revision/runtime/serve arguments/prompt/input manifest, thinking
disabled, and agreement between run and debug decisions. The run SHA-256 is
`efb19311a495a0a4185a7a412bca2681306a280816578966c4fc956875f2bfc6`.

The scientific smoke gate failed. GLM marked all five cases absent at high
confidence. It protected controls 9, 37, 75, and 157 but again missed required
order 177, saying that A was acceptable and no comparison candidate corrected
it. It also described declared CC controls 75 and 157 as MLOs, evidence that it
did not reliably use the panel header. No full-panel bank or repeat was launched.
The result is recorded in
`rotation_orientation_choice_glm46v_smoke_result.json`. This rejects only GLM
on the frozen four-orientation composite. The next one-variable check keeps the
validated pixels and prompt fixed and swaps only to the retained Qwen3.5-27B
backbone; if that also fails, the composite representation rather than GLM alone
is the leading problem.

Froze the one-variable Qwen3.5-27B smoke protocol at
`rotation_orientation_choice_qwen27_smoke_protocol.json` with SHA-256
`611ba463fc7c85027a5da69b8b20cf4cc38517e7ea6d2dc06c4999f180d2be92`;
ten checks passed and all outputs were absent. A fresh broker status showed one
claimed retro-caps task, one unrelated external step, six idle holders, and no
queued Qwen task. The holders remain limited to 16 GiB, so none was entered.
Submitted the restartable Qwen smoke exactly once through the same profile-derived
typed-H200 opportunistic route and repository `--no-srun` launcher. Job
`12907649` requests one H200, 16 CPUs, and 128 GiB RAM. Logs are under
`/ess/scratch/scratch1/annawoodard/prima_view_auto_qc/submitit_runs/qwen27_mirai_rotation_v4_choice_smoke_20260714_152630/submitit_logs`.
Next check: clean scheduler/runtime completion, exact 5/5 artifact audit, then the
unchanged order-177/C-mechanism plus four-control smoke gate.

Job `12907649` completed `0:0` in 4m55s. The temporary server-side
`EngineDeadError` was normal teardown after all five HTTP 200 responses, not a
runtime failure; the earlier interim interpretation was corrected before
scoring. Fourteen artifact checks passed with exact 5/5 run/debug coverage and
run SHA-256
`f88dfd5960fd863d9b6575134ff8fe27be0ac1ad45c4b8b0468e2fafc49a5897`.

The Qwen scientific gate failed in the opposite degenerate direction from GLM.
It marked all five views present at high confidence and always said candidate D
corrected A. Thus it violated all protected controls 9, 37, 75, and 157; for
order 177 it also chose D rather than the known 180-degree candidate C. The
result is recorded in `rotation_orientation_choice_qwen27_smoke_result.json`.
No full-panel work was launched.

This exposes a concrete representation confound: all four candidates were fit
inside landscape panel boxes, so the 90-degree landscape candidates occupied
roughly twice the effective anatomy pixel area of the portrait A/C candidates.
The systematic D choice is consistent with scale/layout bias rather than learned
orientation. Per the frozen plan, stop the four-candidate composite. The next
lower-complexity check narrows the component to the observed mechanism—gross
180-degree inversion—and compares equal-scale square A=current versus B=180
degrees only. This changes the information structure and removes the diagnosed
size confound rather than sweeping prompt wording or another model.

Implemented the lower-complexity pairwise mode in the canonical orientation
builder and committed/pushed it with the component prompt as `e2d5722` (`Add
equal-scale inversion QC panels`). The real five-view pair bank is 1440x720
grayscale, uses equal square candidate bounds, preserves canonical target paths,
and has ordered SHA-256
`f63b4f421d1fefee150fd8360fbe6a58f7eaf4912edf010832b69d209e21228a`.
Visual inspection confirms B is upright for inverted order 177 while A is
upright for control order 9.

Froze `inversion_pair_qwen27_smoke_protocol.json` with SHA-256
`161e84834c176aae9b8e9f33ed09b11d20af3d8411dc7326f8b5546797e0f2a3`;
ten source, model, prompt, panel, predecessor, and absent-output checks passed.
A fresh broker status showed all eight holders idle but still limited to 16 GiB,
below the Qwen checkpoint's 28.75 GiB weight size, so no holder was entered.
Submitted exactly one restartable typed-H200 opportunistic job through the
repository `--no-srun` launcher: `12912760`, with 16 CPUs and 128 GiB RAM. Logs
are under
`/ess/scratch/scratch1/annawoodard/prima_view_auto_qc/submitit_runs/qwen27_mirai_inversion_pair_smoke_20260714_154042/submitit_logs`.
Next check: exact completion and artifact audit, then require high-confidence
order-177 YES with B-corrects-A evidence and parsed NO for all four controls.

Job `12912760` completed `0:0` in 1m49s and passed fourteen exact artifact
checks. The run SHA-256 is
`7f54e581277bc3a48a13b7029825ec6459949b87570ba46aced4cf97c1d317ea`.
Equal-scale pairwise comparison removed the prior systematic candidate-D bias
and recovered order 177 correctly at high confidence with B-corrects-A evidence.
It nevertheless failed specificity: controls 9, 37, and 157 were also called
inverted at high confidence, while only 75 remained negative. No full-panel
work was launched. The result is recorded in
`inversion_pair_qwen27_smoke_result.json`.

Sensemaking: compared with the four-way run, the target mechanism and one
control improved when candidate scale was equalized, so the representation
change mattered. The remaining error is polarity calibration rather than a
parser or target-visibility failure. The cheapest discriminative prompt-only
falsifier is to freeze all current variables and add a four-example bank made
from accepted MLO views outside the five challenge views/exams: two panels with
A upright labeled absent and two with A deterministically inverted labeled
present. This tests whether explicit contrastive polarity resolves the exact
three control errors. If it does not pass the unchanged five-view gate, stop
the pairwise inversion branch rather than tune prompt wording or model scale.

## 2026-07-14 16:01 CDT — Frozen inversion-polarity exemplar bank

Implemented the predeclared final prompt-only falsifier without changing the
Qwen backbone, target prompt, target pair pixels, decoding, parser, confidence
gate, or five scored views. The generic few-shot path now applies the exact
target prompt to each labeled reference and emits the reference's validated
single-line evidence phrase instead of a context-free generic label. The
orientation builder can also apply explicit, provenance-recorded rotations to
candidate A by review order, and the few-shot experiment builder can enforce
view- and audit-exam disjointness from a scored manifest plus an operational
source-label requirement.

Selected four standard, high-confidence human-accepted MLO sources from four
distinct audit exams outside every exam in the five-view challenge. The prompt
order alternates two deterministic A=180-degree present references and two
A=upright absent references, with one left and one right MLO in each class.
Source review orders are 14 (present), 29 (absent), 60 (present), and 10
(absent). Visual inspection of all four rendered pair panels confirmed the
assigned A/B polarity and equal-scale layout.

The 202-row synthetic source representation is
`qc_redo/review_batches/mirai_input_whole_exam_fallback_audit/inversion_polarity_exemplar_source_manifest.parquet`
with manifest SHA-256
`3b3b6b80ac1dc35b7a8927bb3f130fc134cd9dc21b883bdb168211fc0b2a5957`,
provenance SHA-256
`9c7114ed553efcbb8f61b22fc329008dfd8ae7522e88bef753e70ee17a642a86`,
and ordered panel-bank SHA-256
`47a5287427cbcf81211ea0ed4a6bf582513cf58529d48789f213295db7e8828e`.
The four-example bank is under
`qc_redo/auto_qc_development/mirai_input_llm_challenge/inversion_polarity_four_shot_experiment/exemplars`,
with manifest SHA-256
`55777cced68b1dc2a33552c4a942bdaaa6430c50226f758aa39ebcecb6b1ae94`
and experiment-provenance SHA-256
`6ced0792dc42066cb063bb7a4b896e2b4b8a33c2aaebdb99e39f34cdb98eb34e`.
The canonical loader accepted all four, verified both labels, canonical image
paths, and no scored-view overlap; builder provenance records exact audit-exam
disjointness and required operational label `absent`.

Validation used `ruff format`, `ruff check`, `py_compile`, the real 202-panel
producer, the real four-example experiment producer, canonical few-shot loading,
and visual inspection. No tests were written or run. Next check: commit and push
only the four source files, freeze the exact four-shot five-view protocol at that
commit and these artifact hashes, then use the shared-H200 broker rules before
launching exactly one restartable Qwen smoke.

Committed and pushed the generalized few-shot and synthetic-rotation
infrastructure as `75b5572` (`Add disjoint synthetic QC exemplars`) on
`feature/durable-dicom-lineage`. Froze
`inversion_pair_qwen27_four_shot_smoke_protocol.json` at that commit with
SHA-256
`452bffdb6f2a3e90ba59dc8f82bcd6a8234464d8c04c0bf8bde33c2b1cd83c33`;
fifteen code, prompt, model, scored-input, exemplar, disjointness, and absent-
output checks passed.

At 16:08 CDT, shared-H200 broker status enumerated all eight holders as idle,
with no external steps, compute owners, claims, or queued tasks. Every holder is
limited to 4 CPUs and 16 GiB RAM, below the Qwen checkpoint's 28.75 GiB weight
size and this run's 128 GiB host-memory requirement, so no holder was claimed or
entered. A matching Slurm duplicate check was also empty. Submitted exactly one
restartable typed-H200 opportunistic job via the repository `--no-srun`
submitter: job `12913275`, partitions `zhoulabq,catherineq,siweiq`, QOS
`opportunistic`, 1x `nvidia_h200-141gb`, 16 CPUs, 128 GiB RAM. Logs are under
`/ess/scratch/scratch1/annawoodard/prima_view_auto_qc/submitit_runs/qwen27_mirai_inversion_pair_four_shot_smoke_20260714_160935/submitit_logs`.
Next check: scheduler/runtime health, exact five-view and debug provenance, then
the frozen order-177 mechanistic positive plus four-control negative gate.

Job `12913275` completed `0:0` in 1m42s with five successful HTTP responses.
Forty artifact checks passed: exact 5/5 run and debug coverage, unique response
IDs, frozen model/revision/runtime/serve configuration/prompt/input/decoding,
the exact four-example order/labels/roles/image hashes, scored-set disjointness,
canonical target lineage, and run/debug agreement. The run SHA-256 is
`e26f8a9b39bb78a43101c76a8f0ad9f736b49ad81988c7a76c3aeb6a15cc3a2b`.

The frozen scientific gate failed. Order 177 remained a high-confidence true
positive with the required B-corrects-A mechanism. Control 75 remained negative,
and the examples repaired prior false positive 157; however, upright MLO controls
9 and 37 remained high-confidence positives. The model reused the positive
exemplar evidence phrase verbatim for the target and both remaining false
positives. The exact result is recorded in
`inversion_pair_qwen27_four_shot_smoke_result.json`.

Per the frozen stopping rule, no full pair bank or repeat was launched. Stop the
pairwise inversion branch rather than tuning its wording, example selection, or
model scale. This does not alter the retained deterministic-plus-modular baseline,
the validated tubing candidate-and-veto, the exposure development candidate, or
the broader open-source VLM program. Next check: return to the residual broad-QC
errors, identify the next visually coherent non-inversion mechanism, and design
one smallest discriminative prompt/model experiment rather than another
orientation variant.

## 2026-07-14 16:34 CDT — Prosthesis/expander residual reset

Audited the remaining primary broad-QC false negatives in same-exam context.
Orders 46 and 138 are the right MLO and RCC from one exam and both have a large,
smooth, nearly homogeneous unilateral reconstructed/prosthetic breast appearance;
the left CC/MLO controls 153 and 13 retain ordinary internal texture. Order 34
has a similar unilateral smooth prosthetic/reconstruction appearance, while
same-exam controls 157, 48, and 165 were accepted. Order 132 remains a separate
cropping/positioning-boundary case and is absent under secondary adjudication,
so it is not a good positive target for the next component.

This changes the interpretation of the prior "exposure" delta. Its prompt caught
order 46 because the anterior breast looked featureless white, but it missed 34
and 138 and explicitly says implants are another target. The three views are a
more coherent implant/reconstruction family than an exposure family. DICOM
`has_implant` is false for these exams, reinforcing that the tag is not a usable
reference label. The existing implant prompt also contains a concrete rubric
gap: it says tissue expanders without a separately visible prosthesis are
negative, whereas the frozen human rubric excludes every visible tissue
expander.

Implemented the smallest prompt-only reset as target `visible breast prosthesis
or tissue expander`, explicitly covering an unmistakable large coherent
prosthetic/expander volume even when a separate shell, valve, or reservoir is not
visible, while protecting dense breast, ordinary brightness, clips/scars alone,
cropping, and compression hardware. Committed and pushed the prompt plus updated
few-shot documentation as `9db479a` (`Broaden prosthesis QC target`).

Built an eleven-view directional bank with obvious implant order 12 and hard
orders 34, 46, and 138 as required positives; protected negatives are 13, 48,
132, 153, 155, 157, and 165. The manifest SHA-256 is
`497c1734a97c802c6e2b0b0f62d3786c7e19b6f75987e31ef83a973ac249d0fc`.
Froze `prosthesis_expander_prompt_qwen27_smoke_protocol.json` at commit
`9db479a` with SHA-256
`5fd573b4879f9f0a24c0f6591a4332d6d55177935c14653eda6e53ff14fb56b7`;
seventeen source, model, prompt, manifest, order, and absent-output checks passed.
The only experimental change from the existing implant run is the target
definition/prompt. Advancement requires all four positives high-confidence YES
with prosthesis-specific evidence and all seven controls parsed NO. Next check:
shared-broker owner/process enumeration and duplicate audit, then exactly one
restartable Qwen27 smoke.

At 16:40 CDT, the shared-H200 broker enumerated all eight holders as idle with
no external steps, active compute owners, claims, or queued tasks. Every holder
still has only 4 CPUs and 16 GiB RAM, below this vLLM run's 128 GiB host-memory
request, so no shared holder was claimed or entered. A matching Slurm duplicate
audit was empty. Rechecked the frozen commit, source owners, model config and
snapshot provenance, prompt, manifest, provenance, and absent outputs; all
twelve final preflight checks passed. Submitted exactly one restartable typed-
H200 opportunistic run through the repository `--no-srun` submitter: job
`12914526`, partitions `zhoulabq,catherineq,siweiq`, QOS `opportunistic`, 1x
`nvidia_h200-141gb`, 16 CPUs, and 128 GiB RAM. Submitit logs are under
`/ess/scratch/scratch1/annawoodard/prima_view_auto_qc/submitit_runs/qwen27_mirai_prosthesis_expander_prompt_smoke_20260714_164335/submitit_logs`.
Next check: scheduler/runtime health, exact eleven-view and debug provenance,
then the frozen four-positive/seven-control gate.

Job `12914526` completed `0:0` in 1m39s. Forty-eight artifact checks passed:
exact 11/11 run and debug coverage, unique response IDs, frozen model revision,
runtime and serving configuration, decoding, prompt and canonical input lineage,
run/debug agreement, and clean server startup. The run SHA-256 is
`c1de77803f1fce272db7231b6711baed172d6481119683e7399984e5e9461368`.

The frozen smoke gate failed. Obvious implant order 12 was a high-confidence
true positive with direct smooth-contour evidence, and all seven protected
controls were correctly negative. Hard operational positives 34, 46, and 138
were each high-confidence negative with no target evidence. These are exactly
the same eleven decisions made by the predecessor implant prompt, so broadening
the isolated-view definition did not change model behavior. The exact result is
recorded in `prosthesis_expander_prompt_qwen27_smoke_result.json` (SHA-256
`12847de44a4909dd1dd50f486684ff11fdedb6ff7d5f7a5ee7f7f912f20cb093`).

Per the frozen rule, do not launch a 202-view prosthesis component and do not
add examples automatically. This falsifies the claim that these three residuals
are recognizable prosthesis/expander cases from isolated views under the current
model/prompt path; it does not prove the operational labels are wrong. Retain
the deterministic-plus-modular baseline. Next check: build the smallest same-
exam comparative experiment that can distinguish a genuinely unilateral
prosthetic/reconstructed appearance from exposure or ordinary asymmetry before
assigning another residual mechanism.

Visual inspection of the frozen same-exam composites corrected the next
hypothesis. Orders 46 and 138 are paired right MLO/RCC views with the same broad,
nearly featureless white breast region, while their left-side references retain
ordinary internal texture. This is stronger evidence for the already successful
exposure/processing component than for a classic implant shell. Order 34 remains
a cause-ambiguous boundary and is protected rather than required in the next
mechanism check. The earlier isolated exposure arm is the nearest reference: it
recovered order 46 without a new false positive or safety regression but missed
138 and therefore failed its frozen two-recovery gate.

Built an eleven-view subset from the already-frozen target-dominant same-exam
context bank. Required exposure positives are 46 and 138; protected controls are
12, 13, 34, 48, 132, 153, 155, 157, and 165, covering a classic implant,
ordinary same-exam views, a cause-ambiguous boundary, cropping, gross compression
hardware, and views whose references contain the failure while the target does
not. The challenge manifest SHA-256 is
`840d262628adfd5d8c8e6cd7baf05274e0d26507ecee3c3f5d2c2138e7ab0718`.
Froze `exposure_context_qwen27_smoke_protocol.json` with SHA-256
`4d008f90cb284bc9df6a36935d25e3c08bf326d42bdf88081e8cfc8076721d14`;
seventeen commit, source-owner, model, prompt, prior-run, context-bank,
challenge-lineage, and absent-output checks passed. The only changed variable is
the model image: isolated canonical target to the prebuilt target-dominant
same-exam composite. Next check: shared-broker status and duplicate audit, then
exactly one restartable Qwen27 context smoke.

At 17:02 CDT, broker status reported no running shared persistent H200 holders,
no queued broker tasks, and recommended the restartable opportunistic route; a
matching Slurm duplicate audit was empty. Submitted exactly one typed-H200 job
through the repository `--no-srun` launcher: job `12914793`, partitions
`zhoulabq,catherineq,siweiq`, QOS `opportunistic`, 1x
`nvidia_h200-141gb`, 16 CPUs, and 128 GiB RAM. Submitit logs are under
`/ess/scratch/scratch1/annawoodard/prima_view_auto_qc/submitit_runs/qwen27_mirai_exposure_context_smoke_20260714_170322/submitit_logs`.
Next check: scheduler/runtime health, exact eleven-view context and canonical
lineage, then the frozen two-positive/nine-control gate.

Job `12914793` completed `0:0` in 2m39s with eleven HTTP successes. Fifty-nine
artifact checks passed, including exact 11/11 run and debug coverage, unique
response IDs, frozen context-bank and subset lineage, valid 1600x1200 context
images, unchanged canonical evaluation paths, recorded model image column
`model_image_path` and exact model-input manifest hash, identical non-input
settings to the isolated exposure reference, and run/debug agreement. The run
SHA-256 is
`dd9c355bed8db2131a36fe9fb4f09a7847dbfef0cfac51f9d59c32c277eadb0c`.

The passive-context gate failed. All eleven decisions were high-confidence NO.
All nine protected controls remained negative, but order 138 remained missed and
the known isolated-view recovery at order 46 regressed from high-confidence YES
to high-confidence NO. The exact result is recorded in
`exposure_context_qwen27_smoke_result.json` (SHA-256
`a82eeae4f04bb723b5f03905e6237b5f43e4c51388d4f7bb098fbecf6a33290f`).
Reject only the passive-context delta; retain the immutable baseline and the
clean isolated-view exposure development candidate.

The regression of a known positive despite proven context-input lineage is
consistent with an instruction-to-representation mismatch: the unchanged prompt
says only "this mammogram" and never tells the model that the large labeled
TARGET panel, not the three normal references collectively, owns the decision.
The bounded next falsifier is therefore one context-explicit prompt on the same
frozen bank, model, and decoding. Do not add examples, change scale, or change
model family. If the explicit prompt still misses either required positive or
flags a protected target, stop this exposure-context branch.

Implemented the bounded context-explicit prompt. It assigns the decision only
to the large labeled TARGET VIEW, treats smaller same-exam panels as fallible
comparison aids, requires evidence to begin with `TARGET`, and says that a
failure visible only in a reference cannot reject the target. Added the stable
target-ownership rule to the same-exam context documentation. No examples,
model, model input, scale, decoding, threshold, or challenge roles changed.
Repository formatting and linting passed; no tests were written or run.
Committed and pushed these two tracked files as `debc0f4` (`Clarify same-exam
exposure QC`).

Froze `exposure_context_explicit_qwen27_smoke_protocol.json` at that commit with
SHA-256
`361645a84feb72582704a34733f52c8604e181cc49afce29bdc0e939e6cbfc81`;
eighteen commit, source-owner, model, prompt, predecessor, context-bank,
challenge-lineage, and absent-output checks passed. Advancement still requires
both orders 46 and 138 as high-confidence TARGET-specific positives and all nine
protected targets negative. Next check: shared-broker status and duplicate
audit, then exactly one final restartable exposure-context smoke.

At 17:15 CDT, broker status enumerated seven idle holders with no external
steps, active owners, claims, or queued tasks. Each holder again had only 4 CPUs
and 16 GiB RAM, below the 128 GiB vLLM request, so none was claimed or entered.
A matching Slurm duplicate audit was empty. Submitted exactly one restartable
typed-H200 run through the repository `--no-srun` launcher: job `12915036`,
partitions `zhoulabq,catherineq,siweiq`, QOS `opportunistic`, 1x
`nvidia_h200-141gb`, 16 CPUs, and 128 GiB RAM. Submitit logs are under
`/ess/scratch/scratch1/annawoodard/prima_view_auto_qc/submitit_runs/qwen27_mirai_exposure_context_explicit_smoke_20260714_171621/submitit_logs`.
Next check: scheduler/runtime health, exact context provenance, then the final
two-positive/nine-control stopping gate.

Job `12915036` completed `0:0` in 2m09s with eleven HTTP successes. Sixty-one
artifact checks passed, including exact 11/11 run and debug coverage, frozen
context/canonical lineage, identical model images, model, decoding, and runtime
to the passive predecessor, the exact new prompt as the sole changed factor,
unique responses, and run/debug agreement. The run SHA-256 is
`986d11dec07c5cb51c9065f7d1bae6f6654c64304f5e5543f42e4603b632ebd0`.

The context-explicit prompt repaired the passive-context regression: known
order 46 returned to high-confidence YES with TARGET-owned featureless-white
saturation evidence. Every protected target remained high-confidence NO, and
all eleven responses followed the required TARGET evidence format. Order 138,
however, remained a high-confidence NO with the explicit judgment that internal
target texture was visible. The eleven decisions therefore exactly match the
original isolated exposure prompt; context added no recovery. The frozen gate
failed, and the exact result is recorded in
`exposure_context_explicit_qwen27_smoke_result.json` (SHA-256
`fa43eee4951d2a309870c003c1f47c15c1080cef9a660e1697cee2f32a9a6cf9`).

Stop the exposure-context branch without further wording, examples, scale, or
model changes. Retain the immutable baseline and the clean isolated-view
exposure development candidate. The stable engineering lesson remains valid:
multi-panel prompts need explicit target ownership, but same-exam context does
not make order 138 an exposure/processing positive for this model.

Registered a no-new-inference consolidation of the two useful prompt-only
deltas before deriving it: the completed modular-baseline-plus-tubing-veto
visual run and the clean isolated exposure run, combined by high-confidence OR
and then ORed with frozen deterministic DICOM eligibility. This is explicitly a
post-selection development reference, not a retrospective claim that the old
exposure two-recovery arm passed. The first evaluator invocation failed before
writing an evaluation because the candidate protocol omitted the evaluator's
required `reference` field. Added the unchanged primary/adjudicated reference
definition, rehashed the protocol to
`0e13430581dcf2dba071683f430ba98e2d1483907ca9a325d8b7bc2b5094f7db`,
and resumed only the missing evaluations. The combined visual/system artifacts
had not consumed the protocol and remained valid.

Twenty-six artifact checks passed. The best observed candidate recovered exact
baseline misses 24 and 46, introduced zero false positives, reduced unsafe
selected slots by two and unsafe accepted exams by two, and left false
exhaustion unchanged. Primary scoring is TP 89, FN 4, FP 3, TN 106,
sensitivity 0.9570, specificity 0.9725, four unsafe selected slots, two false
exhaustions, and two unsafe accepted exams. Secondary adjudicated scoring is TP
91, FN 3, FP 1, TN 107, sensitivity 0.9681, specificity 0.9907, three unsafe
selected slots, one false exhaustion, and one unsafe accepted exam. Both
view-rate thresholds pass, but both operational whole-exam gates fail.

The exact result is
`arms/best_prompt_only_tubing_veto_exposure/result.json` (SHA-256
`55aa5d81adb34bd778c376ed3c41c522c024e18dc4379b0310b9677fee53d353`).
Primary residual false negatives are 34, 132, 138, and 177; secondary residual
false negatives are 34, 138, and 177. Primary false positives are 147, 148, and
165, but only 165 remains a false positive after adjudication. Retain this as
the nearest prompt-only development reference without replacing the immutable
baseline or claiming prospective success.

## 2026-07-14 17:52 CDT — Broad-alert usability veto falsified

Tested the smallest candidate-and-veto rescue for the old high-recall broad
integrity detector. Its 66 high-confidence positives include residual failures
34, 132, and 138 but also 33 ordinary-view false positives whose rationales
nearly all claim gross crop. The new conservative observer answers YES only
when the large TARGET panel is unmistakably an ordinary usable whole-breast
view; only high-confidence YES can veto a broad alert, and uncertainty preserves
the exclusion. The balanced twelve-view smoke contains six ordinary crop
look-alikes and six failure-preservation targets spanning a detector seam,
feature loss, prosthesis, and gross compression hardware. Committed and pushed
the frozen observer prompt as `efbca2b` (`Add conservative whole-breast usability
veto`).

Broker status enumerated eight idle persistent H200 holders, but every holder
had only 4 CPUs and 16 GiB host memory, below the 16-CPU/128-GiB vLLM request;
none was claimed or entered. A duplicate audit was empty. Submitted exactly one
restartable typed-H200 opportunistic job through the repository `--no-srun`
launcher. Job `12915662` completed `0:0` in 2m05s on `zhoulabq`, and all twelve
API requests succeeded. Ninety-six run, debug, prompt, model, context-input,
canonical-lineage, parser, and evidence-ownership checks passed. The run SHA-256
is `bb002901a58d9ddf737d09afd8736304cabfd9b7fe0580b6a6eea0efb42fe18a`.

The smoke failed 7/12. Only three of six required usable targets were
high-confidence YES; ordinary targets 37, 48, and 157 remained falsely excluded.
More importantly, definite feature-loss target 46 and prosthesis-family target
129 were incorrectly declared usable at high confidence. This is not repairable
by raising the existing high-confidence threshold. A metadata-only protocol
erratum records that the runner hashes prompt text after stripping terminal
whitespace; the committed raw prompt file and every inference input were
unchanged. The exact failed result is
`broad_alert_usable_judge_qwen27_smoke_result.json` (SHA-256
`5184a22e265d70553937fd46e1c950788a7f93e6bd0c934151cdb82c692cc204`).

Stop this broad candidate-and-usability-veto branch. Do not run the 66-view
observer and do not tune its wording, examples, scale, or model family. Retain
the best tubing-veto plus isolated-exposure prompt-only system. Next check: test
one cause-agnostic, appearance-defined component for the coherent smooth,
near-homogeneous whole-breast residual family before concluding that Qwen lacks
the needed visual sensitivity.

## 2026-07-14 18:08 CDT — Cause-agnostic phenotype smoke passed; full continuation running

Reused the exact eleven isolated-view prosthesis smoke bank but changed the
scientific target from a guessed cause to the visible phenotype `large smooth
near-homogeneous breast region`. The prompt treats classic prosthesis,
reconstruction, saturation, and processing as possible causes but asks only
whether one broad smooth region occupies a substantial fraction of the breast
with little internal parenchymal texture. Dense tissue, ordinary low contrast,
crop, black background, pectoral muscle, markers, and compression hardware alone
are explicit negatives. Committed and pushed the prompt as `4575f9a` (`Add
cause-agnostic breast texture QC target`).

Froze the eleven-view protocol before inference. The only experimental change
from the failed prosthesis smoke is the target/prompt; image bank, model,
decoding, and roles are unchanged. Broker status showed eight idle holders but
all remained limited to 4 CPUs/16 GiB, so none was claimed or entered. Job
`12915796` was the only copy and completed `0:0` in 1m34s on one typed
opportunistic H200. Seventy-three artifact checks passed. The run SHA-256 is
`7aea3f66cba45ce5cef23a9cd62b9cc3f328aa7d19edf0ea4541f231e94d09f5`.

The frozen smoke passed exactly 11/11. Defining orders 12, 34, 46, and 138 were
all high-confidence YES; protected orders 13, 48, 132, 153, 155, 157, and 165
were all high-confidence NO. This changed all three hard cause-specific misses
without sacrificing a registered look-alike, supporting wrong cause attribution
rather than basic visual blindness on this small selected bank. The exact result
is `smooth_homogeneous_region_qwen27_smoke_result.json` (SHA-256
`8f8bab8fa445bcf5d9c6f6d2a635b69f0da908dee8ea9ff8d4fa912329e26ca9`).

Froze the full-202 continuation protocol at SHA-256
`2c4faea8d30bc617a1a0f72b437064cfc61e323044626affed5234b6e83b0497`.
Advancement requires exact smoke-role preservation, recovery of adjudicated
residuals 34 and 138 from the current-best system, zero introduced primary and
adjudicated false positives, and no whole-exam safety regression. The same
broker and duplicate audit remained clean, with holders still undersized.
Submitted exactly one restartable typed-H200 opportunistic job through the
repository `--no-srun` launcher: job `12915798`. Next check: exact 202-view run
and debug provenance, then component, primary, adjudicated, delta, and
whole-exam safety scoring.

Job `12915798` completed `0:0` in 3m26s with exact 202/202 inference coverage.
The run passed 1,825 model, prompt, canonical-path, debug, parser, response-ID,
and decoding checks and retained all eleven registered smoke roles. It emitted
25 high-confidence component positives. The component run SHA-256 is
`4c5ac2baa7c28aee02a8fa7138669d5920a357ada8008b997caacdeeccff0d2d`.

Both the primary and adjudicated incremental gates passed. The appearance
component recovered exact residual orders 34 and 138, introduced zero new false
positives under either label state, reduced unsafe selected slots by two, and
did not change false exhausted slots or unsafe accepted exams. Primary scoring
is TP 91/FN 2/FP 3/TN 106, sensitivity 0.9785, specificity 0.9725, two unsafe
selected slots, two false exhaustions, and two unsafe accepted exams. Secondary
adjudicated scoring is TP 93/FN 1/FP 1/TN 107, sensitivity 0.9894, specificity
0.9907, one unsafe selected slot, one false exhaustion, and one unsafe accepted
exam. The remaining adjudicated false negative is exact order 177; the remaining
adjudicated false positive is 165.

The exact combined result is
`arms/best_prompt_only_tubing_veto_exposure_smooth_region/result.json`
(SHA-256
`d0ef37217cd118eb7e00d1063a7eabb99e455738d0c974201980fb2de18ea7dd`).
Retain it as the new best observed post-selection prompt-only development system,
without replacing the immutable original baseline or claiming prospective
success. The final operational gate still fails solely on one adjudicated unsafe
view/exam. Next check: select a genuinely different medical VLM family for the
already-frozen five-view 180-degree inversion mechanism; do not return to Qwen
scale, pair wording, few-shot references, or GLM variants already falsified.

## 2026-07-14 18:49 CDT — Independent Mistral inversion smoke failed; MedGemma access is the next blocker

Probed the preferred genuinely different medical model,
`google/medgemma-27b-it` at pinned revision
`2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f`, before downloading any large
files. Hugging Face denied even `config.json`: the repository is gated and this
host has no authenticated accepted-license session. `UFNLP/MedGPT-oss` is also
gated. The ungated Microsoft LLaVA-Med 1.5 checkpoint was not pursued because
its custom `LlavaMistralForCausalLM` architecture is absent from the pinned
vLLM registry and its CLIP tower reduces the exact 1440x720 pair to 336-pixel
inputs.

Selected one bounded ungated alternative: Apache-2.0
`mistralai/Mistral-Small-3.2-24B-Instruct-2506` at revision
`95a6d26c4bfb886c58daf9d3f7332c857cb27b43`. Its independent Pixtral tower
accepts 1540-pixel images and the architecture is native in vLLM 0.24.0. Added
explicit model weight formats and download exclusions in commits `f3772fb` and
`4b0be60`, preventing the model repository's duplicate Hugging Face and
canonical Mistral representations from both being downloaded. The canonical
datamover owner transferred and size-verified one 48,022,792,280-byte
`consolidated.safetensors`; repository revision and weight validation passed.

Shared-pool status showed two idle H200 holders, but every holder still had only
4 CPUs/16 GiB, unsafe for a model documented and observed to consume about 55
GB/44.76 GiB on GPU plus loader memory. No broker slot was claimed or entered.
After a clean duplicate audit, submitted one typed-H200 opportunistic smoke via
the repository `--no-srun` launcher.

Initial job `12919843` loaded successfully but failed HTTP 400 before scoring
any view because the native Mistral tokenizer rejects Qwen-specific
request-level `chat_template_kwargs`. Preserved the failed run, server log, and
original protocol under
`failed_attempts/mistral24b_request_template_kwargs_12919843`. Made request
chat-template kwargs an explicit required per-model registry policy, recorded
them in run provenance and the exact challenge scorer, retained
`enable_thinking=false` for Qwen/GLM, and sent none for Mistral. Commit
`319fc76` was pushed. This runtime-only amendment changed no model weights,
images, prompt, output parser, labels, thresholds, or advancement rule.

The amended frozen protocol SHA-256 is
`14291d7ba95dbd31602103dd15182ff4f010d5d6651e1210b20b51ec9f8f4cd2`.
Retry job `12919987` completed `0:0` in 1m29s; all five requests returned 200.
Forty-eight artifact checks passed with exact 5/5 manifest and debug coverage,
five unique response IDs, and matching model revision, runtime, serving args,
request policy, prompt, and input-manifest provenance.

The frozen smoke failed. Mistral returned high-confidence NO for all five
orders: protected controls 9, 37, 75, and 157 passed, but required inversion 177
was missed. All five explanations repeated that A was upright because the
pectoral muscle was at the bottom, an internally inconsistent interpretation of
the prompt's superior/inferior anatomy rather than a confidence-threshold edge
case. The exact result is `inversion_pair_mistral24b_smoke_result.json`
(SHA-256
`cbb4c13af38c899134021be10e8149c540167611a271dbfe9f1ecf727d352a06`).

Stop the Mistral pair branch without a 202-view run. Retain the current best
tubing-veto plus exposure plus smooth-region prompt-only development system and
its one residual adjudicated unsafe view/exam. Next check: obtain authenticated
accepted-license access to MedGemma 27B, then run the same frozen five-view
directional smoke; do not resume Qwen, GLM, Mistral, prompt-wording, scale, or
few-shot variants already falsified.

## 2026-07-14 — MedGemma frozen inversion-pair smoke

Authenticated Hugging Face access to gated `google/medgemma-27b-it` succeeded.
Pinned revision `2d3e00ea38b50018bf5dd3aa1009457cd2d5a48f` was downloaded and
verified as twelve safetensor shards totaling 54,864,980,440 bytes. Registered
the model and its model-specific vLLM policy in Prima commit `2d7005e`, pushed
only to `uchicago-dsi/prima` on `feature/durable-dicom-lineage`. The installed
`dsi-local-llms==0.1.1` dependency and the `local-llms` repository were not
modified or pushed.

The shared H200 broker had no slot with sufficient CPU/RAM, so no shared holder
was claimed. After a clean duplicate audit, submitted one opportunistic typed
H200 job through the Prima `--no-srun` launcher. Job `12922532` completed `0:0`
in 2m47s. Gemma3 loaded 51.54 GiB of weights, all five HTTP requests succeeded,
and the only engine error occurred during managed teardown after all responses
and artifact writes.

All 38 artifact checks passed: exact manifest and panel hashes, exact 5/5 run
and debug coverage, five unique response IDs, run/debug agreement, pinned model
revision/runtime/serve/request policy, exact prompt and input manifest, source
commit hashes, and preserved canonical image lineage. The frozen smoke failed
scientifically. MedGemma returned the identical high-confidence YES decision
and nipple-based `B corrects A` rationale for the true inversion at order 177
and all four protected negatives (9, 37, 75, 157). This is a comparison-position
or prompt-template shortcut, not useful inversion discrimination.

The exact result is `inversion_pair_medgemma27b_smoke_result.json` (SHA-256
`0fe21e0c0a4ef512315cd8b6ad8525d754aa3fdbbf6ea2a00a9548939cbe686f`).
Do not launch the 202-view MedGemma pair run. Retain the current best
tubing-veto plus exposure plus smooth-region prompt-only development system and
do not resume the falsified Qwen, GLM, Mistral, or MedGemma inversion-pair
branches without a materially different representation or hypothesis.

### Counterbalanced MedGemma falsifier

Ran one materially different, predeclared candidate-position falsifier before
closing MedGemma orientation work. The same five source images were rebuilt so
candidate A was rotated 180 degrees and candidate B became the original
orientation. Exact pixel comparison confirmed that the image-bearing A/B
regions exchanged positions. The primary gate was restricted before inference
to MLO orders 9, 37, and 177; CC orders 75 and 157 were off-gate diagnostics.
The frozen protocol SHA-256 was
`ef8dd6d974a5c32a82432dff7f444d542b4e8e22dc597057db4ccf5de4284d14`.

The shared broker had four idle H200 holders, but each had only 4 CPUs/16 GiB,
below the verified 16 CPU/128 GiB MedGemma serving requirement. No holder was
claimed. Job `12922544` therefore used the restartable typed-H200 opportunistic
path through the Prima `--no-srun` launcher and completed `0:0` in 2m40s.
All five requests succeeded and 37 artifact checks passed, including exact
swapped pixels, manifest/debug coverage, disjoint response IDs, pinned runtime,
prompt, model revision, and canonical lineage.

MedGemma returned the same high-confidence `B corrects A` decision and identical
nipple rationale for every swapped panel. The prespecified consistency rule
flagged no original MLO views and therefore still missed order 177. This
directly demonstrates candidate-position bias rather than orientation
equivariance. Exact result:
`inversion_pair_swapped_medgemma27b_smoke_result.json` (SHA-256
`9e95ac902066b570ae0f6c8c1aa22c314520b0f54cac23bb687322da5a028155`).
Stop MedGemma orientation work without a full run or prompt tuning; retain the
current best system.

## 2026-07-14 — Neutral Qwen axillary-landmark grounding smoke

Question: can the retained Qwen3.5-27B VLM localize one visible MLO landmark in
display coordinates without being asked to classify orientation? This was a
rut-breaker reset from repeated direct/candidate-choice prompts, not another
wording variant. Raw-image inspection showed recognizable axillary nodes in
band 1 for upright controls 9 and 37 and band 4 for inverted order 177; exact
180-degree variants reverse those locations.

Added a reusable four-band single-image renderer, provenance-preserving builder,
and neutral landmark prompt in Prima commit `894b502` (`Add landmark-grid QC
probes`), pushed only to `uchicago-dsi/prima`. `local-llms` was not changed. The
six derived panels retain canonical source lineage, use opaque synthetic IDs,
and do not expose the rotation or QC label. Visual inspection confirmed the
frozen 1-to-4 node mapping for all three pairs. Validation used Ruff,
`py_compile`, CLI help, the real-input producer, panel hashes, and visual
inspection; no tests were written or run.

The protocol was frozen before inference at SHA-256
`bf1406b67e7b924a1df79664d646f95993dd24518b5202f54bcab76406e5fab3`.
Twenty-eight preflight checks passed. All eight broker holders were idle but
limited to 4 CPUs/16 GiB, below the verified Qwen 16 CPU/128 GiB serving need,
so no shared holder was claimed. Exactly one restartable typed-H200
opportunistic job was submitted through the Prima `--no-srun` launcher. Job
`12923770` completed `0:0` in 1m42s with six successful HTTP responses.

Forty artifact checks passed: exact 6/6 manifest/run/debug coverage, unique
response IDs, run/debug agreement, pinned model/revision/runtime/serve/request
policy and prompt, source commit hashes, original/180 variant provenance, and
canonical lineage. The scientific gate failed 0/6 exact rows. Qwen answered YES
at high confidence for every panel but reported band 1 for five of six,
reported band 2 once, and never used the clearly visible node cue. No source
pair had the required exact 1-to-4 reversal. The deterministic rule therefore
flagged nothing and still missed order 177.

The result is
`mlo_axillary_landmark_qwen27_smoke_result.json` (SHA-256
`3eb46b580c855459e5634881b7229d62c3a29163df0d7c55f883b2051163c7c1`).
Interpretation: Qwen substituted a fixed upper-image pectoral/axillary semantic
prior instead of grounding the requested nodes. Stop this zero-shot Qwen
four-band axillary reset without wording, band-count, examples, or model sweep;
do not launch the full MLO bank. This rejects only that additive grounding
mechanism. Retain the tubing-veto plus exposure plus smooth-region best system
and the broader open-weight VLM program.

## 2026-07-14 — Supervised Qwen2-VL-2B MLO-orientation falsifiers

Question: does a small supervised adapter recover 180-degree MLO orientation
from pixels after zero-shot prompting and neutral landmark grounding failed?
Built a provenance-preserving development bank from 27 distinct, human-reviewed
MLO sources with no patient, exam, or view overlap with the 202-view audit. Each
source contributes its original and exact 180-degree rotation: 19 source pairs
for training, eight for validation, and three audit-derived challenge pairs
(upright controls 9 and 37 plus inverted order 177). Text was suppressed before
rotation, anatomy was retained, and exact rotation was verified for all 30
pairs. The manifest SHA-256 is
`c97023e611a1c3f08f4028e73a6804aa4125a244a3343ca35cd70e123dd588fb`;
the frozen bank SHA-256 is
`76ed89b843c3569d39734a58495f36193cd17e0befce842ce359f655d7c91ba4`.

Added the deterministic builder, Qwen2-VL LoRA trainer, and approved launcher
in Prima commit `d487315`, pushed only to `uchicago-dsi/prima`. Three runtime
integration failures occurred before any optimizer step and were archived with
their logs and exact fixes: Torch 2.0 required eager attention (`c1296b6`),
PEFT's generic `all-linear` selector incorrectly included a Conv3d module so
the trainer now enumerates only `torch.nn.Linear` targets (`f7610aa`), and the
installed Accelerate rejects `data_seed` so the frozen global seed is used
instead (`0f8cf73`). These were runtime-only amendments; the data bank and
scientific gate did not change. All compute used claimed Prima shared-H200
broker tasks; no raw job attachment or duplicate Slurm route was used.

The first arm used full labels `UPRIGHT` and `INVERTED` under protocol SHA-256
`e455ca243bdc63d86f708b8dbe6a4723bda2f6e8070e8b2daac35ea21c5a6bf`.
Request `prima-mlo-orientation-lora-qwen2vl2b-2c2d2fc898` completed, but both
the base and adapted model generated `UPRIGHT` for every row. Adapted accuracy
was 19/38 train, 8/16 validation, and 3/6 challenge with zero correct source
pairs. The near-chance validation loss was misleading: `UPRIGHT` has four
supervised suffix tokens and `INVERTED` five, so an image-blind model uncertain
only on the first token has expected mean loss `2*ln(2)/9 = 0.1540`, essentially
the observed 0.1491. This arm learned label continuations and formatting, not
the image decision. Exact result summary SHA-256:
`042ba14e3752e23c9dd100ae559554290f5ccc12aff2f916f821fadd61ac91c5`.

Commit `7e0c80a` changed only the objective to balanced one-token verbalizers
`A=UPRIGHT` and `B=INVERTED`; preflight confirmed exactly token 32 or 33 was
supervised on every row. The frozen protocol SHA-256 is
`48c087a8c72a2adb51d93925457a603de0cdfa7f9e304b67e17911aa78df4f88`.
Request `prima-mlo-orientation-single-token-qwen2vl2b-3ee751b8c1` completed in
190.3 seconds. This arm learned nontrivial signal: train accuracy 34/38 (15/19
exact pairs), validation accuracy 12/16 with inversion sensitivity 1.0 and
upright specificity 0.5 (4/8 exact pairs), and validation loss fell from 0.6970
to 0.5640. It nevertheless failed the safety-relevant challenge: it emitted
`B` for all six images, correctly catching inverted order 177 while falsely
rejecting both upright controls and the corrected rotation of order 177. Exact
result-summary SHA-256:
`b54f22cfeefe850fc3f6d17052b7675cc3551ab5f63880e8fdfaf3303e4033dc`.

Stop Qwen2-VL-2B on this bank: no prompt, token-map, rank, learning-rate, epoch,
or example tuning. Retain the existing prompt-only QC baseline unchanged and
make no prospective claim. A successor model using the identical one-token
bank must fit every training pair, solve at least seven of eight validation
pairs, and preserve all three challenge polarity reversals before any larger
evaluation. Change the model only; do not simultaneously expand the bank.

## 2026-07-14 — Frozen Qwen2-VL-7B capacity ablation

Question: was the partial Qwen2-VL-2B result limited by decoder/multimodal
capacity rather than the visual representation or data bank? Compared official
pinned configs before selecting the arm. Qwen2-VL-2B and Qwen2-VL-7B both use
a 32-layer, 1280-wide, 16-head visual encoder; the 7B checkpoint changes the
multimodal output width from 1536 to 3584 and enlarges the decoder. This makes
7B a clean capacity ablation, not a different visual-architecture test.
Qwen3-VL-4B would change the visual stack but is unsupported by the current
repo-mandated Prima training runtime, so it was held as the next decision.

Generalized the trainer's pinned snapshot validation to accept an explicit
expected Qwen2-VL repository, added the pinned 7B registry entry, validated
against the existing real 2B snapshot, and pushed Prima commit `bdce77b` only
to `uchicago-dsi/prima`. The local-llms repository was not modified or pushed.
Pinned `Qwen/Qwen2-VL-7B-Instruct` revision
`eed13092ef92e448dd6875b2a00151bd3f7db0ac` (Apache-2.0). The login-node Hub
route projected roughly three hours at about 1.5 MB/s aggregate, so it was
stopped without producing a snapshot. The resumable five-worker datamover route
then transferred and size-verified all five shards (16,582,831,200 bytes) in
about 20 seconds. Snapshot provenance SHA-256:
`5bcfb67b7812f79c5a1947a39355d99c563d086d506380ef20c11b0040c8033e`.

Real-snapshot preflight verified all 60 manifest rows, one supervised token per
class (`A` id 32 and `B` id 33), 326 exact Linear adapter targets (130 vision,
196 language, no Conv3d), and 25,579,520 trainable parameters. The frozen
protocol SHA-256 was
`d0ecc02fe5e53f8d5fff77543122a7bc0557608521890a410ee77ac7c5fcbd2e`.
All eight broker slots were idle and free of external steps, but each had only
16 GiB host RAM, less than the checkpoint plus Python and load overhead. With
no broker claim or duplicate request, submitted the same deterministic task via
the typed `nvidia_h200-141gb` opportunistic route using `slurm_use_srun=false`,
16 CPUs, and 128 GiB. Job `12924003` ran in `zhoulabq` and completed `0:0` in
5m39s; peak batch RSS was about 1.14 GiB after direct GPU placement.

The 7B arm failed the frozen mechanism gate. It fit only 22/38 training labels
and 7/19 training pairs, versus 34/38 and 15/19 for 2B. Validation was 11/16
labels, inversion sensitivity 0.625, upright specificity 0.75, and 4/8 exact
pairs, tying the 2B pair count. Challenge performance was 4/6 labels and 1/3
exact pairs: original inverted order 177 was caught, but its corrected rotation
was still rejected; one of the two upright-control pairs reversed correctly.
Final validation loss was 0.6897 versus 0.5640 for 2B. The exact result SHA-256
is `d796b4e2f25a59e99036a1e65ffc9388167b8178fdb44f022197bf766e1f3a26`;
the checked summary SHA-256 is
`487a5ea65dde2fcdff32ecf7f977057b18c181eb7e74f534aad13061dd0ba5ff`.

Conclusion: increasing Qwen2-VL decoder capacity does not recover stable
180-degree orientation from the shared visual family under the frozen
one-token supervision. Stop both Qwen2-VL sizes on this bank—no prompt,
verbalizer, rank, learning-rate, epoch, augmentation, or further same-family
scale tuning. Retain the current prompt-only QC baseline unchanged. The next
information-changing arm should use a genuinely different visual architecture
with the exact frozen bank before changing data scale; Qwen3-VL-4B is the
leading open-weight candidate, conditional on an explicitly approved training
runtime path.

## 2026-07-15 — SmolVLM visual-architecture orientation arm

Question: after Qwen2-VL decoder scale failed, can a genuinely different visual
architecture learn the same frozen one-token MLO-orientation task without a new
training environment? Selected `HuggingFaceTB/SmolVLM-Instruct` revision
`81cd9a775a4d644f2faf4e7becff4559b46b14c7`: Apache-2.0, native support in the
pinned Transformers 4.46 runtime, and a 27-layer shape-optimized SigLIP visual
encoder under Idefics3. Its scale is close to the Qwen2-VL-2B reference, so the
arm changes visual architecture and its required processor rather than decoder
scale.

Generalized the trainer to the maintained native Idefics3 class and processor,
with exact expected-architecture snapshot validation and per-architecture
vision/language target checks. Prima commit `d38fbc0` added the model arm. The
SmolVLM repository also publishes many large ONNX exports; an initial staging
attempt exposed that datamover mode replaced model-specific ignore patterns.
Commits `37aa085` and `fb15586` added `onnx/*` to the pinned model policy and
fixed the shared downloader to combine those exclusions with temporary weight
exclusions. The corrected route retained no ONNX files and size-verified the
single 4,492,630,912-byte PyTorch weight. Snapshot provenance SHA-256:
`fc26a4ea0fb7aa57bb851560635bb10ae27d6a9731ae02d21a2b7ba9107e82bd`.

Real processor preflight found two maintained-runtime integration details
before GPU use. Pillow 9.0 did not expose the `ImagePalette` module required by
Idefics3, so the Prima environment and training requirements now pin Pillow
10.4.0. SmolVLM's assistant template renders the output as one space-prefixed
token (`A` id 330 or `B` id 389), whereas raw tokenization gives another ID.
Commit `aee4a38` derives the supervised token from the two fully rendered
assistant templates, requires exactly one token difference, and masks all
other suffix/prompt/image tokens. Real-input preflight verified one supervised
position for both SmolVLM and the prior Qwen model, 17 native 384-pixel image
tiles per unchanged 896-pixel input, and 331 Linear targets (162 vision, 168
language, one connector) with 13,490,944 trainable parameters. No tests were
written or run. The frozen base protocol SHA-256 was
`db2962ec6e13b1217f5c15b0b2b020b97854b3c39bb42c91253e1b88a1f214b2`.

The broker atomically claimed request
`prima-mlo-orientation-smolvlm-f61e919d27` on priority slot `12890338_0` after
confirming no queued task, external step, or compute owner. The first base
forward failed before any prediction or optimizer step because Torch 2.0.1's
CUDA `torch.triu` does not implement bfloat16 for the Idefics3 eager causal
mask. Preserved the empty output plus exact broker/log hashes under
`failed_attempts/smolvlm_bf16_torch20_causal_mask`. Commit `d0c4519` made
precision explicit. The frozen float16-only amendment SHA-256 was
`5b91a8cf112e28297d9a84c5f081b6deb26ecd3a4ca3a1535104bff543ffe376`;
all model/data/objective/adapter/optimization/seed/gate settings remained fixed.

Broker request `prima-mlo-orientation-smolvlm-fp16-a55cd3924c` then claimed the
same free priority slot as step `12890342.17` and completed with code 0 in
6m20s. Float16 was numerically noisy: two logged NaN gradient norms and
validation losses 0.6743, 0.6175, 5.9073, 2.7576, 2.4778, 0.5921, 0.7480, and
0.7844. It nevertheless recovered finite training, wrote every artifact, and
fit all 38 training labels and all 19 source pairs exactly.

The scientific gate failed, but this is the strongest orientation mechanism
result so far. Validation was 13/16 labels, inversion sensitivity 0.75, upright
specificity 0.875, and 5/8 exact pairs. Challenge was 5/6 labels and 2/3 exact
pairs. Both directions of the critical order-177 pair were correct: the real
inverted view was rejected and its corrected rotation accepted. Order 9 also
reversed exactly. The only challenge error was false rejection of original
upright order 37; its rotated inverted version was correct. Exact result
SHA-256: `559b2811d0d5f6b2e89bd67da7b5a228278ca42c4277b286e1e73bb18215f2e4`.
Checked summary SHA-256:
`c6d608974edf3eb61810cabcc1faaa2e8276e5e02c2fe0d469b9a2b9eda920a9`.

Interpretation: replacing the visual architecture materially changed the
learnability of mammography orientation; this is not a total representation
failure. The remaining problem is patient/source generalization, with float16
instability as a secondary runtime confound. Do not promote the adapter or
tune prompt, template tokens, tiling, rank, learning rate, epochs, precision,
or checkpoint on this now-unblinded bank. Retain the current whole-exam QC
baseline unchanged. The next information-changing arm should enlarge the
patient-disjoint supervised MLO source bank while holding SmolVLM and training
settings fixed; do not also change architecture. It must preserve exact train
fit, solve at least seven of eight newly frozen validation pairs, and preserve
all three existing challenge reversals before any audit integration.

## 2026-07-15 — Expanded orientation bank, paired rule, and DICOM label audit

Expanded the patient-disjoint SmolVLM orientation bank while holding model,
prompt, representation, one-token objective, LoRA policy, optimization, seed,
and challenge fixed. The expanded dataset has 59 train, eight validation, and
three challenge source pairs; manifest SHA-256
`a290674559bce34ab81203c5d411e43bde4a6bba56d03aeb77adb2eb4968c8a6`
and ordered PNG-bank SHA-256
`7e6433d1c1259b7edbf9e877fc2233473d2478bdc9b9315e8400bf3675a7c74c`.
The expanded adapter fit all 59 training pairs. Greedy scoring solved seven of
eight validation pairs and two of three challenge pairs. A frozen, parameter-
free paired rule then compared each image's A-minus-B logit margin with the
margin of its exact 180-degree partner. Its zero-threshold contrast solved all
eight development-validation pairs and all three challenge pairs. The paired
rule is part of the VLM inference mechanism; it is not an embedding model,
learned classifier, or fitted threshold.

Built a new source-disjoint holdout with 32 validation and three challenge
sources before scoring. Its original manifest SHA-256 was
`7f86b8d2d83dd54ec515e33937e8c8b10d0905f543ab576672a96c25d0881d05`.
The formal frozen result, SHA-256
`e9d34a9690d5e730d32098df001056a03161cb7e04c3ac53dc11a166aa74b048`,
failed at 30/32 paired validation sources and passed 3/3 challenge sources.
Inspection showed that both nominal failures had superior anatomy at the
bottom in the source PNGs. This exposed a dataset-builder error: every source
original had been hard-coded as UPRIGHT.

Added a PHI-safe DICOM audit in Prima commit `a684c31`. It resolves every source
through the durable archive/member locator, validates SOP identity, and reads
only headers. Per DICOM PatientOrientation, the second value is the positive
image-column direction from top to bottom: principal F means UPRIGHT and
principal H means INVERTED. The fresh audit covered all 35 evaluated sources,
found 32 UPRIGHT and three INVERTED, no unknowns, and independently identified
exactly the two nominal fresh failures as inverted originals. The expanded-bank
audit found 57 UPRIGHT and two INVERTED train originals, eight UPRIGHT
validation originals, and the already known two-upright/one-inverted challenge.
Thus two of 59 training pairs were also label-reversed.

Prima commit `a8e8cbd` changed the builder to require durable DICOM resolution
and derive every original label from PatientOrientation; metadata remains
ground truth only and the model still receives pixels only. Rebuilt the same
banks with no compatibility shim. The corrected expanded manifest SHA-256 is
`2c77069f04756bd93a10b2c5048acfbefd0520222e4a42a172547b298d45da2e`;
all 140 PNGs are byte-identical and exactly four train rows changed labels. The
corrected fresh manifest SHA-256 is
`14bdf8834fc9af08b0751c6e9f134bdc259439cd49c6144ccf6bef91f178f428`;
all 204 PNGs are byte-identical and four train plus four validation rows changed
labels.

Prima commit `603d311` added a deterministic label-only adjudication scorer. It
verified all 70 evaluated PNGs byte-identical and rescored the already frozen
predictions without inference or threshold fitting. The adjudicated artifact
SHA-256 is
`8bb44a090a78ba987a1af36a88cd0f81e67727927710e01160fa1c232b77befe`:
greedy validation is 63/64 labels and 31/32 pairs; paired validation is 64/64
labels and 32/32 pairs; paired challenge is 6/6 labels and 3/3 pairs. Preserve
the original formal failure and describe this only as independent post-hoc
ground-truth adjudication.

Froze one clean-label retraining arm at protocol SHA-256
`a2dd54b6304d6f8a7b0ec36d741082a7e3c4d8293ac760880119c5faaf7cad7b`.
Only the four corrected training-row labels change. The broker audited all
shared slots and atomically claimed priority slot `12890338_0` for request
`prima-mlo-orientation-smolvlm-dicom-labels-v1-2c5338f3db`; no raw Slurm
attachment or duplicate opportunistic job was used. The next check is the
single-run train-fit gate followed by zero-threshold paired scoring on the
unchanged corrected fresh holdout.

The clean-label training request completed normally and released its broker
claim. It reproduced the fixed fp16 warm-up NaN gradient at step 15, recovered,
and finished all 472 steps in 961.8 seconds with final validation loss 0.0522.
Adapter weights SHA-256:
`029390d38302623f35b55634f61c009692a971cf7105a3ac15f7596fd9449fc0`.
It fit all 118 train labels and all 59 train pairs. The older trainer's greedy
gate still reported failure at 15/16 development-validation labels and 5/6
challenge labels; this was expected to be non-decisive because the frozen arm
specified the separately scored paired-logit mechanism.

Froze the pairwise evaluation at SHA-256
`c7a0c5a59b0e52185585fa711537b2d9ffd05412fd83b263d5e51bf1a4f6b6f0`
and ran both datasets sequentially through broker request
`prima-mlo-orientation-smolvlm-dicom-labels-score-v1-b7321ebdc8`. The
corrected adapter passed every frozen paired gate: 118/118 train labels and
59/59 pairs; 16/16 development-validation labels and 8/8 pairs; 64/64 fresh
validation labels and 32/32 pairs; and 6/6 challenge labels and 3/3 pairs in
both evaluations. There were no ties and the minimum absolute paired contrast
was 3.953125. Greedy fresh scoring also improved from the old adjudicated
63/64 labels and 31/32 pairs to 64/64 labels and 32/32 pairs.

The checked summary is
`smolvlm_2b_dicom_label_result_summary.json`, SHA-256
`2f4482407158803a5bd0a6771eae3168b5589311d01cc2ab775f7fc5af53aa16`.
Interpretation: correcting the contradictory supervision preserves the perfect
paired mechanism and modestly improves the greedy readout. This passes the
orientation mechanism gate but does not promote the whole QC system. Retain
the tubing-veto plus exposure plus smooth-region baseline while testing
orientation as one additive MLO rejection signal across all 98 MLO views in
the already-adjudicated whole-exam audit. The operational falsifier is failure
to recover the inversion residual or any new unsafe selected slot/exam; a new
patient/exam-disjoint prospective audit remains required after development
integration.

Built a complete paired evaluation bank for all 98 MLO views in the 202-view
whole-exam audit. DICOM ground truth contains 97 UPRIGHT and one INVERTED
original; all six protected challenge PNGs are byte-identical to the prior
bank. The audit-bank manifest SHA-256 is
`0b26720826246f0053a7944dea03aa174ce68c784ba78d70e55807cd60cbb66a`
and ordered PNG-bank SHA-256 is
`bfc889c48e4050c1c1dcea528d3f6dcc6856008d398ace972612edcac393b421`.
Prima commit `ec1da6c` added the generic audit-bank builder, `audit` split
support, and a converter from paired orientation logits to a complete view-QC
run; it was pushed only to the Prima feature branch.

Froze the sign-only full-audit protocol at SHA-256
`ddb1384bf2705a96c0845ed53efa4c2c27792d5584b90b839c02c83af3b03955`.
Broker request `prima-mlo-orientation-smolvlm-whole-audit-v1-fa691c49cb`
completed and released its priority claim. The paired result SHA-256 is
`b0d47984d95782e1ba9d38926d07adc2fee70f227bc5152f63a270f5bd262843`.
It scored 194/196 labels and 97/98 pairs with no ties. On original audit views,
it correctly rejected the true inversion at review order 177 but also rejected
one DICOM-upright, adjudicated-usable MLO at order 10. The original-view score
was therefore 97/98 and the strict zero-new-false-positive component gate
failed. Per the frozen stop rule, did not combine this failed component and
retained the whole-exam baseline unchanged.

Visual anatomy, PatientOrientation column `F`, and the adjudicated human label
all independently confirm order 10 is upright/usable. Its paired contrast is
only -1.6484375, versus -33.390625 for the true inversion. The source-disjoint
fresh holdout, recorded before full-audit inference, had no errors and minimum
absolute contrast 3.953125. The checked failure summary SHA-256 is
`14414be5be23aa8aa19874a2ca89b715db2b3cbe6864ceccf6e9602bc0846e95`.
This falsifies only treating every sign-negative contrast as a high-confidence
rejection; it does not falsify the VLM orientation representation. The next
single development arm is a confidence-aware abstention policy whose floor is
fixed at the pre-audit holdout minimum 3.953125: reject only when contrast is at
most -3.953125. Keep model, adapter, images, and logits fixed, then require the
incremental and existing whole-exam operational gates. This arm is motivated
post hoc by the audit and cannot substitute for new prospective validation.

Prima commit `2111eea` added an explicit required minimum-inversion contrast to
the orientation-to-view-QC converter and was pushed only to Prima. Froze the
confidence-aware arm before applying it at protocol SHA-256
`14c80d507172b5f1f2bbbb40cd090300f8144a8f1932d81d4d788241a9ea56ad`.
The action floor is exactly 3.953125, the weakest correct paired decision among
the 70 pre-audit fresh-validation/challenge rows. Model, adapter, images,
paired transform, and all 196 full-audit logits remained fixed.

The action policy saw two sign-negative original MLOs, abstained on weak order
10, and added exactly one high-confidence rejection at order 177. Recombined
the retained visual run, deterministic eligibility, and fallback selection
from scratch, then evaluated the adjudicated 202 views, 128 exact slots, and 32
exams under the existing gate. The incremental arm passed: it recovered the
one baseline false negative at order 177, introduced zero false positives,
reduced unsafe selected slots from one to zero, left false exhaustion at one,
and reduced unsafe accepted exams from one to zero.

The combined candidate passes the existing whole-exam gate with TP 94, FN 0,
FP 1, TN 107, sensitivity 1.0, specificity 0.99074, zero unsafe selected slots,
one false-exhausted slot, and zero unsafe accepted exams. The remaining false
positive/false-rejected exam is unchanged from the retained baseline. The
checked result summary SHA-256 is
`fb56cb25230bf011fcb5022fe65622156243b2fe03f8c0890759853160f6da63`;
the exact replay README SHA-256 is
`fdcfbe2330faac313478d847f211a0327a7692c2d280b06deacd07bdf06798db`.

Interpretation: this is the first complete passing whole-exam development
system, but the confidence-policy idea was selected after seeing the audit
error. Retain it as the best observed post-selection development candidate,
preserve the prior baseline, and make no prospective claim. The next required
step is a new patient/exam-disjoint whole-exam audit whose human labels remain
blinded to model outputs and whose gate is frozen before candidate inference.

## 2026-07-15 — Prospective whole-exam candidate frozen and inferred

Registered the exact post-selection development winner before prospective
inference at
`qc_redo/auto_qc_validation/mirai_input_prospective_whole_exam_v1/candidate_spec.json`
(SHA-256
`d706cba4b1261ce5bd362519592a3cfc4f355c8f32e24cf6713384a29767c164`).
Prima commit `4676614` makes the whole-exam audit builder require this hashed
candidate specification. Built a new patient/exam/view-disjoint panel of 32
patients, 32 exams, 128 exact L/R CC/MLO slots, and 174 candidate views. It has
all four required slots per exam, complete same-slot rank sequences, 17
deterministically nonstandard DICOM candidates, and no overlap with any prior
vertical-line, spot-view, binary-pilot, or whole-exam source panel. Blinded
manifest SHA-256:
`03b2946ac816820dcc0ecfda0ffd7322cdc40040b972f422232c758a700bd3ac`.
Human state and append-only event log were empty at freeze.

Because all 88 natural MLO originals in the whole panel were DICOM-upright,
registered a separate matched natural-orientation holdout before inference:
32 DICOM-INVERTED and 32 DICOM-UPRIGHT patient/exam-disjoint MLO sources,
matched exactly on laterality, manufacturer, model, and implant status and
verified against each raw durable DICOM. Broker inference plus frozen action
scoring passed 64/64: TP 32, FN 0, FP 0, TN 32, with all 64 rotation pairs
exact. Action-score SHA-256:
`5786137314b638dc5203c90ade716c03d1d9dfde7117e8aef4c2556fecd33c35`.
The weakest absolute natural contrast was about 28.59, more than seven times
the frozen 3.953125 action floor. This is mechanistically consistent with a
strong inversion signal, but the matched holdout is one Hologic model and one
laterality; it supports this action rule here, not vendor-wide orientation
universality. A future source family with substantially weaker paired
contrast or a natural false positive would falsify that broader transfer.

Ran the whole panel through all nine frozen Qwen components and the paired
SmolVLM orientation layer. Every Qwen run has exact 174-view coverage, 174
debug records, the frozen revision, and byte-identical stripped prompt text.
The whole-panel orientation run also has exact 88-pair coverage and correctly
abstains everywhere under the frozen floor. All GPU work went through the
shared H200 broker with `--project prima`; no raw `srun --jobid` step was
attached. Six initial Qwen requests failed before inference because the long
runtime scratch path plus vLLM UUID exceeded the Unix-socket path limit.
Preserved those failures, changed only runtime scratch to short
`/scratch/annawoodard/tmp/pq-<port>` paths, and completed all retries. This
fix changed neither model nor scientific inputs.

Materialized the tubing observer's exact nine high-confidence candidates
outside the blinded browser directory using the new generic provenance owner
`qc/build_view_auto_qc_veto_manifest.py`. Prima commit `2a6fffd` was pushed to
the Prima feature branch only; `local-llms` was neither committed nor pushed.
The first veto broker admission asked for eight CPUs and was rejected before
launch because no holder had that CPU headroom. Retried the identical frozen
command with four CPUs, matching `OMP_NUM_THREADS=4`; request
`prima-prospective-tubing-mimic-veto-cpu4-00c591d27d` completed normally. The
observer vetoed five obvious mimics and retained four candidate tubing calls.

Froze the ten exact component hashes before assembly at
`system_assembly_protocol.json` (SHA-256
`36c416034206c75d697698d006e758bb6c71a3420432e9c4bcdc7c174df0f079`).
Rebuilt the visual logical OR, deterministic-DICOM OR, and immutable-rank
same-slot fallback from scratch. The hidden system has 174 views, 128 slots,
and 32 exams; structural checks prove exact component-union equivalence,
deterministic-OR-visual equivalence, contiguous candidate ranks, and no
system-positive selected candidate. Combined-system SHA-256:
`312bacd9c8dcde5ed0fb8ac11049c531faf89f05af34f7c88706634e5648a407`.
The raw model disposition counts are deliberately not interpreted as accuracy:
the only falsifier is the frozen human-reference gate, and looking at aggregate
prediction prevalence cannot establish sensitivity, specificity, or safe
fallback behavior.

Activated the empty blinded reviewer in persistent tmux session
`prima-qc-gallery` on the stable loopback port 8767. API verification showed
schema 3 and 0/174 labels. `qc_redo/CURRENT_VIEW_QC.md` and the panel README
record the fixed tunnel and exact restart command. The evaluator will fail
closed until all 174 binary decisions exist and every low-confidence flag is
adjudicated. Then freeze the state hash and score the registered candidate
exactly once against sensitivity >= 0.95, specificity >= 0.90, zero unsafe
selected slots, at most one false-exhausted slot, and zero unsafe accepted
exams. Until then, retain the development baseline and make no prospective
performance claim.

## 2026-07-17 — Prospective whole-exam reference frozen and candidate scored once

The blinded reviewer completed all 174/174 binary labels with zero unresolved
low-confidence flags. Append-only event replay exactly reproduced the state.
Stopped the gallery, verified that port 8767 was released, and created the
read-only canonical archive at
`qc_redo/annotation_archive/mirai_input_whole_exam_prospective_v1_reference`.
The archived manifest/state/events SHA-256 values are respectively
`03b2946ac816820dcc0ecfda0ffd7322cdc40040b972f422232c758a700bd3ac`,
`824104051200dc55f840948e5ef9a104088276a3d172569ef3d970a5bff3fe6f`,
and `20ee70799a13417d4c0cbfa93953556e9bff7fac9ae7abea179eafce295e7cb9`.
The archive contains 362 independently verified files and is read-only.

Registered an exact one-shot score execution protocol before opening the
reference, then evaluated the unchanged hidden system once. It scored TP 70,
FN 0, FP 4, TN 100: sensitivity 1.0, specificity 0.96154, zero unsafe selected
slots, four false-exhausted slots, and zero unsafe accepted exams. It therefore
failed only the registered maximum of one false-exhausted slot. This is a real
prospective failure even though its view-level sensitivity and specificity
passed. Gate SHA-256:
`e4d6a278a7ad16408b04d493bdf6771ca51d0daf538979ee1b61950636ee799b`.

All four false positives came from the broad gross-device observer and were
ordinary postoperative external wire/scar markers; one was also called by the
vertical-seam observer. The gross observer's rationales repeatedly described a
"hooked wire traversing breast tissue" despite its frozen prompt explicitly
listing ordinary wire skin/scar/nipple markers as negative. Removing that
component post hoc would reduce false exhaustion to one but create one unsafe
miss: it uniquely detects an unmistakable pacemaker view. Therefore this
falsifies the gross-device observer's specificity on ordinary wire markers,
not the unchanged seam, film, implant, special-view, tubing-veto, exposure,
smooth-region, orientation, deterministic-DICOM, fallback, or open-VLM
families. Retain the immutable registered result and do not promote it.

Nearest reference: the preceding 120-view development panel already contained
three identical wire-marker false positives, but its view-level gate tolerated
them. The new exact-slot fallback gate correctly exposed their operational
cost. Mechanistic explanation: a single broad observer is sensitive to both
true gross hardware and long radiopaque external markers; asking a separate
conservative mimic question may recover specificity without discarding the
true device signal. Falsifier: a candidate-only mimic observer that misses a
known marker or vetoes a true gross device.

## 2026-07-17 — Gross-device external-marker veto mechanism check

Froze one zero-shot candidate-only veto delta at protocol SHA-256
`78bfbf95d3ec9015759c5b31f70c30064d9381db7641882f6686225aed55953d`.
Everything except the new target "obvious allowed external wire marker with no
gross device" remained unchanged. The exact input was all 20 high-confidence
gross-device candidates from the old development and completed prospective
panels. Both inference tasks used the pinned `prima-vllm` environment through
the shared H200 broker with `--project prima`; requests
`prima-gross-device-marker-veto-development-9e1e9b0ee1` and
`prima-gross-device-marker-veto-prospective-development-3b8b4acb25` completed
with exact 7/7 and 13/13 coverage and clean broker claim release.

The frozen primary target-specific rule failed: the veto removed all 7/7
registered marker cases but also reviews 19 and 59, which the protocol had
listed among 13 gross-device positives because they were overall QC failures.
Same-exam inspection shows that review 59 is a superficial postoperative scar
wire present around surgical clips in both projections; review 19 is an
external U-shaped line on an independently nonstandard special view. Thus the
protocol conflated whole-system exclusion with target-specific device truth.
Preserve that failed primary result; do not edit the protocol or claim a
perfect target-specific score. The second broker log emitted an EngineDeadError
only during intentional zero-timeout server shutdown, after all 13 atomic
records were persisted; the command exited zero and the slot was released.

Operationally, replacing only the gross-device component improved the old
development panel from TP 53/FN 1/FP 3/TN 63 to TP 53/FN 1/FP 0/TN 66. On the
completed prospective panel treated strictly as development data, it improved
the whole system from TP 70/FN 0/FP 4/TN 100 and four false exhaustions to TP
70/FN 0/FP 1/TN 103 and one false exhaustion, with zero unsafe selected slots
and zero unsafe accepted exams. The unchanged seam observer causes the one
remaining error; the registered gate already allows it, so do not broaden the
change or tune seam here. Posthoc gate SHA-256:
`2ef757aa36cd5e311b9cefdc1e863e48ac9702d3ccfaaa8847dbc666b539b1ef`.

Sense check: the direction and localization match the mechanism—only the known
wire-marker errors changed, all gross hardware remains covered by the system,
and the two disputed vetoes remain independently excluded by frozen layers.
The strongest counterexample is a future true localization wire that the veto
mistakes for a scar marker and that no other layer detects. Therefore the
posthoc rescue supports freezing a successor hypothesis but cannot promote it.
Retain the original registered system as the immutable comparison. Next require
a brand-new patient/exam-disjoint whole-exam audit with the same point gates,
model outputs hidden from human review, and enriched representation of both
ordinary scar-wire markers and true gross devices.

## 2026-07-17 — Prospective whole-exam v2 audit frozen and hidden system assembled

Froze the one-delta successor before sampling or inference at candidate-spec
SHA-256
`2fb557fddb06c2176896cf9462c43954ec6380a341df89056d7e2cdf62d5efe2`.
The only change from the failed v1 candidate is the high-confidence
external-wire-marker veto after the unchanged sensitive gross-device observer.
All other prompts, models, thresholds, deterministic eligibility rules,
orientation action, and exact-slot fallback policy are inherited unchanged.

Built a new 32-patient/32-exam, 128-slot, 213-candidate whole-exam panel with
seed 20260717. It has the registered 8 seam-challenge, 2 film, 8 implant, 7
multi-candidate-control, and 7 single-candidate-control exams. Patient, exam,
and view overlap is zero against every prior development, prospective, and
orientation panel supplied to the builder. Every slot has contiguous immutable
ranks and exactly one rank-1 source; all original DICOM archive/member lineage
and hashes are present. Manifest, source-manifest, group-manifest, and DICOM
eligibility-audit SHA-256 values are respectively
`5d0a0e837202cd3671b257cd1cc66362b5fcf716e957cd306c48c506208b1e2f`,
`f496e50461b7cf624d74701a5670925a52fb604b9686b6732469949686caa29d`,
`2bce86cde166ef45634523bcb5e814ae586927547c324bc003894f56cdadc9f4`,
and `d59c2e2f3e5d1d90ce220d8064354606bd3aa2b70571b4765e6cf6c977c58864`.
The browser manifest exposes no patient, exam, SOP, or source identifier.

Froze Qwen and orientation execution protocols before inference while the
reference state was empty. Their SHA-256 values are
`a16c474046a00e9c3c10de784c959382332d14120ad583190fc156c65ad5e634`
and `14b75c520bfb09edb2435a1deca4eb54179a7cc541775fc1efba64f659f04794`.
All GPU work used the shared H200 broker with `--project prima`; no raw
`srun --jobid` step was attached. Two initial queued launcher processes for
tubing and exposure failed before GPU admission when the broker's
`squeue --json` query timed out. Neither produced inference output. Retried the
identical frozen commands through the broker; both completed normally. Every
primary component, both candidate-only vetoes, and the paired SmolVLM
orientation run have exact expected coverage.

Dependency resolution selected 15 high-confidence gross-device candidates;
the frozen external-marker veto acted on 13 and retained 2. It selected 11
high-confidence tubing candidates; the unchanged non-tubing-mimic veto acted
on 3 and retained 8. The natural raw-DICOM orientation bank has 102 exact MLO
pairs and the frozen action rejected zero original sources. These are hidden
model-disposition counts, not accuracy measurements, and no per-view output was
opened or compared with a reference.

Froze the ten exact component hashes in the system-assembly protocol at
SHA-256
`2012c550439ee3eb278788695765a6a23a7d71cda8ce667720c686fd30a205e8`
while the human reference still had zero labels. Structural validation proves
exact 213-view component coverage, exact ten-way visual logical-OR behavior,
exact deterministic-DICOM OR visual behavior, 128 same-slot rank-preserving
fallback decisions, and no system-positive selected candidate. The combined
visual run, combined system run, and fallback-decision SHA-256 values are
`cb735610545df5f764e4e2b0a31a53892e4cdd6ea2c763448b18789ebe0d56cc`,
`45a2750fb2cb3b83ef714c61abedefd3452155bb86eab0a79f00639e671b978f`,
and `6144580564817a12468c7c1c486ef8fdf0be406208b0e6c1cf8e3e0ac23fb8ed`.
The hidden system marks 72 views visually and 92 after deterministic DICOM
eligibility; its fallback dispositions are 76 original retained, 5 alternate
retained, and 47 slots exhausted. Those prevalences have no truth meaning.

Started and API-verified the blinded gallery in tmux session
`prima-qc-gallery` on the stable loopback port 8767. It has 213 items, schema 3,
the full 4,452-character rubric, and zero initial labels. Model and fallback
outputs remain outside the browser directory. Do not score, inspect errors, or
tune this candidate until the complete human state is archived. The only
prospective falsifier remains the registered gate: sensitivity at least 0.95,
specificity at least 0.90, zero unsafe selected slots, at most one false
exhaustion, and zero unsafe accepted exams.

## 2026-07-24 — Prospective v2 reference scored and nonstandard-angle source traced

### Question

Did the frozen external-marker-veto successor pass its prospective whole-exam
gate, and can the nonstandard or "wrong-angle" views seen during review be
removed programmatically?

### Action

Verified completion of all 213 blinded labels with zero unresolved
low-confidence flags and exact replay of all 214 append-only events. Created
and verified the read-only canonical archive at
`qc_redo/annotation_archive/mirai_input_whole_exam_prospective_v2_reference`,
then stopped the gallery and confirmed port 8767 was released. Froze
`score_execution_protocol.json` before opening the reference/model comparison
and ran the registered evaluator exactly once.

After scoring, joined only hashed view identity across the archived labels,
raw-DICOM eligibility audit, candidate ranks, safe acquisition tags, and
component decisions. Compared `ViewPosition`, explicit mammography view
modifiers, `PartialView`, `PositionerPrimaryAngle`, field-of-view rotation,
patient orientation, series/protocol-derived categories, and the declared
L/R CC/MLO slot. No patient, exam, accession, SOP, or source path was printed.

### Evidence

- Archived manifest/state/events SHA-256:
  `5d0a0e837202cd3671b257cd1cc66362b5fcf716e957cd306c48c506208b1e2f`,
  `22a2f52abf8e6219a5bbc5ad7286141ebc69de10a21b50942cc6ec1ed4f5569c`,
  and
  `0947f3621097f2b1324b2cc21f56b120771075af92c3e854ba642775a25a9cba`.
- Score inputs were rehashed against the frozen execution protocol immediately
  before evaluation. Evaluation artifacts:
  `qc_redo/auto_qc_validation/mirai_input_prospective_whole_exam_v2/evaluation/`.
- Registered result: TP 81, FN 1, FP 11, TN 120; sensitivity 0.98780,
  specificity 0.91603; one unsafe selected slot, five false-exhausted slots,
  and zero unsafe accepted exams.
- The source candidate table was written on 2026-07-11. Commit `717aa54`,
  which added strict standard-view eligibility, landed on 2026-07-12. Thus
  this panel intentionally re-audited candidates from a cache that predates the
  current source rule.
- The raw-DICOM audit identified 31 deterministic exclusions: 20
  implant-displaced, seven magnification, two spot-compression plus
  implant-displaced, one magnification plus implant-displaced, and one partial
  spot-compression view. Sixteen were also human "Do not use" labels; the
  deterministic union correctly excludes all 31 even when the special
  acquisition is not visually obvious.
- Every raw `ViewPosition` in this panel was CC or MLO. `PositionerPrimaryAngle`
  was present for 120/213 views. Human-use and human-fail distributions
  overlapped throughout the ordinary CC and MLO ranges; the six MLO sign
  exceptions were all human-use views. No safe numeric-angle threshold was
  supported. Series/protocol-derived unexpected strings were also more common
  among human-use than human-fail views.
- Among 182 DICOM-eligible views, the human marked 51 as visual failures. The
  frozen VLM system detected 50. The single miss was selected review 132, an
  L-MLO source with a DICOM implant flag and otherwise ordinary standard-view
  tags; no frozen component fired.

### Result

The v2 successor failed its registered operational gate despite passing both
view-level point thresholds. The deterministic programmatic filter already
covers explicit non-CC/MLO positions and coded special-purpose acquisitions;
the reason these appeared in the browser was the deliberately comprehensive
audit design plus a source cache older than that rule, not absence of the
filter. A fresh preprocessing rebuild under current code will remove those
sources from the eligible candidate inventory and persist them in the
diagnostic-exclusion pool.

For views whose pixels look like the wrong projection while the DICOM says
ordinary CC/MLO and supplies no modifier, the available angle tags do not
separate failures from usable views. A hard tag threshold would reject known
usable views. Those cases require a pixel-level projection/nonstandard-view
observer or focused adjudication; they should not motivate a brittle numeric
angle rule.

### Conclusion

Explicitly tagged special views are programmatically filterable and already
covered; the stale candidate cache and human-panel composition made them
visible. The remaining visually wrong-looking projection problem is not
solved by the current angle tags. Inference confidence: high for the coded
special-view conclusion, medium for the visually miscategorized subset because
the binary human state does not record failure reasons.

### Falsifier

A rebuilt candidate inventory that still contains a source rejected by
`mirai_source_eligibility_reasons`, or a prospectively labeled
projection-mismatch set on which a predeclared DICOM angle rule separates all
failures without rejecting usable views, would falsify the corresponding
conclusion.

### Decision Impact

Do not promote the v2 external-marker-veto successor. Retain the unchanged
modular prompt/VLM family and its best prior baseline while designing the next
single-delta candidate. Before final dataset QC, rebuild the SoT candidate
inventory under the current eligibility policy. Future residual visual-review
panels should omit deterministic DICOM exclusions from the human workload and
audit those exclusions separately.

## 2026-07-24 — Current-policy SoT promoted and residual-review contract validated

### Question

Can explicitly tagged wrong-angle and special-purpose views be removed
programmatically from the authoritative Mirai candidate inventory, while
retaining durable DICOM lineage and avoiding unnecessary human annotation?

### Action

Changed the prospective whole-exam audit builder so
`candidate_manifest.parquet` retains every system candidate, while the browser
`manifest.parquet` contains only DICOM-eligible residual views. The group and
source manifests retain all system candidates and nullable human review order.
Changed the evaluator to require human labels exactly for that eligible
residual and to combine them with deterministic DICOM exclusions at scoring
time. Rebuilt the completed v2 panel with this contract and replayed a state
containing only the 182 residual labels.

Added fixed `--today` propagation to the deferred sharded-preprocessing merge
so Mirai label emission is reproducible. Rebuilt from the unchanged explicit
10,722-source production allowlist into isolated staging with 16 CPU shards.
All shard artifact contracts completed, but Slurm left 11 allocations in a
stale running state after clean script completion and intermittently failed
controller queries. After verifying every required shard output, canceled the
stale allocations and original merge and ran recovery merge job `13180339`
behind an `afterany` artifact barrier.

Four source archives contained two complete exams apiece, exposing an old
one-pass cache bug: the writer passed more than one four-view exam to the
single-exam Zarr writer. Updated `pipelines/preprocess.py` to group selected
rows by `(patient_id, exam_id)`, then reprocessed only the four affected
sources in jobs `13180395` and `13180396`. Integrated the eight recovered exam
caches and regenerated the full Mirai CSV with fixed date `2026-07-24`.

Validated staging, created checksumed rollback material, rewrote manifest
paths, promoted the rebuild, and reran the same contract against production.
Full run provenance is
`/scratch/annawoodard/prima_current_policy_rebuild/README.md`; staged and final
reports are `staged_validation_report.json` and
`final_validation_report.json` in that directory.

### Evidence

- Residual-contract replay reproduced the frozen v2 view confusion,
  per-stratum outcomes, slot/exam outcomes, and gate exactly from 182 human
  labels plus 31 deterministic DICOM exclusions. The old 213-row candidate
  manifest remained byte-identical; the new residual browser manifest has 182
  rows.
- Final production SoT has 42,804 selected views, 55,356 eligible fallback
  candidates, 18,934 deterministic exclusions, and 10,701 complete exams.
  The manifest has exactly 42,804 rows and 10,701 Zarr groups; the Mirai CSV
  also has 42,804 rows.
- Every selected exam has exactly one L/R CC/MLO quad. Candidate ranks are
  contiguous, rank one equals the selected source, candidate and exclusion
  source sets are disjoint, and manifest slots equal selected SoT slots.
- Relative to 59,617 prior candidate sources, 55,356 remain candidates and
  4,251 moved to deterministic exclusions. Ten eligible remnants belong to
  five exams rendered incomplete by those exclusions; all ten original
  headers were readable and verified as ordinary CC/MLO, so this is an
  exam-completeness disposition rather than unexplained source loss.
- The new exclusion pool adds 14,683 diagnostic sources that were never in the
  prior canonical candidate table. Its major explicit categories include ML
  projections, spot compression, magnification, implant-displaced views, and
  partial views.
- The frozen v2 sample cross-check is exact: all 31 DICOM-ineligible views are
  exclusions and none are candidates; all 182 eligible views remain candidates
  and none are exclusions.
- `ops/audit_dicom_lineage.py` passed structural lineage for all 42,804
  selected rows and verified SOP UID plus SHA-256 for a deterministic
  128-source sample. The final validator also opened 100 deterministic Zarr
  groups and verified all four arrays and declared shapes.
- Production paths are `/gpfs/data/huo-lab/Image/ChiMEC/MG/sot` and
  `/gpfs/data/huo-lab/Image/ChiMEC/MG/out`. Rollback paths are
  `/gpfs/data/huo-lab/Image/ChiMEC/MG/sot_pre_current_policy` and the checksumed
  scratch `pre_promotion_backup`.

### Result

The authoritative candidate inventory now enforces the current raw-DICOM
eligibility policy. Explicitly coded noncanonical and special-purpose views no
longer enter Mirai fallback candidates, and every rejected diagnostic source
retains archive/member lineage plus an exclusion reason. Human prospective
review can now focus on the residual pixel-visible QC problem without asking
the annotator to rediscover deterministic metadata exclusions.

### Conclusion

The programmatic part of the wrong-angle problem is resolved in production.
This does not solve visually miscategorized ordinary CC/MLO views whose DICOM
headers carry no disqualifying signal; those remain appropriate targets for
the prompt/VLM QC observer. The residual-only annotation contract keeps that
distinction explicit.

### Falsifier

Any production candidate that fails `mirai_source_eligibility_reasons`, any
selected view without a resolvable and hash-valid source, any manifest slot
without a readable four-array exam cache, or any future residual panel that
requires a human label for a deterministic DICOM exclusion would falsify this
completion claim.

### Decision Impact

Use `candidate_manifest.parquet` for model inference and the residual
`manifest.parquet` for blinded human review in future whole-exam audits. Build
the next prompt-only QC experiment from the promoted production candidate
inventory; do not reuse the stale pre-policy panel as the dataset candidate
source.

## 2026-07-24 — Frozen v1 baseline re-audit on the promoted residual inventory

### Question

Does the unchanged frozen v1 prompt/VLM system meet the registered whole-exam
gate after deterministic source eligibility has been promoted into the
production candidate inventory?

### Nearest Reference

The retained comparison is the immutable v1 prospective candidate, not its v2
successor. V1 scored TP 70, FN 0, FP 4, TN 100 with zero unsafe selected slots,
four false-exhausted slots, and zero unsafe accepted exams; it failed only the
maximum-false-exhaustion gate. The one-delta v2 external-marker-veto successor
regressed to TP 81, FN 1, FP 11, TN 120, one unsafe selected slot, and five
false-exhausted slots. This experiment therefore changes no model, prompt,
example bank, threshold, component, orientation floor, or fallback rule. It
tests the unchanged v1 baseline under the corrected production input contract.

### Action

Froze the one-arm registration at
`qc_redo/auto_qc_validation/mirai_input_production_residual_v1/candidate_spec.json`
(SHA-256
`4612b99dffd8ea335f5c7a4e0b6e04381a67a7e6f6b3537550897f9c20fe1f35`).
Built a restricted prior-reference exclusion universe covering every
source-linked development/prospective panel and every archived annotation
reference. Its 1,586 views map to 754 patients and 982 exams; archive coverage
is complete and the table SHA-256 is
`07ccec76761a025816352e6387997e762619270a7e6704287618e63d1d951dba`.

Prepared a current-policy render campaign from the exact promoted candidate
inventory. All 55,356 current view hashes were already present in the prior
source-verified campaign, so the new campaign uses exact hard-linked PNGs.
Sixteen parallel shard passes and the final exact inventory validator both
passed 55,356/55,356 with zero failures.

Built the new patient/exam/view-disjoint audit at
`qc_redo/review_batches/mirai_input_production_residual_v1`. The registered
32-exam strata are six seam-fallback challenges, two film challenges, one
implant challenge, 11 multi-candidate controls, and 12 single-candidate
controls. Only one implant exam remains after excluding every prior reference
patient; the design records that limitation rather than reusing a patient.
The panel has 32 patients, 32 exams, 128 exact L/R CC/MLO slots, and 181
candidates. Candidate and residual-review manifests are byte-identical because
all 181 raw sources pass the current DICOM eligibility rule.

Derived the frozen 181-row multiscale input bank and 94 exact MLO
original/180-degree pairs. The MLO originals contain 93 DICOM-upright and one
DICOM-inverted source. Independently re-materialized and verified SOP identity
plus full DICOM SHA-256 for all 181 panel sources; durable result SHA-256:
`e8fa20ea9d57fb6f22b7985f8bdf9ad650156ed45418f4e25b479af01512c9e7`.

Froze Qwen and orientation protocols while the human state contained zero
labels. Their SHA-256 values are
`e740870f1cbaeb0195e97b7da07588ecbf92ae0188ba6fe1b861ea356ccdecdc`
and
`d5138c190292ce61e2dba662217a5256a0a47fa1a23b37cb6c80f7e483ae2c9e`.
The shared H200 broker reported no persistent holder, so no broker task was
queued. Submitted the nine checkpointed Qwen components and one atomic,
rerunnable orientation evaluation exactly once through Prima's native
opportunistic launchers on `zhoulabq,catherineq,siweiq` with
`qos=opportunistic`. Qwen job IDs are 13184009, 13184010, 13184011, 13184013,
13184014, 13184015, 13184016, 13184017, and 13184018; orientation is 13184028.
All ten were running on `zhoulabq` H200s at the first health check. No raw
`srun --jobid` step was attached and no task was duplicated between the broker
and opportunistic queue.

The first Qwen submission attempt failed before queueing because direct
invocation of the pinned environment's Python did not put its `vllm`
executable on `PATH`. Changed only the launcher plumbing to
`micromamba run -p prima-vllm`, verified vLLM 0.24.0 and OpenAI 2.45.0, and
then submitted the unchanged frozen commands successfully.

Started the empty blinded reviewer in tmux session `prima-qc-gallery` on
loopback port 8767. API checks returned 181 items, zero labels, the full
4,452-character rubric, and the expected independent low-confidence workflow.

### Evidence

- Candidate/review manifest SHA-256:
  `0a09579e13cb6a367b479327256d5ad6b0cdc37fa74dcdc5d208e97358d4a9b9`.
- Source/group/eligibility SHA-256:
  `d8b9db3e719a3aa84e0493bd3f2738ea622b30fd14a3f8febe7b769a9ba5db90`,
  `b0970cc45c4a5b458e907ca256f0c4e5a916c6601d7f2dc5fe57aee1cfc4edf9`,
  and
  `f29672ae706fe1276dc451d9c8afbb5311e04b63a2a4a97800b3676e2084771f`.
- Prior patient, exam, and exact-view overlap are all zero.
- Every exam has four exact slots; all candidate ranks are contiguous and
  rank one is the unique original selection.
- All 181 browser images validate at the registered pixel ceiling.
- All 181 raw headers are eligible; deterministic residual exclusions are
  zero.
- Empty state/events SHA-256:
  `2e52c643c4a5ab45f10a8cf147420523030a32a568251d69a5b413d247423281`
  and
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Multiscale manifest and ordered bank SHA-256:
  `e777d855e1b5b25a6dc72a9e4e21df3eef993ef271d9a2024bf99339f988c737`
  and
  `978ea77cb649a698cf7532a120d46e3ecbd078bfcd72c9aec747f3479e4e4fb8`.
- Orientation manifest and ordered bank SHA-256:
  `8cb51f2e6a2ad9fe20ab7da88d4c317ced61b194fa0a50bfaafc1de9ac40e916`
  and
  `1a234c61dfdd469f9152c3a353b231532fecd982bb78d612dbc7651582a92331`.

### Result

The fresh residual audit and both frozen execution inputs are valid, disjoint,
and blinded. GPU inference is active and the human reviewer is ready. No
accuracy, prediction prevalence, or error interpretation is permitted until
the complete human reference is frozen.

### Conclusion

This is the smallest decision-changing check: one unchanged baseline on the
corrected production inventory. It directly distinguishes whether the prior
operational failure was dominated by stale deterministic-special candidates
or remains a visual false-exhaustion problem. It does not promote v1 in
advance and does not rehabilitate the regressed v2 successor.

### Falsifier

Any prior-patient overlap, any candidate failing current raw-DICOM eligibility,
any incomplete same-slot rank sequence, any changed v1 model artifact, or a
one-shot score below sensitivity 0.95, specificity 0.90, zero unsafe selected
slots, at most one false-exhausted slot, and zero unsafe accepted exams
falsifies the corresponding claim.

### Next Check

Monitor the ten opportunistic jobs by campaign family. Require exact 181-view
coverage for every Qwen component and exact 188-row/94-pair orientation
coverage. Resolve the tubing-veto dependency without opening the human
reference, freeze system-assembly hashes, and then score exactly once only
after all 181 binary labels and every low-confidence adjudication are complete.

### Completion update

All frozen inference and assembly work completed without opening the human
reference. The seven initially healthy Qwen component jobs, orientation job
`13184028`, and tubing-veto job `13184917` all finished `COMPLETED/0`.
The initial seam and film jobs loaded all model weights in about 839 seconds
but exceeded their 900-second server-health deadline while profiling and
failed before scoring any view. Increased only the launcher startup allowance
to 1,800 seconds and resubmitted the unchanged checkpointed tasks as
`13185584` and `13185592`. Both retries finished `COMPLETED/0`; the warm node
page cache reduced weight loading to about 20 seconds. No prompt, model,
example, threshold, representation, or panel input changed.

The tubing candidate and exact candidate-only mimic-veto runs were gated with
the unchanged v1 rule. Every final component run covers the exact same 181
view IDs and image paths as the frozen candidate manifest. The orientation
result covers the registered 188 rows and 94 MLO pairs, and its converted
action run covers all 181 candidates.

Froze system assembly with zero human labels at
`qc_redo/auto_qc_validation/mirai_input_production_residual_v1/system_assembly_protocol.json`
(SHA-256
`a17478b14a1527b0c6a753ca27a145ae92eaf0d3ade221ac40fd3c22fb769c23`).
The ten frozen component SHA-256 values are:

- vertical seam:
  `813216cb836cc85e211742c4260f8b2a946d4eb82b641f74c22f6cc1ded8e877`
- digitized hard-copy film:
  `003e53bdaa9ee22cccdf879eb95a53083c4f2731df9c3cc5a69e2c4db02062b6`
- breast implant:
  `66704baec236e8955fe2a98272a6fbd9ee19d905628cf3b0fba8f525b0b9db42`
- spot compression or magnification:
  `d8cafd9561129c46d87dcf8d3e022eb830847cc72ef4cd0444dd0f683fc13fbe`
- spot compression:
  `41f4581cb779009f36702d7f6386e57b5431c0e9a6c01e3494a74ea14e61b60e`
- gross implanted or procedural device:
  `ecb54692143e279a6cf42916b1d115d1db31fe8ae89249066d612919c2cf72e3`
- gated catheter or procedural tubing:
  `a30b9ecace10aff76381bf405d07b098a7b31eb2bc9a9986fe24366fdb76956f`
- exposure or processing failure:
  `054a87bdfe1baec70e168d1fd47ebdf6e0fe2948428654bb9580cc03d14ceb91`
- smooth homogeneous region:
  `ded79be4eaa1e604ccd9445268aa460f7fa1190371b0a86b102d7a6dba91869d`
- confident paired SmolVLM MLO orientation:
  `d08d5f9678509d66a0dc9a6e98937aed21ff853134ddd1223c19066107e8950a`

Assembled the system exactly once. Output SHA-256 values are:

- visual logical-OR run:
  `9802e259df9ab2c7cdde6840c7a76504b8849a1d7102f33bb649f7986d9daaa8`
- deterministic-or-visual system run:
  `9a63afa57c6340ae6ccba3a9fb65971d5a4ff31cf9164a7632d36b9518ab00e0`
- rank-preserving fallback decisions:
  `e980a4b465c7d3b1144dce316ad3f68ed33bd09c3dd26358df27a3dd977d01fd`

Structural validation confirmed exact ten-component logical-OR behavior,
181-view system coverage, exact source-linked image paths, equality of visual
and hybrid decisions because all 181 current-production candidates are
DICOM-eligible, 128 contiguous fallback groups, and selection of the first
system-negative immutable-ranked candidate whenever one exists. All assembled
artifacts remain permission-restricted outside the browser directory. The
browser contains no model predictions, and the human state and events were
still the registered empty hashes at assembly freeze.

The hidden baseline is now ready for blinded reference annotation. Do not
inspect prediction prevalence or score any endpoint until all 181 binary
labels are complete and every low-confidence item has been adjudicated. Then
freeze the reference files and run the registered evaluator exactly once.

## 2026-07-24 — Current-production residual score and film metadata audit

### Question

Does the unchanged frozen v1 prompt/VLM baseline pass after the production
DICOM-eligibility correction, and can digitized hard-copy film be removed
programmatically before visual QC?

### Action

Verified all 181 residual labels and exact replay of the append-only event
history. One label remained low confidence. It was review 158, the grossly
rotated image the reviewer identified at completion, but it had been saved as
usable. Inspected that single image before opening model outputs, adjudicated
it as `Do not use` from the reviewer's explicit completion note, and cleared
the confidence flag through the annotation API. The final reference has 32
visual failures, 149 usable views, zero low-confidence flags, and 183 events.

Stopped the gallery, confirmed port 8767 was released, and created and
verified the read-only canonical reference at
`qc_redo/annotation_archive/mirai_input_production_residual_v1_reference`.
Archive metadata SHA-256 is
`74494517610ab6073f1085048798a635cbbab1de86031c40fa44faedacc52339`.

Froze
`qc_redo/auto_qc_validation/mirai_input_production_residual_v1/score_execution_protocol.json`
before opening the reference/model comparison. Its SHA-256 is
`7bb2dcbbe00944432f51a412b6ddd16c9aff8f9f4854e89e59e9fe8c7ed5bfba`.
Rehashed every registered input, required an absent output directory, and ran
the registered evaluator exactly once.

After scoring, attributed disagreements to the ten frozen component outputs.
Also joined the completed whole-QC reference and the independent 120-view
film-specific reference to raw DICOM headers. Missing wide-tag rows were read
directly through durable archive/member lineage. Only aggregate tag behavior
was reported; no patient, exam, accession, SOP, or source path was printed.

### Evidence

- View confusion: TP 30, FN 2, FP 9, TN 140.
- Sensitivity: 0.9375 (Wilson 95% CI 0.7985–0.9827).
- Specificity: 0.9396 (Wilson 95% CI 0.8892–0.9679).
- Precision: 0.7692; negative predictive value: 0.9859.
- Operational readouts: one unsafe selected slot, seven false-exhausted
  slots, and one unsafe accepted exam.
- Exam outcomes: 20 complete-safe accepted, six false rejected, five safely
  routed out, and one unsafe accepted.
- Gate SHA-256:
  `9905f8800470e3b219ce95c03b4080de52c86ffa05eebfff880a635a00e7c3e5`.
- Metrics SHA-256:
  `fd98f6e17feda556fd2eab84b8799a1800c3fa3f8262df1c90328c3ccf17999a`.

Both false negatives are rank 1 and rank 2 of the same L-MLO fallback slot
and show the same large device generator partially cropped into the superior
corner. Representative browser images are
`images/8192ae024b04ebb8c98eabd39bd756b5f1ef5630815c153318efff4f4b88b7de.png`
and
`images/f5dce05a756b74b9750fb0d53f1fecaaca3d324ec9fa4ac1fd1b6b4feb8b2691.png`
under the review batch. No frozen component acted on either image. The
adjudicated rotated view at review 158 was correctly detected by the
orientation component.

Eight of the nine false positives include the gross-device component, two of
those also include tubing, and the remaining false positive is the seam
component. This localizes both the unsafe miss and most false exhaustion to
the gross-device decision boundary, although the nonexpert binary reference
remains the strongest alternative explanation for some apparent device false
positives.

The independent film reference contains 60 visible-film positives and 60
controls:

- `DetectorType == FILM`: TP 60, FN 0, FP 0, TN 60.
- `BurnedInAnnotation == YES`: TP 59, FN 1, FP 0, TN 60.
- Secondary Capture SOP Class: TP 0, FN 60, FP 0, TN 60.
- All film positives were tagged `DetectorType == FILM`; controls were
  `DIRECT` or `SCINTILLATOR`.

In the new whole-QC cohort, all nine views from the two film-enriched exams
were human failures, all nine had `DetectorType == FILM`, and all nine were
already caught by the visual film component. No usable view carried the FILM
tag. Replacing those visual decisions with the deterministic rule would
therefore leave this cohort's system score unchanged.

The current production SoT contains 2,632 selected FILM-tagged views spanning
658 exams; every one of those exams has all four selected views tagged FILM.
Those exams contain 2,657 candidate rows. The existing downstream
`scanned_film` analysis filter already removes exams whose
`DetectorType == FILM`, but `mirai_source_eligibility_reasons` does not yet
apply that rule. Consequently these known film exams remain in the
authoritative candidate inventory and consume visual-QC work.

### Result

The candidate failed the registered gate. It passed the specificity point
threshold but missed the sensitivity threshold, and it failed all three
operational requirements that were not allowed to regress: zero unsafe
selected slots, at most one false exhaustion, and zero unsafe accepted exams.
The corrected production input policy did not rescue the unchanged v1
baseline on this disjoint cohort.

The FILM metadata rule is supported as a deterministic CHiMEC source
exclusion. It is stronger and more specific than using
`BurnedInAnnotation`, which should remain a separate markup/vendor signal.
Adding `DetectorType == FILM` to source eligibility would remove the 658
known film exams before rendering or VLM inference while continuing to
persist their original DICOM lineage and exclusion reason.

### Sensemaking

- Observation: corrected source eligibility removed tagged special views, but
  the unchanged v1 system still produced two misses in one device-bearing
  slot and seven false exhaustions.
- Expected: the production correction would preserve zero unsafe selections
  and reduce false exhaustion to at most one.
- Nearest reference: the frozen v1 prospective run, because this is the exact
  same model/prompt/threshold/fallback system on a new disjoint production
  inventory. It had TP 70, FN 0, FP 4, TN 100, zero unsafe selections, four
  false exhaustions, and zero unsafe accepted exams.
- Best explanation: the upstream correction solved deterministic
  special-acquisition leakage, but the residual gross-device observer remains
  too broad on several allowed views while missing a partially cropped
  generator in two same-slot candidates.
- Falsifier: independent expert review that reverses the two device failures
  or most gross-device false positives would materially weaken that
  localization.
- Decision impact: do not promote the assembled v1 candidate. Preserve the
  film and orientation results, promote the validated FILM tag separately,
  and test one gross-device delta on new data rather than tuning on this
  reference.
- Baseline disposition: the immutable v1 result remains the historical
  nearest reference, but neither v1 nor the regressed v2 successor is
  acceptable for production promotion. The prompt-only modular family remains
  active; this result does not support switching to embeddings.
- Falsification scope: this rejects the hypothesis that the production
  eligibility correction alone would make frozen v1 pass. It does not reject
  the validated film classifier, the paired orientation component, or the
  broader prompt-only approach.

Inference confidence is high for the frozen metrics and the within-CHiMEC
FILM rule, and medium for the gross-device causal interpretation because the
reference annotator is not a mammography QC expert.

### Skeptical review

The strongest confound is reference interpretation, especially whether
partially visible devices should fail and whether the eight gross-device
disagreements are true model false positives. The cheapest confidence-improving
check is a small, new, expert- or jointly-adjudicated corner-device challenge
set containing partial generators and the current hard negatives. Recommendation:
proceed with deterministic FILM filtering, but treat gross-device prompt
revision as a new registered experiment rather than retroactively rescoring
this cohort.

## 2026-07-24 — FILM policy promotion and disjoint gross-device challenge

### Question

Can the independently validated exact `DetectorType == FILM` rule be promoted
into authoritative source eligibility without changing any non-FILM source,
and can one new prompt-only experiment isolate the partially cropped
gross-device miss from the known hard-negative false positives?

### Registered decision

Promote only normalized `DetectorType == FILM`; do not use
`BurnedInAnnotation` as an exclusion. Keep the unchanged
`visible_gross_implanted_or_procedural_device_v1` prompt as the immutable
control. Compare it with one zero-shot candidate whose only semantic change is
to inspect all image borders, count a recognizable partially cropped device
housing, and explicitly reject unstructured bright corners, ordinary markers,
detector/paddle edges, and anatomy. Do not use embeddings, the failed
external-marker veto, examples, or a prompt sweep.

The new model-mining population must be patient-, exam-, and view-disjoint
from all prior reference panels. The blinded target-specific panel is
registered at 160 views with one view per patient/exam and fixed enrichment
strata for revised-only edge calls, other revised-only calls, both-positive
calls, baseline-only calls, lower-confidence calls, and both-negative random
controls. Model arms, predictions, rationales, and strata remain hidden from
the reviewer.

### Action

Added the exact FILM source rule to
`prima/view_selection.py` and updated `docs/mirai_input_qc_targets.md`.
Targeted formatting, lint, and compile validation passed. A real-header source
smoke check verified that a SHA-256-checked FILM DICOM receives exactly
`DetectorType is FILM`, while a checked DIRECT DICOM remains eligible.

Prepared a one-delta prompt, human target rubric, and reusable disjoint
challenge builder:

- `qc/targets/visible_gross_implanted_or_procedural_device_v2_edge_crops.txt`
- `qc/targets/visible_gross_implanted_or_procedural_device_human_v1.txt`
- `qc/build_gross_device_edge_challenge.py`

The builder reuses hard links to the already source-verified rendered
candidate bank, excludes prior reference patients/exams before inference,
freezes the exact prompt/model/selection/gate contract, and creates a blinded
panel only after both full mining runs complete.

Submitted the isolated 16-shard CPU rebuild with fixed label date
`2026-07-24`:

- preprocessing shards: `13189364`, `13189366`–`13189380`
- dependency-gated merge: `13189381`
- staged SoT:
  `/gpfs/data/huo-lab/Image/ChiMEC/MG/sot_rebuild_film_policy`
- staged derived output:
  `/gpfs/data/huo-lab/Image/ChiMEC/MG/out_rebuild_film_policy`
- provenance and monitoring root:
  `/scratch/annawoodard/prima_film_policy_rebuild`

Production paths remain untouched pending aggregate, Zarr, manifest,
source-transition, exact FILM-delta, and sampled DICOM-lineage validation.
Promotion is prepared as a reversible directory swap with policy-specific SoT
and Zarr rollback roots plus a checksumed metadata backup.

### Current evidence

All 16 shard jobs entered `RUNNING`, log timestamps and exam counters continued
to advance, and the staged Zarr-group count increased on successive checks.
No traceback, OOM, missing-space, or explicit runtime error has appeared.

### Sensemaking

- Observation: the FILM rule is exact on two independent references and on
  real source headers, while the full rebuild is progressing normally.
- Expected: every old FILM candidate moves to deterministic exclusions with
  the exact reason, no non-FILM source receives that reason, and only exams
  made incomplete by FILM removal leave the selected inventory.
- Nearest reference: the immediately preceding current-policy production SoT
  and its successful staged validation, not an older preprocessing cache.
- Falsifier: any retained FILM candidate, any non-FILM source with the FILM
  reason, a new candidate source, or a selected-source change outside an exam
  containing FILM blocks promotion.
- Decision impact: babysit the rebuild to completion, validate before and after
  promotion, then freeze and launch only the paired prompt-only mining arms.

Inference confidence is high for the FILM policy and medium for the edge-crop
prompt hypothesis until the new blinded target-specific labels exist.

### Promotion and challenge launch update

All preprocessing jobs completed successfully, and staged, exact-delta, and
lineage validation passed before promotion. The authoritative production
inventory now contains 40,172 selected views from 10,043 canonical exams,
52,699 eligible candidate sources, and 21,591 deterministic exclusion rows.
The policy moved all 2,657 formerly eligible FILM sources out of candidacy.
Across the complete source inventory, 2,850 sources now carry
`DetectorType is FILM`, including 193 that already had another deterministic
reason.

Post-promotion validation reopened all 2,850 FILM-reason sources from their
original archive members and confirmed `DetectorType == FILM`. It also
reopened 128 deterministic retained-candidate controls and confirmed that
they were non-FILM. No non-FILM candidate moved to exclusions, no FILM source
remained a candidate, and the independent lineage audit passed for 128
selected views by SOP UID and SHA-256. Rollback copies remain at
`sot_pre_film_policy`, `out/zarr_pre_film_policy`, and the checksumed
`pre_promotion_backup` under the rebuild provenance root.

The gross-device challenge was then frozen before inference. Its mining
population contains 26,450 source-verified rendered views from 5,168 exams
and 1,201 patients, with zero patient, exam, or view overlap with either prior
reference source table. The images are hard links to the existing verified
render bank. The immutable specification is
`qc_redo/auto_qc_development/gross_device_edge_crop_challenge_v1/candidate_spec.json`.

The shared H200 broker reported no persistent holder, so the restartable
per-view shards were submitted once through the native opportunistic launcher
with the registered `zhoulabq,catherineq,siweiq` fallback. No raw
`srun --jobid` attachment or duplicate broker claim was used.

- baseline jobs: `13190487`, `13190514`, `13190515`, `13190518`
- candidate jobs: `13190521`, `13190523`, `13190524`, `13190526`

Each arm uses four deterministic shards. Per-view results are checkpointed in
`qc_redo/auto_qc_development/gross_device_edge_crop_challenge_v1/mining_runs`.
The first launcher attempt failed before submission because the pinned
environment's `vllm` executable was not on `PATH`; after that was corrected,
one job submitted and the next failed before submission because its unique
executable `TMPDIR` did not yet exist. The seven remaining scratch directories
were created, and only the seven not-yet-queued jobs were submitted.

All seven running shards reached model inference and began writing durable
`view_suggestions` checkpoints. Candidate shard 003 remained queued for the
next available H200. Dependency-gated CPU follow-ons were submitted so an
incomplete arm cannot silently produce a review panel:

- baseline merge: `13190615`
- candidate merge: `13190616`
- blinded panel assembly: `13190617`

The assembly job depends on both successful merges and validates complete
coverage, frozen prompt hashes, and unchanged inference settings before
selecting the registered 160-view panel.
