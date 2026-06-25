# Randi Actual-Data Handoff - 2026-06-25

## Goal

Move Prima vertical-line auto-QC from the DSI copied montage/cache packet to the other cluster with H200s and the actual mammography data, then validate on a human-reviewable batch before a full scan.

## Current State

- Repo branch on DSI before transfer: `main`; handoff commit should include `AGENTS.md`, `README.md`, unified QC-state analysis changes, and the repo-local notebooks under `logs/`.
- Canonical DSI notebook: `/home/annawoodard/prima/logs/lab_notebook_dsi.md`.
- Randi notebook path to use after transfer: `/home/annawoodard/prima/logs/lab_notebook_randi.md`.
- Best dev-set detector prompt: Qwen3.5-397B FP8 repair wrapper with bf16 experts, `prompt_mode=marker_classifier`, `prompt_variant=confidence_specificity`, target tag `vertical line (detector artifact)`.
- Dev-set evidence: final gray-seam prompt on the repaired 38-exam QC set scored `TP=8 FP=0 FN=0 TN=30`.
- DSI held-out job `858828` completed on `2026-05-15` with `ExitCode=0:0`, scored 120 copied-cache exams, and suggested 3 vertical-line positives; human review state is still empty, so sensitivity/specificity is not yet estimated.

## Evidence Paths On DSI

- Dev-set final run: `/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_confidence_grayseam_targetshots_rescue6h_20260430/qwen397b_vertical_line_ablation_confidence_grayseam_targetshots_rescue6h_20260430.json`.
- Dev-set QC truth: `/net/projects2/annawoodard/qc_redo/qc_state_repaired.json`.
- Held-out copied-cache run: `/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_heldout_validation_20260515/qwen397b_vertical_line_heldout_validation_20260515.json`.
- Held-out review runbook: `/home/annawoodard/prima/logs/vertical_line_heldout_validation_20260515_review.md`.
- Held-out manual review state: `/net/projects2/annawoodard/qc_redo/qc_state_vertical_line_heldout_validation_20260515.json` currently has 0 reviewed exams.
- PI-facing progress update: `/home/annawoodard/prima/logs/qwen397b_vertical_line_progress_update_20260515.md`.

## Environment Reproduction On Randi

Use the cluster's actual paths; do not reuse `/net/projects2/annawoodard/qc_export` except as historical evidence.

```bash
cd /home/annawoodard/prima
git fetch origin
git checkout main
git pull --ff-only

# Create once if missing.
micromamba create -y -f env.yaml
eval "$(micromamba shell hook -s bash)"
micromamba activate prima
pip install -e .
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install the model stack into this env if not already present.
python - <<'PY'
import importlib.util
missing=[m for m in ["torch","transformers","accelerate","safetensors"] if importlib.util.find_spec(m) is None]
print("missing", missing)
PY
```

If any model-stack package is missing, install maintained packages into `prima` rather than adding local shims:

```bash
pip install "torch" "transformers" "accelerate" "safetensors"
```

Then verify:

```bash
python -m py_compile auto_annotate_qc.py submit_auto_qc.py qc/qc_gallery.py scripts/build_qwen35_fp8_repair.py
python submit_auto_qc.py --help
python qc/qc_gallery.py --help
sinfo -s
sacctmgr -nP show assoc user="$USER" format=Cluster,Account,User,Partition,QOS,DefaultQOS,MaxJobs,MaxSubmitJobs,MaxTRES,GrpTRES,Priority
```

## Qwen Repair Wrapper

Native Qwen3.5-397B FP8 MoE was unstable on DSI around layer 14, with CUDA/device-side assert/CUBLAS/deep-gemm failures. The reliable path was the PRIMA repair wrapper plus bf16 experts.

```bash
python scripts/build_qwen35_fp8_repair.py \
  --base-model-path <RANDI_MODELS>/Qwen3.5-397B-A17B-FP8 \
  --output-model-path <RANDI_MODELS>/Qwen3.5-397B-A17B-FP8-prima-repair \
  --force-bf16-experts
```

Run inference with both the repaired wrapper path and the env override:

```bash
env CUDA_LAUNCH_BLOCKING=1 PRIMA_QWEN35_FP8_FORCE_BF16_EXPERTS=1 \
  python submit_auto_qc.py ...
```

## What Changes With Actual Data

- Replace copied-cache paths with Randi actual-data paths:
  - raw DICOM root: discover on Randi, likely the cluster-local equivalent of `/gpfs/data/huo-lab/Image/ChiMEC/MG`
  - views parquet: use `<RAW>/sot/views.parquet` if present, otherwise regenerate with `pipelines/preprocess.py`
  - QC montage export dir: create a Randi-local export dir, for example `<RANDI_WORK>/qc_export`
  - auto-QC run root: create a Randi-local run root, for example `<RANDI_WORK>/qc_redo/auto_qc_runs`
- Do not treat the DSI 7,832 cached montages as the full dataset; that was a copied subset/cache packet.
- Rebuild or verify cached four-view montages from actual data before broad Qwen inference:

```bash
python qc/qc_gallery.py --estimate-only --raw <RAW> --views <RAW>/sot/views.parquet --output <RANDI_QC_EXPORT>
python qc/qc_gallery.py --preprocess-all --raw <RAW> --views <RAW>/sot/views.parquet --output <RANDI_QC_EXPORT>
```

If `<RAW>/sot/views.parquet` is missing or stale, run preprocessing on the actual raw root first; for large actual-data runs prefer `pipelines/run_preprocess_sharded.py --help` over a headnode/full serial run.

## Recommended Next Action

1. On Randi, verify Slurm H200 policy and update any submission flags; do not copy DSI's `general`/`interactive` assumptions without `sacctmgr`/`sinfo`.
2. Rebuild the Qwen repair wrapper in the Randi model directory with `--force-bf16-experts`.
3. Generate/verify actual-data QC montages and `views.parquet`.
4. Run a small smoke job on 1-3 actual-data exams with the repaired wrapper and `PRIMA_QWEN35_FP8_FORCE_BF16_EXPERTS=1`.
5. Build a 100-300 exam actual-data review batch with positive-enriched vertical-line heuristic candidates plus random negatives, run the model, serve it in `qc/qc_gallery.py`, and hand review both model positives and negatives.
6. Evaluate with `scripts/evaluate_auto_qc_ablation.py`; only scale to full actual-data seam discovery after the reviewed batch gives acceptable sensitivity and specificity.

Concrete first command for the new agent:

```bash
cd /home/annawoodard/prima && git status --short --branch && sinfo -s && python submit_auto_qc.py --help
```
