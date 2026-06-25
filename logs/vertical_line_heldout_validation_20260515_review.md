# Vertical-Line Held-Out Validation Review - 2026-05-15

## State

Held-out validation batch completed as Slurm job `858828`.

- Slurm policy checked on `2026-05-15`: use partition `general` and QoS `general` for this production validation run. The submission omits `--qos`, so Slurm applies the user's default associated QoS `general`.
- Scheduler result: `COMPLETED`, `ExitCode=0:0`, elapsed `08:49:39`, node `p002`.
- Model output summary: `120` exam rows, `3` exams with one suggested tag, all `vertical line (detector artifact)`.
- Manual review state remains empty as of `2026-06-25`, so sensitivity and specificity have not been estimated.
- Requested resources: one `general` node, `4xh200`, `16` CPUs, `720G` memory, `12:00:00` walltime, excluding `o001`.
- Exam list: `/net/projects2/annawoodard/qc_redo/debug_exam_lists/vertical_line_heldout_validation_20260515.txt`
- Sample manifest: `/net/projects2/annawoodard/qc_redo/debug_exam_lists/vertical_line_heldout_validation_20260515_manifest.csv`
- Sample summary: `/net/projects2/annawoodard/qc_redo/debug_exam_lists/vertical_line_heldout_validation_20260515_summary.json`
- Auto-QC run file, created after job completion: `/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_heldout_validation_20260515/qwen397b_vertical_line_heldout_validation_20260515.json`
- Debug dump dir: `/net/projects2/annawoodard/qc_redo/auto_qc_debug/vertical_line_heldout_validation_20260515`
- Submitit logs: `/net/projects2/annawoodard/qc_redo/submitit_runs/qwen397b_vertical_line_ablation_heldout_validation_20260515_20260515_141435/submitit_logs`
- Manual review QC state: `/net/projects2/annawoodard/qc_redo/qc_state_vertical_line_heldout_validation_20260515.json`

## Sample Design

The batch contains `120` previously unreviewed cached montages:

- `80` high vertical-line heuristic candidates
- `20` mid-ranked vertical-line heuristic candidates
- `20` random exams from the 1,000-exam prescreen

The purpose is to get a reviewable held-out batch with enough likely positives to test sensitivity while still including model negatives for specificity and false-negative checks.

## Historical Job Check

```bash
micromamba run -p /net/projects2/annawoodard/micromamba/envs/prima \
  sacct -j 858828 --format=JobID,JobName%80,State,ExitCode,Elapsed,Start,End,NodeList -P
```

Do not resubmit this DSI run to `normal` or `burst`; those are stale queue names for this workspace. On another cluster, recheck `sinfo` and `sacctmgr` rather than copying DSI QoS assumptions.

## Start Review Server

```bash
micromamba run -p /net/projects2/annawoodard/micromamba/envs/prima \
  python qc/qc_gallery.py \
  --serve \
  --port 5000 \
  --preprocessed-only \
  --views /net/projects2/annawoodard/qc_export/views_for_qc.parquet \
  --output /net/projects2/annawoodard/qc_redo/review_galleries/vertical_line_heldout_validation_20260515 \
  --exam-list /net/projects2/annawoodard/qc_redo/debug_exam_lists/vertical_line_heldout_validation_20260515.txt \
  --qc-file /net/projects2/annawoodard/qc_redo/qc_state_vertical_line_heldout_validation_20260515.json \
  --auto-run-file /net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_heldout_validation_20260515/qwen397b_vertical_line_heldout_validation_20260515.json
```

Use the gallery to mark exams:

- `g`: reviewed good / no QC finding
- `a`: annotate, then choose `vertical line (detector artifact)` if present

The review state is separate from the canonical repaired QC state so this validation can be analyzed cleanly.

## Analysis After Review

After manual review, compare model predictions to the validation QC state:

```bash
micromamba run -p /net/projects2/annawoodard/micromamba/envs/prima \
  python scripts/evaluate_auto_qc_ablation.py \
  --qc-file /net/projects2/annawoodard/qc_redo/qc_state_vertical_line_heldout_validation_20260515.json \
  --tag 'vertical line (detector artifact)' \
  --run-file heldout=/net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_heldout_validation_20260515/qwen397b_vertical_line_heldout_validation_20260515.json \
  --out-csv /net/projects2/annawoodard/qc_redo/auto_qc_runs/vertical_line_heldout_validation_20260515/heldout_metrics_after_review.csv
```

Primary readouts:

- recall on reviewed positives
- precision among model positives
- false-positive rate among reviewed negatives
- whether misses or false positives share a visual mechanism
