#!/bin/bash
# Submit ChiMEC breast-MR HFDP risk export loop.
#
# Keeps breast MR controls plus case exams before diagnosis and diagnosis-like
# case exams near t=0 for export from iBroker.

#SBATCH --job-name=scrape-risk-mr
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/scrape-risk-mr-%j.out
#SBATCH --error=logs/scrape-risk-mr-%j.err

set -euo pipefail

mkdir -p logs

export IBROKER_USERNAME="${IBROKER_USERNAME:-annawoodard@uchicago.edu}"
export IBROKER_PASSWORD="${IBROKER_PASSWORD:-16352a}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate selenium-ff

cd /gpfs/data/huo-lab/Image/annawoodard/prima
export PYTHONUNBUFFERED=1

DX_LIKE_WINDOW_DAYS="${DX_LIKE_WINDOW_DAYS:-30}"
BATCH_SIZE="${BATCH_SIZE:-200}"
MAX_EXPORTS_PER_HOUR="${MAX_EXPORTS_PER_HOUR:-200}"
LOOP_WAIT="${LOOP_WAIT:-5m}"

stdbuf -oL -eL python -u exports/export_chimec.py \
  --auto-confirm \
  --modality MR \
  --cohort priority \
  --target-profile breast-risk-hfdp \
  --risk-dx-like-window-days "${DX_LIKE_WINDOW_DAYS}" \
  --batch-size "${BATCH_SIZE}" \
  --max-exports-per-hour "${MAX_EXPORTS_PER_HOUR}" \
  --loop-wait "${LOOP_WAIT}"
