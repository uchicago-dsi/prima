#!/bin/bash
#SBATCH --job-name=scrape-ibroker
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/scrape-ibroker-%j.out
#SBATCH --error=logs/scrape-ibroker-%j.err

set -euo pipefail

mkdir -p logs

export IBROKER_USERNAME=annawoodard@uchicago.edu IBROKER_PASSWORD=16352a

eval "$(micromamba shell hook --shell bash)"
micromamba activate selenium-ff

cd /gpfs/data/huo-lab/Image/annawoodard/prima
export PYTHONUNBUFFERED=1
stdbuf -oL -eL python -u exports/export_chimec.py \
  --auto-confirm \
  --refresh-metadata \
  --refresh-mode fresh \
  --refresh-workers 4 \
  --refresh-checkpoint-batch-size 50 \
  --refresh-max-new-batches-per-cycle 20 \
  --max-exports-per-hour 200 \
  --reconcile-disk-ibroker \
  --reconcile-output-csv data/chimec_disk_ibroker_reconciliation.csv \
  --batch-size 200 \
  --loop-wait 5m
