#!/bin/bash
#SBATCH --job-name=preprocess_regen
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/preprocess-regen-%j.out
#SBATCH --error=logs/preprocess-regen-%j.err

set -eo pipefail

cd /gpfs/data/huo-lab/Image/annawoodard/prima
mkdir -p logs

eval "$(micromamba shell hook -s bash)"
micromamba activate prima

export PYTHONUNBUFFERED=1

RAW_DIR=/gpfs/data/huo-lab/Image/ChiMEC/MG
SOT_DIR="${RAW_DIR}/sot"
OUT_DIR="${RAW_DIR}/out"
CHECKPOINT_DIR=/gpfs/data/huo-lab/Image/annawoodard/prima/data/discovery_checkpoints
LABELS_CSV=/gpfs/data/phs/groups/Projects/Huo_projects/SPORE/annawoodard/Phenotype_ChiMEC_2025Oct4.csv
BACKUP_SUFFIX=$(date +%Y%m%d-%H%M%S)

echo "[regen] starting sharded rebuild at $(date)"
echo "[regen] raw: ${RAW_DIR}"

# Backup existing outputs before removing
if [[ -d "${SOT_DIR}" ]] || [[ -d "${OUT_DIR}" ]] || [[ -d "${CHECKPOINT_DIR}" ]]; then
  echo "[regen] backing up previous outputs to *_backup_${BACKUP_SUFFIX}"
  [[ -d "${SOT_DIR}" ]]      && mv "${SOT_DIR}"      "${SOT_DIR}_backup_${BACKUP_SUFFIX}"
  [[ -d "${OUT_DIR}" ]]      && mv "${OUT_DIR}"      "${OUT_DIR}_backup_${BACKUP_SUFFIX}"
  [[ -d "${CHECKPOINT_DIR}" ]] && mv "${CHECKPOINT_DIR}" "${CHECKPOINT_DIR}_backup_${BACKUP_SUFFIX}"
fi

echo "[regen] launching sharded preprocessing"
stdbuf -oL -eL python -u pipelines/run_preprocess_sharded.py \
  --raw "${RAW_DIR}" \
  --workers 32 \
  --num_shards 32 \
  --genotyped-only \
  --labels "${LABELS_CSV}" \
  --partition tier1q

echo "[regen] done at $(date)"
