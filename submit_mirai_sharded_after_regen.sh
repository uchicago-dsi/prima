#!/bin/bash
# launcher job: runs on CPU partition and submits shard jobs to GPU partition
#SBATCH --job-name=mirai_sharded
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mirai-sharded-launch-%j.out
#SBATCH --error=logs/mirai-sharded-launch-%j.err

set -eo pipefail

cd /gpfs/data/huo-lab/Image/annawoodard/prima
mkdir -p logs

eval "$(micromamba shell hook -s bash)"
micromamba activate prima

export PYTHONUNBUFFERED=1

RAW_DIR=/gpfs/data/huo-lab/Image/ChiMEC/MG
META_CSV="${RAW_DIR}/out/mirai_manifest.csv"
PRED_CSV="${RAW_DIR}/out/validation_output.csv"
SHARD_PARTITION=gpuq

echo "[mirai] launching sharded workflow at $(date)"
echo "[mirai] metadata_path=${META_CSV}"
echo "[mirai] prediction_save_path=${PRED_CSV}"
echo "[mirai] shard_partition=${SHARD_PARTITION}"

stdbuf -oL -eL python -u pipelines/run_mirai_sharded.py \
  --max_samples_per_shard 500 \
  --partition "${SHARD_PARTITION}" \
  --metadata_path "${META_CSV}" \
  --prediction_save_path "${PRED_CSV}" \
  --model_name mirai_full \
  --img_encoder_snapshot /gpfs/data/huo-lab/Image/annawoodard/.mirai/snapshots/mgh_mammo_MIRAI_Base_May20_2019.p \
  --transformer_snapshot /gpfs/data/huo-lab/Image/annawoodard/.mirai/snapshots/mgh_mammo_cancer_MIRAI_Transformer_Jan13_2020.p \
  --calibrator_snapshot /gpfs/data/huo-lab/Image/annawoodard/.mirai/snapshots/calibrators/Mirai_calibrator_mar12_2022.p \
  --batch_size 8 \
  --num_workers 4 \
  --dataset csv_mammo_risk_all_full_future \
  --img_mean 7047.99 \
  --img_size 1664 2048 \
  --img_std 12005.5 \
  --test \
  --save_hiddens \
  --cuda

echo "[mirai] sharded launcher complete at $(date)"
