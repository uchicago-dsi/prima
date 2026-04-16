#!/bin/bash
#SBATCH --job-name=sync_local
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=sync_%j.log
#SBATCH --error=sync_%j.err

set -e

# # Activate environment
# eval "$(micromamba shell hook -s bash)"
# micromamba activate prima

# # Change to script directory
# cd /gpfs/data/huo-lab/Image/annawoodard/prima

# # Run sync with 4 workers, whole-file optimization
# python ops/sync_local.py --workers 4 --whole-file --no-auto-restart 2>&1 | tee sync_local.log

# Sleep to keep job alive for reconnection
echo "Sync completed. Sleeping for 1 hour to allow reconnection..."
sleep 36000

