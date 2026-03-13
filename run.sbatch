#!/bin/bash
#SBATCH --job-name=stego_ge
#SBATCH --output=run-%j.log
#SBATCH --error=run-%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
echo "submit_dir=$SLURM_SUBMIT_DIR"
echo "pwd_before=$(pwd)"
cd "$SLURM_SUBMIT_DIR"
echo "pwd_after=$(pwd)"

module purge
module load Python/Anaconda

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate stego
else
  source activate stego
fi

python -u generate_dataset.py

