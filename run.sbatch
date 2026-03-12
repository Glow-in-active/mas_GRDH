#!/bin/bash
#SBATCH --job-name=stego_gen
#SBATCH --output=run-%j.log
#SBATCH --error=run-%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

module load Python/Anaconda
source activate stego

