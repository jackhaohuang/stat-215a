#!/bin/bash
#SBATCH --job-name=lab3_algorithm1
#SBATCH --partition=low
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=algorithm1_%j.log


source ~/.bashrc
conda activate 215a

python algorithm1.py \
    --data ../data/lingBinary.Rdata \
    --output ../results/S.csv \
    --k-max 10 \
    --B 100 \
    --m 0.8 \
    --n-jobs "${SLURM_CPUS_PER_TASK}"
