#!/bin/bash
# NOTE: this is just for producing whatever figures are in
# your report. run.sh does NOT need to produce S.csv
# (that is what the SCF job script, run_algorithm1.sbatch, is for --
#  it was run on the SCF cluster with `sbatch run_algorithm1.sbatch`
#  to produce ../results/S.csv, committed to the repo since rerunning
#  it is expensive).
conda activate 215a

sbatch run_algorithm1.sbatch

# Render the report notebook, re-executing all cells in place.
jupyter nbconvert --to notebook --execute --inplace lab3.ipynb
