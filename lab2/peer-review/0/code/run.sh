#!/bin/bash
# Regenerate every figure in figs/ and every number in report/values/
# Run from anywhere:  bash -i code/run.sh   (or: source code/run.sh)
# `-i` makes `conda activate` available in a non-interactive shell

cd "$(dirname "${BASH_SOURCE[0]:-$0}")" || exit 1
conda activate 215a
export OMP_NUM_THREADS=1 # single-threaded k-means is bit-reproducible

python data_summary.py # cleaning funnel -> values/data_summary.tex
python eda.py # task 2: question pairs -> figs/eda_*.pdf
python dimred.py # task 3: PCA -> figs/dr_*.pdf
python cluster.py # task 4: clustering -> figs/cl_*.pdf
python stability.py # task 5: robustness -> figs/st_*.pdf
