#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")"

conda activate stat215a

jupyter nbconvert --to notebook --execute --inplace lab1.ipynb

conda deactivate
