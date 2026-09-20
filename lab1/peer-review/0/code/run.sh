#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate 215a
jupyter nbconvert --to notebook --execute --inplace lab1.ipynb
