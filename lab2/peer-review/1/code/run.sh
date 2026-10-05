#!/bin/bash

conda activate 215a

clean.py
python clean.ipynb
python clean.py
python lab2.ipynb
conda deactivate
