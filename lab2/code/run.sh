#!/bin/bash
# add stuff below here...

conda activate 215a

# Re-run every cell of the report notebook in order, saving all outputs
# (tables, figures) back into the notebook itself.
jupyter nbconvert --to notebook --execute --inplace lab2.ipynb

# Render the executed notebook to the PDF report (no code shown, per
# lab2.ipynb's front-matter `execute: echo: false`).
quarto render lab2.ipynb --to pdf
mv lab2.pdf ../report/lab2.pdf
