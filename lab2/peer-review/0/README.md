# Lab 2: American English dialect survey

A reference analysis of the lexical questions (Q50–Q121) of Bert Vaux's
Dialect Survey

## Contents

```
code/       analysis scripts, run.sh, environment.yaml
documents/  Nerbonne & Kretzschmar (2003, 2006)
figs/       all figures (generated)
report/     lab2.tex, values/*.tex (generated numbers), lab2.pdf
```

## Reproducing the report

```bash
conda env create -f code/environment.yaml
bash -i code/run.sh                         # figs/ and report/values/
cd report && latexmk -pdf lab2.tex          # the pdf
```

or `make PY=python` from this folder (analysis + pdf).

| script | lab task | output |
|---|---|---|
| `clean.py` | shared loading, cleaning, one-hot encoding | — (imported by all) |
| `data_summary.py` | cleaning funnel | `values/data_summary.tex` |
| `eda.py` | 2: Q050 × Q105, many questions | `figs/eda_*.pdf`, `values/eda.tex` |
| `dimred.py` | 3: PCA, centering vs scaling | `figs/dr_*.pdf`, `values/dr.tex` |
| `cluster.py` | 4: k-means (people), Ward (grid cells) | `figs/cl_*.pdf`, `values/cluster.tex` |
| `stability.py` | 5: random starts, bootstrap, subsampling | `figs/st_*.pdf`, `values/stability.tex` |

Every number in the report is a LaTeX macro written by these scripts, so the
text cannot drift from the code. All randomness is seeded (`clean.SEED = 215`);
k-means runs single-threaded (`OMP_NUM_THREADS=1`), so reruns give
byte-identical values files and figures.

## Main judgment calls

- Drop respondents without coordinates, outside the contiguous US, or with
  more than 10% of questions unanswered (47,471 -> 44,557).
- Drop later copies of duplicated answer vectors and keep unique respondents
  plus the first copy of each vector, because later copies sit on neighboring
  IDs but carry other people's locations; the result with all rows is a
  sensitivity check.
- PCA on the centered, unscaled 468-column binary matrix.
