# make          run the analysis, then build report/lab2.pdf
# make report   rebuild only the pdf from existing figures/values
# make clean    remove LaTeX build products

PY ?= python
SCRIPTS := data_summary eda dimred cluster stability

.PHONY: all analysis report clean

all: analysis report

analysis:
	cd code && export OMP_NUM_THREADS=1 && for s in $(SCRIPTS); do $(PY) $$s.py || exit 1; done

report:
	cd report && latexmk -pdf -interaction=nonstopmode lab2.tex

clean:
	cd report && latexmk -c lab2.tex
	rm -rf code/__pycache__
