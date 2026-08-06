#!/bin/bash
set -e
python3 splice.py
python3 patch_intro.py
python3 patch_final.py
python3 insert_figs.py
cp main_new.tex build.tex && rm -f build.aux build.bbl
pdflatex -interaction=nonstopmode build.tex >/dev/null 2>&1
bibtex build >/dev/null 2>&1
for i in 1 2 3; do pdflatex -interaction=nonstopmode build.tex >/dev/null 2>&1; done
pdflatex -interaction=nonstopmode build.tex > final.log 2>&1
echo "errors=$(grep -cE '^! ' final.log) undefined=$(grep -ciE 'undefined (reference|citation|control)' final.log) pages=$(pdfinfo build.pdf|awk '/^Pages/{print $2}') cites=$(grep -c bibitem build.bbl)"
cp build.pdf main_new.pdf
