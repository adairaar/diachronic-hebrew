#!/bin/bash
# Full manuscript build from result files.  No hand-maintained numbers.
#
#   make_numbers.py  -> numbers.tex   (every quantity in the text)
#   make_tables.py   -> tables/*.tex
#   make_figs.py     -> figures/*.pdf
#
# Each generator exits non-zero if a result file is missing, so a stale
# manuscript cannot be built silently.
set -e
cd "$(dirname "$0")"

python3 make_numbers.py
python3 make_tables.py
python3 make_si.py
python3 make_figs.py

rm -f main.aux main.bbl main.blg main.out
pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
bibtex main >/dev/null 2>&1 || true
for i in 1 2; do pdflatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true; done
pdflatex -interaction=nonstopmode main.tex > build.log 2>&1 || true

echo "---------------------------------------------------------------"
echo "errors      : $(grep -cE '^! ' build.log || true)"
echo "undefined   : $(grep -ciE 'undefined (reference|citation|control sequence)' build.log || true)"
echo "overfull    : $(grep -c 'Overfull' build.log || true)"
echo "pages       : $(pdfinfo main.pdf 2>/dev/null | awk '/^Pages/{print $2}')"
echo "citations   : $(grep -c bibitem main.bbl 2>/dev/null || echo 0)"
echo "---------------------------------------------------------------"
grep -E '^! |Undefined control sequence|Citation .* undefined|Reference .* undefined' build.log | head -20 || true
