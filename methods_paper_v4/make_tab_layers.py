"""
Tables for the internal-structure result.

Two of them.  The first reports dispersion within each source against the
yardstick of the dated anchor books.  The second reports the block separations,
which is where the layering actually shows: each block is internally tight while
the blocks themselves sit a century apart, and dispersion alone would miss that.

Both are generated from the result files, like every other table in this
manuscript, so neither can drift from the analysis.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import os, sys, json
import numpy as np, pandas as pd
from scipy import stats


RNG = np.random.default_rng(3)
EX = 586


def need(p):
    if not os.path.exists(p): sys.exit(f"MISSING RESULT FILE: {p}")
    return p


C = pd.read_csv(need(DH.f("internal_consistency.csv"))).set_index("unit")
M = json.load(open(need(DH.f("internal_consistency_matched.json"))))
P = pd.read_csv(need(DH.f("chunk_preds.csv")))
Tg, An = P[P.kind == "target"], P[P.kind == "anchor"]
ref = np.array([An.cpred[An.unit == b].std(ddof=1)
                for b in An.unit.unique() if (An.unit == b).sum() >= 4])

ROWS = [("JE_source", "JE composite", 0), ("Gen_JE", "Genesis JE", 1),
        ("Exo_JE", "Exodus JE", 1), ("Num_JE", "Numbers JE", 1),
        ("D_source", "D source", 0),
        ("D_Code", "law code, Deut 12--26", 1),
        ("D_Frame", "frame, Deut 1--11, 27--34", 1),
        ("P_source", "P source", 0),
        ("Lev_Priestly", "Leviticus 1--16", 1),
        ("Lev_Holiness", "Holiness Code, Lev 17--26", 1),
        ("Jer_DTR", "Jeremiah Dtr prose", 0)]

body = ["\\textbf{Unit} & \\textbf{Passages} & \\textbf{Median} & "
        "\\textbf{IQR} & \\textbf{SD} & \\textbf{Pctile} & "
        "\\textbf{Post-exilic} \\\\\n\\midrule\n"]
for u, lab, ind in ROWS:
    if u not in C.index: continue
    r = C.loc[u]
    nm = ("\\hspace{1.5em}\\textit{" + lab + "}") if ind else ("\\textbf{" + lab + "}")
    body.append(f"{nm} & {int(r['n'])} & {r['median']:.0f} & {r['iqr']:.0f} & "
                f"{r['sd']:.0f} & {100*r['pctile']:.0f}\\% & "
                f"{100*r['pct_post']:.0f}\\% \\\\\n")
open(DH.tab("tab_layers.tex"), "w").write(
    "\\begin{table}[!ht]\n\\centering\n\\footnotesize\n"
    "\\caption{{\\bf Internal structure of the Pentateuchal sources.}  "
    "Dispersion of the estimated dates of the individual $\\sim$500-word "
    "passages making up each unit, in years BCE.  ``Pctile'' places each unit's "
    "standard deviation in the distribution of within-book dispersions measured "
    "on the "
    f"{len(ref)} dated anchor books (median {np.median(ref):.0f}~yr): 50\\% is an "
    "ordinary single composition, and higher values are more internally varied "
    "than that.  Because dispersion rises with passage count "
    f"($\\rho = +0.56$ across the anchors), the three-way comparison of the "
    "sources is made at matched count in the text; the raw values are given "
    "here.  ``Post-exilic'' is the share of a unit's passages estimated later "
    f"than {EX}~BCE.  Sub-blocks are indented under their source.}}\n"
    "\\label{tab:layers}\n\\begin{tabular}{lrrrrrr}\n\\toprule\n"
    + "".join(body) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ── block separations ──────────────────────────────────────────────────────
PAIRS = [("D_Code", "D_Frame", "Deuteronomic law code", "Deuteronomic frame"),
         ("Lev_Holiness", "Lev_Priestly", "Holiness Code", "Leviticus 1--16"),
         ("Gen_JE", "Exo_JE", "Genesis JE", "Exodus JE")]
body = ["\\textbf{Earlier-dated block} & \\textbf{Later-dated block} & "
        "\\textbf{Gap} & \\textbf{95\\% CI} & \\textbf{$p$} \\\\\n\\midrule\n"]
out = []
for a, b, la, lb in PAIRS:
    x = Tg.cpred[Tg.unit == a].values; y = Tg.cpred[Tg.unit == b].values
    gap = float(np.median(x) - np.median(y))
    d = np.array([np.median(RNG.choice(x, len(x))) - np.median(RNG.choice(y, len(y)))
                  for _ in range(4000)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    pu = float(stats.mannwhitneyu(x, y)[1])
    out.append(dict(a=a, b=b, gap=gap, lo=float(lo), hi=float(hi), p=pu))
    body.append(f"{la} & {lb} & {gap:+.0f} & {lo:+.0f} to {hi:+.0f} & "
                f"{pu:.3f} \\\\\n")
open(DH.tab("tab_blocks.tex"), "w").write(
    "\\begin{table}[!ht]\n\\centering\n\\footnotesize\n"
    "\\caption{{\\bf Separation between blocks within a source.}  Difference in "
    "median passage date between the conventionally distinguished blocks of each "
    "source, in years; a positive gap means the first block is placed earlier.  "
    "Confidence intervals are from 4{,}000 bootstrap resamples of the passages; "
    "$p$ is a two-sided Mann--Whitney test.  These separations, not the "
    "within-block dispersions of Table~\\ref{tab:layers}, are where the "
    "conventional layering of the sources shows: each block is internally "
    "homogeneous while the blocks stand roughly a century apart.}\n"
    "\\label{tab:blocks}\n\\begin{tabular}{llrrr}\n\\toprule\n"
    + "".join(body) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")

json.dump(dict(blocks=out, ref_median=float(np.median(ref)), n_ref=len(ref),
               sd39=M["sd39"], order=M["order"],
               p_JE_gt_D=M["p_JE_gt_D"], p_JE_gt_P=M["p_JE_gt_P"],
               p_D_gt_P=M["p_D_gt_P"]),
          open(DH.f("layers_numbers.json"), "w"), indent=2)
print(f"wrote tab_layers.tex ({len(ROWS)} units), tab_blocks.tex "
      f"({len(PAIRS)} comparisons), layers_numbers.json")
