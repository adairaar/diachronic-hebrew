"""
Specification curve figure: all 48 analyses at once.

The estimates in this project move when the specification moves.  Reporting one
of them and burying the rest in an appendix invites the reader to treat the
reported number as special, which it is not.  The figure shows every
specification, sorted by how well it orders the anchors once genre is held
fixed, so the reader can see both the spread and whether the better-validated
analyses say anything different from the worse ones.

Sorted by validation quality rather than by effect size, which is the usual
convention, because the question this corpus raises is not "how big is the
effect" but "do the analyses that actually work agree".
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE, PURPLE = "#2166AC", "#D95F02", "#7570B3"   # validated, see palette check
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d8d8d4"
SURF = "#fcfcfb"
EX = 586
PASS = 0.20

D = pd.read_csv(DH.f("spec_curve.csv"))
D = D.sort_values("rho_genre").reset_index(drop=True)
D["ok"] = D.rho_genre > PASS
x = np.arange(len(D))

SRC = [("JE_source", "JE composite", BLUE),
       ("D_source", "D source", ORANGE),
       ("P_source", "P source", PURPLE)]

fig = plt.figure(figsize=(7.5, 8.6))
gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.95], hspace=0.16,
                      left=0.215, right=0.985, top=0.938, bottom=0.075)
fig.patch.set_facecolor(SURF)

for i, (col, name, c) in enumerate(SRC):
    ax = fig.add_subplot(gs[i])
    ax.set_facecolor(SURF)
    lo, hi = D[f"{col}_lo"].values, D[f"{col}_hi"].values
    pr = D[f"{col}_pred"].values
    ax.axhspan(EX, 1250, color=GRID, alpha=0.35, lw=0, zorder=0)
    ax.axhline(EX, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    for j in x:
        ax.plot([j, j], [lo[j], hi[j]], color=c, lw=2.0,
                alpha=0.85 if D.ok[j] else 0.22, solid_capstyle="round",
                zorder=3)
    ax.scatter(x[D.ok], pr[D.ok.values], s=17, color=c, zorder=4,
               edgecolors=SURF, linewidths=0.8)
    ax.scatter(x[~D.ok], pr[(~D.ok).values], s=17, facecolors="none",
               edgecolors=c, linewidths=1.1, zorder=4)
    ax.set_xlim(-1, len(D))
    ax.set_ylim(-40, 1150)
    ax.tick_params(labelsize=8, colors=MUTED, length=3)
    ax.set_xticks([])
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    npre = int((pr[D.ok.values] > EX).sum()); npass = int(D.ok.sum())
    ax.text(-0.20, 0.95, name, transform=ax.transAxes, fontsize=10,
            color=INK, fontweight="bold", va="top", ha="left")
    ax.text(-0.20, 0.76,
            f"post-exilic in\n{npass-npre} of {npass} validated\nspecifications",
            transform=ax.transAxes, fontsize=7.5, color=MUTED,
            va="top", ha="left", linespacing=1.5)
    ax.text(-0.035, 1.0, "BCE", transform=ax.transAxes, fontsize=7.5,
            color=MUTED, ha="right", va="center")
    if i == 0:
        ax.text(0.5, 1065, "shaded band: pre-exilic, before 586 BCE",
                fontsize=7.5, color=MUTED, ha="left", va="center")

# ── the analytic choices, one row per level ─────────────────────────────────
axm = fig.add_subplot(gs[3])
axm.set_facecolor(SURF)
ROWS = ([(f"{s} words", D["size"] == s) for s in (400, 500, 700, 1000)]
        + [(f"keep {int(100*f)}% of features", D.frac == f) for f in (1.0, 0.75, 0.5)]
        + [("inverse-density weights", D.alpha == 1.0),
           ("unweighted", D.alpha == 0.0),
           ("variance-matched", D.calib == "var"),
           ("uncalibrated", D.calib == "none")])
for r, (lab, mask) in enumerate(ROWS):
    yy = len(ROWS) - 1 - r
    axm.scatter(x[mask.values], np.full(mask.sum(), yy), s=11,
                color=INK, alpha=0.8, zorder=3)
    axm.scatter(x[~mask.values], np.full((~mask).sum(), yy), s=4,
                color=GRID, zorder=2)
axm.set_yticks(range(len(ROWS)))
axm.set_yticklabels([l for l, _ in ROWS][::-1], fontsize=8, color=MUTED)
axm.set_xlim(-1, len(D)); axm.set_ylim(-0.7, len(ROWS) - 0.3)
axm.set_xlabel("48 specifications, ordered by genre-controlled Spearman "
               "$\\rho$ (worst to best)", fontsize=9, color=MUTED)
axm.tick_params(labelsize=8, colors=MUTED, length=0)
for s in ("top", "right", "left", "bottom"):
    axm.spines[s].set_visible(False)
for g in range(0, len(ROWS)):
    axm.axhline(g, color=GRID, lw=0.4, zorder=1, alpha=0.5)

leg = [Line2D([], [], marker="o", ls="none", color=MUTED, markersize=5,
              label="passes validation ($\\rho$ | genre > 0.20)"),
       Line2D([], [], marker="o", ls="none", markerfacecolor="none",
              markeredgecolor=MUTED, markersize=5,
              label="fails validation"),
       Line2D([], [], color=MUTED, lw=2, label="68% jackknife+ interval")]
fig.legend(handles=leg, loc="upper right", bbox_to_anchor=(0.985, 0.998),
           frameon=False, fontsize=8, ncol=3, labelcolor=MUTED,
           handletextpad=0.5, columnspacing=1.4)

fig.savefig(DH.fig("fig_speccurve.pdf"), facecolor=SURF)
fig.savefig(DH.fig("fig_speccurve.png"), dpi=190, facecolor=SURF)
print("wrote fig_speccurve.pdf and .png")
for col, name, _ in SRC:
    P = D[D.ok]
    print(f"  {name:<14} median {P[f'{col}_pred'].median():>4.0f}  "
          f"range {P[f'{col}_pred'].min():>4.0f}-{P[f'{col}_pred'].max():>4.0f}  "
          f"post-exilic {int((P[f'{col}_pred'] < EX).sum())}/{len(P)}")
