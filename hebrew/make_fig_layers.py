"""
Figure: the internal structure of each source.

One row per unit, one dot per ~500-word passage, so the reader sees the actual
distribution rather than a summary of it.  Sources carry their sub-blocks
directly beneath them, which is where the layering shows: each block is
internally tight while the blocks sit a century apart.

The reference band is the middle half of the within-book dispersions measured on
the dated anchors, so "is this source unusually varied" can be read off the page
instead of taken on trust.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import numpy as np, pandas as pd, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE, PURPLE = "#2166AC", "#D95F02", "#7570B3"
INK, MUTED, GRID, SURF = "#1a1a1a", "#6b6b6b", "#d8d8d4", "#fcfcfb"
EX = 586
RNG = np.random.default_rng(5)

P = pd.read_csv(DH.f("chunk_preds.csv"))
T = P[P.kind == "target"]
A = P[P.kind == "anchor"]
C = pd.read_csv(DH.f("internal_consistency.csv")).set_index("unit")

ROWS = [
    ("JE_source", "JE composite", BLUE, 0),
    ("Gen_JE", "Genesis JE", BLUE, 1),
    ("Exo_JE", "Exodus JE", BLUE, 1),
    ("Num_JE", "Numbers JE", BLUE, 1),
    (None, None, None, None),
    ("D_source", "D source", ORANGE, 0),
    ("D_Code", "law code, Deut 12–26", ORANGE, 1),
    ("D_Frame", "frame, Deut 1–11 + 27–34", ORANGE, 1),
    (None, None, None, None),
    ("P_source", "P source", PURPLE, 0),
    ("Lev_Priestly", "Leviticus 1–16", PURPLE, 1),
    ("Lev_Holiness", "Holiness Code, Lev 17–26", PURPLE, 1),
]

fig, ax = plt.subplots(figsize=(7.6, 5.4))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
fig.subplots_adjust(left=0.30, right=0.90, top=0.90, bottom=0.115)

ax.axvspan(EX, 1400, color=GRID, alpha=0.35, lw=0, zorder=0)
ax.axvline(EX, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)

ylabels, ypos = [], []
y = len(ROWS)
for u, lab, c, indent in ROWS:
    y -= 1
    if u is None:
        continue
    v = T.cpred[T.unit == u].values
    if len(v) == 0:
        continue
    jit = RNG.uniform(-0.17, 0.17, len(v))
    ax.scatter(v, y + jit, s=13, color=c, alpha=0.40 if indent else 0.55,
               edgecolors="none", zorder=3)
    q1, md, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([q1, q3], [y, y], color=c, lw=3.2, solid_capstyle="butt",
            alpha=0.95, zorder=4)
    ax.plot([md, md], [y - 0.30, y + 0.30], color=SURF, lw=3.4, zorder=5)
    ax.plot([md, md], [y - 0.27, y + 0.27], color=INK, lw=1.6, zorder=6)
    ax.text(1005, y, f"{C.loc[u,'sd']:.0f}", fontsize=8, color=MUTED,
            va="center", ha="right")
    ax.text(1075, y, f"{int(C.loc[u,'n'])}", fontsize=8, color=MUTED,
            va="center", ha="right")
    ylabels.append(("      " if indent else "") + lab); ypos.append(y)

ax.set_yticks(ypos); ax.set_yticklabels(ylabels, fontsize=9, color=INK)
for t, (u, lab, c, ind) in zip(ax.get_yticklabels(),
                               [r for r in ROWS if r[0] is not None]):
    if ind: t.set_color(MUTED); t.set_fontsize(8.5)
ax.set_ylim(-1.45, len(ROWS) + 0.15)
ax.set_xlim(-40, 1090)
ax.set_xticks([0, 200, 400, 600, 800, 1000])
ax.set_xlabel("estimated date of each ~500-word passage (BCE)", fontsize=9,
              color=MUTED)
ax.tick_params(labelsize=8, colors=MUTED, length=3)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.text(1005, len(ROWS) - 0.25, "SD", fontsize=8, color=INK, ha="right",
        va="center", fontweight="bold")
ax.text(1075, len(ROWS) - 0.25, "n", fontsize=8, color=INK, ha="right",
        va="center", fontweight="bold")
ax.text(600, len(ROWS) - 0.25, "shaded: pre-exilic", fontsize=8,
        color=MUTED, ha="left", va="center")

# the anchor yardstick, drawn as a scale bar rather than asserted in prose
ref = np.array([A.cpred[A.unit == b].std(ddof=1)
                for b in A.unit.unique() if (A.unit == b).sum() >= 4])
lo, hi = np.percentile(ref, [25, 75])
ax.text(-30, -1.05,
        f"within-book SD of the 17 dated anchor books: "
        f"median {np.median(ref):.0f} yr (middle half {lo:.0f}–{hi:.0f})",
        fontsize=8, color=MUTED, ha="left", va="center")

leg = [Line2D([], [], marker="o", ls="none", color=MUTED, markersize=5,
              alpha=.6, label="one passage"),
       Line2D([], [], color=MUTED, lw=3.2, label="interquartile range"),
       Line2D([], [], color=INK, lw=1.6, label="median")]
ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.0, 1.12),
          frameon=False, fontsize=8, ncol=3, labelcolor=MUTED,
          handletextpad=0.6, columnspacing=1.6)

fig.savefig(DH.fig("fig_layers.pdf"), facecolor=SURF)
fig.savefig(DH.fig("fig_layers.png"), dpi=190, facecolor=SURF)
print("wrote fig_layers.pdf and .png")
