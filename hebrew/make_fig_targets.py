"""Figure: undated units under the recommended model."""
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, RED, GREEN = "#2166AC", "#B2182B", "#1B7837"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#dcdcdc"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "xtick.major.width": 0.6, "xtick.major.size": 2.5})

R = pd.read_csv("/home/claude/target_predictions_final.csv").set_index("unit")
NM = {"Song_Sea": "Song of the Sea", "Song_Deborah": "Song of Deborah",
      "D_Song": "Song of Moses", "JE_source": "JE source", "P_source": "P source",
      "D_source": "D source", "Gen_JE": "Genesis JE", "Exo_JE": "Exodus JE",
      "Num_JE": "Numbers JE", "D_Code": "D Code", "D_Frame": "D Frame",
      "Lev_Holiness": "Holiness Code", "Lev_Priestly": "Leviticus P",
      "Jer_DTR": "Jeremiah Dtr"}
GROUPS = [("Archaic poems", ["Song_Deborah", "Song_Sea", "D_Song"], GREEN),
          ("Documentary sources", ["JE_source", "D_source", "P_source"], BLUE),
          ("Sub-strata", ["Gen_JE", "Exo_JE", "Num_JE", "D_Code", "D_Frame",
                          "Lev_Holiness", "Lev_Priestly", "Jer_DTR"], MUTED),
          ("Pentateuch books", ["Genesis", "Exodus", "Leviticus", "Numbers",
                                "Deuteronomy"], RED)]

rows = []
for g, units, col in GROUPS:
    for u in units:
        if u in R.index: rows.append((g, u, col))
n = len(rows)
fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=300)

ax.axvspan(1000, 586, color=MUTED, alpha=0.05, lw=0)
ax.axvline(586, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.axvline(332, color=GRID, lw=0.8, zorder=0)
ax.axvspan(760, 167, color=GREEN, alpha=0.04, lw=0)
ax.annotate("586\nexile", xy=(586, n + 0.5), ha="center", va="bottom",
            fontsize=6, color=INK, linespacing=1.1)
ax.annotate("332", xy=(332, n + 0.5), ha="center", va="bottom", fontsize=6, color=MUTED)
ax.annotate("training range 760–167 BCE", xy=(463, -1.35), ha="center",
            fontsize=6, color=GREEN)

y = n - 1
labels = []
for g, u, col in rows:
    r = R.loc[u]
    ax.plot([r.hi90, r.lo90], [y, y], color=col, lw=1.0, alpha=0.35,
            solid_capstyle="round", zorder=2)
    ax.plot([r.hi68, r.lo68], [y, y], color=col, lw=2.6, alpha=0.75,
            solid_capstyle="round", zorder=3)
    ax.scatter([r.pred], [y], s=22, color=col, edgecolor="white", lw=0.6, zorder=4)
    ax.annotate(f"{r.p_post:.2f}", xy=(150, y), ha="right", va="center",
                fontsize=6, color=col)
    labels.append(NM.get(u, u))
    y -= 1
ax.annotate("$P$(post-\nexilic)", xy=(150, n + 0.3), ha="right", va="bottom",
            fontsize=6, color=MUTED, linespacing=1.1)

# group separators
yy = n - 0.5
for g, units, col in GROUPS:
    k = sum(1 for gg, u, c in rows if gg == g)
    if k == 0: continue
    ax.annotate(g, xy=(-0.34, yy - k / 2 + 0.5), xycoords=("axes fraction", "data"),
                ha="center", va="center", fontsize=6.6, color=col,
                fontweight="bold", rotation=90)
    yy -= k
    if yy > 0: ax.axhline(yy, color=GRID, lw=0.6, zorder=0)

ax.set_yticks(range(n)); ax.set_yticklabels(labels[::-1], fontsize=7)
ax.set_ylim(-1.8, n + 1.4)
ax.set_xlim(1010, 140)
ax.set_xticks([1000, 800, 600, 400, 200])
ax.set_xlabel("date (BCE)")
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("Undated units under the recommended model", loc="left",
             fontsize=9, fontweight="bold", pad=16)
ax.legend(handles=[Line2D([], [], color=MUTED, lw=2.6, label="68% conformal"),
                   Line2D([], [], color=MUTED, lw=1.0, alpha=0.4, label="90% conformal")],
          loc="lower left", frameon=False, fontsize=6.2, ncol=2,
          handletextpad=0.4, borderpad=0.1, bbox_to_anchor=(0.0, -0.13))
fig.subplots_adjust(left=0.30)
fig.savefig("/home/claude/fig_targets.png", dpi=300, facecolor="white",
            bbox_inches="tight", pad_inches=0.05)
print("wrote fig_targets.png")
