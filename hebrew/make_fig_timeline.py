"""
Figure 2: corpus and targets on one timeline.

Left column: the 25 dated units with their scholarly date and stated sigma.
Right column: the 19 undated units with conformal 68% ranges from both
model families, ordered by the generative point estimate.

Replaces fig1_corpus_timeline and fig_s12_corpus_timeline, both of which
plotted HB-VI MAP estimates produced under the prior-leakage design.
"""
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

BLUE, RED = "#2166AC", "#B2182B"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#dcdcdc"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "xtick.major.width": 0.6, "xtick.major.size": 2.5})

B = "/mnt/user-data/uploads/Diachronic Hebrew"
import json
man = json.load(open(f"{B}/hebrew/corpus_manifest_v2.json"))
NM = {"Isaiah_1":"Isaiah 1–39","Isaiah_2":"Isaiah 40–55","Isaiah_3":"Isaiah 56–66",
      "Zechariah_1":"Zechariah 1–8","Zechariah_2":"Zechariah 9–14","Daniel":"Daniel (Heb.)",
      "Jer_oracle":"Jeremiah oracle","Jer_DTR":"Jeremiah Dtr","Song_Sea":"Song of the Sea",
      "Song_Deborah":"Song of Deborah","D_source":"D source","P_source":"P source",
      "JE_source":"JE source","D_Code":"D Code","D_Frame":"D Frame","D_Song":"D Song",
      "Lev_Holiness":"Holiness Code","Lev_Priestly":"Leviticus P","Gen_JE":"Genesis JE",
      "Exo_JE":"Exodus JE","Num_JE":"Numbers JE"}
nm = lambda i: NM.get(i, i.replace("_", " "))

dated = [(t["id"], t["date_bce"], t["date_sigma"], k)
         for k in ("training", "holdouts") for t in man[k]]
dated.sort(key=lambda r: -r[1])
g = pd.read_csv("/home/claude/targets_generative.csv").set_index("id")
r = pd.read_csv("/home/claude/targets_ridge.csv").set_index("id")
tg = g.join(r, lsuffix="_g", rsuffix="_r").sort_values("point_g", ascending=False)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.5, 5.0), dpi=300,
                             gridspec_kw=dict(wspace=0.52, width_ratios=[1, 1]))
XLO, XHI = 950, -80

def bands(ax, n):
    for x0, x1, lab in [(950, 586, "pre-exilic"), (586, 539, "exilic"),
                        (539, 332, "Persian"), (332, -80, "Hellenistic")]:
        ax.axvspan(x1, x0, color=MUTED, alpha=0.055, lw=0)
        ax.axvline(x1, color=GRID, lw=0.7, zorder=0)
        ax.annotate(lab, xy=((x0+x1)/2, n - 0.25), ha="center", va="bottom",
                    fontsize=5.6, color=MUTED, rotation=0)

# ── left: dated corpus ───────────────────────────────────────────────────────
n1 = len(dated)
bands(a1, n1)
for i, (uid, d0, s, role) in enumerate(dated):
    y = n1 - 1 - i
    held = role == "holdouts"
    a1.plot([d0 - s, d0 + s], [y, y], color=MUTED, lw=1.3, solid_capstyle="round", zorder=2)
    a1.scatter([d0], [y], s=20, marker="D" if held else "o",
               facecolor="white" if held else INK, edgecolor=INK, lw=0.7, zorder=3)
a1.set_yticks(range(n1))
a1.set_yticklabels([nm(u) + ("  †" if k == "holdouts" else "")
                    for u, _, _, k in dated][::-1], fontsize=6.2)
a1.set_ylim(-0.7, n1 + 0.4); a1.set_title("A   Dated corpus (n = 25)", loc="left",
                                          fontsize=8, fontweight="bold", pad=6)

# ── right: targets ───────────────────────────────────────────────────────────
n2 = len(tg)
bands(a2, n2)
for i, (uid, row) in enumerate(tg.iterrows()):
    y = n2 - 1 - i
    a2.plot([row["lo68_g"], row["hi68_g"]], [y + 0.16, y + 0.16], color=BLUE,
            lw=2.1, alpha=0.75, solid_capstyle="round", zorder=2)
    a2.scatter([row["point_g"]], [y + 0.16], s=13, color=BLUE, zorder=3,
               edgecolor="white", lw=0.4)
    a2.plot([row["lo68_r"], row["hi68_r"]], [y - 0.16, y - 0.16], color=RED,
            lw=2.1, alpha=0.75, solid_capstyle="round", zorder=2)
    a2.scatter([row["point_r"]], [y - 0.16], s=13, color=RED, zorder=3,
               marker="^", edgecolor="white", lw=0.4)
a2.set_yticks(range(n2)); a2.set_yticklabels([nm(u) for u in tg.index][::-1], fontsize=6.2)
a2.set_ylim(-0.7, n2 + 0.4)
a2.set_title("B   Undated units, conformal 68% ranges", loc="left",
             fontsize=8, fontweight="bold", pad=6)

for ax in (a1, a2):
    ax.set_xlim(XLO, XHI); ax.set_xlabel("date (BCE)")
    for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([900, 700, 500, 300, 100]); ax.axvline(0, color=MUTED, lw=0.5, ls=":", zorder=0)

a1.legend(handles=[Line2D([], [], marker="o", color=MUTED, mfc=INK, mec=INK,
                          lw=1.3, ms=4, label="scholarly date $\\pm\\sigma$"),
                   Line2D([], [], marker="D", color="none", mfc="white", mec=INK,
                          ms=4, lw=0, label="held out of both models (†)")],
          loc="upper right", frameon=False, fontsize=5.9, handletextpad=0.4,
          borderpad=0.1, labelspacing=0.3)
a2.legend(handles=[Line2D([], [], color=BLUE, lw=2.1, marker="o", ms=3.4,
                          mec="white", label="generative"),
                   Line2D([], [], color=RED, lw=2.1, marker="^", ms=3.4,
                          mec="white", label="ridge")],
          loc="upper right", frameon=False, fontsize=5.9, handletextpad=0.4,
          borderpad=0.1, labelspacing=0.3)

fig.savefig("/home/claude/paper/figures/fig2_timeline.png", dpi=300,
            facecolor="white", bbox_inches="tight", pad_inches=0.05)
print("wrote fig2_timeline.png", f"({n1} dated, {n2} targets)")
