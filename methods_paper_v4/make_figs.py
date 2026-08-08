"""
All manuscript figures.

Palette: #2166AC / #D95F02 / #7570B3, validated colorblind-safe (deutan,
protan and tritan separation all pass) -- the project's previous blue/red/green
palette failed deuteranopia separation at delta-E 2.5 and has been retired.

Where four or more groups appear, identity is carried by direct labels and
physical separation rather than by hue, and color is reserved for the one
semantic contrast that matters: which side of the exile a unit falls on.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import json, os, sys
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

G = DH.GREEK


BLUE, ORANGE, PURPLE = "#2166AC", "#D95F02", "#7570B3"
INK, MUTED, GRID, SURF = "#1a1a1a", "#5a5a5a", "#dcdcdc", "white"
EXILE = 586

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelcolor": INK, "savefig.facecolor": SURF,
})


def need(p):
    if not os.path.exists(p): sys.exit(f"MISSING RESULT FILE: {p}")
    return p


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


# ══════════════════════════════════════════════════════════════════════
# Fig 1. Leave-one-book-out calibration
# ══════════════════════════════════════════════════════════════════════
B = pd.read_csv(need(DH.f("final_lobo_books.csv")))
m = json.load(open(need(DH.f("final_lobo_metrics.json"))))
q68, q90 = m["q68"], m["q90"]

fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=400)
lo, hi = 820, 120
ax.fill_between([lo, hi], [lo + q90, hi + q90], [lo - q90, hi - q90],
                color=BLUE, alpha=0.07, lw=0, zorder=0)
ax.fill_between([lo, hi], [lo + q68, hi + q68], [lo - q68, hi - q68],
                color=BLUE, alpha=0.13, lw=0, zorder=0)
ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8, ls=(0, (4, 3)), zorder=1)
ax.axvline(EXILE, color=GRID, lw=0.7, zorder=0)
ax.axhline(EXILE, color=GRID, lw=0.7, zorder=0)

correct = (B.pred > EXILE) == (B.truth > EXILE)
ax.scatter(B.truth[correct], B.pred[correct], s=17, color=BLUE,
           edgecolor="white", lw=0.5, zorder=3, label="side of exile correct")
ax.scatter(B.truth[~correct], B.pred[~correct], s=22, color=ORANGE, marker="D",
           edgecolor="white", lw=0.5, zorder=4, label="side of exile wrong")
for _, r in B.iterrows():
    if abs(r.resid) > 150 or r.book in ("Amos", "Daniel"):
        ax.annotate(r.book.replace("_", " "), xy=(r.truth, r.pred),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=5.4, color=MUTED)

ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("anchored date (BCE)"); ax.set_ylabel("estimated date (BCE)")
ax.set_xticks([800, 600, 400, 200]); ax.set_yticks([800, 600, 400, 200])
ax.set_aspect("equal")
despine(ax)
ax.legend(loc="lower right", frameon=False, fontsize=5.8, handletextpad=0.3,
          borderpad=0.4, labelspacing=0.25)
ax.set_title("Leave-one-book-out calibration", loc="left", fontsize=8.5,
             fontweight="bold", pad=8)
ax.annotate(f"MAE {m['mae']:.0f} yr   $\\rho$ = {m['rho']:+.2f}\n"
            f"bands: 68% and 90% conformal",
            xy=(0.035, 0.975), xycoords="axes fraction", ha="left", va="top",
            fontsize=5.8, color=MUTED, linespacing=1.4)
fig.savefig(DH.fig("fig_calibration.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("fig_calibration")

# ══════════════════════════════════════════════════════════════════════
# Fig 2. Undated units
# ══════════════════════════════════════════════════════════════════════
T = pd.read_csv(need(DH.f("target_predictions_final.csv"))).set_index("unit")
P = pd.read_csv(need(DH.f("poem_predictions.csv"))).set_index("unit")

NM = {"Song_Deborah": "Song of Deborah", "Song_Sea": "Song of the Sea",
      "D_Song": "Song of Moses", "JE_source": "JE composite",
      "Gen_JE": "Genesis JE", "Exo_JE": "Exodus JE", "Num_JE": "Numbers JE",
      "D_source": "D source", "D_Code": "D law code", "D_Frame": "D frame",
      "P_source": "P source", "Lev_Priestly": "Leviticus P",
      "Lev_Holiness": "Holiness Code", "Jer_DTR": "Jeremiah Dtr prose",
      "Genesis": "Genesis", "Exodus": "Exodus", "Leviticus": "Leviticus",
      "Numbers": "Numbers", "Deuteronomy": "Deuteronomy"}
GROUPS = [("Archaic poems", ["Song_Deborah", "Song_Sea", "D_Song"]),
          ("Documentary sources", ["JE_source", "D_source", "P_source"]),
          ("Sub-strata", ["Gen_JE", "Exo_JE", "Num_JE", "D_Code", "D_Frame",
                          "Lev_Holiness", "Lev_Priestly", "Jer_DTR"]),
          ("Pentateuch as received", ["Genesis", "Exodus", "Leviticus",
                                      "Numbers", "Deuteronomy"])]
# poems are shown at verse precision
POEM_SRC = {"Song_Sea": "SongSea_poem", "Song_Deborah": "SongDeborah_poem",
            "D_Song": "SongMoses_poem"}

rows = [(g, u) for g, us in GROUPS for u in us if u in T.index]
n = len(rows)
fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=400)
ax.axvspan(1000, EXILE, color=MUTED, alpha=0.045, lw=0, zorder=0)
ax.axvline(EXILE, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1)
ax.annotate("586 BCE\nexile", xy=(EXILE, n + 0.4), ha="center", va="bottom",
            fontsize=6, color=INK, linespacing=1.15)

y = n - 1
labels = []
for g, u in rows:
    if u in POEM_SRC:
        r = P.loc[POEM_SRC[u]]
        pred, lo68, hi68 = r.pred, r.lo68, r.hi68
        lo90, hi90 = pred - q90, pred + q90
        ppost = r.p_post
    else:
        r = T.loc[u]
        pred, lo68, hi68, lo90, hi90, ppost = (r.pred, r.lo68, r.hi68,
                                               r.lo90, r.hi90, r.p_post)
    col = BLUE if pred < EXILE else ORANGE
    ax.plot([hi90, lo90], [y, y], color=col, lw=1.0, alpha=0.30,
            solid_capstyle="round", zorder=2)
    ax.plot([hi68, lo68], [y, y], color=col, lw=2.6, alpha=0.80,
            solid_capstyle="round", zorder=3)
    ax.scatter([pred], [y], s=20, color=col, edgecolor="white", lw=0.6, zorder=4)
    ax.annotate(f"{ppost:.2f}", xy=(112, y), ha="right", va="center",
                fontsize=6, color=INK)
    labels.append(NM.get(u, u)); y -= 1
ax.annotate("$P$(post-\nexilic)", xy=(112, n + 0.25), ha="right", va="bottom",
            fontsize=6, color=MUTED, linespacing=1.15)

yy = n - 0.5
for g, us in GROUPS:
    k = sum(1 for gg, u in rows if gg == g)
    if not k: continue
    ax.annotate(g, xy=(-0.285, yy - k / 2 + 0.5),
                xycoords=("axes fraction", "data"), ha="center", va="center",
                fontsize=6.8, color=INK, fontweight="bold", rotation=90)
    yy -= k
    if yy > 0: ax.axhline(yy, color=GRID, lw=0.6, zorder=0)

ax.set_yticks(range(n)); ax.set_yticklabels(labels[::-1], fontsize=7)
ax.set_ylim(-1.6, n + 1.2); ax.set_xlim(1010, 100)
ax.set_xticks([1000, 800, 600, 400, 200])
ax.set_xlabel("date (BCE)")
despine(ax, keep=("bottom",))
ax.tick_params(axis="y", length=0)
ax.set_title("Undated units with conformal intervals", loc="left",
             fontsize=9, fontweight="bold", pad=14)
ax.legend(handles=[
    Line2D([], [], color=MUTED, lw=2.6, label="68% conformal"),
    Line2D([], [], color=MUTED, lw=1.0, alpha=0.35, label="90% conformal"),
    Line2D([], [], color=BLUE, lw=0, marker="o", ms=3.5, label="estimate post-exilic"),
    Line2D([], [], color=ORANGE, lw=0, marker="o", ms=3.5, label="estimate pre-exilic")],
    loc="lower left", frameon=False, fontsize=6, ncol=4, handletextpad=0.4,
    columnspacing=1.1, borderpad=0.1, bbox_to_anchor=(0.0, -0.115))
fig.subplots_adjust(left=0.27)
fig.savefig(DH.fig("fig_targets.pdf"), bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("fig_targets")

# ══════════════════════════════════════════════════════════════════════
# Fig 3. What archaizing buys (two panels: Hebrew synthetic, Greek observed)
# ══════════════════════════════════════════════════════════════════════
A = pd.read_csv(need(DH.f("archaize_results.csv")))
GA = pd.read_csv(need(DH.g("greek_atticizers.csv"))).sort_values("truth")
gm = json.load(open(need(DH.g("greek_metrics.json"))))

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=400,
                             gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.62})

# -- panel A: substitution rate vs apparent shift
dens = {}
for u in A.unit.unique():
    s = A[A.unit == u]
    dens[u] = 1000 * float(s[s.rate == 1.0].n_sub.iloc[0]) / float(s.n_words.iloc[0])
informative = [u for u in A.unit.unique() if dens[u] >= 5]
ends = []
for u in A.unit.unique():
    s = A[A.unit == u].sort_values("rate")
    base = float(s[s.rate == 0].pred.iloc[0])
    solid = u in informative
    a1.plot(s.rate, s.pred - base, lw=1.5 if solid else 0.9,
            color=BLUE if solid else MUTED, alpha=0.9 if solid else 0.40,
            marker="o" if solid else None, ms=2.6, zorder=3 if solid else 2)
    if solid:
        ends.append((float(s[s.rate == 1.0].pred.iloc[0]) - base, u))
# stagger direct labels so they cannot collide with each other
ends.sort()
prev = -99
for val, u in ends:
    ypos = max(val, prev + 3.4); prev = ypos
    a1.annotate(u, xy=(1.02, ypos), fontsize=5.8, color=BLUE, va="center",
                ha="left", zorder=5)
a1.axhline(0, color=INK, lw=0.7)
a1.set_xlim(-0.03, 1.30); a1.set_ylim(-6, 44)
a1.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
a1.set_xlabel("fraction of LBH forms replaced by CBH")
a1.set_ylabel("apparent antiquity gained (yr)")
despine(a1)
a1.spines["bottom"].set_bounds(0, 1.0)
a1.set_title("A  Hebrew: simulated archaizing", loc="left", fontsize=8,
             fontweight="bold", pad=7)
a1.annotate("solid: books with enough eligible\ntokens for the result to be informative",
            xy=(0.02, 0.97), xycoords="axes fraction", va="top", ha="left",
            fontsize=5.6, color=MUTED, linespacing=1.35)

# -- panel B: Greek Atticizer displacement
k = len(GA)
yy = np.arange(k)
XLO, XHI = -560, 160
a2.axvspan(-gm["mae"], gm["mae"], color=MUTED, alpha=0.085, lw=0, zorder=0)
a2.axvline(-gm["mae"], color=MUTED, lw=0.5, ls=(0, (2, 2)), zorder=1)
a2.axvline(0, color=INK, lw=0.8, zorder=2)
for i, (_, r) in enumerate(GA.iterrows()):
    col = PURPLE if r["shift"] < 0 else ORANGE
    a2.plot([0, r["shift"]], [yy[i], yy[i]], color=col, lw=1.4, alpha=0.75,
            solid_capstyle="round", zorder=3)
    a2.scatter([r["shift"]], [yy[i]], s=16, color=col, edgecolor="white",
               lw=0.5, zorder=4)
a2.set_yticks(yy)
WORK = {"dionysius_roman_antiquities": "Dion. Hal., Rom. Ant.",
        "dionysius_thucydides": "Dion. Hal., On Thucydides",
        "plutarch_lives": "Plutarch, Lives",
        "plutarch_moralia": "Plutarch, Moralia"}
a2.set_yticklabels(
    [f"{WORK.get(r.text, r.author)} ({int(r.truth):+d})" for _, r in GA.iterrows()],
    fontsize=5.8)
a2.set_ylim(-1.9, k + 0.4); a2.set_xlim(XLO, XHI)
a2.set_xticks([-500, -400, -300, -200, -100, 0, 100])
a2.set_xlabel("apparent date minus true date (yr)")
despine(a2, keep=("bottom",))
a2.spines["bottom"].set_bounds(XLO, XHI)
a2.tick_params(axis="y", length=0)
a2.set_title("B  Greek: historical archaizing", loc="left", fontsize=8,
             fontweight="bold", pad=7)
a2.annotate(f"shaded band: the model's own $\\pm${gm['mae']:.0f} yr error "
            f"on non-archaizing Greek",
            xy=(0.5, -0.175), xycoords="axes fraction", ha="center", va="top",
            fontsize=5.6, color=MUTED)
a2.annotate("$\\leftarrow$ looks older than it is", xy=(XLO + 15, k + 0.1),
            fontsize=5.8, color=PURPLE, va="top", ha="left")
fig.savefig(DH.fig("fig_archaize.pdf"), bbox_inches="tight", pad_inches=0.03)
plt.close(fig)
print("fig_archaize")

# ══════════════════════════════════════════════════════════════════════
# Fig 4. Where the model's leverage lives
# ══════════════════════════════════════════════════════════════════════
FAM = pd.read_csv(need(DH.f("sensitivity_families.csv")))
fam = [(r.family, r.total, int(r["count"]), r.share) for _, r in FAM.iterrows()]
fam.sort(key=lambda x: -x[3])

fig, ax = plt.subplots(figsize=(3.6, 2.5), dpi=400)
names = [f[0] for f in fam][::-1]
shares = [f[3] for f in fam][::-1]
counts = [f[2] for f in fam][::-1]
ypos = np.arange(len(names))
ax.barh(ypos, shares, height=0.62, color=BLUE, alpha=0.85,
        edgecolor="white", lw=0.8, zorder=3)
for i, (s, c) in enumerate(zip(shares, counts)):
    ax.annotate(f"{s:.1f}%  ({c} feat.)", xy=(s + 0.8, i), va="center",
                fontsize=5.8, color=INK)
ax.set_yticks(ypos); ax.set_yticklabels(names, fontsize=6.4)
ax.set_xlim(0, 62); ax.set_xlabel("share of total leverage on the date estimate")
despine(ax, keep=("bottom",))
ax.tick_params(axis="y", length=0)
ax.set_title("Where the model's leverage lives", loc="left", fontsize=8.5,
             fontweight="bold", pad=7)
fig.savefig(DH.fig("fig_leverage.pdf"), bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("fig_leverage")
print("\nfigures ->", DH.FIGURES)
