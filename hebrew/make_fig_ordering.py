"""
Figure 1: the ordering result.

Panel A  Hebrew: true vs leave-one-out predicted date, with the conformal
         68% band and the period boundaries the model cannot resolve.
Panel B  Greek: the same, showing the effect replicates and absolute
         accuracy fails harder over a longer span.
Panel C  Permutation nulls for pairwise ordering accuracy, both languages,
         with the observed values marked.

Two model families are distinguished by hue AND marker shape, so identity
never depends on colour alone.  Palette validated (CVD dE 21.1 protan).
"""
import numpy as np, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, RED = "#2166AC", "#B2182B"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d8d8d8"
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})

d = np.load("/home/claude/nulls.npz", allow_pickle=True)
fig = plt.figure(figsize=(7.5, 2.95), dpi=300)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.34,
                      left=0.065, right=0.985, top=0.87, bottom=0.17)

# ── shared scatter renderer ──────────────────────────────────────────────────
def scatter(ax, true, preds, q68, lo, hi, bounds, band, xlabel):
    """lo/hi are the axis endpoints in plot order, so time always runs
    left-to-right and bottom-to-top regardless of BCE/CE sign convention."""
    ax.fill_between([lo, hi], [lo - q68, hi - q68], [lo + q68, hi + q68],
                    color=MUTED, alpha=0.11, lw=0, zorder=0)
    ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), lw=0.8, color=MUTED, zorder=1)
    if band:                              # highlight an unresolvable window
        ax.axvspan(band[0], band[1], color=MUTED, alpha=0.16, lw=0, zorder=0)
    for b in bounds:
        ax.axvline(b, color=GRID, lw=0.6, zorder=0)
        ax.axhline(b, color=GRID, lw=0.6, zorder=0)
    for (nm, p), col, mk in [(preds[0], BLUE, "o"), (preds[1], RED, "^")]:
        ax.scatter(true, p, s=17, facecolor=col, edgecolor="white",
                   linewidth=0.5, marker=mk, alpha=0.9, zorder=3, label=nm)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel); ax.set_ylabel("predicted date")
    ax.set_aspect("equal", adjustable="box")
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# Panel A — Hebrew (BCE, axes inverted so time runs left to right)
axA = fig.add_subplot(gs[0, 0])
ht = d["heb_true"]
resid = ht - d["heb_pred_ridge"]; ar = np.sort(np.abs(resid[np.isfinite(resid)]))
qh = ar[min(int(np.ceil((len(ar)+1)*0.68))-1, len(ar)-1)]
scatter(axA, ht, [("generative", d["heb_pred_generative"]), ("ridge", d["heb_pred_ridge"])],
        qh, 900, 100, [586, 332], (586, 539), "scholarly date (BCE)")
axA.set_title("A   Hebrew  (n = 25)", loc="left", fontsize=8, fontweight="bold", pad=9)
axA.annotate("exilic\n(47 yr)", xy=(562, 168), ha="center", va="center",
             fontsize=5.8, color=MUTED, linespacing=1.15)
axA.annotate("586", xy=(586, 858), ha="center", va="center", fontsize=5.8, color=MUTED,
             bbox=dict(fc="white", ec="none", pad=0.8))
axA.annotate("332", xy=(332, 858), ha="center", va="center", fontsize=5.8, color=MUTED,
             bbox=dict(fc="white", ec="none", pad=0.8))
axA.set_xticks([800, 600, 400, 200])

# Panel B — Greek (CE)
axB = fig.add_subplot(gs[0, 1])
gt = d["grk_true"]
rg = gt - d["grk_pred_ridge"]; ag = np.sort(np.abs(rg[np.isfinite(rg)]))
qg = ag[min(int(np.ceil((len(ag)+1)*0.68))-1, len(ag)-1)]
scatter(axB, gt, [("generative", d["grk_pred_generative"]), ("ridge", d["grk_pred_ridge"])],
        qg, -500, 500, [-323, 0], None, "established date (CE)")
axB.set_title("B   Greek  (n = 63)", loc="left", fontsize=8, fontweight="bold", pad=9)
for b, lab in [(-323, "323 BCE"), (0, "1 CE")]:
    axB.annotate(lab, xy=(b, 470), xytext=(0, -8), textcoords="offset points",
                 ha="center", va="top", fontsize=5.8, color=MUTED)
leg = axB.legend(loc="lower right", frameon=True, fontsize=6.4, handletextpad=0.3,
                 borderpad=0.35, labelspacing=0.25)
leg.get_frame().set_facecolor("white"); leg.get_frame().set_edgecolor("none")
leg.get_frame().set_alpha(0.88)

# Panel C — permutation nulls for pairwise ordering
axC = fig.add_subplot(gs[0, 2])
rows = [("Hebrew, generative", d["heb_null_generative"], float(d["heb_obs_generative"][0]), BLUE),
        ("Hebrew, ridge",      d["heb_null_ridge"],      float(d["heb_obs_ridge"][0]),      BLUE),
        ("Greek, generative",  d["grk_null_generative"], float(d["grk_obs_generative"][0]), RED),
        ("Greek, ridge",       d["grk_null_ridge"],      float(d["grk_obs_ridge"][0]),      RED)]
for i, (lab, null, obs, col) in enumerate(rows):
    y = len(rows) - 1 - i
    q = np.percentile(null, [2.5, 25, 75, 97.5])
    axC.plot([q[0], q[3]], [y, y], color=GRID, lw=1.1, solid_capstyle="round", zorder=1)
    axC.plot([q[1], q[2]], [y, y], color=MUTED, lw=3.4, solid_capstyle="round",
             alpha=0.45, zorder=2)
    axC.plot([null.mean()], [y], marker="|", ms=7, color=MUTED, mew=1.0, zorder=3)
    axC.scatter([obs], [y], s=34, facecolor=col, edgecolor="white", lw=0.6,
                zorder=5, clip_on=False)
    p = (np.sum(null >= obs) + 1) / (len(null) + 1)
    axC.annotate(f"$p$ = {p:.3f}", xy=(obs, y), xytext=(7, -0.5),
                 textcoords="offset points", fontsize=6.0, color=INK,
                 ha="left", va="center")
axC.axvline(0.5, color=MUTED, ls=(0, (4, 3)), lw=0.8, zorder=0)
axC.set_yticks(range(len(rows))); axC.set_yticklabels([r[0] for r in rows][::-1], fontsize=6.6)
axC.set_ylim(-0.5, len(rows) - 0.15)
axC.set_xlim(0.18, 0.86); axC.set_xlabel("pairwise ordering accuracy")
axC.set_title("C   Observed vs permutation null", loc="left", fontsize=8,
              fontweight="bold", pad=9)
for s in ("top", "right", "left"): axC.spines[s].set_visible(False)
axC.tick_params(axis="y", length=0)
axC.annotate("chance", xy=(0.5, len(rows) - 0.45), ha="center", va="center",
             fontsize=6.0, color=MUTED)

fig.savefig("/home/claude/paper/figures/fig1_ordering.png", dpi=300,
            facecolor="white", bbox_inches="tight", pad_inches=0.04)
print("wrote fig1_ordering.png")
for lab, null, obs, _ in rows:
    print(f"  {lab:<22} obs {obs*100:5.1f}%   null {null.mean()*100:5.1f}% "
          f"+/-{null.std()*100:4.1f}   p={(np.sum(null>=obs)+1)/(len(null)+1):.4f}")
