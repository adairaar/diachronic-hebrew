"""
Figure 3: is P(post-exilic) calibrated?

Panel A  Every dated unit: true date against the model's P(post-exilic),
         from a fully nested leave-one-out.  The transition should be sharp
         far from the boundary and flat near it.  The band marks where the
         P source estimate falls, i.e. the regime the paper's claim occupies.
Panel B  Reliability diagram: predicted probability against observed
         frequency, with the corpus base rate for comparison.
"""
import numpy as np, pandas as pd, matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, RED = "#2166AC", "#B2182B"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d8d8d8"
mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5})

G = pd.read_csv("/home/claude/postexilic_calibration_generative.csv")
R = pd.read_csv("/home/claude/postexilic_calibration_ridge.csv")

fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.5, 3.05), dpi=300,
                               gridspec_kw=dict(wspace=0.30, width_ratios=[1.45, 1]))

# ── Panel A ──────────────────────────────────────────────────────────────────
axA.axvspan(586, 100, color=MUTED, alpha=0.05, lw=0)
axA.axvline(586, color=INK, lw=0.9, ls=(0, (4, 3)))
axA.axvspan(344, 264, color=BLUE, alpha=0.10, lw=0)          # P source estimates
axA.annotate("P source\nestimate", xy=(304, 0.06), ha="center", va="bottom",
             fontsize=5.8, color=BLUE, linespacing=1.15)
axA.annotate("586 BCE", xy=(586, 1.03), ha="center", va="bottom",
             fontsize=6, color=INK)
axA.annotate("pre-exilic", xy=(760, 1.03), ha="center", va="bottom",
             fontsize=5.8, color=MUTED)
axA.annotate("post-exilic", xy=(330, 1.03), ha="center", va="bottom",
             fontsize=5.8, color=MUTED)

for D, col, mk, lab in [(G, BLUE, "o", "generative"), (R, RED, "^", "ridge")]:
    wrong = ((D.p_post > 0.8) & (D.actually_post == 0)) | \
            ((D.p_post < 0.2) & (D.actually_post == 1))
    axA.scatter(D.truth[~wrong], D.p_post[~wrong], s=20, marker=mk, facecolor=col,
                edgecolor="white", lw=0.5, alpha=0.9, zorder=3, label=lab)
    axA.scatter(D.truth[wrong], D.p_post[wrong], s=54, marker=mk, facecolor="none",
                edgecolor=col, lw=1.3, zorder=4, clip_on=False)
    for _, r in D[wrong].iterrows():
        axA.annotate(f"{r['id']}\n(confident, wrong)", xy=(r.truth, r.p_post),
                     xytext=(8, -4), textcoords="offset points", ha="left", va="top",
                     fontsize=5.6, color=col, linespacing=1.1)

axA.set_xlim(820, 120); axA.set_ylim(-0.04, 1.04)
axA.set_xlabel("true date (BCE)"); axA.set_ylabel("model $P$(post-exilic)")
axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
for s in ("top", "right"): axA.spines[s].set_visible(False)
axA.set_title("A   Every dated unit, nested leave-one-out", loc="left",
              fontsize=8, fontweight="bold", pad=12)
axA.legend(loc="lower left", frameon=False, fontsize=6.4, handletextpad=0.3,
           borderpad=0.1, labelspacing=0.25)

# ── Panel B ──────────────────────────────────────────────────────────────────
BINS = [(0.0, 0.4), (0.4, 0.7), (0.7, 1.01)]
axB.plot([0, 1], [0, 1], ls=(0, (4, 3)), lw=0.8, color=MUTED, zorder=1)
base = G.actually_post.mean()
axB.axhline(base, color=GRID, lw=0.8, zorder=0)
axB.annotate(f"corpus base rate {base:.2f}", xy=(0.03, base), xytext=(0, 3),
             textcoords="offset points", fontsize=5.8, color=MUTED)

for D, col, mk, lab in [(G, BLUE, "o", "generative"), (R, RED, "^", "ridge")]:
    xs, ys, ns = [], [], []
    for lo, hi in BINS:
        s = D[(D.p_post >= lo) & (D.p_post < hi)]
        if len(s):
            xs.append(s.p_post.mean()); ys.append(s.actually_post.mean()); ns.append(len(s))
    axB.plot(xs, ys, color=col, lw=1.1, alpha=0.55, zorder=2)
    axB.scatter(xs, ys, s=[14 + 5.5 * k for k in ns], marker=mk, facecolor=col,
                edgecolor="white", lw=0.6, zorder=3, label=lab)
    off = (7, -8) if lab == "generative" else (-7, 7)
    ha = "left" if lab == "generative" else "right"
    for x, y, k in zip(xs, ys, ns):
        axB.annotate(f"n={k}", xy=(x, y), xytext=off, textcoords="offset points",
                     fontsize=5.6, color=col, ha=ha)

axB.set_xlim(0, 1.02); axB.set_ylim(-0.04, 1.08)
axB.set_xlabel("mean predicted $P$(post-exilic)")
axB.set_ylabel("observed fraction post-exilic")
for s in ("top", "right"): axB.spines[s].set_visible(False)
axB.set_title("B   Reliability", loc="left", fontsize=8, fontweight="bold", pad=12)

def skill(D):
    b = D.actually_post.mean()
    return 1 - np.mean((D.p_post - D.actually_post) ** 2) / np.mean((b - D.actually_post) ** 2)
axB.annotate(f"Brier skill\n{skill(G):+.2f} / {skill(R):+.2f}", xy=(0.97, 0.06),
             ha="right", va="bottom", fontsize=6.2, color=INK, linespacing=1.2)

fig.savefig("/home/claude/paper/figures/fig3_calibration.png", dpi=300,
            facecolor="white", bbox_inches="tight", pad_inches=0.04)
print("wrote fig3_calibration.png")
for D, nm in [(G, "generative"), (R, "ridge")]:
    hi = D[D.p_post >= 0.9]
    print(f"  {nm:<12} Brier skill {skill(D):+.3f}  |  P>=0.90 calls "
          f"{hi.actually_post.sum()}/{len(hi)} correct")
