"""
02_compare.py
=============
Side-by-side comparison: Hierarchical Bayes VI  vs.  MLE-MVN (Script 10).

Metrics reported
----------------
  * MAP date error vs. scholarly date (all texts + holdouts only)
  * 68% CI width (narrower = more confident)
  * 68% CI coverage of the scholarly date
  * MAE and RMSE on holdouts
  * Per-text delta: HB_MAP − MLE_MAP

Produces
--------
  results/comparison_table.csv      — combined per-text table
  results/comparison_summary.txt    — printed summary
  results/comparison_plot.png       — scatter plot + coverage bars
"""

import os
import sys
import json
import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

HERE      = os.path.dirname(os.path.abspath(__file__))
PARENT    = os.path.dirname(HERE)
HB_CSV    = os.path.join(HERE,   "results", "hb_vi_dating.csv")
MLE_CSV   = os.path.join(PARENT, "results", "hard_register_dating.csv")
ELBO_FILE = os.path.join(HERE,   "results", "hb_vi_elbo.txt")
RESULTS   = os.path.join(HERE,   "results")


def ce(year: float) -> str:
    y = int(round(year))
    return f"{abs(y)} BCE" if y < 0 else f"{y} CE"


def rmse(errors):
    return float(np.sqrt(np.mean(np.array(errors) ** 2)))


def main():
    # ── Load both result tables ───────────────────────────────────────────────
    if not os.path.exists(HB_CSV):
        print(f"Missing {HB_CSV}\nRun 01_hb_vi_dating.py first.")
        sys.exit(1)
    if not os.path.exists(MLE_CSV):
        print(f"Missing {MLE_CSV}\nRun ../10_hard_register_dating.py first.")
        sys.exit(1)

    hb  = pd.read_csv(HB_CSV).set_index("id")
    mle = pd.read_csv(MLE_CSV).set_index("id")

    # Align on common texts
    common = hb.index.intersection(mle.index)
    hb  = hb.loc[common]
    mle = mle.loc[common]

    # ── Build combined table ──────────────────────────────────────────────────
    combined = pd.DataFrame({
        "group"         : hb["group"],
        "holdout"       : hb["holdout"].astype(bool),
        "scholarly_date": hb["scholarly_date"],
        "hb_map"        : hb["hb_map"],
        "hb_ci68_lo"    : hb["hb_ci68_lo"],
        "hb_ci68_hi"    : hb["hb_ci68_hi"],
        "hb_ci68_width" : hb["hb_ci68_hi"] - hb["hb_ci68_lo"],
        "hb_post_std"   : hb["hb_post_std"],
        "hb_in68"       : hb["hb_scholar_in68"].astype(bool),
        "mle_map"       : mle["map_date"],
        "mle_ci68_lo"   : mle["ci68_lo"],
        "mle_ci68_hi"   : mle["ci68_hi"],
        "mle_ci68_width": mle["ci68_hi"] - mle["ci68_lo"],
        "mle_in68"      : mle["scholar_in_68"].astype(bool),
        "hb_error"      : hb["hb_map"] - hb["scholarly_date"],
        "mle_error"     : mle["map_date"] - hb["scholarly_date"],
        "delta_map"     : hb["hb_map"] - mle["map_date"],   # HB minus MLE
    })

    combined.to_csv(os.path.join(RESULTS, "comparison_table.csv"))

    lines = []
    bar = "=" * 70

    lines.append(bar)
    lines.append("COMPARISON: Hierarchical Bayes VI  vs.  MLE-MVN (Script 10)")
    lines.append(bar)

    # ── Overall metrics (all texts) ───────────────────────────────────────────
    for label, subset in [("ALL TEXTS", combined), ("HOLDOUTS ONLY", combined[combined["holdout"]])]:
        n = len(subset)
        lines.append(f"\n── {label} (n={n}) ──────────────────────────────────────────────")

        for method, map_col, in68_col, width_col, err_col in [
            ("HB-VI  ", "hb_map",  "hb_in68",  "hb_ci68_width", "hb_error"),
            ("MLE-MVN", "mle_map", "mle_in68", "mle_ci68_width", "mle_error"),
        ]:
            errs    = subset[err_col].values
            widths  = subset[width_col].values
            coverage = subset[in68_col].mean()
            mae   = float(np.mean(np.abs(errs)))
            rmse_ = rmse(errs)
            w_med = float(np.median(widths))
            lines.append(
                f"  {method}:  MAE={mae:6.1f} yr  RMSE={rmse_:6.1f} yr  "
                f"68%-cov={coverage:.0%}  median CI width={w_med:.0f} yr"
            )

    # ── Per-text holdout comparison ───────────────────────────────────────────
    lines.append(f"\n── PER-HOLDOUT DETAIL ──────────────────────────────────────────────")
    holdouts = combined[combined["holdout"]].sort_values("scholarly_date")

    fmt = "{:45s}  {:>8s}  {:>10s} {:>4s}  {:>10s} {:>4s}  {:>8s}"
    lines.append(fmt.format(
        "Text", "Scholarly",
        "HB MAP", "In?",
        "MLE MAP", "In?",
        "HB-MLE"))
    lines.append("─" * 105)
    for eid, row in holdouts.iterrows():
        hb_in  = "✓" if row["hb_in68"]  else "✗"
        mle_in = "✓" if row["mle_in68"] else "✗"
        delta  = row["delta_map"]
        delta_s = f"{delta:+.0f}yr"
        lines.append(fmt.format(
            eid,
            ce(row["scholarly_date"]),
            ce(row["hb_map"]),  hb_in,
            ce(row["mle_map"]), mle_in,
            delta_s,
        ))

    # ── CI width comparison ───────────────────────────────────────────────────
    lines.append(f"\n── CI WIDTH COMPARISON (median, all texts by group) ─────────────────")
    for grp in combined["group"].unique():
        sub = combined[combined["group"] == grp]
        hb_w  = sub["hb_ci68_width"].median()
        mle_w = sub["mle_ci68_width"].median()
        ratio = hb_w / mle_w if mle_w > 0 else float("nan")
        lines.append(f"  {grp:12s}:  HB={hb_w:.0f} yr   MLE={mle_w:.0f} yr   ratio={ratio:.2f}")

    lines.append(f"\n── INTERPRETATION ───────────────────────────────────────────────────")
    lines.append(
        "  HB-VI propagates uncertainty in (α, β, σ) into date posteriors.\n"
        "  Wider CI vs. MLE = model correctly inflating uncertainty where\n"
        "  the regression parameters themselves are uncertain.\n"
        "  Narrower CI = HB regularised noisy features, tightening inference.\n"
        "  68% CI coverage near 68% → well-calibrated posterior uncertainty."
    )

    summary_text = "\n".join(lines)
    print(summary_text)

    out_txt = os.path.join(RESULTS, "comparison_summary.txt")
    with open(out_txt, "w") as fh:
        fh.write(summary_text + "\n")
    print(f"\nSummary → {out_txt}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not HAS_PLT:
        print("matplotlib not available — skipping plots.")
        return

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    colors = {"classical": "#2196F3", "atticizing": "#FF9800", "koine": "#4CAF50"}
    marker_style = {"training": "o", "holdout": "D"}

    # ── Panel A: Scholarly vs MAP (both methods) ──────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.set_title("A — Scholarly date vs. MAP date: HB-VI (filled) and MLE-MVN (open)",
                   fontsize=11)

    for eid, row in combined.iterrows():
        col = colors.get(row["group"], "grey")
        mk  = "D" if row["holdout"] else "o"
        sz  = 100 if row["holdout"] else 60
        # HB
        ax_a.scatter(row["scholarly_date"], row["hb_map"],
                     color=col, marker=mk, s=sz, zorder=5,
                     label=f"{row['group']} HB" if eid == combined[combined["group"]==row["group"]].index[0] else "")
        # MLE (open)
        ax_a.scatter(row["scholarly_date"], row["mle_map"],
                     facecolors="none", edgecolors=col, marker=mk, s=sz, linewidths=1.5, zorder=4)
        # Error bar (HB)
        ax_a.plot([row["scholarly_date"]] * 2,
                  [row["hb_ci68_lo"], row["hb_ci68_hi"]],
                  color=col, lw=0.8, alpha=0.5, zorder=3)

    lo, hi = combined["scholarly_date"].min() - 50, combined["scholarly_date"].max() + 50
    ax_a.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect")
    ax_a.set_xlabel("Scholarly date (CE)")
    ax_a.set_ylabel("Model MAP date (CE)")
    ax_a.axhline(0, color="grey", lw=0.5, ls=":")
    ax_a.axvline(0, color="grey", lw=0.5, ls=":")
    ax_a.legend(fontsize=8, ncol=3)

    # ── Panel B: Error distribution ───────────────────────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    ax_b.set_title("B — MAP error distribution (HB blue, MLE orange)", fontsize=11)
    bins = np.linspace(-300, 300, 25)
    ax_b.hist(combined["hb_error"],  bins=bins, alpha=0.6, color="#2196F3", label="HB-VI")
    ax_b.hist(combined["mle_error"], bins=bins, alpha=0.6, color="#FF9800", label="MLE-MVN")
    ax_b.axvline(0, color="k", lw=1, ls="--")
    ax_b.set_xlabel("MAP − Scholarly date (yr)")
    ax_b.set_ylabel("Count")
    ax_b.legend(fontsize=9)

    # ── Panel C: CI width comparison ─────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.set_title("C — 68% CI width: HB-VI vs. MLE-MVN (all texts)", fontsize=11)
    for _, row in combined.iterrows():
        mk = "D" if row["holdout"] else "o"
        ax_c.scatter(row["mle_ci68_width"], row["hb_ci68_width"],
                     c=colors.get(row["group"], "grey"),
                     marker=mk, s=70, alpha=0.8, zorder=5)
    lim_lo = min(combined["mle_ci68_width"].min(), combined["hb_ci68_width"].min()) - 5
    lim_hi = max(combined["mle_ci68_width"].max(), combined["hb_ci68_width"].max()) + 5
    ax_c.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label="Equal width")
    ax_c.set_xlabel("MLE-MVN 68% CI width (yr)")
    ax_c.set_ylabel("HB-VI 68% CI width (yr)")
    ax_c.legend(fontsize=9)

    # Colour legend patch
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_els = ([Patch(facecolor=c, label=g) for g, c in colors.items()] +
                  [Line2D([0],[0], marker="D", color="w", markerfacecolor="grey",
                          markersize=8, label="Holdout"),
                   Line2D([0],[0], marker="o", color="w", markerfacecolor="grey",
                          markersize=8, label="Training")])
    fig.legend(handles=legend_els, loc="lower center",
               ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.01))

    # Add ELBO inset to Panel A if available
    if os.path.exists(ELBO_FILE):
        elbo = np.loadtxt(ELBO_FILE)
        ax_inset = ax_a.inset_axes([0.75, 0.05, 0.22, 0.35])
        ax_inset.plot(elbo, lw=0.8, color="#1565C0")
        ax_inset.set_xlabel("Iteration", fontsize=7)
        ax_inset.set_ylabel("ELBO", fontsize=7)
        ax_inset.set_title("Training curve", fontsize=7)
        ax_inset.tick_params(labelsize=6)

    plt.suptitle(
        "Hierarchical Bayes VI  vs.  MLE-MVN: Greek diachronic dating\n"
        "(filled = HB-VI, open = MLE-MVN; diamonds = holdout texts)",
        fontsize=12, y=1.01)

    out_png = os.path.join(RESULTS, "comparison_plot.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot → {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
