#!/usr/bin/env python3
"""
consolidate_v2.py — single master results table + comparison figure (corpus v2)
===============================================================================
Merges every v2 model output into one authoritative table so the manuscript can
be rebuilt from a single file, and draws the HB-VI vs MLE-MVN comparison.

Inputs
  results_v2/dating_v3.csv            pre-specified MLE-MVN (13 features)
  results_v2/hard_register_v2.csv     register-conditioned MLE-MVN (degenerate;
                                      retained for the comparison figure only)
  results_v2/archaism_genre_v2.csv    archaism index + genre correction
  hierarchical_bayes/results_v2/hb_vi_dating.csv
  hierarchical_bayes/results_v2/prior_sensitivity.csv

Outputs
  results_v2/master_results_v2.csv
  results_v2/fig_model_comparison_v2.png
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R    = os.path.join(HERE, "results_v2")
HB   = os.path.join(HERE, "hierarchical_bayes", "results_v2")


def load(path, idx, cols):
    if not os.path.exists(path):
        print(f"  [missing] {path}")
        return pd.DataFrame()
    d = pd.read_csv(path)
    if idx not in d.columns:
        d = d.rename(columns={d.columns[0]: idx})
    d = d.set_index(idx)
    return d[[c for c in cols if c in d.columns]]


def main():
    mvn = load(os.path.join(R, "dating_v3.csv"), "unit",
               ["n_words", "role", "scholarly", "map_bce", "ci68_lo", "ci68_hi",
                "ci68_width", "map_resist", "delta_arch"])
    mvn = mvn.rename(columns={"map_bce": "mvn_map", "ci68_lo": "mvn_lo",
                              "ci68_hi": "mvn_hi", "ci68_width": "mvn_width"})

    ag = load(os.path.join(R, "archaism_genre_v2.csv"), "unit",
              ["map_genre", "genre_shift", "lbh_score", "lbh_class"])

    hr = load(os.path.join(R, "hard_register_v2.csv"), "unit",
              ["map_bestgroup", "best_group"])
    hr = hr.rename(columns={"map_bestgroup": "reg_mvn_map"})

    hb = load(os.path.join(HB, "hb_vi_dating.csv"), "id",
              ["group", "hb_map_bce", "hb_ci68_lo_ce", "hb_ci68_hi_ce",
               "hb_post_std", "hb_scholar_in68"])
    if not hb.empty:
        hb["hb_lo"] = -hb["hb_ci68_hi_ce"]
        hb["hb_hi"] = -hb["hb_ci68_lo_ce"]
        hb = hb.drop(columns=["hb_ci68_lo_ce", "hb_ci68_hi_ce"])
        hb = hb.rename(columns={"hb_map_bce": "hb_map"})

    ps = load(os.path.join(HB, "prior_sensitivity.csv"), "id",
              ["map_A", "map_B", "map_C", "shift_AB", "verdict"])
    ps = ps.rename(columns={"map_A": "ps_modeA", "map_B": "ps_modeB",
                            "map_C": "ps_lik_only", "shift_AB": "ps_shift",
                            "verdict": "ps_verdict"})

    for _d in (mvn, ag, hr, hb, ps):
        if not _d.empty:
            _d.drop(index=_d.index[_d.index.duplicated()], inplace=True)
    m = mvn.join([ag, hr, hb, ps], how="outer")
    m = m[~m.index.duplicated(keep="first")]
    m.index.name = "unit"
    m.to_csv(os.path.join(R, "master_results_v2.csv"))

    # ── Console summary ───────────────────────────────────────────────────────
    def fmt(v, w=5):
        return f"{v:>{w}.0f}" if pd.notna(v) else " " * (w - 1) + "-"

    print("MASTER RESULTS — corpus v2   (all dates BCE)\n")
    print(f"{'unit':15s}{'schol':>7s}{'MVN':>7s}{'68% CI':>15s}"
          f"{'HB-VI':>7s}{'genre':>7s}{'resist':>8s}{'Darch':>7s}  prior-sens")
    print("-" * 92)

    order = (["Jer_oracle", "Haggai", "Habakkuk", "Daniel", "--",
              "P_source", "JE_source", "D_source", "--",
              "D_Code", "D_Frame", "Lev_Priestly", "Lev_Holiness", "Jer_DTR", "--",
              "Song_Sea", "Song_Deborah", "D_Song"])
    for u in order:
        if u == "--":
            print("-" * 92); continue
        if u not in m.index:
            continue
        r = m.loc[u]
        ci = (f"[{r['mvn_lo']:.0f}–{r['mvn_hi']:.0f}]"
              if pd.notna(r.get("mvn_lo")) else "-")
        print(f"{u:15s}{fmt(r.get('scholarly'),7)}{fmt(r.get('mvn_map'),7)}"
              f"{ci:>15s}{fmt(r.get('hb_map'),7)}{fmt(r.get('map_genre'),7)}"
              f"{fmt(r.get('map_resist'),8)}{fmt(r.get('delta_arch'),7)}"
              f"  {r.get('ps_verdict','') if pd.notna(r.get('ps_verdict')) else ''}")

    # ── Holdout comparison ────────────────────────────────────────────────────
    hold = [u for u in ["Habakkuk", "Jer_oracle", "Haggai", "Daniel"] if u in m.index]
    if hold:
        print("\n\nHOLDOUT COMPARISON (never trained by either model)\n")
        print(f"{'text':14s}{'scholarly':>11s}{'HB-VI':>8s}{'err':>7s}"
              f"{'reg-MVN':>10s}{'err':>7s}{'presp-MVN':>11s}{'err':>7s}")
        print("-" * 76)
        eh, er, ep = [], [], []
        for u in hold:
            r = m.loc[u]; s = r["scholarly"]
            for col, acc in (("hb_map", eh), ("reg_mvn_map", er), ("mvn_map", ep)):
                if pd.notna(r.get(col)):
                    acc.append(abs(r[col] - s))
            print(f"{u:14s}{s:>11.0f}{fmt(r.get('hb_map'),8)}"
                  f"{fmt(r.get('hb_map')-s if pd.notna(r.get('hb_map')) else np.nan,7)}"
                  f"{fmt(r.get('reg_mvn_map'),10)}"
                  f"{fmt(r.get('reg_mvn_map')-s if pd.notna(r.get('reg_mvn_map')) else np.nan,7)}"
                  f"{fmt(r.get('mvn_map'),11)}"
                  f"{fmt(r.get('mvn_map')-s if pd.notna(r.get('mvn_map')) else np.nan,7)}")
        print("-" * 76)
        print(f"{'MAE':14s}{'':>11s}{np.mean(eh):>8.1f}{'':>7s}"
              f"{np.mean(er):>10.1f}{'':>7s}{np.mean(ep):>11.1f}")
        print("\n  Caution: HB-VI holdout priors are tight (sigma 10-30 yr) and the")
        print("  likelihood-only column in prior_sensitivity.csv is far from these")
        print("  MAPs, so the low HB-VI MAE is substantially prior-carried.")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    tr = m[(m["role"] == "train") & m["scholarly"].notna()]
    for col, lbl, c, mk in (("hb_map", "HB-VI", "#1565C0", "o"),
                            ("mvn_map", "MLE-MVN (pre-specified)", "#E65100", "^")):
        s = tr.dropna(subset=[col]) if col in tr else pd.DataFrame()
        if len(s):
            ax.scatter(s["scholarly"], s[col], c=c, marker=mk, s=45,
                       alpha=.85, label=lbl, zorder=3)
    lo, hi = 130, 800
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect", zorder=1)
    for u in hold:
        if u in m.index and pd.notna(m.loc[u, "scholarly"]):
            r = m.loc[u]
            ax.scatter(r["scholarly"], r.get("hb_map"), facecolors="none",
                       edgecolors="#1565C0", s=170, lw=2, zorder=4)
            ax.annotate(u, (r["scholarly"], r.get("hb_map")), fontsize=7,
                        xytext=(5, 5), textcoords="offset points")
    ax.set_xlim(hi, lo); ax.set_ylim(hi, lo)
    ax.set_xlabel("Scholarly date (BCE)"); ax.set_ylabel("Model MAP (BCE)")
    ax.set_title("A — Training fit; circled = holdouts", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1]
    tgt = ["P_source", "JE_source", "D_source", "D_Code", "D_Frame",
           "Lev_Priestly", "Lev_Holiness", "Song_Sea", "Jer_DTR"]
    tgt = [t for t in tgt if t in m.index]
    y = np.arange(len(tgt))
    for off, col, lbl, c in ((-0.22, "mvn_map", "MLE-MVN", "#E65100"),
                             (0.0, "hb_map", "HB-VI", "#1565C0"),
                             (0.22, "map_genre", "genre-corrected", "#2E7D32")):
        v = [m.loc[t].get(col, np.nan) for t in tgt]
        ax.scatter(v, y + off, c=c, s=42, label=lbl, zorder=3)
    for i, t in enumerate(tgt):
        r = m.loc[t]
        if pd.notna(r.get("mvn_lo")):
            ax.plot([r["mvn_lo"], r["mvn_hi"]], [i - 0.22] * 2,
                    c="#E65100", lw=1.4, alpha=.5, zorder=2)
    ax.axvline(586, color="k", ls=":", lw=1)
    ax.text(586, len(tgt) - .3, " exile", fontsize=7, va="top")
    ax.set_yticks(y); ax.set_yticklabels(tgt, fontsize=8)
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_xlabel("Date (BCE)")
    ax.set_title("B — Targets across models", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="x")

    plt.tight_layout()
    p = os.path.join(R, "fig_model_comparison_v2.png")
    plt.savefig(p, dpi=300, bbox_inches="tight"); plt.close()
    print(f"\nSaved → {R}/master_results_v2.csv")
    print(f"Saved → {p}")


if __name__ == "__main__":
    main()
