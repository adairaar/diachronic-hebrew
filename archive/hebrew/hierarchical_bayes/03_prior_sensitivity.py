"""
03_prior_sensitivity.py
=======================
Prior sensitivity analysis for the Hebrew HB-VI dating model.

Motivation
----------
After fitting (01_hb_vi_dating.py), every date posterior combines:

    p(d | x)  ∝  likelihood(x | d)  ×  prior(d)

If the prior is tight (σ = 30–80 yr), the posterior MAP can be dominated
by the scholarly opinion encoded in that prior rather than by the linguistic
data.  This script loads the already-fitted VI parameters and re-dates all
sub-sources, unknown-register texts, and holdouts under:

  (A) Scholarly prior  — the original, literature-derived N(μ_scholar, σ_scholar²)
  (B) Agnostic prior   — N(575 BCE, 400²):  essentially flat over the
                          training range [760–330 BCE] with ±2σ reaching
                          [175–975] BCE — wide enough that it imposes no
                          meaningful preference within the plausible range.

Comparing (A) vs (B) reveals which results are data-driven (shift < 30 yr)
and which are prior-dominated (shift > 80 yr).

It also reports the *likelihood-only* MAP: the date that maximises the
log-likelihood alone, with no prior correction.  This is the most direct
answer to "what do the linguistic features say, ignoring all prior belief?"

Outputs
-------
  results/prior_sensitivity.csv   — full per-text table
  results/prior_sensitivity.txt   — printed summary
  results/prior_sensitivity.png   — dot-plot comparing the two estimates
"""

from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp as lse

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

HERE     = os.path.dirname(os.path.abspath(__file__))
PARENT   = os.path.dirname(HERE)
PARAMS   = os.path.join(HERE, "results", "hb_vi_params.npz")
FEAT_CSV = os.path.join(PARENT, "data", "feature_matrix.csv")
EXTR_CSV = os.path.join(HERE,  "results", "extracted_features.csv")
RESULTS  = os.path.join(HERE,  "results")

# Agnostic prior: wide Gaussian roughly flat over the whole training range
AGNOSTIC_CENTER_BCE = 575     # midpoint of training range (~760–330 BCE)
AGNOSTIC_SIGMA_BCE  = 400     # ±2σ → [175, 975] BCE

N_MC_INFER = 200
N_GRID     = 1000

GROUPS = ["SBH", "Transitional", "LBH"]

SUBSOURCE_META = {
    "Jer_oracle"   : (605,  30, "SBH",          "Jer oracles (non-Dtr)"),
    "Jer_DTR"      : (570,  60, "Transitional",  "Jer Dtr prose"),
    "D_Code"       : (620,  40, "SBH",           "Deut 12-26 (code)"),
    "D_Frame"      : (620,  40, "SBH",           "Deut framing"),
    "D_Song"       : (900, 300, "SBH",           "Song of Moses (Deut 32)"),
    "Lev_Holiness" : (580,  80, "SBH",           "Holiness Code (Lev 17-26)"),
    "Lev_Priestly" : (520,  80, "SBH",           "Priestly (non-H)"),
    "Song_Sea"     : (1100, 300,"SBH",           "Song of the Sea (Ex 15)"),
    "Song_Deborah" : (1150, 200,"SBH",           "Song of Deborah (Judg 5)"),
}

# Main-matrix holdouts and unknown-register texts
HOLDOUT_META = {
    "Habakkuk": (605, 20, "SBH"),
    "Haggai":   (520,  5, "Transitional"),
    "Daniel":   (167, 10, "LBH"),
}
UNKNOWN_META = {
    "D_source":  (625, 100, "LBH"),   # group assigned by best-group in script 01
    "P_source":  (600, 100, "LBH"),
    "JE_source": (800,  75, "LBH"),
}


# ══════════════════════════════════════════════════════════════════════════════
# VI helpers (duplicated from 01 to keep this script self-contained)
# ══════════════════════════════════════════════════════════════════════════════

def unpack(phi, R, F):
    RF = R * F; i = 0
    m_a    = phi[i:i+RF].reshape(R, F); i += RF
    rho_a  = phi[i:i+RF].reshape(R, F); i += RF
    m_b    = phi[i:i+RF].reshape(R, F); i += RF
    rho_b  = phi[i:i+RF].reshape(R, F); i += RF
    m_ls   = phi[i:i+RF].reshape(R, F); i += RF
    rho_ls = phi[i:i+RF].reshape(R, F); i += RF
    m_mb   = phi[i:i+F];                i += F
    rho_mb = phi[i:i+F];                i += F
    m_lsb  = phi[i];                    i += 1
    rho_lsb= phi[i];                    i += 1
    return m_a, rho_a, m_b, rho_b, m_ls, rho_ls, m_mb, rho_mb, m_lsb, rho_lsb


def date_posterior_raw(x_obs_norm, reg_id, phi, R, F,
                       prior_mean_norm, prior_sigma_norm,
                       n_samples=500, n_grid=3000,
                       date_lo_norm=-4.0, date_hi_norm=4.0,
                       rng=None):
    """Returns (d_grid_norm, posterior, log_likelihood_grid)."""
    if rng is None:
        rng = np.random.default_rng(42)

    (m_a, rho_a, m_b, rho_b, m_ls, rho_ls, *_) = unpack(phi, R, F)
    s_a  = np.exp(rho_a);  s_b  = np.exp(rho_b);  s_ls = np.exp(rho_ls)
    alpha_s = m_a + s_a  * rng.standard_normal((n_samples, R, F))
    beta_s  = m_b + s_b  * rng.standard_normal((n_samples, R, F))
    σ_s     = np.exp(m_ls + s_ls * rng.standard_normal((n_samples, R, F)))

    alpha_r = alpha_s[:, reg_id, :]
    beta_r  = beta_s[:,  reg_id, :]
    σ_r     = σ_s[:,    reg_id, :]

    d_lo   = min(prior_mean_norm - 4*prior_sigma_norm, date_lo_norm) - 0.5
    d_hi   = max(prior_mean_norm + 4*prior_sigma_norm, date_hi_norm) + 0.5
    d_grid = np.linspace(d_lo, d_hi, n_grid)

    pred    = alpha_r[np.newaxis,:,:] + beta_r[np.newaxis,:,:] * d_grid[:,np.newaxis,np.newaxis]
    x_bc    = x_obs_norm[np.newaxis, np.newaxis, :]
    σ_bc    = σ_r[np.newaxis, :, :]
    mask_f  = (~np.isnan(x_obs_norm))[np.newaxis, np.newaxis, :]
    log_lik_f = -0.5 * ((x_bc - pred) / σ_bc)**2 - np.log(σ_bc)
    log_lik_f = np.where(mask_f, log_lik_f, 0.0)
    log_lik   = lse(np.sum(log_lik_f, axis=2), axis=1) - np.log(n_samples)  # [D]

    log_prior = -0.5 * ((d_grid - prior_mean_norm) / prior_sigma_norm)**2
    log_post  = log_lik + log_prior
    log_post -= np.max(log_post)
    post = np.exp(log_post); post /= post.sum()
    return d_grid, post, log_lik


def summarise(d_grid, post, date_mean, date_std):
    dates_ce = d_grid * date_std + date_mean
    map_bce  = -float(dates_ce[np.argmax(post)])
    cdf      = np.cumsum(post)
    lo68_bce = -float(dates_ce[np.searchsorted(cdf, 0.16)])
    hi68_bce = -float(dates_ce[np.searchsorted(cdf, 0.84)])
    pmean    = np.dot(post, dates_ce)
    pstd     = np.sqrt(np.dot(post, (dates_ce - pmean)**2))
    # In BCE: lo68_bce is older (larger), hi68_bce is younger (smaller)
    return dict(map_bce=map_bce, lo68_bce=lo68_bce, hi68_bce=hi68_bce, pstd=pstd)


def fmt_bce(bce: float) -> str:
    return f"{int(round(bce))} BCE" if bce > 0 else f"{int(round(-bce))} CE"


def group_idx(g): return GROUPS.index(g)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not os.path.exists(PARAMS):
        print(f"Missing {PARAMS}\nRun 01_hb_vi_dating.py first.")
        sys.exit(1)

    # ── Load fitted parameters ────────────────────────────────────────────────
    data       = np.load(PARAMS, allow_pickle=True)
    phi        = data["phi"]
    feat_cols  = list(data["feat_cols"])
    feat_means = data["feat_means"]
    feat_stds  = data["feat_stds"]
    date_mean  = float(data["date_mean"])
    date_std   = float(data["date_std"])
    F = len(feat_cols)
    R = len(GROUPS)

    print("=" * 70)
    print("Prior sensitivity analysis  —  Hebrew HB-VI")
    print(f"Agnostic prior: N({AGNOSTIC_CENTER_BCE} BCE, {AGNOSTIC_SIGMA_BCE}²)")
    print("=" * 70)

    def norm_d(bce): return (-bce - date_mean) / date_std   # BCE → normalised CE

    agnostic_mean_n  = norm_d(AGNOSTIC_CENTER_BCE)
    agnostic_sigma_n = AGNOSTIC_SIGMA_BCE / date_std

    # ── Load feature matrices ─────────────────────────────────────────────────
    df_main = pd.read_csv(FEAT_CSV, index_col="id")
    df_extr = pd.read_csv(EXTR_CSV, index_col="id")

    def get_x_norm(eid, source="main"):
        src = df_main if source == "main" else df_extr
        row = src.loc[eid]
        x_raw  = pd.to_numeric(row[feat_cols], errors="coerce").values.astype(np.float64)
        return (x_raw - feat_means) / feat_stds

    rng = np.random.default_rng(77)

    # Date grid bounds in normalised units covering plausible range
    d_lo_n = norm_d(900)   # 900 BCE
    d_hi_n = norm_d(150)   # 150 BCE

    rows = []

    def analyse_text(eid, x_norm, grp, scholar_bce, scholar_sig_bce, source_label):
        ri = group_idx(grp)
        smn = norm_d(scholar_bce)
        ssn = scholar_sig_bce / date_std

        # ── (A) Scholarly prior ───────────────────────────────────────────────
        dg, pA, ll = date_posterior_raw(
            x_norm, ri, phi, R, F, smn, ssn,
            n_samples=N_MC_INFER, n_grid=N_GRID,
            date_lo_norm=d_lo_n, date_hi_norm=d_hi_n, rng=rng)
        rA = summarise(dg, pA, date_mean, date_std)

        # ── (B) Agnostic prior ────────────────────────────────────────────────
        dg2, pB, ll2 = date_posterior_raw(
            x_norm, ri, phi, R, F,
            agnostic_mean_n, agnostic_sigma_n,
            n_samples=N_MC_INFER, n_grid=N_GRID,
            date_lo_norm=d_lo_n, date_hi_norm=d_hi_n, rng=rng)
        rB = summarise(dg2, pB, date_mean, date_std)

        # ── Likelihood-only MAP ───────────────────────────────────────────────
        # Re-run with an EXTREMELY wide prior so posterior ≈ likelihood
        dg3, pC, ll3 = date_posterior_raw(
            x_norm, ri, phi, R, F,
            agnostic_mean_n, 10.0,   # σ_n=10 → essentially uniform
            n_samples=N_MC_INFER, n_grid=N_GRID,
            date_lo_norm=d_lo_n, date_hi_norm=d_hi_n, rng=rng)
        rC = summarise(dg3, pC, date_mean, date_std)

        shift_AB = rA["map_bce"] - rB["map_bce"]   # how much does the prior shift things?
        verdict  = ("prior-dominated" if abs(shift_AB) > 80
                    else "mildly prior-influenced" if abs(shift_AB) > 30
                    else "data-driven")

        rows.append(dict(
            id=eid, group=grp, source=source_label,
            scholar_bce=scholar_bce, scholar_sig=scholar_sig_bce,
            map_A=rA["map_bce"], lo68_A=rA["lo68_bce"], hi68_A=rA["hi68_bce"],
            map_B=rB["map_bce"], lo68_B=rB["lo68_bce"], hi68_B=rB["hi68_bce"],
            map_C=rC["map_bce"],    # likelihood-only
            shift_AB=shift_AB,
            verdict=verdict,
        ))
        return rA, rB, rC, shift_AB, verdict

    # ── Holdouts ──────────────────────────────────────────────────────────────
    print("\n── HOLDOUTS (with & without scholarly prior) ──────────────────")
    print(f"{'Text':20s}  {'Scholar':9s}  {'MAP(A)':9s}  {'MAP(B)':9s}  {'MAP(lik)':9s}  {'Shift':6s}  Verdict")
    print("─" * 95)
    for eid, (bce, sig, grp) in HOLDOUT_META.items():
        x_n = get_x_norm(eid, "main")
        rA, rB, rC, shift, verdict = analyse_text(eid, x_n, grp, bce, sig, "holdout")
        # Error with agnostic prior = more honest holdout validation
        err_B = rB["map_bce"] - bce
        print(f"  {eid:18s}  {bce:>5.0f} BCE  "
              f"{fmt_bce(rA['map_bce']):>9s}  "
              f"{fmt_bce(rB['map_bce']):>9s}  "
              f"{fmt_bce(rC['map_bce']):>9s}  "
              f"{shift:>+5.0f}yr  {verdict}")

    # ── Unknown-register main texts ───────────────────────────────────────────
    print("\n── UNKNOWN-REGISTER TEXTS ─────────────────────────────────────")
    print(f"{'Text':20s}  {'Scholar':9s}  {'MAP(A)':9s}  {'MAP(B)':9s}  {'MAP(lik)':9s}  {'Shift':6s}  Verdict")
    print("─" * 95)
    for eid, (bce, sig, grp) in UNKNOWN_META.items():
        x_n = get_x_norm(eid, "main")
        rA, rB, rC, shift, verdict = analyse_text(eid, x_n, grp, bce, sig, "unknown")
        print(f"  {eid:18s}  {bce:>5.0f} BCE  "
              f"{fmt_bce(rA['map_bce']):>9s}  "
              f"{fmt_bce(rB['map_bce']):>9s}  "
              f"{fmt_bce(rC['map_bce']):>9s}  "
              f"{shift:>+5.0f}yr  {verdict}")

    # ── Sub-sources ───────────────────────────────────────────────────────────
    print("\n── SUB-SOURCES ────────────────────────────────────────────────")
    print(f"{'Text':18s}  {'Prior(σ)':10s}  {'MAP(A)':9s}  {'MAP(B)':9s}  {'MAP(lik)':9s}  {'Shift':6s}  Verdict")
    print("─" * 100)
    for eid, (bce, sig, grp, desc) in SUBSOURCE_META.items():
        if eid not in df_extr.index:
            print(f"  {eid}: not found in extracted features, skipping.")
            continue
        x_n = get_x_norm(eid, "extr")
        rA, rB, rC, shift, verdict = analyse_text(eid, x_n, grp, bce, sig, "sub-source")
        print(f"  {eid:16s}  {bce:>5.0f}±{sig:<3.0f}   "
              f"{fmt_bce(rA['map_bce']):>9s}  "
              f"{fmt_bce(rB['map_bce']):>9s}  "
              f"{fmt_bce(rC['map_bce']):>9s}  "
              f"{shift:>+5.0f}yr  {verdict}")

    # ── Summary interpretation ────────────────────────────────────────────────
    df_rows = pd.DataFrame(rows)
    n_driven  = (df_rows["verdict"] == "data-driven").sum()
    n_mild    = (df_rows["verdict"] == "mildly prior-influenced").sum()
    n_dom     = (df_rows["verdict"] == "prior-dominated").sum()

    lines = [
        "",
        "=" * 70,
        "PRIOR SENSITIVITY SUMMARY",
        "=" * 70,
        f"  Data-driven          (shift < 30 yr): {n_driven}",
        f"  Mildly prior-influenced (30–80 yr):   {n_mild}",
        f"  Prior-dominated      (shift > 80 yr): {n_dom}",
        "",
        "Columns:",
        "  MAP(A) = posterior MAP with scholarly prior (as reported by 01_hb_vi_dating.py)",
        "  MAP(B) = posterior MAP with agnostic prior  N(575 BCE, 400²)",
        "  MAP(lik) = likelihood-only MAP (no prior — what the data alone says)",
        "  Shift  = MAP(A) − MAP(B)  [positive = scholarly prior pushed older]",
        "",
        "Interpretation notes:",
        "  * If MAP(B) ≈ MAP(lik), the agnostic prior is wide enough to be negligible.",
        "  * If MAP(B) ≈ MAP(A), the scholarly prior was not driving the result.",
        "  * Large shifts flag texts where scholarly opinion, not the data,",
        "    determines the model's conclusion.",
        "  * Archaic poems (Song_Sea, Song_Deborah) are bounded by the training",
        "    ceiling (~760 BCE); MAP(lik) near this ceiling means the data can only",
        "    say 'consistent with oldest SBH' — it cannot reach truly pre-monarchic",
        "    dates. Use with caution.",
    ]
    note = "\n".join(lines)
    print(note)

    out_txt = os.path.join(RESULTS, "prior_sensitivity.txt")
    full_text = note
    with open(out_txt, "w") as fh:
        fh.write(df_rows.to_string() + "\n\n" + full_text + "\n")
    print(f"\nFull output → {out_txt}")

    df_rows.to_csv(os.path.join(RESULTS, "prior_sensitivity.csv"), index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not HAS_PLT:
        return

    subsrc = df_rows[df_rows["source"] == "sub-source"].copy()
    other  = df_rows[df_rows["source"] != "sub-source"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Prior sensitivity: scholarly prior (●) vs. agnostic prior (▲)\n"
                 "bars = 68% CI for agnostic prior; vertical line = likelihood-only MAP",
                 fontsize=11)

    colors_g = {"SBH": "#1565C0", "Transitional": "#FF8F00", "LBH": "#2E7D32",
                "LBH": "#2E7D32"}

    for ax, subset, title in [
        (axes[0], other,  "Holdouts and unknown-register texts"),
        (axes[1], subsrc, "Sub-sources (extracted from BHSA)"),
    ]:
        labels = list(subset["id"])
        y = np.arange(len(labels))

        for yi, (_, row) in enumerate(subset.iterrows()):
            col = colors_g.get(row["group"], "grey")
            # CI bar (agnostic)
            ax.plot([row["hi68_B"], row["lo68_B"]], [yi, yi],
                    color=col, lw=5, alpha=0.35, solid_capstyle="butt")
            # Scholarly MAP
            ax.scatter([row["map_A"]], [yi], color=col, s=80, zorder=6, marker="o")
            # Agnostic MAP
            ax.scatter([row["map_B"]], [yi], color=col, s=80, zorder=7, marker="^",
                       facecolors="none", linewidths=1.8)
            # Likelihood-only MAP
            ax.axvline(row["map_C"], color=col, lw=0.8, ls=":", alpha=0.5)
            # Scholarly date tick
            ax.scatter([row["scholar_bce"]], [yi], color="black", s=25, marker="|",
                       zorder=8, linewidths=2)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_xaxis()
        ax.set_xlabel("Date (BCE, older → left)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.2)

    # Shared legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    leg = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="grey",
               markersize=9, label="MAP: scholarly prior"),
        Line2D([0],[0], marker="^", color="grey", markerfacecolor="none",
               markersize=9, markeredgewidth=1.8, label="MAP: agnostic prior"),
        Line2D([0],[0], color="grey", lw=0.8, ls=":", label="Likelihood-only MAP"),
        Patch(facecolor="grey", alpha=0.35, label="68% CI (agnostic prior)"),
        Line2D([0],[0], marker="|", color="black", markersize=10,
               markeredgewidth=2, label="Scholarly date"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_png = os.path.join(RESULTS, "prior_sensitivity.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot → {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
