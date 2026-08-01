"""
06_holdout_validation.py
========================
Validates the trained MVN dating model against all holdout texts defined
in corpus_manifest.json (holdout=True).

Currently includes:
  - Polybius, Histories         (~160 BCE, σ=25)
  - Gospel of Mark              (~70 CE,   σ=15)
  - Gospel of Matthew           (~85 CE,   σ=15)
  - Gospel of Luke              (~120 CE,  σ=30)
  - Diogenes Laërtius, Lives    (~230 CE,  σ=30)

For each holdout the script:
  1. Loads the full posterior (computed by 05_mvn_dating.py).
  2. Reports MAP date, 68% and 95% credible intervals.
  3. Tests whether the scholarly date falls within the credible intervals.
  4. Computes a calibration score: how many standard deviations is the
     MAP date from the scholarly consensus?
  5. Generates a validation dotplot (results/plots/holdout_validation.png).

Holdout list is read dynamically from corpus_manifest.json — no need to
update this script when the manifest changes.

Usage
-----
    python 06_holdout_validation.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

HERE       = os.path.dirname(os.path.abspath(__file__))
RESULTS    = os.path.join(HERE, "results")
PLOTS_DIR  = os.path.join(HERE, "results", "plots")
POST_DIR   = os.path.join(HERE, "results", "posteriors")
MANIFEST   = os.path.join(HERE, "corpus_manifest.json")

# ---------------------------------------------------------------------------
# Holdout metadata — loaded dynamically from corpus_manifest.json
# ---------------------------------------------------------------------------
def _load_holdouts_from_manifest(manifest_path: str) -> list:
    """Build the HOLDOUTS list from corpus_manifest.json holdout entries."""
    with open(manifest_path, encoding="utf-8") as f:
        corpus = json.load(f)
    holdouts = []
    for entry in corpus:
        if not entry.get("holdout", False):
            continue
        holdouts.append({
            "id"               : entry["id"],
            "author"           : entry["author"],
            "work"             : entry["work"],
            "scholarly_date_ce": entry["date_ce"],
            "scholarly_sigma"  : entry["date_sigma"],
            "register"         : entry.get("register", "unknown"),
            "notes"            : entry.get("notes", ""),
        })
    # Sort chronologically
    holdouts.sort(key=lambda h: h["scholarly_date_ce"])
    return holdouts

HOLDOUTS = _load_holdouts_from_manifest(MANIFEST)


def load_posterior(eid: str):
    """
    Load posterior JSON.
    Returns (date_grid, posterior_combined, posterior_lik_only).
    posterior_combined  = prior × likelihood  (main displayed result)
    posterior_lik_only  = likelihood only, no prior (honest model test)
    """
    path = os.path.join(POST_DIR, f"{eid}.json")
    if not os.path.exists(path):
        return None, None, None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    dg      = np.array(data["date_grid"])
    post    = np.array(data["posterior"])
    post_lo = np.array(data.get("posterior_lik_only", data["posterior"]))
    return dg, post, post_lo


def credible_interval(posterior: np.ndarray, date_grid: np.ndarray,
                      frac: float = 0.68) -> tuple[float, float]:
    """Equal-tailed credible interval (used for combined posterior only)."""
    cdf  = np.cumsum(posterior)
    lo   = date_grid[np.searchsorted(cdf, (1 - frac) / 2)]
    hi   = date_grid[np.searchsorted(cdf, 1 - (1 - frac) / 2)]
    return float(lo), float(hi)


def hdi(posterior: np.ndarray, date_grid: np.ndarray,
        frac: float = 0.68) -> tuple[float, float]:
    """
    Highest Density Interval — the shortest contiguous interval (by date span)
    that contains at least `frac` of the posterior mass.

    Unlike the equal-tailed CI, the HDI always covers the densest region of
    the distribution.  For posteriors truncated at a grid boundary (the true
    peak lies off-grid, so mass piles up against the edge), the HDI honestly
    reports that the densest region is at the boundary — rather than giving
    a centred interval that spans equally into low-density territory on the
    other side.

    Algorithm: for each starting index i, binary-search for the smallest hi
    such that mass[i:hi] >= frac; keep the (i, hi) pair with minimum span.
    O(n log n).
    """
    n   = len(posterior)
    cum = np.cumsum(posterior)

    best_span = n          # measured in grid steps; minimise this
    best_lo   = 0
    best_hi   = n - 1

    for i in range(n):
        lo_mass = cum[i - 1] if i > 0 else 0.0
        target  = lo_mass + frac
        hi      = int(np.searchsorted(cum, target, side="left"))
        if hi >= n:
            break                             # not enough mass remains
        mass = cum[hi] - lo_mass
        if mass >= frac - 1e-9:
            span = hi - i
            if span < best_span:
                best_span = span
                best_lo   = i
                best_hi   = hi

    return float(date_grid[best_lo]), float(date_grid[best_hi])


def posterior_mean_std(posterior: np.ndarray, date_grid: np.ndarray) -> tuple[float, float]:
    mu  = float(np.sum(date_grid * posterior))
    var = float(np.sum((date_grid - mu)**2 * posterior))
    return mu, float(np.sqrt(var))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load training dating results for context
    res_path = os.path.join(RESULTS, "dating_results.csv")
    if os.path.exists(res_path):
        df_results = pd.read_csv(res_path)
    else:
        df_results = pd.DataFrame()
        print("[WARN] dating_results.csv not found — run 05_mvn_dating.py first.")

    print("=" * 72)
    print("HOLDOUT VALIDATION REPORT")
    print("=" * 72)

    validation_records = []

    for ho in HOLDOUTS:
        eid         = ho["id"]
        schol_date  = ho["scholarly_date_ce"]
        schol_sigma = ho["scholarly_sigma"]
        author      = ho["author"]

        dg, post, post_lo = load_posterior(eid)
        if dg is None:
            print(f"\n  {author}: posterior not found (run 05_mvn_dating.py).")
            continue

        def fmt(y):
            return f"{abs(int(y))} {'BCE' if y < 0 else 'CE'}"

        # ── Combined posterior (prior × likelihood) ───────────────────────
        map_date      = float(dg[np.argmax(post)])
        post_mu, post_std = posterior_mean_std(post, dg)
        ci68_lo, ci68_hi  = credible_interval(post, dg, 0.68)
        ci95_lo, ci95_hi  = credible_interval(post, dg, 0.95)
        in_68 = ci68_lo <= schol_date <= ci68_hi
        in_95 = ci95_lo <= schol_date <= ci95_hi
        combined_sigma = np.sqrt(schol_sigma**2 + post_std**2)
        z_score = (map_date - schol_date) / combined_sigma

        # ── Likelihood-only posterior (no prior, honest model power test) ─
        map_lo        = float(dg[np.argmax(post_lo)])
        mu_lo, std_lo = posterior_mean_std(post_lo, dg)
        # Use HDI (highest density interval) for likelihood-only.
        # Equal-tailed CI is inappropriate here: for distributions whose
        # peak falls off-grid (truncated Gaussian ramping to the boundary),
        # equal-tailed CI gives an artificially centred interval that makes
        # the result look better than it is.  HDI gives the shortest interval
        # that actually contains the densest probability mass.
        lo68_lo, lo68_hi = hdi(post_lo, dg, 0.68)
        lo95_lo, lo95_hi = hdi(post_lo, dg, 0.95)
        lo_in_68 = lo68_lo <= schol_date <= lo68_hi
        # z-score for likelihood-only: use model std (not prior-dominated)
        lo_z = (map_lo - schol_date) / np.sqrt(schol_sigma**2 + std_lo**2)
        # Prior contribution: what fraction of precision comes from prior?
        prior_prec = 1.0 / schol_sigma**2
        total_prec = 1.0 / post_std**2 if post_std > 0 else np.inf
        prior_pct  = 100 * prior_prec / total_prec if np.isfinite(total_prec) else 100

        print(f"\n{'─'*72}")
        print(f"  {author}  —  {ho['work']}")
        print(f"  Scholarly date   : {fmt(schol_date)} ± {schol_sigma} yr")
        print()
        print(f"  ── Combined posterior (prior × likelihood) ──")
        print(f"  MAP date (model) : {fmt(map_date)}")
        print(f"  Posterior mean   : {fmt(post_mu)} ± {int(post_std)} yr")
        print(f"  68% CI           : [{fmt(ci68_lo)}, {fmt(ci68_hi)}]")
        print(f"  95% CI           : [{fmt(ci95_lo)}, {fmt(ci95_hi)}]")
        print(f"  Scholarly in 68%?: {'YES ✓' if in_68 else 'NO'}")
        print(f"  Scholarly in 95%?: {'YES ✓' if in_95 else 'NO'}")
        print(f"  Z-score          : {z_score:+.2f} σ")
        print(f"  Prior precision  : {prior_pct:.0f}% of total  "
              f"(prior σ={schol_sigma} yr vs posterior σ={int(post_std)} yr)")
        print()
        print(f"  ── Likelihood only (no prior — raw model test) ──")
        print(f"  MAP (lik. only)  : {fmt(map_lo)}")
        print(f"  Posterior mean   : {fmt(mu_lo)} ± {int(std_lo)} yr")
        print(f"  68% HDI          : [{fmt(lo68_lo)}, {fmt(lo68_hi)}]")
        print(f"  95% HDI          : [{fmt(lo95_lo)}, {fmt(lo95_hi)}]")
        print(f"  Scholarly in 68%HDI?: {'YES ✓' if lo_in_68 else 'NO  ← model alone cannot pin this date'}")
        print(f"  Z-score (lik.)   : {lo_z:+.2f} σ  (|z| < 2 = model usefully constrains date)")
        print(f"  Notes: {ho['notes']}")

        validation_records.append({
            "id"               : eid,
            "author"           : author,
            "work"             : ho["work"],
            "scholarly_date_ce": schol_date,
            "scholarly_sigma"  : schol_sigma,
            # Combined (prior × likelihood)
            "map_date_ce"      : round(map_date),
            "posterior_mean_ce": round(post_mu),
            "posterior_std"    : round(post_std),
            "ci68_lo"          : round(ci68_lo),
            "ci68_hi"          : round(ci68_hi),
            "ci95_lo"          : round(ci95_lo),
            "ci95_hi"          : round(ci95_hi),
            "in_68ci"          : in_68,
            "in_95ci"          : in_95,
            "z_score"          : round(z_score, 3),
            "prior_pct_precision": round(prior_pct, 1),
            # Likelihood only
            "lik_only_map_ce"  : round(map_lo),
            "lik_only_mean_ce" : round(mu_lo),
            "lik_only_std"     : round(std_lo),
            "lik_only_ci68_lo" : round(lo68_lo),
            "lik_only_ci68_hi" : round(lo68_hi),
            "lik_only_ci95_lo" : round(lo95_lo),
            "lik_only_ci95_hi" : round(lo95_hi),
            "lik_only_in_68ci" : lo_in_68,
            "lik_only_z_score" : round(lo_z, 3),
        })

    # Save validation table
    val_path = os.path.join(RESULTS, "holdout_validation.csv")
    pd.DataFrame(validation_records).to_csv(val_path, index=False)
    print(f"\n\nValidation table → {val_path}")

    # ── Holdout validation plot ───────────────────────────────────────────────
    # The plot's purpose is to test model quality honestly:
    #   • Main curve   = likelihood-only posterior (no prior) — this is the
    #                    genuine prediction the model makes from features alone.
    #   • Shaded band  = scholarly date ± sigma — this is what we're testing
    #                    against.
    #   • The combined posterior (prior × likelihood) trivially peaks at the
    #     scholarly date because the prior dominates, so it is NOT plotted here
    #     to avoid creating a false impression of model accuracy.
    if not HAS_MPL or not validation_records:
        print("Skipping plot (matplotlib unavailable or no results).")
        return

    fig, axes = plt.subplots(len(HOLDOUTS), 1, figsize=(12, 3.5 * len(HOLDOUTS)),
                             sharex=False)
    if len(HOLDOUTS) == 1:
        axes = [axes]

    palette = {"Polybius": "#2196F3", "Luke (anon.)": "#4CAF50",
               "Diogenes Laërtius": "#FF5722"}

    for ax, ho, rec in zip(axes, HOLDOUTS, validation_records):
        eid   = ho["id"]
        color = palette.get(ho["author"], "#607D8B")

        dg, _post_combined, post_lo = load_posterior(eid)
        if dg is None:
            ax.text(0.5, 0.5, "No posterior data", transform=ax.transAxes, ha="center")
            continue

        # ── Main: likelihood-only posterior ──────────────────────────────
        ax.fill_between(dg, post_lo, alpha=0.22, color=color)
        ax.plot(dg, post_lo, color=color, lw=1.5,
                label=f"Likelihood (no prior)  mean={fmt(rec['lik_only_mean_ce'])} σ≈{int(rec['lik_only_std'])} yr")

        # HDI shading (densest region — honest about where mass actually is)
        h68_lo = rec["lik_only_ci68_lo"]
        h68_hi = rec["lik_only_ci68_hi"]
        hdi_mask = (dg >= h68_lo) & (dg <= h68_hi)
        h68_lo_lbl = f"{abs(int(h68_lo))} {'BCE' if h68_lo<0 else 'CE'}"
        h68_hi_lbl = f"{abs(int(h68_hi))} {'BCE' if h68_hi<0 else 'CE'}"
        ax.fill_between(dg, post_lo, where=hdi_mask, alpha=0.50, color=color,
                        label=f"68% HDI: {h68_lo_lbl} – {h68_hi_lbl}")

        # Likelihood MAP
        map_lo      = rec["lik_only_map_ce"]
        map_lo_lbl  = f"{abs(map_lo)} {'BCE' if map_lo<0 else 'CE'}"
        ax.axvline(map_lo, color=color, lw=1.8, ls="-",
                   label=f"Likelihood MAP: {map_lo_lbl}")

        # Mark the grid right edge so the reader can see truncation.
        # If the HDI abuts the boundary, add a hatched region to signal
        # that probability mass extends beyond the grid.
        truncated = abs(h68_hi - dg[-1]) < 10   # HDI reaches the right wall
        ax.axvline(dg[-1], color="gray", lw=1.2, ls=":", alpha=0.7,
                   label=f"Grid boundary ({int(dg[-1])} CE)")
        if truncated:
            # Hatch beyond the boundary to show the curve is cut off
            ax.axvspan(dg[-1], dg[-1] + 80, alpha=0.12, color="gray",
                       hatch="///", label="Truncated (peak off-grid →)")

        # ── Scholarly date reference ──────────────────────────────────────
        schol = rec["scholarly_date_ce"]
        sigma = rec["scholarly_sigma"]
        ax.axvline(schol, color="black", lw=2, ls="--",
                   label=f"Scholarly: {abs(schol)} {'BCE' if schol<0 else 'CE'} ±{sigma} yr")
        ax.axvspan(schol - sigma, schol + sigma,
                   alpha=0.13, color="black", zorder=0)

        ax.set_ylabel("Prob. density", fontsize=9)
        ax.set_yticks([])
        ax.set_title(f"{ho['author']}  —  {ho['work']}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper left")

        # Annotation box: honest HDI-based assessment
        lo_in68    = rec["lik_only_in_68ci"]
        in_hdi_str = "✓ within 68% HDI" if lo_in68 else "✗ outside 68% HDI"
        trunc_str  = "\n⚠ HDI truncated at grid boundary" if truncated else ""
        ax.text(0.98, 0.97,
                f"Likelihood z = {rec['lik_only_z_score']:+.2f} σ\n"
                f"Scholarly date {in_hdi_str}{trunc_str}\n"
                f"(prior = {rec['prior_pct_precision']:.0f}% of combined precision)",
                transform=ax.transAxes, va="top", ha="right", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.85))

    axes[-1].set_xlabel("Date (CE; negative = BCE)", fontsize=10)
    fig.suptitle(
        "Holdout Validation — Model Likelihood Only (no prior)\n"
        "Shaded band = scholarly consensus ±1σ; dashed line = scholarly MAP",
        fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = os.path.join(PLOTS_DIR, "holdout_validation.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Validation plot → {plot_path}")


if __name__ == "__main__":
    main()
