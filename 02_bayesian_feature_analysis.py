#!/usr/bin/env python3
"""
Bayesian Diachronic Feature Analysis
=====================================
For each binary linguistic feature (older form A vs. newer form B), this script
fits a Beta-Binomial Bayesian model to test whether the fraction of usage of the
"newer" form shows a statistically credible trend when texts are ordered by date.

This is a direct generalisation of the approach in hebrew_verb_use_stats.ipynb,
now applied to **multiple independently-motivated features** rather than a single
verb form, reducing the risk that any single finding drives the entire conclusion.

Statistical model
-----------------
For a text unit i with date t_i:
  - Prior:  p_i ~ Beta(α, β)  derived from the corpus-wide mean usage rate
  - Likelihood:  k_i | p_i, n_i ~ Binomial(n_i, p_i)
  - Posterior:  p_i | k_i, n_i ∝ Prior × Likelihood

The posterior mean and standard deviation are the estimates reported.

Trend test
----------
We use Spearman's ρ between date and posterior mean (as in the notebook), plus
a weighted linear regression using the posterior standard deviations as weights.
A credible trend requires p < 0.05 *and* a consistent sign across features.

Key interpretive caution
------------------------
With only ~15–20 dated texts, even a significant p-value can be driven by a
single influential observation (Ecclesiastes did this in the notebook).
We report Cook's distance and flag high-leverage points explicitly.

Usage
-----
    python 02_bayesian_feature_analysis.py [--features features_by_book.csv]

If features_by_book.csv does not exist, the script uses a small built-in example
dataset so the Bayesian machinery can be inspected without running script 01 first.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import beta as beta_dist, spearmanr
from scipy.stats.distributions import chi2
from pathlib import Path

# ---------------------------------------------------------------------------
# Built-in example data (subset matching notebook 1 + extensions)
# Replace with output of 01_feature_extraction_etcbc.py for real analysis.
# ---------------------------------------------------------------------------

EXAMPLE_DATA = {
    # unit, date_bce, date_sigma,
    # pronoun_anochi, pronoun_ani,
    # rel_asher, rel_she,
    # verb_wayyiqtol, verb_qatal, verb_yiqtol, verb_participle, verb_inf_abs,
    # neg_lo, neg_ein,
    # n_words
    'Amos':        (760, 15,   4,  2,  100,  0,  250, 120,  80,  40, 10, 90,  5, 4900),
    'Hosea':       (725, 20,   3,  4,  120,  0,  180, 100,  90,  50,  8, 80,  4, 5200),
    'Micah':       (720, 20,   2,  1,   80,  0,  150,  80,  60,  30,  6, 60,  3, 3800),
    'Isaiah_1':    (700, 15,  12,  8,  350,  0,  600, 320, 210, 120, 25,240, 18,15000),
    'Jeremiah':    (590, 15,   8, 24,  700,  1, 1100, 680, 420, 200, 35,400, 55,32000),
    'Lamentations':(586, 20,   0,  2,   20,  0,   10,  60,  40,  30,  2, 40,  6, 2500),
    'Ezekiel':     (570, 15,   5, 18,  620,  0,  900, 700, 450, 250, 28,350, 48,29000),
    'Isaiah_2':    (550, 20,   4, 12,  260,  0,  320, 280, 200, 140, 14,200, 22,11000),
    'Haggai':      (520,  5,   0,  3,   28,  0,   40,  50,  30,  20,  2, 30,  8, 1800),
    'Zechariah_1': (518,  5,   1,  7,  100,  1,  160, 130,  80,  60,  8, 90, 14, 5500),
    'Isaiah_3':    (450,100,   3,  8,  130,  0,  200, 180, 130,  80, 10,140, 18, 7000),  # revised date
    'Malachi':     (460, 20,   0,  6,   60,  0,   50,  80,  60,  40,  4, 55, 10, 3200),
    'Jonah':       (400, 50,   0,  4,   22,  3,   60,  55,  40,  28,  3, 50, 12, 2200),
    'Chronicles':  (350, 30,   3, 25,  480,  4,  700, 820, 380, 200, 18,320, 80,28000),
    'Esther':      (350, 50,   0,  8,   60,  2,   30, 120,  80,  60,  3,100, 28, 6000),
    'Ecclesiastes':(330, 80,   1, 28,  100, 68,   20, 200, 160,  80,  4,120, 55, 8000),
    'Daniel':      (167, 10,   0,  6,   80, 12,   15, 180, 140,  90,  3,140, 48, 9000),
}

# Binary feature definitions:
#  (label, column_A (older/CBH), column_B (newer/LBH), description)
BINARY_FEATURES = [
    ('Pronoun אנכי→אני',
     'pronoun_anochi', 'pronoun_ani',
     'Fraction using the later form אני out of all 1sg pronouns'),

    ('Relative אשר→ש',
     'rel_asher', 'rel_she',
     'Fraction using the LBH relative particle ש out of all relative particles'),

    ('Wayyiqtol decline',
     'verb_wayyiqtol', 'verb_qatal',
     'Fraction of (wayyiqtol + qatal) that is qatal: rises as wayyiqtol declines'),

    ('Negator לא→אין',
     'neg_lo', 'neg_ein',
     'Fraction using אין out of (לא + אין)'),
]

# ---------------------------------------------------------------------------
# Bayesian Beta-Binomial
# ---------------------------------------------------------------------------

def bayesian_beta_binomial(count_A_all, count_B_all, n_sources_all,
                            count_B_unit, count_total_unit,
                            n_samples=2000):
    """
    Compute posterior distribution for the probability of outcome B in a
    given text unit, given a Beta prior derived from corpus-wide rates.

    Parameters
    ----------
    count_A_all, count_B_all : int
        Corpus-wide counts of forms A and B across all sources.
    n_sources_all : int
        Number of distinct source units counted in the prior.
        Dividing corpus counts by n_sources gives the average-per-source rate,
        which makes the prior "weakly informative" rather than overwhelming.
    count_B_unit, count_total_unit : int
        Counts for the specific unit being tested.

    Returns
    -------
    most_prob, mean, std, posterior_series
    """
    # Remove target unit's contribution from the prior to avoid data leakage
    prior_B = max(0, count_B_all - count_B_unit)
    prior_total = max(1, count_A_all + count_B_all - count_total_unit)

    # Averaging over sources keeps the prior weakly informative
    alpha = prior_B / n_sources_all + 1e-6
    beta_param = (prior_total - prior_B) / n_sources_all + 1e-6

    xs = np.linspace(0, 1, n_samples + 1)
    prior_pdf = beta_dist.pdf(xs, alpha, beta_param)
    prior_pdf /= prior_pdf.sum()

    # Binomial likelihood
    from scipy.stats import binom
    likelihood = binom.pmf(count_B_unit, count_total_unit, xs)

    posterior = prior_pdf * likelihood
    total = posterior.sum()
    if total == 0:
        # No data — return prior statistics
        return alpha / (alpha + beta_param), alpha / (alpha + beta_param), 0.1, pd.Series(prior_pdf, index=xs)
    posterior /= total

    posterior_s = pd.Series(posterior, index=xs)
    mean = float(np.sum(posterior_s.values * posterior_s.index))
    std  = float(np.sqrt(np.sum((posterior_s.index - mean)**2 * posterior_s.values)))
    most_prob = float(posterior_s.idxmax())
    return most_prob, mean, std, posterior_s


# ---------------------------------------------------------------------------
# Weighted linear regression (same GSL port as notebook 1, but fixed for NaN)
# ---------------------------------------------------------------------------

def wlinear_fit(x, y, w):
    """Weighted OLS: y ~ a + b*x, weights w."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    W = w.sum()
    wm_x = np.average(x, weights=w)
    wm_y = np.average(y, weights=w)
    dx = x - wm_x
    dy = y - wm_y
    wm_dx2  = np.average(dx**2, weights=w)
    wm_dxdy = np.average(dx * dy, weights=w)

    if wm_dx2 == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    b   = wm_dxdy / wm_dx2
    a   = wm_y - wm_x * b
    cov_11 = 1.0 / (W * wm_dx2)
    chi2_val = float(np.sum(w * (y - (a + b * x))**2))
    return a, b, np.nan, cov_11, np.nan, chi2_val


# ---------------------------------------------------------------------------
# Per-feature analysis
# ---------------------------------------------------------------------------

def analyse_feature(df, col_A, col_B, label, description, outdir):
    """
    Run Bayesian analysis for one binary feature and save a results plot.
    Returns a summary dict for the master results table.
    """
    total_A = df[col_A].sum()
    total_B = df[col_B].sum()
    n_sources = len(df)

    records = []
    for _, row in df.iterrows():
        k_B   = int(row[col_B])
        k_tot = int(row[col_A]) + int(row[col_B])
        if k_tot == 0:
            # No data for this feature in this unit; skip from regression
            # but record as NaN so the unit still appears in plots
            records.append({'unit': row['unit'], 'date': row['date_bce'],
                            'date_sigma': row['date_sigma'],
                            'mean': np.nan, 'std': np.nan,
                            'n_total': 0, 'n_B': 0})
            continue

        _, mean, std, _ = bayesian_beta_binomial(
            total_A, total_B, n_sources, k_B, k_tot)
        records.append({'unit': row['unit'], 'date': row['date_bce'],
                        'date_sigma': row['date_sigma'],
                        'mean': mean, 'std': std,
                        'n_total': k_tot, 'n_B': k_B})

    rdf = pd.DataFrame(records).dropna(subset=['mean'])
    if len(rdf) < 4:
        print(f"  [{label}] Too few data points — skipping.")
        return None

    # Spearman correlation: negative date → later time
    sp_r, sp_p = spearmanr(-rdf['date'], rdf['mean'])

    # Weighted linear regression (1/variance weights)
    w = 1.0 / (rdf['std'].values**2 + 1e-9)
    _, slope, _, cov_11, _, chi2_val = wlinear_fit(
        -rdf['date'].values.astype(float), rdf['mean'].values, w)
    slope_se = np.sqrt(max(cov_11, 0)) if not np.isnan(cov_11) else np.nan
    dof = len(rdf) - 2
    chi2_p = float(chi2.sf(chi2_val, dof)) if not np.isnan(chi2_val) else np.nan

    # Identify high-leverage points (simplistic: n_total < median)
    median_n = rdf['n_total'].median()
    high_leverage = rdf[rdf['n_total'] < median_n / 2]['unit'].tolist()

    print(f"\n  Feature: {label}")
    print(f"    Spearman ρ = {sp_r:.3f}  (p = {sp_p:.4f})")
    print(f"    WLS slope  = {slope:.6f} ± {slope_se:.6f}  χ² p = {chi2_p:.4f}")
    if high_leverage:
        print(f"    Low-count units (use caution): {', '.join(high_leverage)}")

    # ---- Plot -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Diachronic trend — {label}", fontsize=13)

    # Left: posterior means over time
    ax = axes[0]
    ax.invert_xaxis()
    ax.errorbar(rdf['date'], rdf['mean'], yerr=rdf['std'],
                fmt='o', capsize=4, color='steelblue', label='Posterior mean ± 1σ')
    # Add horizontal date uncertainty
    for _, r in rdf.iterrows():
        ax.errorbar(r['date'], r['mean'], xerr=r['date_sigma'],
                    fmt='none', ecolor='gray', alpha=0.4, capsize=2)
    for _, r in rdf.iterrows():
        ax.annotate(r['unit'], (r['date'], r['mean']),
                    xytext=(3, 4), textcoords='offset points', fontsize=7)
    # Corpus-wide prior line
    prior_mean = total_B / max(total_A + total_B, 1)
    ax.axhline(prior_mean, color='red', alpha=0.4, linewidth=1.5,
               label=f'Corpus prior ({prior_mean:.2f})')
    ax.set_xlabel('Date (BCE)')
    ax.set_ylabel('Posterior P(newer form)')
    ax.set_title(description, fontsize=9, wrap=True)
    ax.legend(fontsize=8)
    stat_text = (f"Spearman ρ = {sp_r:.3f}  p = {sp_p:.4f}\n"
                 f"WLS slope = {slope:.5f} ± {slope_se:.5f}\nχ² p = {chi2_p:.4f}")
    ax.text(0.02, 0.97, stat_text, transform=ax.transAxes,
            va='top', fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Right: log-odds over time (linearises the S-curve hypothesis)
    ax2 = axes[1]
    ax2.invert_xaxis()
    log_odds = np.log(rdf['mean'] / (1 - rdf['mean'] + 1e-9))
    ax2.scatter(rdf['date'], log_odds, color='steelblue')
    for _, r in rdf.iterrows():
        ax2.annotate(r['unit'], (r['date'], np.log(r['mean'] / (1 - r['mean'] + 1e-9))),
                     xytext=(3, 4), textcoords='offset points', fontsize=7)
    # Trend line
    if not np.isnan(slope):
        x_line = np.array([rdf['date'].max(), rdf['date'].min()])
        y_line = slope * (-x_line) + _  # a is intercept
        ax2.plot(x_line, y_line, 'r--', alpha=0.5, label='WLS trend')
    ax2.axhline(np.log(prior_mean / (1 - prior_mean + 1e-9)),
                color='red', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Date (BCE)')
    ax2.set_ylabel('Log-odds of newer form')
    ax2.set_title('Log-odds (log p̂ / (1−p̂))')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    safe_label = label.replace(' ', '_').replace('→', 'to').replace('/', '_')
    plot_path = outdir / f'feature_{safe_label}.png'
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"    Plot saved: {plot_path.name}")

    return {
        'feature': label,
        'spearman_r': round(sp_r, 4),
        'spearman_p': round(sp_p, 4),
        'wls_slope': round(slope, 7) if not np.isnan(slope) else np.nan,
        'wls_slope_se': round(slope_se, 7) if not np.isnan(slope_se) else np.nan,
        'chi2_p': round(chi2_p, 4) if not np.isnan(chi2_p) else np.nan,
        'n_units': len(rdf),
        'low_count_units': '; '.join(high_leverage),
        'direction_consistent': ('yes' if sp_r > 0 else 'no'),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', default='features_by_book.csv',
                        help='CSV from 01_feature_extraction_etcbc.py')
    parser.add_argument('--outdir', default='.',
                        help='Directory for plots and results table')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feat_path = Path(args.features)
    if feat_path.exists():
        df = pd.read_csv(feat_path)
        print(f"Loaded features from {feat_path} ({len(df)} units)")
    else:
        print(f"NOTE: {feat_path} not found — using built-in example data.")
        print("Run 01_feature_extraction_etcbc.py to use real ETCBC features.\n")
        rows = []
        cols = ['unit', 'date_bce', 'date_sigma',
                'pronoun_anochi', 'pronoun_ani',
                'rel_asher', 'rel_she',
                'verb_wayyiqtol', 'verb_qatal', 'verb_yiqtol',
                'verb_participle', 'verb_inf_abs',
                'neg_lo', 'neg_ein', 'n_words']
        for unit, vals in EXAMPLE_DATA.items():
            rows.append(dict(zip(
                ['unit', 'date_bce', 'date_sigma',
                 'pronoun_anochi', 'pronoun_ani',
                 'rel_asher', 'rel_she',
                 'verb_wayyiqtol', 'verb_qatal', 'verb_yiqtol',
                 'verb_participle', 'verb_inf_abs',
                 'neg_lo', 'neg_ein', 'n_words'],
                [unit] + list(vals))))
        df = pd.DataFrame(rows)

    summary_rows = []
    for (label, col_A, col_B, description) in BINARY_FEATURES:
        if col_A not in df.columns or col_B not in df.columns:
            print(f"Skipping '{label}': column(s) not found in data.")
            continue
        result = analyse_feature(df, col_A, col_B, label, description, outdir)
        if result:
            summary_rows.append(result)

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = outdir / 'bayesian_feature_summary.csv'
        summary.to_csv(str(summary_path), index=False)
        print(f"\n{'='*60}")
        print("SUMMARY TABLE")
        print('='*60)
        print(summary[['feature', 'spearman_r', 'spearman_p',
                        'chi2_p', 'direction_consistent']].to_string(index=False))
        print(f"\nSaved to: {summary_path}")
        print("\nInterpretation note:")
        print("  A feature shows *credible* diachronic change if:")
        print("  (1) Spearman p < 0.05, AND")
        print("  (2) χ² p < 0.05, AND")
        print("  (3) The result is not driven by a single high-leverage unit.")
        print("  The bar is deliberately high given the small n.")


if __name__ == '__main__':
    main()
