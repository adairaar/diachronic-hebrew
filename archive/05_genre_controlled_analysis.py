#!/usr/bin/env python3
"""
Genre-Controlled Diachronic Analysis with Leave-One-Out Robustness
====================================================================
Runs the Bayesian feature analysis from script 02, restricted to a single
genre, and adds a leave-one-out (LOO) sensitivity test to every result.

Why genre control?
------------------
Script 02 showed that the wayyiqtol decline feature has the WRONG sign when
all books are included: Chronicles (~350 BCE) uses wayyiqtol heavily because
it is narrative prose, outweighing its late date.  Genre drives wayyiqtol
density more strongly than period does, so mixing genres contaminates the
temporal signal.

Restricting to prophecy gives 15 dated units spanning ~760–460 BCE, all in
roughly the same rhetorical register, so inter-book stylistic differences are
more likely to reflect temporal change than genre change.

Why leave-one-out?
------------------
In notebook 1 (hebrew_verb_use_stats.ipynb), the conclusion about the ידע
verb hung almost entirely on Ecclesiastes.  A result that evaporates when
one data point is removed is not reliable evidence for diachronic change,
regardless of its p-value.

The LOO test makes this explicit:
  - For each feature, compute Spearman ρ on the full genre-filtered set.
  - Then remove each unit in turn and recompute ρ and p.
  - A result is **LOO-robust** if:
      (a) ρ keeps the same sign across ALL LOO subsets, AND
      (b) p < 0.05 in ≥ 80% of LOO subsets.
  - Any feature that fails (b) is "fragile"; the driving unit is identified.

Features tested
---------------
All binary pairs already in features_by_book.csv:
  1. Pronoun  אנכי → אני      (older → newer form of "I")
  2. Relative אשר  → ש        (CBH → LBH relative particle)
  3. Wayyiqtol decline         (wayyiqtol fraction of finite verbs → rises as wayq falls)
  4. Negator  לא   → אין      (standard negator → negative existential)
Plus rate-based features (no binary denominator):
  5. Infinitive absolute rate  (per 1,000 words; expected to decline in LBH)
  6. Participle rate           (per 1,000 words; expected to rise in LBH)

Usage
-----
    python 05_genre_controlled_analysis.py [--features features_by_book.csv]
                                           [--genre prophecy]
                                           [--outdir .]

Genre options: prophecy (default), narrative, poetry, wisdom, all
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import beta as beta_dist, spearmanr, binom
from scipy.stats.distributions import chi2
from pathlib import Path

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Genre classification (same as script 03)
# ---------------------------------------------------------------------------
GENRE_MAP = {
    'Amos':         'prophecy',
    'Hosea':        'prophecy',
    'Micah':        'prophecy',
    'Isaiah_1':     'prophecy',
    'Isaiah_2':     'prophecy',
    'Isaiah_3':     'prophecy',
    'Zephaniah':    'prophecy',
    'Nahum':        'prophecy',
    'Habakkuk':     'prophecy',
    'Jeremiah':     'prophecy',
    'Lamentations': 'poetry',
    'Ezekiel':      'prophecy',
    'Haggai':       'prophecy',
    'Zechariah_1':  'prophecy',
    'Malachi':      'prophecy',
    'Jonah':        'narrative',
    'Chronicles':   'narrative',
    'Esther':       'narrative',
    'Ecclesiastes': 'wisdom',
    'Daniel':       'mixed',
}

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
# Each entry is a dict with keys:
#   label        : display name
#   kind         : 'binary' (col_A vs col_B) or 'rate' (per-1k-words)
#   col_A        : column for older/CBH count (binary) or count column (rate)
#   col_B        : column for newer/LBH count (binary only)
#   description  : one-line linguistic description
#   direction    : 'increase' if newer/LBH values should be higher over time

FEATURES = [
    dict(
        label='Pronoun אנכי→אני',
        kind='binary',
        col_A='pronoun_anochi',
        col_B='pronoun_ani',
        description='Fraction of 1sg pronouns that are the newer אני (vs. archaic אנכי)',
        direction='increase',
    ),
    dict(
        label='Relative אשר→ש',
        kind='binary',
        col_A='rel_asher',
        col_B='rel_she',
        description='Fraction of relative particles that are ש (vs. standard אשר)',
        direction='increase',
    ),
    dict(
        label='Wayyiqtol fraction',
        kind='binary',
        col_A='verb_wayyiqtol',   # wayyiqtol = numerator here means we track its *decline*
        col_B='verb_qatal',       # frac = qatal / (wayq+qatal): rises as wayq falls
        description='Fraction of (wayyiqtol+qatal) that is qatal — rises as wayyiqtol declines',
        direction='increase',
    ),
    dict(
        label='Negator לא→אין',
        kind='binary',
        col_A='neg_lo',
        col_B='neg_ein',
        description='Fraction of (לא + אין) that is the newer negative-existential אין',
        direction='increase',
    ),
    dict(
        label='Infinitive absolute rate',
        kind='rate',
        col_A='verb_inf_abs',
        description='Infinitive absolute forms per 1,000 words (expected to decline in LBH)',
        direction='decrease',
    ),
    dict(
        label='Participle rate',
        kind='rate',
        col_A='verb_participle',
        description='Active + passive participle forms per 1,000 words (expected to rise in LBH)',
        direction='increase',
    ),
]


# ---------------------------------------------------------------------------
# Bayesian Beta-Binomial (same as script 02, extracted here for independence)
# ---------------------------------------------------------------------------

def bayesian_posterior(count_A_total, count_B_total, n_sources,
                       count_B_unit, count_total_unit, n_samples=1500):
    """
    Beta prior (derived from corpus-wide averages) × Binomial likelihood.
    Returns (mean, std, posterior_series).
    """
    prior_B = max(0, count_B_total - count_B_unit)
    prior_N = max(1, count_A_total + count_B_total - count_total_unit)
    alpha = prior_B / n_sources + 1e-6
    b_param = (prior_N - prior_B) / n_sources + 1e-6

    xs = np.linspace(0, 1, n_samples + 1)
    prior_pdf = beta_dist.pdf(xs, alpha, b_param)
    s = prior_pdf.sum()
    if s == 0 or not np.isfinite(s):
        prior_pdf = np.ones(n_samples + 1) / (n_samples + 1)
    else:
        prior_pdf /= s

    like = binom.pmf(count_B_unit, count_total_unit, xs)
    post = prior_pdf * like
    total = post.sum()
    if total == 0 or not np.isfinite(total):
        mean = alpha / (alpha + b_param)
        return mean, 0.15, pd.Series(prior_pdf, index=xs)
    post /= total
    post_s = pd.Series(post, index=xs)
    mean = float(np.sum(post_s.values * post_s.index))
    std = float(np.sqrt(np.sum((post_s.index - mean) ** 2 * post_s.values)))
    return mean, std, post_s


# ---------------------------------------------------------------------------
# Leave-one-out Spearman analysis
# ---------------------------------------------------------------------------

def leave_one_out(units, dates, values, min_n=5):
    """
    Compute Spearman ρ with each unit removed in turn.

    Parameters
    ----------
    units  : list of str   — unit names
    dates  : array-like    — date (BCE; larger = older)
    values : array-like    — the statistic being tested (e.g. posterior mean)
    min_n  : int           — minimum remaining n to bother computing

    Returns
    -------
    DataFrame with columns: removed_unit, rho, p_value, delta_rho
    """
    dates = np.array(dates, dtype=float)
    values = np.array(values, dtype=float)

    # Full-set correlation
    full_r, full_p = spearmanr(-dates, values)

    rows = []
    for i, unit in enumerate(units):
        mask = np.ones(len(units), dtype=bool)
        mask[i] = False
        if mask.sum() < min_n:
            continue
        r, p = spearmanr(-dates[mask], values[mask])
        rows.append({
            'removed_unit': unit,
            'date_bce': int(dates[i]),
            'rho_full': round(full_r, 4),
            'rho_loo':  round(r, 4),
            'p_loo':    round(p, 4),
            'delta_rho': round(r - full_r, 4),
            'sign_change': (np.sign(r) != np.sign(full_r)),
            'p_crosses_05': ((full_p < 0.05) != (p < 0.05)),
        })

    loo_df = pd.DataFrame(rows)

    # Robustness summary
    n_total = len(loo_df)
    n_sig = (loo_df['p_loo'] < 0.05).sum() if n_total > 0 else 0
    any_sign_change = loo_df['sign_change'].any() if n_total > 0 else False
    frac_sig = n_sig / n_total if n_total > 0 else 0.0

    robust = (not any_sign_change) and (frac_sig >= 0.8)
    return loo_df, full_r, full_p, robust, frac_sig


# ---------------------------------------------------------------------------
# Per-feature analysis
# ---------------------------------------------------------------------------

def analyse_feature(df, feat, outdir):
    """
    Run Bayesian posterior + LOO analysis for one feature on the filtered df.
    Returns a summary dict.
    """
    label = feat['label']
    kind = feat['kind']
    n_sources = len(df)

    # --- Compute per-unit values (posterior mean for binary; rate for rate) ---
    unit_values = []
    unit_stds   = []

    if kind == 'binary':
        col_A, col_B = feat['col_A'], feat['col_B']
        if col_A not in df.columns or col_B not in df.columns:
            print(f"  [{label}] columns not found — skipping.")
            return None

        total_A = df[col_A].sum()
        total_B = df[col_B].sum()

        for _, row in df.iterrows():
            kA = int(row[col_A])
            kB = int(row[col_B])
            n  = kA + kB
            if n == 0:
                unit_values.append(np.nan)
                unit_stds.append(np.nan)
            else:
                mean, std, _ = bayesian_posterior(total_A, total_B, n_sources, kB, n)
                unit_values.append(mean)
                unit_stds.append(std)

    else:  # rate
        col = feat['col_A']
        if col not in df.columns:
            print(f"  [{label}] column '{col}' not found — skipping.")
            return None
        for _, row in df.iterrows():
            rate = row[col] / max(row['n_words'], 1) * 1000
            unit_values.append(rate)
            unit_stds.append(np.nan)

    df = df.copy()
    df['_value'] = unit_values
    df['_std']   = unit_stds
    valid = df.dropna(subset=['_value'])

    if len(valid) < 5:
        print(f"  [{label}] Only {len(valid)} valid units — skipping.")
        return None

    dates  = valid['date_bce'].values
    values = valid['_value'].values
    units  = valid['unit'].values
    stds   = valid['_std'].values

    # --- Spearman on full filtered set ---
    full_r, full_p = spearmanr(-dates, values)

    # --- LOO ---
    loo_df, _, _, robust, frac_sig = leave_one_out(list(units), dates, values)

    # Most influential unit (largest |delta_rho|)
    if len(loo_df) > 0:
        worst_idx = loo_df['delta_rho'].abs().idxmax()
        worst_unit = loo_df.loc[worst_idx, 'removed_unit']
        worst_delta = loo_df.loc[worst_idx, 'delta_rho']
    else:
        worst_unit, worst_delta = 'N/A', 0.0

    direction_ok = (full_r > 0) if feat['direction'] == 'increase' else (full_r < 0)

    print(f"\n  Feature: {label}")
    print(f"    Genre-filtered n    = {len(valid)}")
    print(f"    Spearman ρ          = {full_r:.3f}  (p = {full_p:.4f})")
    print(f"    Expected direction  : {'✓' if direction_ok else '✗'} ({feat['direction']})")
    print(f"    LOO-robust          : {'YES' if robust else 'NO'}  "
          f"({frac_sig*100:.0f}% of LOO subsets p<0.05)")
    print(f"    Most influential    : {worst_unit} (Δρ = {worst_delta:+.3f})")

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"{label}  |  genre-filtered ({len(valid)} units)", fontsize=12)

    # Panel 1: posterior means (or rates) over time
    ax = axes[0]
    ax.invert_xaxis()
    if kind == 'binary':
        ax.errorbar(dates, values, yerr=np.where(np.isnan(stds), 0, stds),
                    fmt='o', color='steelblue', capsize=3, label='Posterior mean ± 1σ')
        # Prior line
        prior_p = df[feat['col_B']].sum() / max(df[feat['col_A']].sum() + df[feat['col_B']].sum(), 1)
        ax.axhline(prior_p, color='red', alpha=0.3, linewidth=1.5, label=f'Corpus prior ({prior_p:.2f})')
    else:
        ax.scatter(dates, values, color='steelblue', s=60)
    for date, val, unit in zip(dates, values, units):
        ax.annotate(unit, (date, val), xytext=(3, 4), textcoords='offset points', fontsize=7)
    ax.set_xlabel('Date (BCE)')
    ax.set_ylabel(label)
    ax.set_title(feat['description'], fontsize=8, wrap=True)
    stat_txt = (f"ρ={full_r:.3f}  p={full_p:.4f}\n"
                f"Direction: {'✓' if direction_ok else '✗'}\n"
                f"LOO-robust: {'YES' if robust else 'NO'}")
    ax.text(0.02, 0.97, stat_txt, transform=ax.transAxes, va='top', fontsize=8,
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if robust else 'lightyellow', alpha=0.8))
    ax.legend(fontsize=7)

    # Panel 2: LOO ρ waterfall
    ax2 = axes[1]
    if len(loo_df) > 0:
        loo_sorted = loo_df.sort_values('date_bce', ascending=False)
        colors = ['tomato' if pc else 'steelblue' for pc in loo_sorted['p_crosses_05']]
        ax2.barh(loo_sorted['removed_unit'], loo_sorted['rho_loo'], color=colors, alpha=0.8)
        ax2.axvline(full_r, color='black', linewidth=1.5, linestyle='--', label=f'Full ρ={full_r:.3f}')
        ax2.axvline(0, color='gray', linewidth=0.8)
        ax2.set_xlabel('Spearman ρ (leave-one-out)')
        ax2.set_title('LOO sensitivity\n(red = removing this unit changes p<0.05 conclusion)')
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data for LOO', ha='center', va='center',
                 transform=ax2.transAxes)

    # Panel 3: log-odds (for binary) or residuals from trend line (for rate)
    ax3 = axes[2]
    ax3.invert_xaxis()
    if kind == 'binary':
        eps = 1e-9
        log_odds = np.log(np.clip(values, eps, 1 - eps) / (1 - np.clip(values, eps, 1 - eps)))
        ax3.scatter(dates, log_odds, color='steelblue', s=60)
        for date, lo, unit in zip(dates, log_odds, units):
            ax3.annotate(unit, (date, lo), xytext=(3, 3), textcoords='offset points', fontsize=7)
        # Trend line
        finite = np.isfinite(log_odds)
        if finite.sum() >= 3:
            m, b = np.polyfit(-dates[finite], log_odds[finite], 1)
            x_line = np.linspace(dates.min(), dates.max(), 100)
            ax3.plot(x_line, m * (-x_line) + b, 'r--', alpha=0.5, label='OLS trend')
        ax3.set_ylabel('Log-odds of newer form')
        ax3.set_title('Log-odds (linearises logistic hypothesis)')
        ax3.legend(fontsize=8)
    else:
        # Rate with trend
        m, b = np.polyfit(-dates, values, 1)
        x_line = np.linspace(dates.min(), dates.max(), 100)
        ax3.scatter(dates, values, color='steelblue', s=60)
        ax3.plot(x_line, m * (-x_line) + b, 'r--', alpha=0.5, label='OLS trend')
        for date, val, unit in zip(dates, values, units):
            ax3.annotate(unit, (date, val), xytext=(3, 3), textcoords='offset points', fontsize=7)
        ax3.set_ylabel(f'{label} (per 1k words)')
        ax3.set_title('Rate trend')
        ax3.legend(fontsize=8)
    ax3.set_xlabel('Date (BCE)')

    plt.tight_layout()
    safe = label.replace(' ', '_').replace('→', 'to').replace('/', '_').replace('×', 'x')
    out_path = outdir / f'genre_{safe}.png'
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"    Plot saved: {out_path.name}")

    # Save LOO table
    if len(loo_df) > 0:
        loo_path = outdir / f'loo_{safe}.csv'
        loo_df.to_csv(str(loo_path), index=False)

    return {
        'feature':          label,
        'n_units':          len(valid),
        'spearman_r':       round(full_r, 4),
        'spearman_p':       round(full_p, 4),
        'direction_ok':     direction_ok,
        'loo_robust':       robust,
        'pct_loo_sig':      round(frac_sig * 100, 1),
        'most_influential': worst_unit,
        'delta_rho':        round(worst_delta, 4),
        'verdict':          ('CREDIBLE' if robust and full_p < 0.05 and direction_ok
                             else 'FRAGILE' if full_p < 0.05
                             else 'NOT SIGNIFICANT'),
    }


# ---------------------------------------------------------------------------
# Within-genre PCA (minimal, since script 03 covers this in full)
# ---------------------------------------------------------------------------

def genre_pca(df, outdir, genre_label):
    """Quick PCA for the genre-filtered set."""
    cols = ['rate_verb_wayyiqtol', 'rate_verb_participle', 'rate_verb_inf_abs',
            'rate_verb_yiqtol', 'rate_neg_ein', 'frac_ani', 'frac_she',
            'frac_non_wayyiqtol', 'frac_ein']
    available = [c for c in cols if c in df.columns]
    if len(available) < 3 or len(df) < 4:
        return

    X = df[available].values.astype(float)
    # Impute column medians
    for j in range(X.shape[1]):
        nans = np.isnan(X[:, j])
        if nans.any():
            X[nans, j] = np.nanmedian(X[:, j])

    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    cov = np.cov(Xs.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0)
    eigvecs = eigvecs[:, order]
    scores = Xs @ eigvecs
    explained = eigvals / max(eigvals.sum(), 1e-12)

    dates = df['date_bce'].values
    units = df['unit'].values
    norm = (dates - dates.min()) / max(dates.max() - dates.min(), 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'PCA — {genre_label} books only  (n={len(df)})', fontsize=12)

    cmap = plt.cm.RdYlBu_r
    ax = axes[0]
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=norm, cmap=cmap, s=80, zorder=3)
    plt.colorbar(sc, ax=ax, label='Relative date (blue=older, red=newer)')
    for i, u in enumerate(units):
        ax.annotate(u, (scores[i, 0], scores[i, 1]),
                    xytext=(4, 4), textcoords='offset points', fontsize=7)
    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)
    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)')
    ax.set_title('Biplot by date (blue→red = old→new)')

    ax2 = axes[1]
    ax2.invert_xaxis()
    ax2.scatter(dates, scores[:, 0], c=norm, cmap=cmap, s=80)
    for i, u in enumerate(units):
        ax2.annotate(u, (dates[i], scores[i, 0]),
                     xytext=(3, 4), textcoords='offset points', fontsize=7)
    from scipy.stats import linregress
    m, b, r, p, _ = linregress(-dates, scores[:, 0])
    x_tr = np.linspace(dates.min(), dates.max(), 100)
    ax2.plot(x_tr, m * (-x_tr) + b, 'r--', alpha=0.5, label=f'OLS r={r:.2f} p={p:.3f}')
    sp_r, sp_p = spearmanr(-dates, scores[:, 0])
    ax2.set_title(f'PC1 vs date  Spearman ρ={sp_r:.2f} p={sp_p:.3f}')
    ax2.set_xlabel('Date (BCE)'); ax2.set_ylabel('PC1 score')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    out_path = outdir / f'genre_pca_{genre_label.replace(" ", "_")}.png'
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"\nGenre PCA plot saved: {out_path.name}")
    print(f"  PC1 explains {explained[0]*100:.1f}% of within-genre variance")
    print(f"  Spearman ρ (date vs PC1) = {sp_r:.3f}  (p = {sp_p:.4f})")

    print("\n  Top PC1 loadings:")
    pc1 = (eigvecs[:, 0])
    for idx in np.argsort(np.abs(pc1))[::-1][:5]:
        print(f"    {available[idx]:30s}  {pc1[idx]:+.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--features', default='features_by_book.csv',
                        help='CSV from 01_feature_extraction_etcbc.py')
    parser.add_argument('--genre', default='prophecy',
                        choices=['prophecy', 'narrative', 'poetry', 'wisdom', 'mixed', 'all'],
                        help='Genre to filter to (default: prophecy)')
    parser.add_argument('--outdir', default='.',
                        help='Directory for plots and tables')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load features
    feat_path = Path(args.features)
    if not feat_path.exists():
        print(f"ERROR: {feat_path} not found.")
        print("Run script 01 first:  python3 01_feature_extraction_etcbc.py")
        import sys; import sys; sys.exit(1)

    df_all = pd.read_csv(feat_path)
    print(f"Loaded {len(df_all)} units from {feat_path}")

    # Add genre column
    df_all['genre'] = df_all['unit'].map(GENRE_MAP).fillna('unknown')

    # Filter
    if args.genre == 'all':
        df = df_all.copy()
        genre_label = 'all genres'
    else:
        df = df_all[df_all['genre'] == args.genre].copy()
        genre_label = args.genre

    print(f"\nGenre filter: '{args.genre}' → {len(df)} units")
    if len(df) < 5:
        print("ERROR: fewer than 5 units after filtering — analysis not meaningful.")
        import sys; sys.exit(1)

    print(f"Units included: {', '.join(df.sort_values('date_bce', ascending=False)['unit'].tolist())}")

    # Run per-feature analysis
    summary_rows = []
    for feat in FEATURES:
        result = analyse_feature(df, feat, outdir)
        if result:
            summary_rows.append(result)

    # Summary table
    summary = pd.DataFrame(summary_rows)
    summary_path = outdir / f'genre_summary_{args.genre}.csv'
    summary.to_csv(str(summary_path), index=False)

    print(f"\n{'='*70}")
    print(f"SUMMARY — {genre_label.upper()}  ({len(df)} units)")
    print('='*70)
    display_cols = ['feature', 'n_units', 'spearman_r', 'spearman_p',
                    'direction_ok', 'pct_loo_sig', 'most_influential', 'verdict']
    print(summary[display_cols].to_string(index=False))
    print(f"\nSaved to: {summary_path}")

    # Genre PCA
    genre_pca(df, outdir, genre_label)

    # Interpretation guide
    credible = summary[summary['verdict'] == 'CREDIBLE']
    fragile  = summary[summary['verdict'] == 'FRAGILE']
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print('='*70)
    if len(credible) > 0:
        print(f"\nCREDIBLE diachronic signals ({len(credible)} feature(s)):")
        for _, r in credible.iterrows():
            print(f"  • {r['feature']}")
            print(f"    ρ={r['spearman_r']:.3f}, p={r['spearman_p']:.4f}, "
                  f"LOO-robust, direction consistent")
    else:
        print("\nNo features pass all three criteria (p<0.05 + correct direction + LOO-robust).")

    if len(fragile) > 0:
        print(f"\nFRAGILE signals — significant but NOT LOO-robust ({len(fragile)} feature(s)):")
        for _, r in fragile.iterrows():
            print(f"  • {r['feature']}")
            print(f"    ρ={r['spearman_r']:.3f}, p={r['spearman_p']:.4f}, "
                  f"driven primarily by: {r['most_influential']} (Δρ={r['delta_rho']:+.3f})")
            print(f"    Only {r['pct_loo_sig']:.0f}% of LOO subsets remain p<0.05")

    print(f"\nStatistical cautions:")
    print(f"  n = {len(df)} units — with this sample size, p-values should be treated as")
    print(f"  rough guides, not precise measures.  LOO robustness is the stronger test.")
    print(f"  A CREDIBLE result here means the signal is consistent and not driven by")
    print(f"  one outlier — it does not rule out confounds we haven't measured.")


if __name__ == '__main__':
    main()
