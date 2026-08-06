#!/usr/bin/env python3
"""
Sliding Window Diachronic Analysis
====================================
Orders the dated text units by their estimated date and applies a sliding
window over this temporal sequence to track how feature rates evolve.

Rationale
---------
Bayesian Beta-Binomial analysis (script 02) treats each book as an
independent observation.  But if the *rate of change* itself changed
(e.g., accelerated post-exile), a sliding window can reveal that.

The window approach also mitigates the influence of any single book:
because each book contributes to multiple windows, its effect is
"smoothed" across adjacent periods.

Two analysis modes
------------------
1.  **Fixed-width window in time** (default):  pool all books within
    ± W years of each centre point and compute bootstrapped feature means.

2.  **Fixed-count window**: move a window of N books along the
    date-sorted sequence.  Simpler, but ignores unequal temporal spacing.

Date uncertainty
----------------
Instead of treating dates as exact, we use Monte Carlo sampling:
  - Each book's date is drawn from N(μ, σ) according to its date_sigma.
  - We compute the feature rate for each draw and summarize across draws.
This gives honest uncertainty bands that widen when many books have
uncertain dates (as happens in the post-exilic period).

Usage
-----
    python 04_sliding_window_analysis.py [--features features_by_book.csv]
                                         [--window 75]   # years half-width
                                         [--n_mc 500]    # Monte Carlo draws
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# Same example data as scripts 02 and 03
EXAMPLE_DATA = {
    'Amos':        (760, 15,  4,  2, 100,  0, 250, 120,  80,  40, 10, 90,  5, 4900),
    'Hosea':       (725, 20,  3,  4, 120,  0, 180, 100,  90,  50,  8, 80,  4, 5200),
    'Micah':       (720, 20,  2,  1,  80,  0, 150,  80,  60,  30,  6, 60,  3, 3800),
    'Isaiah_1':    (700, 15, 12,  8, 350,  0, 600, 320, 210, 120, 25,240, 18,15000),
    'Jeremiah':    (590, 15,  8, 24, 700,  1,1100, 680, 420, 200, 35,400, 55,32000),
    'Lamentations':(586, 20,  0,  2,  20,  0,  10,  60,  40,  30,  2, 40,  6, 2500),
    'Ezekiel':     (570, 15,  5, 18, 620,  0, 900, 700, 450, 250, 28,350, 48,29000),
    'Isaiah_2':    (550, 20,  4, 12, 260,  0, 320, 280, 200, 140, 14,200, 22,11000),
    'Haggai':      (520,  5,  0,  3,  28,  0,  40,  50,  30,  20,  2, 30,  8, 1800),
    'Zechariah_1': (518,  5,  1,  7, 100,  1, 160, 130,  80,  60,  8, 90, 14, 5500),
    'Isaiah_3':    (500, 30,  3,  8, 130,  0, 200, 180, 130,  80, 10,140, 18, 7000),
    'Malachi':     (460, 20,  0,  6,  60,  0,  50,  80,  60,  40,  4, 55, 10, 3200),
    'Jonah':       (400, 50,  0,  4,  22,  3,  60,  55,  40,  28,  3, 50, 12, 2200),
    'Chronicles':  (350, 30,  3, 25, 480,  4, 700, 820, 380, 200, 18,320, 80,28000),
    'Esther':      (350, 50,  0,  8,  60,  2,  30, 120,  80,  60,  3,100, 28, 6000),
    'Ecclesiastes':(330, 80,  1, 28, 100, 68,  20, 200, 160,  80,  4,120, 55, 8000),
    'Daniel':      (167, 10,  0,  6,  80, 12,  15, 180, 140,  90,  3,140, 48, 9000),
}

# Features we will track, defined as (numerator column, denominator expression, label)
TRACKED_FEATURES = [
    # (col, 'per_word', label)  → rate per 1000 words
    # (col_A, col_B, 'fraction', label)  → col_B / (col_A + col_B)
    ('verb_wayyiqtol', None,           'per_word', 'Wayyiqtol rate (per 1k words)'),
    ('verb_participle', None,          'per_word', 'Participle rate (per 1k words)'),
    ('pronoun_anochi', 'pronoun_ani',  'fraction', 'Fraction אני (newer 1sg pronoun)'),
    ('rel_asher',      'rel_she',      'fraction', 'Fraction ש (LBH relative particle)'),
    ('neg_lo',         'neg_ein',      'fraction', 'Fraction אין (negative existential)'),
]


def compute_feature_values(df):
    """
    Compute a scalar feature value per book for each tracked feature.
    Returns df with new columns added.
    """
    df = df.copy()
    for spec in TRACKED_FEATURES:
        label = spec[-1]
        if spec[2] == 'per_word':
            col = spec[0]
            if col in df.columns:
                df[label] = df[col] / df['n_words'].clip(lower=1) * 1000
            else:
                df[label] = np.nan
        else:  # fraction
            col_A, col_B = spec[0], spec[1]
            if col_A in df.columns and col_B in df.columns:
                denom = df[col_A] + df[col_B]
                df[label] = np.where(denom > 0, df[col_B] / denom, np.nan)
            else:
                df[label] = np.nan
    return df


def fixed_time_window(df, feat_label, centre_dates, half_width, weights='token'):
    """
    For each centre date, pool all books within ±half_width years and
    compute the weighted mean feature value.

    weights='token'  → weight each book by its token count (n_words)
    weights='unit'   → equal weight per book
    """
    means, lower, upper, n_books = [], [], [], []
    for c in centre_dates:
        mask = (df['date_bce'] >= c - half_width) & (df['date_bce'] <= c + half_width)
        sub = df[mask].dropna(subset=[feat_label])
        if len(sub) == 0:
            means.append(np.nan)
            lower.append(np.nan)
            upper.append(np.nan)
            n_books.append(0)
            continue
        if weights == 'token':
            w = sub['n_words'].values.astype(float)
        else:
            w = np.ones(len(sub))
        w /= w.sum()
        vals = sub[feat_label].values
        wmean = np.dot(w, vals)
        wstd = np.sqrt(np.dot(w, (vals - wmean)**2))
        means.append(wmean)
        lower.append(wmean - wstd)
        upper.append(wmean + wstd)
        n_books.append(len(sub))
    return np.array(means), np.array(lower), np.array(upper), np.array(n_books)


def monte_carlo_date_window(df, feat_label, centre_dates, half_width, n_mc=500):
    """
    Resample book dates from N(date_bce, date_sigma) and recompute
    fixed-time-window means across all draws.  Returns percentile bands.

    For small corpora, date uncertainty can substantially shift which
    books fall in each window.
    """
    all_means = np.full((n_mc, len(centre_dates)), np.nan)

    for draw in range(n_mc):
        # Perturb dates
        perturbed = df.copy()
        perturbed['date_bce'] = (
            df['date_bce'] +
            np.random.randn(len(df)) * df['date_sigma'].fillna(10)
        )
        means, _, _, _ = fixed_time_window(
            perturbed, feat_label, centre_dates, half_width, weights='token')
        all_means[draw, :] = means

    p50 = np.nanpercentile(all_means, 50, axis=0)
    p16 = np.nanpercentile(all_means, 16, axis=0)
    p84 = np.nanpercentile(all_means, 84, axis=0)
    return p50, p16, p84


def plot_sliding_window(df, outdir, half_width=75, n_mc=300):
    """
    Create a figure with one panel per tracked feature showing the
    sliding-window trend and Monte Carlo uncertainty bands.
    """
    df = compute_feature_values(df)
    feat_labels = [spec[-1] for spec in TRACKED_FEATURES
                   if spec[-1] in df.columns]

    # Centre dates: span the range of the corpus with enough resolution
    date_min = df['date_bce'].min() - half_width
    date_max = df['date_bce'].max() + half_width
    centre_dates = np.arange(date_min, date_max + 1, 25)  # every 25 years

    n_feats = len(feat_labels)
    fig, axes = plt.subplots(n_feats, 1, figsize=(12, 4 * n_feats), sharex=True)
    if n_feats == 1:
        axes = [axes]
    fig.suptitle(f'Sliding window analysis  (half-width = ±{half_width} yr, '
                 f'token-weighted,  MC n={n_mc})', fontsize=13)

    for ax, feat_label in zip(axes, feat_labels):
        # Deterministic window
        means, lo, hi, n_b = fixed_time_window(
            df, feat_label, centre_dates, half_width)

        # Monte Carlo uncertainty from date jitter
        mc_med, mc_lo, mc_hi = monte_carlo_date_window(
            df, feat_label, centre_dates, half_width, n_mc=n_mc)

        # Plot: most recent dates on the right, oldest on the left
        # x-axis is date BCE (inverted so time flows left→right)
        ax.fill_between(centre_dates, mc_lo, mc_hi,
                        alpha=0.2, color='steelblue', label='MC 16–84% band')
        ax.plot(centre_dates, mc_med, 'steelblue', linewidth=2, label='MC median')
        ax.plot(centre_dates, means, 'k--', linewidth=1, alpha=0.6,
                label='Deterministic mean')

        # Scatter individual books
        valid = df.dropna(subset=[feat_label])
        ax.scatter(valid['date_bce'], valid[feat_label],
                   s=40, color='darkorange', zorder=4, alpha=0.8)
        for _, row in valid.iterrows():
            ax.annotate(row['unit'], (row['date_bce'], row[feat_label]),
                        xytext=(3, 3), textcoords='offset points', fontsize=7)

        ax.invert_xaxis()
        ax.set_ylabel(feat_label, fontsize=9)

        # Books-per-window count as secondary axis
        ax2 = ax.twinx()
        ax2.bar(centre_dates, n_b, width=20, alpha=0.12, color='gray',
                label='Books in window')
        ax2.set_ylabel('Books/window', fontsize=7, color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        ax2.set_ylim(0, max(n_b.max() * 4, 4))

        ax.legend(loc='upper left', fontsize=7)

        # Spearman on deterministic means (only where n_b > 0)
        valid_mask = n_b > 0
        if valid_mask.sum() >= 5:
            sp_r, sp_p = spearmanr(-centre_dates[valid_mask], means[valid_mask])
            ax.set_title(f'{feat_label}   Spearman ρ={sp_r:.2f}  p={sp_p:.3f}',
                         fontsize=9)
        else:
            ax.set_title(feat_label, fontsize=9)

    axes[-1].set_xlabel('Date (BCE)  ←  older    newer  →', fontsize=10)
    plt.tight_layout()
    out_path = outdir / 'sliding_window.png'
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Sliding window plot saved: {out_path}")


def fixed_count_window(df, feat_labels, window_size=5):
    """
    Sort books by date and move a window of fixed size, computing
    the mean of each feature in each window.
    Returns a DataFrame of window-level statistics.
    """
    df_sorted = df.sort_values('date_bce', ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    records = []
    for start in range(0, n - window_size + 1):
        window = df_sorted.iloc[start: start + window_size]
        w_dates = window['date_bce'].values
        rec = {
            'window_centre_date': w_dates.mean(),
            'window_date_range': f"{int(w_dates.max())}–{int(w_dates.min())} BCE",
            'books': ', '.join(window['unit'].tolist()),
        }
        for feat in feat_labels:
            if feat in window.columns:
                vals = window[feat].dropna().values
                w = window.loc[window[feat].notna(), 'n_words'].values.astype(float)
                if len(vals) > 0 and w.sum() > 0:
                    rec[feat] = np.average(vals, weights=w)
                else:
                    rec[feat] = np.nan
        records.append(rec)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', default='features_by_book.csv')
    parser.add_argument('--window', type=int, default=75,
                        help='Half-width of time window in years (default: 75)')
    parser.add_argument('--n_mc', type=int, default=300,
                        help='Number of Monte Carlo date-jitter samples (default: 300)')
    parser.add_argument('--outdir', default='.')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feat_path = Path(args.features)
    if feat_path.exists():
        df = pd.read_csv(feat_path)
        print(f"Loaded {len(df)} units from {feat_path}")
    else:
        print(f"NOTE: {feat_path} not found — using built-in example data.")
        cols = ['unit', 'date_bce', 'date_sigma',
                'pronoun_anochi', 'pronoun_ani', 'rel_asher', 'rel_she',
                'verb_wayyiqtol', 'verb_qatal', 'verb_yiqtol',
                'verb_participle', 'verb_inf_abs', 'neg_lo', 'neg_ein', 'n_words']
        rows = [[u] + list(v) for u, v in EXAMPLE_DATA.items()]
        df = pd.DataFrame(rows, columns=cols)

    if 'date_sigma' not in df.columns:
        df['date_sigma'] = 25  # default uncertainty if not provided

    print(f"\nRunning sliding window analysis (±{args.window} yr, MC n={args.n_mc})…")
    plot_sliding_window(df, outdir, half_width=args.window, n_mc=args.n_mc)

    # Also output fixed-count window table
    df2 = compute_feature_values(df)
    feat_labels = [spec[-1] for spec in TRACKED_FEATURES if spec[-1] in df2.columns]
    win_df = fixed_count_window(df2, feat_labels, window_size=5)
    win_path = outdir / 'fixed_count_windows.csv'
    win_df.to_csv(str(win_path), index=False)
    print(f"Fixed-count window table saved: {win_path}")
    print("\nFirst few windows:")
    print(win_df[['window_centre_date', 'window_date_range', 'books']].head(8).to_string(index=False))


if __name__ == '__main__':
    main()
