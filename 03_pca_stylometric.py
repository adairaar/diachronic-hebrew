#!/usr/bin/env python3
"""
PCA Stylometric Analysis of Biblical Hebrew
============================================
Uses Principal Component Analysis (PCA) over a vector of normalized
morphosyntactic feature rates to look for a temporal gradient in the
writing style of the dated Hebrew Bible books.

Why PCA (not word2vec)?
-----------------------
word2vec learns semantic geometry from word co-occurrence statistics.
On a 300k-token corpus it is too data-hungry: the variance of embedding
vectors swamps any real signal.  PCA over interpretable feature rates is:
  (a) Identifiable — each principal component is a weighted sum of features
      we can name and interpret linguistically.
  (b) Data-efficient — only ~15–20 dimensions, not 100–1000.
  (c) Falsifiable — a temporal gradient in PC-space has a clear prediction:
      books should separate along PCs in order of date.

Features used
-------------
Rate columns from script 01 (counts per 1000 words).  Binary pair fractions
are included alongside the raw rates to give the model both absolute and
relative information.

Caveats
-------
Genre confounds PCA severely: narrative prose uses more wayyiqtol than
prophetic poetry regardless of date.  We therefore flag genre for each unit
and check whether any PCA separation reflects date, genre, or both.
The truly diachronically informative features are those that vary *within*
a genre across time, not those that vary *between* genres.

Usage
-----
    python 03_pca_stylometric.py [--features features_by_book.csv]
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Genre labels (narrative prose vs. prophetic poetry vs. wisdom)
# These are crucial for interpreting PCA — genre dominates stylistic distance.
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
    'Jeremiah':     'prophecy',    # mixed, but primarily prophetic
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

GENRE_COLORS = {
    'prophecy':  'steelblue',
    'narrative': 'darkorange',
    'poetry':    'mediumseagreen',
    'wisdom':    'mediumpurple',
    'mixed':     'gray',
}

# ---------------------------------------------------------------------------
# Built-in example dataset (same as script 02) — used if CSV not found
# ---------------------------------------------------------------------------

EXAMPLE_DATA = {
    'Amos':        (760, 15, 'prophecy',  4,  2, 100,  0, 250, 120,  80,  40, 10, 90,  5, 4900),
    'Hosea':       (725, 20, 'prophecy',  3,  4, 120,  0, 180, 100,  90,  50,  8, 80,  4, 5200),
    'Micah':       (720, 20, 'prophecy',  2,  1,  80,  0, 150,  80,  60,  30,  6, 60,  3, 3800),
    'Isaiah_1':    (700, 15, 'prophecy', 12,  8, 350,  0, 600, 320, 210, 120, 25,240, 18,15000),
    'Jeremiah':    (590, 15, 'prophecy',  8, 24, 700,  1,1100, 680, 420, 200, 35,400, 55,32000),
    'Lamentations':(586, 20, 'poetry',    0,  2,  20,  0,  10,  60,  40,  30,  2, 40,  6, 2500),
    'Ezekiel':     (570, 15, 'prophecy',  5, 18, 620,  0, 900, 700, 450, 250, 28,350, 48,29000),
    'Isaiah_2':    (550, 20, 'prophecy',  4, 12, 260,  0, 320, 280, 200, 140, 14,200, 22,11000),
    'Haggai':      (520,  5, 'prophecy',  0,  3,  28,  0,  40,  50,  30,  20,  2, 30,  8, 1800),
    'Zechariah_1': (518,  5, 'prophecy',  1,  7, 100,  1, 160, 130,  80,  60,  8, 90, 14, 5500),
    'Isaiah_3':    (500, 30, 'prophecy',  3,  8, 130,  0, 200, 180, 130,  80, 10,140, 18, 7000),
    'Malachi':     (460, 20, 'prophecy',  0,  6,  60,  0,  50,  80,  60,  40,  4, 55, 10, 3200),
    'Jonah':       (400, 50, 'narrative', 0,  4,  22,  3,  60,  55,  40,  28,  3, 50, 12, 2200),
    'Chronicles':  (350, 30, 'narrative', 3, 25, 480,  4, 700, 820, 380, 200, 18,320, 80,28000),
    'Esther':      (350, 50, 'narrative', 0,  8,  60,  2,  30, 120,  80,  60,  3,100, 28, 6000),
    'Ecclesiastes':(330, 80, 'wisdom',    1, 28, 100, 68,  20, 200, 160,  80,  4,120, 55, 8000),
    'Daniel':      (167, 10, 'mixed',     0,  6,  80, 12,  15, 180, 140,  90,  3,140, 48, 9000),
}


def build_rate_matrix(df):
    """
    Compute per-1000-word rates and binary fractions.
    Returns a feature matrix X (n_units × n_features) and a list of feature names.
    """
    n = df['n_words'].values.astype(float)
    per = 1000.0

    features = {}

    # Absolute rates
    for col in ['verb_wayyiqtol', 'verb_qatal', 'verb_yiqtol',
                'verb_participle', 'verb_inf_abs', 'neg_lo', 'neg_ein', 'yesh']:
        if col in df.columns:
            features[f'rate_{col}'] = df[col].values / n * per

    # Binary pair fractions (the LBH form's share)
    if 'pronoun_anochi' in df.columns and 'pronoun_ani' in df.columns:
        denom = df['pronoun_anochi'].values + df['pronoun_ani'].values
        features['frac_ani'] = np.where(denom > 0,
                                         df['pronoun_ani'].values / denom, np.nan)

    if 'rel_asher' in df.columns and 'rel_she' in df.columns:
        denom = df['rel_asher'].values + df['rel_she'].values
        features['frac_she'] = np.where(denom > 0,
                                         df['rel_she'].values / denom, np.nan)

    wyq = df['verb_wayyiqtol'].values
    qatal = df['verb_qatal'].values if 'verb_qatal' in df.columns else np.zeros(len(df))
    denom = wyq + qatal
    features['frac_non_wayyiqtol'] = np.where(denom > 0,
                                                (denom - wyq) / denom, np.nan)

    if 'neg_lo' in df.columns and 'neg_ein' in df.columns:
        denom = df['neg_lo'].values + df['neg_ein'].values
        features['frac_ein'] = np.where(denom > 0,
                                         df['neg_ein'].values / denom, np.nan)

    feat_names = list(features.keys())
    X = np.column_stack([features[k] for k in feat_names])
    return X, feat_names


def run_pca(X, feat_names):
    """
    Manual PCA (no sklearn dependency).  Centers and scales X, then
    computes covariance eigendecomposition.

    Returns: scores (n × k), loadings (k × p), explained_variance_ratio
    """
    # Drop rows or columns with too many NaNs
    col_nan_frac = np.isnan(X).mean(axis=0)
    keep_cols = col_nan_frac < 0.4
    X = X[:, keep_cols]
    feat_names = [f for f, k in zip(feat_names, keep_cols) if k]

    # Impute remaining NaNs with column median
    for j in range(X.shape[1]):
        nans = np.isnan(X[:, j])
        if nans.any():
            X[nans, j] = np.nanmedian(X[:, j])

    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    # Covariance and eigen-decomposition
    cov = np.cov(Xs.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by descending eigenvalue
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    eigenvalues = np.maximum(eigenvalues, 0)  # numerical safety
    total_var = eigenvalues.sum()
    explained = eigenvalues / max(total_var, 1e-12)

    scores = Xs @ eigenvectors
    return scores, eigenvectors, explained, feat_names


def plot_pca(df, scores, loadings, explained, feat_names, outdir):
    """
    Create four diagnostic plots:
      1. Biplot (PC1 vs PC2) coloured by date
      2. Biplot coloured by genre
      3. PC1 vs date — the key temporal test
      4. Scree plot
    """
    dates = df['date_bce'].values
    units = df['unit'].values
    genres = [GENRE_MAP.get(u, 'unknown') for u in units]

    # Date colormap
    cmap = plt.cm.RdYlBu_r
    norm_date = (dates - dates.min()) / max(dates.max() - dates.min(), 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("PCA Stylometric Analysis — Dated Biblical Hebrew Books", fontsize=14)

    # --- Panel 1: PC1 vs PC2, coloured by date -------------------------
    ax = axes[0, 0]
    sc = ax.scatter(scores[:, 0], scores[:, 1], c=norm_date, cmap=cmap,
                    s=80, zorder=3)
    for i, u in enumerate(units):
        ax.annotate(u, (scores[i, 0], scores[i, 1]),
                    xytext=(4, 4), textcoords='offset points', fontsize=7)
    plt.colorbar(sc, ax=ax, label='Relative date (blue=older, red=newer)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}% variance)')
    ax.set_title('Biplot by date')

    # Overlay top loading vectors
    scale = min(abs(scores[:, :2]).max(axis=0).max() * 0.6, 2.5)
    for j, fname in enumerate(feat_names[:min(len(feat_names), 8)]):
        lx, ly = loadings[0, j] * scale, loadings[1, j] * scale
        if abs(lx) + abs(ly) < 0.3:
            continue
        ax.annotate('', xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='darkred', alpha=0.6))
        ax.text(lx * 1.1, ly * 1.1, fname.replace('rate_', '').replace('frac_', ''),
                fontsize=7, color='darkred', ha='center')

    # --- Panel 2: PC1 vs PC2, coloured by genre -------------------------
    ax = axes[0, 1]
    for u, g, s0, s1 in zip(units, genres, scores[:, 0], scores[:, 1]):
        ax.scatter(s0, s1, color=GENRE_COLORS.get(g, 'gray'), s=80, zorder=3)
        ax.annotate(u, (s0, s1), xytext=(4, 4),
                    textcoords='offset points', fontsize=7)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}% variance)')
    ax.set_title('Biplot by genre')
    patches = [mpatches.Patch(color=c, label=g)
               for g, c in GENRE_COLORS.items() if g in genres]
    ax.legend(handles=patches, fontsize=7, loc='best')

    # --- Panel 3: PC1 vs date (key temporal test) ----------------------
    ax = axes[1, 0]
    ax.invert_xaxis()
    ax.scatter(dates, scores[:, 0], c=norm_date, cmap=cmap, s=80)
    for i, u in enumerate(units):
        ax.annotate(u, (dates[i], scores[i, 0]),
                    xytext=(3, 4), textcoords='offset points', fontsize=7)
    # Simple OLS trendline
    from scipy.stats import linregress
    m, b, r, p, _ = linregress(-dates, scores[:, 0])
    x_trend = np.linspace(dates.min(), dates.max(), 100)
    ax.plot(x_trend, m * (-x_trend) + b, 'r--', alpha=0.5,
            label=f'OLS trend  r={r:.2f}  p={p:.3f}')
    ax.set_xlabel('Date (BCE)')
    ax.set_ylabel('PC1 score')
    ax.set_title('PC1 score vs. date — temporal gradient test')
    ax.legend(fontsize=8)

    # --- Panel 4: Scree plot -------------------------------------------
    ax = axes[1, 1]
    k = min(len(explained), 10)
    ax.bar(range(1, k + 1), explained[:k] * 100, color='steelblue', alpha=0.7)
    ax.plot(range(1, k + 1), np.cumsum(explained[:k]) * 100, 'ro-', label='Cumulative')
    ax.axhline(80, color='gray', linestyle='--', linewidth=0.8, label='80% threshold')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Variance explained (%)')
    ax.set_title('Scree plot')
    ax.legend(fontsize=8)
    ax.set_xticks(range(1, k + 1))

    plt.tight_layout()
    out_path = outdir / 'pca_stylometric.png'
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"PCA plot saved: {out_path}")

    # Print top loadings
    print("\nTop feature loadings on PC1 (most stylistically discriminating):")
    pc1_load = loadings[0, :]
    order = np.argsort(np.abs(pc1_load))[::-1]
    for idx in order[:6]:
        print(f"  {feat_names[idx]:30s}  {pc1_load[idx]:+.3f}")

    from scipy.stats import spearmanr
    sp_r, sp_p = spearmanr(-dates, scores[:, 0])
    print(f"\nSpearman ρ (date vs PC1) = {sp_r:.3f}  (p = {sp_p:.4f})")
    if sp_p < 0.05:
        print("  → Credible temporal gradient on PC1.")
    else:
        print("  → No significant temporal gradient on PC1.")
    print("\nNote: significant PC1–date correlation does not distinguish")
    print("  diachronic change from genre shift.  Inspect genre coloring.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', default='features_by_book.csv')
    parser.add_argument('--outdir', default='.')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feat_path = Path(args.features)
    if feat_path.exists():
        df = pd.read_csv(feat_path)
        if 'genre' not in df.columns:
            df['genre'] = df['unit'].map(GENRE_MAP).fillna('unknown')
        print(f"Loaded {len(df)} units from {feat_path}")
    else:
        print(f"NOTE: {feat_path} not found — using built-in example data.")
        cols = ['unit', 'date_bce', 'date_sigma', 'genre',
                'pronoun_anochi', 'pronoun_ani', 'rel_asher', 'rel_she',
                'verb_wayyiqtol', 'verb_qatal', 'verb_yiqtol',
                'verb_participle', 'verb_inf_abs', 'neg_lo', 'neg_ein', 'n_words']
        rows = [[u] + list(v) for u, v in EXAMPLE_DATA.items()]
        df = pd.DataFrame(rows, columns=cols)

    X, feat_names = build_rate_matrix(df)
    scores, loadings_T, explained, feat_names = run_pca(X, feat_names)

    # loadings_T is (n_components × n_features); we want (n_components × n_features)
    # eigenvectors from np.linalg.eigh are column vectors, so loadings_T.T[i] = PC_i
    # The transpose gives us the standard biplot loading convention
    loadings = loadings_T.T   # shape: (n_components, n_features)

    plot_pca(df, scores, loadings, explained, feat_names, outdir)


if __name__ == '__main__':
    main()
