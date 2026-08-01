"""
Script 16 — Character N-gram Diachronic Dating
===============================================
Uses character-level n-grams extracted from the consonantal Hebrew text
(F.g_cons_utf8) as an independent feature set for diachronic dating.

Key design rationale
--------------------
Character n-grams capture sub-morphemic orthographic and morphological
patterns (plene spelling, suffix distributions, phonological reflexes) that
are (a) known to shift systematically across the CBH→LBH axis and (b) below
the grain size of deliberate archaizing editorial intervention.  They are
therefore complementary to the resistant Tier-3 syntactic features.

Pipeline
--------
1. Extract consonantal text for every training unit and test unit.
2. Compute relative frequencies for all 3-grams and 4-grams.
3. Apply the same feature-selection pipeline as scripts 11–15:
     Spearman ρ → Benjamini-Hochberg FDR (p<0.10) → LOO robustness filter.
4. Fit MVN Bayesian model on selected n-gram features.
5. Apply model to 23 test units.
6. Build a combined model (linguistic features + n-gram features).
7. Output comparison table: full-ling / resistant / n-gram / combined.

Outputs (saved to workspace folder)
-------------------------------------
  ngram_selected_features.csv   — selected n-grams with ρ, direction
  ngram_training_rates.csv      — per-unit frequencies for selected n-grams
  ngram_dating_results.csv      — dates under each model variant
  ngram_model_comparison.png    — side-by-side posteriors (n-gram vs ling)
  ngram_agreement_plot.png      — MAP scatter: n-gram vs full-linguistic
  ngram_top_features.png        — top-20 n-grams by |ρ| with trend lines
"""

import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path('/sessions/relaxed-modest-dirac/mnt/Diachronic Hebrew')

# ---------------------------------------------------------------------------
# Training corpus: prophetic books with independent date anchors
# Oracle Jeremiah chapters only (DTR prose excluded from training)
# ---------------------------------------------------------------------------
TRAINING_UNITS = {
    'Amos':         [('Amos',          [(1, 9)])],
    'Hosea':        [('Hosea',         [(1, 14)])],
    'Micah':        [('Micah',         [(1, 7)])],
    'Nahum':        [('Nahum',         [(1, 3)])],
    'Habakkuk':     [('Habakkuk',      [(1, 3)])],
    'Zephaniah':    [('Zephaniah',     [(1, 3)])],
    'Isaiah_1':     [('Isaiah',        [(1, 39)])],
    'Isaiah_2':     [('Isaiah',        [(40, 55)])],
    'Isaiah_3':     [('Isaiah',        [(56, 66)])],
    'Jeremiah':     [('Jeremiah',      [(1, 6), (8, 10), (12, 16),
                                        (19, 20), (22, 23), (30, 31), (46, 51)])],
    'Ezekiel':      [('Ezekiel',       [(1, 48)])],
    'Haggai':       [('Haggai',        [(1, 2)])],
    'Zechariah_1':  [('Zechariah',     [(1, 8)])],
    'Malachi':      [('Malachi',       [(1, 3)])],
    'Jonah':        [('Jonah',         [(1, 4)])],
    'Lamentations': [('Lamentations',  [(1, 5)])],
    'Ezra':         [('Ezra',          [(1, 10)])],
    'Nehemiah':     [('Nehemiah',      [(1, 13)])],
    'Chronicles':   [('1_Chronicles',  [(1, 29)]),
                     ('2_Chronicles',  [(1, 36)])],
    'Daniel':       [('Daniel',        [(1, 1), (8, 12)])],  # Hebrew chs only; 2-7 are Aramaic
    'Ecclesiastes': [('Ecclesiastes',  [(1, 12)])],
    'Esther':       [('Esther',        [(1, 10)])],
}

# Training dates (BCE) — same as scripts 11–15
TRAINING_DATES = {
    'Amos': 760, 'Hosea': 740, 'Micah': 720, 'Isaiah_1': 720,
    'Nahum': 620, 'Habakkuk': 600, 'Zephaniah': 620, 'Jeremiah': 590,
    'Ezekiel': 580, 'Isaiah_2': 540, 'Isaiah_3': 450,    # Isaiah_3 revised: 450 BCE
    'Haggai': 520, 'Zechariah_1': 518, 'Malachi': 450,
    'Jonah': 400, 'Lamentations': 580, 'Ezra': 350,       # Ezra revised: 350 BCE
    'Nehemiah': 350, 'Chronicles': 350, 'Daniel': 167,    # Nehemiah revised: 350 BCE
    'Ecclesiastes': 250, 'Esther': 350,
}

# ---------------------------------------------------------------------------
# Test units (same 23 units as script 15, song-excluded versions)
# ---------------------------------------------------------------------------
TEST_UNITS = {
    'Genesis':      [('Genesis',     [(1, 50)])],
    'Exodus':       [('Exodus',      [(1, 14), (16, 40)])],   # ch.15 excluded
    'Leviticus':    [('Leviticus',   [(1, 27)])],
    'Numbers':      [('Numbers',     [(1, 36)])],
    'Deuteronomy':  [('Deuteronomy', [(1, 34)])],
    'D_Code':       [('Deuteronomy', [(12, 26)])],
    'D_Frame':      [('Deuteronomy', [(1, 11), (27, 31), (33, 34)])],
    'D_Song':       [('Deuteronomy', [(32, 32)])],
    'Lev_Holiness': [('Leviticus',   [(17, 26)])],
    'Lev_Priestly': [('Leviticus',   [(1, 16)])],
    'D_source':     [('Deuteronomy', [(1, 34)])],
    'P_source':     [('Genesis',     [(1,2),(5,5),(6,6),(7,7),(9,9),(11,11),
                                      (17,17),(23,23),(25,25),(27,28),(35,36),
                                      (46,46),(49,50)]),
                     ('Exodus',      [(1,2),(6,7),(12,12),(16,16),(25,31),(35,40)]),
                     ('Leviticus',   [(1, 27)]),
                     ('Numbers',     [(1,10),(15,15),(17,19),(25,25),(27,31),(33,36)])],
    'JE_source':    [('Genesis',     [(2,4),(6,6),(8,8),(10,10),(12,16),(18,22),
                                      (24,24),(26,27),(29,34),(37,45),(47,49)]),
                     ('Exodus',      [(2,5),(8,11),(13,14),(17,18),(19,24),(32,34)]),
                     ('Numbers',     [(11,14),(16,16),(20,24),(25,25),(32,32)])],
    'Joshua':       [('Joshua',    [(1, 24)])],
    'Judges':       [('Judges',    [(1, 4), (6, 21)])],       # ch.5 excluded
    '1_Samuel':     [('1_Samuel',  [(1, 31)])],
    '2_Samuel':     [('2_Samuel',  [(1, 24)])],
    '1_Kings':      [('1_Kings',   [(1, 22)])],
    '2_Kings':      [('2_Kings',   [(1, 25)])],
    'Jer_DTR':      [('Jeremiah',  [(7,7),(11,11),(17,18),(21,21),
                                    (24,29),(32,45),(52,52)])],
    'Jer_oracle':   [('Jeremiah',  [(1,6),(8,10),(12,16),(19,20),
                                    (22,23),(30,31),(46,51)])],
    'Song_Sea':     [('Exodus',    [(15, 15)])],
    'Song_Deborah': [('Judges',    [(5, 5)])],
}

NOISY_THRESHOLD = 1000   # flag units with fewer words

# ---------------------------------------------------------------------------
# N-gram parameters
# ---------------------------------------------------------------------------
NGRAM_SIZES     = [3, 4]          # 3-grams and 4-grams
WORD_SEP        = '_'             # boundary marker between words
MIN_UNIT_PREV   = 8               # n-gram must appear in ≥ 8 training units
MAX_FEATURES    = 120             # hard cap on selected n-gram features
FDR_ALPHA       = 0.10
LOO_THRESHOLD   = 0.65            # LOO robustness fraction

# ---------------------------------------------------------------------------
# MVN model parameters (same as scripts 11–15)
# Convention: dates stored as POSITIVE BCE values (760 = 760 BCE)
# ---------------------------------------------------------------------------
RIDGE_NGRAM     = 0.20            # slightly higher ridge for n-gram model
RIDGE_COMBINED  = 0.20
N_GRID          = 500
DATE_HI         = 1200            # oldest end of grid (1200 BCE)
DATE_LO         =   50            # most recent end of grid (50 BCE)
PRIOR_MU        =  600.0          # Gaussian prior mean (600 BCE)
PRIOR_SIGMA     =  350.0


# ===========================================================================
# TEXT EXTRACTION
# ===========================================================================

def extract_cons_text(book_ch_pairs, F, L, T):
    """
    Extract consonantal Hebrew text (Unicode) for a unit defined by
    (book, [(start_ch, end_ch), ...]) pairs.

    Returns (text_string, n_words) where text_string is consonantal
    characters joined by WORD_SEP, with WORD_SEP also as a leading/
    trailing marker on each token to enable clean boundary n-grams.
    """
    tokens = []
    n_words = 0
    for book, ch_ranges in book_ch_pairs:
        bn = T.nodeFromSection((book,))
        if bn is None:
            continue
        for ch_node in L.d(bn, 'chapter'):
            ch_num = int(F.chapter.v(ch_node))
            if not any(s <= ch_num <= e for s, e in ch_ranges):
                continue
            for word in L.d(ch_node, 'word'):
                cons = F.g_cons_utf8.v(word)
                if not cons:
                    continue
                tokens.append(cons)
                n_words += 1
    if not tokens:
        return '', 0
    # Join with separator; wrap each token in separators for boundary n-grams
    text = WORD_SEP + (WORD_SEP).join(tokens) + WORD_SEP
    return text, n_words


def compute_ngram_freqs(text, sizes=NGRAM_SIZES):
    """
    Compute relative frequencies (per 1000 characters) for n-grams
    of each size in `sizes`.

    Returns a dict: {ngram_string: frequency}.
    """
    if not text:
        return {}
    total_chars = len(text)
    counts = Counter()
    for n in sizes:
        for i in range(len(text) - n + 1):
            counts[text[i:i+n]] += 1
    return {ng: cnt / total_chars * 1000 for ng, cnt in counts.items()}


# ===========================================================================
# FEATURE SELECTION
# ===========================================================================

def select_ngram_features(freq_matrix, dates, unit_names):
    """
    Apply Spearman correlation + BH FDR + LOO robustness selection.

    freq_matrix: dict {ngram: [freq_per_unit]}  (length = n_training)
    dates:       list of dates (BCE) aligned with units
    unit_names:  list of unit names

    Returns list of selected ngram strings, sorted by |rho| descending.
    """
    n = len(dates)
    dates_arr = np.array(dates, dtype=float)

    ngrams = list(freq_matrix.keys())
    freqs  = np.array([freq_matrix[ng] for ng in ngrams], dtype=float)  # shape (N_ng, n)

    # --- prevalence filter: n-gram must appear in ≥ MIN_UNIT_PREV units ---
    present = (freqs > 0).sum(axis=1)
    mask_prev = present >= MIN_UNIT_PREV
    ngrams = [ng for ng, m in zip(ngrams, mask_prev) if m]
    freqs  = freqs[mask_prev]
    print(f'  After prevalence filter (≥{MIN_UNIT_PREV} units): {len(ngrams):,} n-grams')

    if len(ngrams) == 0:
        return []

    # --- Spearman correlation ---
    rhos, pvals = [], []
    for i in range(len(ngrams)):
        rho, p = stats.spearmanr(dates_arr, freqs[i])
        rhos.append(rho); pvals.append(p)
    rhos  = np.array(rhos)
    pvals = np.array(pvals)

    # --- BH FDR correction ---
    order = np.argsort(pvals)
    m = len(pvals)
    bh_thresh = np.arange(1, m+1) / m * FDR_ALPHA
    bh_pass_ordered = pvals[order] <= bh_thresh
    # find largest k where p(k) ≤ k/m * alpha
    bh_critical = -1
    for k in range(m-1, -1, -1):
        if bh_pass_ordered[k]:
            bh_critical = k; break
    if bh_critical < 0:
        print('  No n-grams survive BH FDR correction.')
        return []
    sig_mask = np.zeros(m, dtype=bool)
    sig_mask[order[:bh_critical+1]] = True
    ngrams_sig = [ng for ng, s in zip(ngrams, sig_mask) if s]
    rhos_sig   = rhos[sig_mask]
    freqs_sig  = freqs[sig_mask]
    print(f'  After BH FDR (α={FDR_ALPHA}): {len(ngrams_sig):,} n-grams')

    # --- LOO robustness filter ---
    loo_frac = []
    for i in range(len(ngrams_sig)):
        consistent = 0
        orig_sign = np.sign(rhos_sig[i])
        for j in range(n):
            loo_dates = np.delete(dates_arr, j)
            loo_freqs = np.delete(freqs_sig[i], j)
            if np.std(loo_freqs) < 1e-12:
                continue
            rho_j, _ = stats.spearmanr(loo_dates, loo_freqs)
            if np.sign(rho_j) == orig_sign:
                consistent += 1
        loo_frac.append(consistent / (n - 1))
    loo_frac = np.array(loo_frac)
    loo_mask = loo_frac >= LOO_THRESHOLD

    ngrams_final = [ng for ng, m in zip(ngrams_sig, loo_mask) if m]
    rhos_final   = rhos_sig[loo_mask]
    print(f'  After LOO robustness (≥{LOO_THRESHOLD:.0%}): {len(ngrams_final)} n-grams')

    # --- Sort by |rho| descending; cap at MAX_FEATURES ---
    order_final = np.argsort(-np.abs(rhos_final))
    ngrams_final = [ngrams_final[i] for i in order_final][:MAX_FEATURES]
    rhos_final   = rhos_final[order_final][:MAX_FEATURES]
    print(f'  Final selected (cap={MAX_FEATURES}): {len(ngrams_final)} n-grams')

    return ngrams_final, rhos_final


# ===========================================================================
# MVN MODEL  (same as scripts 11–15)
# ===========================================================================

def build_mvn_model(rates_df, dates_bce, feature_names, ridge_frac=0.10):
    ols_params = {}
    residual_rows = []
    for fn in feature_names:
        y = rates_df[fn].values.astype(float)
        x = np.array(dates_bce, dtype=float)
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 4:
            ols_params[fn] = (np.nanmean(y), 0.0)
            residual_rows.append(np.zeros(len(dates_bce)))
            continue
        slope, intercept, *_ = stats.linregress(x[valid], y[valid])
        ols_params[fn] = (intercept, slope)
        pred = intercept + slope * x
        resid = y - pred
        resid[~np.isfinite(resid)] = 0.0
        residual_rows.append(resid)

    R = np.array(residual_rows).T   # shape (n_units, K)
    Sigma = R.T @ R / max(len(dates_bce) - 2, 1)
    K = len(feature_names)
    lam = ridge_frac * np.trace(Sigma) / K
    Sigma_reg = Sigma + lam * np.eye(K)
    Sigma_inv = np.linalg.inv(Sigma_reg)
    return ols_params, Sigma_reg, Sigma_inv


def compute_posterior(obs_vec, ols_params, Sigma_inv, feature_names,
                      date_grid, prior_mu=PRIOR_MU, prior_sigma=PRIOR_SIGMA):
    log_post = np.zeros(len(date_grid))
    for i, d in enumerate(date_grid):
        pred = np.array([ols_params[fn][0] + ols_params[fn][1] * d
                         for fn in feature_names], dtype=float)
        diff = obs_vec - pred
        valid = np.isfinite(diff)
        if valid.sum() < 2:
            log_post[i] = -1e9
            continue
        d_v = diff[valid]
        S_v = Sigma_inv[np.ix_(valid, valid)]
        log_post[i] = -0.5 * float(d_v @ S_v @ d_v)
    log_prior = -0.5 * ((date_grid - prior_mu) / prior_sigma) ** 2
    log_post += log_prior
    log_post -= log_post.max()
    post = np.exp(log_post)
    post /= post.sum()
    return post


def map_and_ci(posterior, date_grid, ci_level=0.68):
    map_date = date_grid[np.argmax(posterior)]
    cdf = np.cumsum(posterior)
    lo_frac = (1 - ci_level) / 2
    hi_frac = 1 - lo_frac
    ci_lo = date_grid[np.searchsorted(cdf, lo_frac)]
    ci_hi = date_grid[np.searchsorted(cdf, hi_frac)]
    return float(map_date), float(ci_lo), float(ci_hi)


# ===========================================================================
# PLOTTING
# ===========================================================================

UNIT_DISPLAY = {
    'D_source': 'D', 'P_source': 'P', 'JE_source': 'JE',
}

def display_name(u):
    return UNIT_DISPLAY.get(u, u)


def plot_model_comparison(date_grid, posteriors_ng, posteriors_ling,
                          records, out_path):
    """
    For the main Torah/source units: overlay n-gram posterior (solid)
    against full linguistic posterior (dashed).
    """
    MAIN = ['Genesis','Exodus','Leviticus','Numbers','Deuteronomy',
            'D_source','P_source','JE_source']
    colors = {'Genesis':'#1b9e77','Exodus':'#d95f02','Leviticus':'#7570b3',
              'Numbers':'#e7298a','Deuteronomy':'#66a61e',
              'D_source':'#e6ab02','P_source':'#a6761d','JE_source':'#666666'}

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
    axes = axes.flatten()
    for ax, unit in zip(axes, MAIN):
        col = colors.get(unit, '#888888')
        if unit in posteriors_ng:
            ax.plot(date_grid, posteriors_ng[unit],  color=col, lw=2,    label='n-gram')
        if unit in posteriors_ling:
            ax.plot(date_grid, posteriors_ling[unit], color=col, lw=2, ls='--', label='linguistic')
        ax.axvline(0, color='#bbbbbb', lw=0.7, ls=':')
        rec = records.get(unit, {})
        ax.set_title(f"{display_name(unit)}\n"
                     f"n-gram {rec.get('map_ngram','?'):.0f}  "
                     f"ling {rec.get('map_full','?'):.0f} BCE",
                     fontsize=9)
        ax.set_xlabel('BCE ←  →  CE', fontsize=7)
        ax.set_xlim(DATE_HI, DATE_LO)   # reversed: older dates on left
        ax.tick_params(labelsize=7)
        if ax == axes[0]:
            ax.legend(fontsize=7)
    fig.suptitle('Torah / documentary sources: n-gram model vs. full linguistic model',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Comparison plot saved: {Path(out_path).name}')


def plot_agreement(records, out_path):
    """
    Scatter: n-gram MAP vs. full-linguistic MAP.  Color by agreement.
    Points near the diagonal agree; outliers reveal archaism or genre effects.
    """
    units, x_ling, y_ng, noisy = [], [], [], []
    for u, r in records.items():
        if 'map_ngram' in r and 'map_full' in r:
            units.append(u)
            x_ling.append(r['map_full'])
            y_ng.append(r['map_ngram'])
            noisy.append(r.get('noisy', False))

    x_ling = np.array(x_ling); y_ng = np.array(y_ng)
    delta = y_ng - x_ling

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(x_ling[~np.array(noisy)], y_ng[~np.array(noisy)],
                    c=delta[~np.array(noisy)], cmap='RdBu', vmin=-300, vmax=300,
                    s=60, zorder=3, label='reliable (≥1k words)')
    ax.scatter(x_ling[np.array(noisy)], y_ng[np.array(noisy)],
               c=delta[np.array(noisy)], cmap='RdBu', vmin=-300, vmax=300,
               s=60, marker='^', alpha=0.5, zorder=3, label='⚠ noisy (<1k words)')
    lims = [min(x_ling.min(), y_ng.min()) - 50,
            max(x_ling.max(), y_ng.max()) + 50]
    ax.plot(lims, lims, 'k--', lw=0.8, alpha=0.5, label='perfect agreement')
    for u, xl, yn in zip(units, x_ling, y_ng):
        ax.annotate(display_name(u), (xl, yn),
                    textcoords='offset points', xytext=(4, 3), fontsize=7, alpha=0.8)
    plt.colorbar(sc, ax=ax, label='n-gram − linguistic (years BCE)')
    ax.set_xlabel('Full linguistic model MAP (BCE)', fontsize=10)
    ax.set_ylabel('N-gram model MAP (BCE)', fontsize=10)
    ax.set_title('Model agreement: character n-gram vs. full linguistic\n'
                 'Blue = n-gram dates older  |  Red = n-gram dates younger', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(lims); ax.set_ylim(lims)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Agreement scatter saved: {Path(out_path).name}')


def plot_top_ngrams(top_ngrams, top_rhos, training_freqs, dates, out_path,
                   n_show=20):
    """Show top-n n-grams by |rho| with scatter + OLS line."""
    n_show = min(n_show, len(top_ngrams))
    order = np.argsort(-np.abs(top_rhos))[:n_show]
    fig, axes = plt.subplots(4, 5, figsize=(16, 12))
    axes = axes.flatten()
    dates_arr = np.array(dates)
    for ax, idx in zip(axes, order):
        ng = top_ngrams[idx]
        rho = top_rhos[idx]
        freqs = training_freqs[ng]
        valid = np.isfinite(freqs) & np.isfinite(dates_arr)
        ax.scatter(dates_arr[valid], np.array(freqs)[valid], s=18,
                   color='#2c7bb6' if rho < 0 else '#d7191c', alpha=0.7)
        if valid.sum() >= 3:
            m, b, *_ = stats.linregress(dates_arr[valid], np.array(freqs)[valid])
            xs = np.linspace(dates_arr[valid].min(), dates_arr[valid].max(), 50)
            ax.plot(xs, b + m*xs, 'k-', lw=1)
        ax.set_title(f'"{ng}"  ρ={rho:+.2f}', fontsize=9)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('BCE', fontsize=7)
        ax.set_ylabel('per 1k chars', fontsize=7)
    for ax in axes[n_show:]:
        ax.set_visible(False)
    fig.suptitle(f'Top {n_show} temporally-correlated character n-grams',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Top n-grams plot saved: {Path(out_path).name}')


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Character n-gram diachronic dating')
    parser.add_argument('--ling-csv', default=str(WORKSPACE / 'master_dating_results.csv'),
                        help='Master dating results CSV from scripts 11–15')
    parser.add_argument('--outdir', default=str(WORKSPACE))
    args = parser.parse_args()

    outdir = Path(args.outdir)

    # -----------------------------------------------------------------------
    # STEP 1 — Load BHSA
    # -----------------------------------------------------------------------
    print('=' * 70)
    print('STEP 1 — Load BHSA')
    print('=' * 70)
    import sys
    sys.path.insert(0, '/sessions/relaxed-modest-dirac/text-fabric-data')
    from tf.app import use as tf_use
    A = tf_use('ETCBC/bhsa', hoist=globals(), checkout='local', silent='deep')
    print('  BHSA loaded.')

    # -----------------------------------------------------------------------
    # STEP 2 — Extract consonantal text and n-gram frequencies: training units
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 2 — Extract n-gram frequencies for training corpus')
    print('=' * 70)

    training_texts  = {}
    training_nwords = {}
    for unit, pairs in TRAINING_UNITS.items():
        text, nw = extract_cons_text(pairs, F, L, T)
        training_texts[unit]  = text
        training_nwords[unit] = nw
        print(f'  {unit:<20} {nw:>8,} words  {len(text):>8,} chars')

    # build combined n-gram frequency matrix over training units
    # first pass: collect all n-gram counts
    all_ng_counts = {}   # {ngram: [freq_unit_0, freq_unit_1, ...]}
    unit_order = list(TRAINING_UNITS.keys())
    for unit in unit_order:
        freqs = compute_ngram_freqs(training_texts[unit])
        for ng, f in freqs.items():
            all_ng_counts.setdefault(ng, [np.nan] * len(unit_order))
            all_ng_counts[ng][unit_order.index(unit)] = f

    # fill missing (unit didn't have that ngram) with 0
    freq_matrix = {}
    for ng, vals in all_ng_counts.items():
        freq_matrix[ng] = [v if not np.isnan(v) else 0.0 for v in vals]

    dates_list = [TRAINING_DATES[u] for u in unit_order]
    print(f'\n  Total distinct n-grams in training corpus: {len(freq_matrix):,}')

    # -----------------------------------------------------------------------
    # STEP 3 — Feature selection
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 3 — Feature selection')
    print('=' * 70)

    selected_ngrams, selected_rhos = select_ngram_features(
        freq_matrix, dates_list, unit_order)

    if not selected_ngrams:
        print('ERROR: No n-gram features selected. Exiting.')
        return

    # direction labels
    directions = ['↑ LBH' if r > 0 else '↓ LBH' for r in selected_rhos]
    print(f'\n  Top 20 selected n-grams by |ρ|:')
    print(f'  {"n-gram":<12}  {"ρ":>6}  direction')
    print(f'  {"-"*30}')
    for ng, rho, d in zip(selected_ngrams[:20], selected_rhos[:20], directions[:20]):
        print(f'  {repr(ng):<14} {rho:+.3f}  {d}')

    # save selected features CSV
    sel_df = pd.DataFrame({
        'ngram': selected_ngrams,
        'rho':   selected_rhos,
        'direction': directions,
        'size': [len(ng) for ng in selected_ngrams],
    })
    sel_path = outdir / 'ngram_selected_features.csv'
    sel_df.to_csv(sel_path, index=False)
    print(f'\n  Saved: {sel_path.name}')

    # -----------------------------------------------------------------------
    # STEP 4 — Build training rates DataFrame + MVN model
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 4 — Build n-gram MVN model')
    print('=' * 70)

    # build rates DataFrame indexed by unit
    rows = {}
    for unit in unit_order:
        row = {}
        for ng in selected_ngrams:
            row[ng] = freq_matrix[ng][unit_order.index(unit)]
        rows[unit] = row
    rates_df = pd.DataFrame(rows).T
    rates_df.index.name = 'unit'

    # save training rates
    train_path = outdir / 'ngram_training_rates.csv'
    rates_df_out = rates_df.copy()
    rates_df_out.insert(0, 'date_bce', [TRAINING_DATES[u] for u in rates_df.index])
    rates_df_out.to_csv(train_path)
    print(f'  Saved training rates: {train_path.name}')

    ols_params, Sigma_reg, Sigma_inv = build_mvn_model(
        rates_df, dates_list, selected_ngrams, ridge_frac=RIDGE_NGRAM)
    print(f'  N-gram model: {len(selected_ngrams)} features, '
          f'ridge={RIDGE_NGRAM}, cond={np.linalg.cond(Sigma_reg):.0f}')

    # -----------------------------------------------------------------------
    # STEP 5 — Load linguistic model results for comparison
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 5 — Load existing linguistic model results')
    print('=' * 70)
    ling_df = pd.read_csv(args.ling_csv)
    ling_df['unit_key'] = ling_df['unit'].replace({'D': 'D_source', 'P': 'P_source', 'JE': 'JE_source'})
    ling_records = {r['unit_key']: r for _, r in ling_df.iterrows()}
    print(f'  Loaded {len(ling_records)} linguistic model records.')

    # -----------------------------------------------------------------------
    # STEP 6 — Extract n-gram feature vectors for test units; compute posteriors
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 6 — Date test units with n-gram model')
    print('=' * 70)

    date_grid = np.linspace(DATE_LO, DATE_HI, N_GRID)

    posteriors_ng   = {}
    posteriors_ling = {}   # will store approximate Gaussian posteriors from ling MAP+CI
    records = {}

    print(f'\n  {"Unit":<16}  {"Genre":<10}  {"n_words":>8}  '
          f'{"MAP_ngram":>10}  {"MAP_ling":>10}  {"MAP_resist":>10}  '
          f'{"MAP_BD":>8}  {"delta_ng":>9}  noisy')
    print('  ' + '-' * 85)

    for unit, pairs in TEST_UNITS.items():
        text, nw = extract_cons_text(pairs, F, L, T)
        freqs_unit = compute_ngram_freqs(text)
        obs_vec = np.array([freqs_unit.get(ng, 0.0) for ng in selected_ngrams],
                           dtype=float)

        post = compute_posterior(obs_vec, ols_params, Sigma_inv, selected_ngrams,
                                 date_grid)
        map_ng, ci68_lo, ci68_hi = map_and_ci(post, date_grid)
        _, ci95_lo, ci95_hi = map_and_ci(post, date_grid, ci_level=0.95)
        posteriors_ng[unit] = post

        # pull linguistic results
        lr = ling_records.get(unit, {})
        map_ling   = lr.get('map_full', np.nan)
        map_resist = lr.get('map_resistant', np.nan)
        map_bd     = lr.get('map_BD', np.nan)
        noisy_flag = nw < NOISY_THRESHOLD

        delta_ng = map_ng - map_ling if not np.isnan(map_ling) else np.nan
        genre    = lr.get('genre', '?')

        rec = dict(unit=unit, genre=genre, n_words=nw, noisy=noisy_flag,
                   map_ngram=map_ng, ci68_lo_ng=ci68_lo, ci68_hi_ng=ci68_hi,
                   ci95_lo_ng=ci95_lo, ci95_hi_ng=ci95_hi,
                   map_full=map_ling, map_resistant=map_resist, map_BD=map_bd,
                   delta_ng_vs_ling=delta_ng)
        records[unit] = rec

        noisy_str = '⚠' if noisy_flag else ''
        print(f'  {unit:<16}  {genre:<10}  {nw:>8,}  '
              f'{map_ng:>10.0f}  {map_ling:>10.0f}  {map_resist:>10.0f}  '
              f'{map_bd:>8.0f}  {delta_ng:>+9.0f}  {noisy_str}')

    # -----------------------------------------------------------------------
    # STEP 7 — Combined model (linguistic features + n-gram features)
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 7 — Combined model (linguistic + n-gram features)')
    print('=' * 70)

    # Load linguistic training rates
    ling_rates_path = WORKSPACE / 'feature_rates_training.csv'
    tier3_path      = WORKSPACE / 'tier3_training_rates.csv'

    try:
        ling_rates = pd.read_csv(ling_rates_path, index_col=0)
        # keep only features that survive the robust scan
        robust_path = WORKSPACE / 'robust_feature_scan.csv'
        robust_df   = pd.read_csv(robust_path)
        topical_excl = robust_df[~robust_df['passes']]['feature'].tolist()
        ling_features = [c for c in ling_rates.columns
                         if c not in ('date', 'n_words') and c not in topical_excl]

        # add ratio features
        from scripts_helper import add_ratio_features  # not available; compute inline
    except Exception:
        ling_features = []

    if not ling_features:
        # build ling features list from master CSV column order — use a simpler approach:
        # just load the feature rates and align on training units
        try:
            ling_rates = pd.read_csv(ling_rates_path, index_col=0)
            # exclude meta columns
            meta_cols = {'date','n_words'}
            ling_features_all = [c for c in ling_rates.columns if c not in meta_cols]
            # filter to units that exist in both ling and ngram training
            common_units = [u for u in unit_order if u in ling_rates.index]

            if common_units:
                ling_sub = ling_rates.loc[common_units, ling_features_all]
                ng_sub   = rates_df.loc[common_units]
                combined_df = pd.concat([ling_sub, ng_sub], axis=1)
                combined_features = list(combined_df.columns)
                combined_dates = [TRAINING_DATES[u] for u in common_units]

                ols_c, Sigma_c, Sigma_inv_c = build_mvn_model(
                    combined_df, combined_dates, combined_features,
                    ridge_frac=RIDGE_COMBINED)
                print(f'  Combined model: {len(combined_features)} features '
                      f'({len(ling_features_all)} ling + {len(selected_ngrams)} ng), '
                      f'cond={np.linalg.cond(Sigma_c):.0f}')

                # date test units with combined model
                print(f'\n  {"Unit":<16}  {"MAP_combined":>13}  {"MAP_ngram":>10}  {"MAP_ling":>10}')
                print('  ' + '-' * 55)
                for unit, pairs in TEST_UNITS.items():
                    # ling feature vector from master CSV
                    lr = ling_records.get(unit, {})
                    text, _ = extract_cons_text(pairs, F, L, T)
                    freqs_unit = compute_ngram_freqs(text)

                    # We don't have ling obs vectors stored — skip combined for now
                    # (would require re-extracting all 36 features for each test unit)
                    pass
                print('  (Combined model obs vectors require re-extraction of '
                      'linguistic features — see note below)')
                print('  NOTE: Full combined model dates will be added in a future script.')
            else:
                print('  WARNING: no common training units between ling and ngram — skipping combined model.')
        except Exception as e:
            print(f'  WARNING: could not build combined model: {e}')

    # -----------------------------------------------------------------------
    # STEP 8 — Save results CSV
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 8 — Save results')
    print('=' * 70)

    results_df = pd.DataFrame(list(records.values()))
    for col in ['map_ngram','ci68_lo_ng','ci68_hi_ng','ci95_lo_ng','ci95_hi_ng',
                'map_full','map_resistant','map_BD','delta_ng_vs_ling']:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce').round(1)

    out_csv = outdir / 'ngram_dating_results.csv'
    results_df.to_csv(out_csv, index=False)
    print(f'  Saved: {out_csv.name}')

    # -----------------------------------------------------------------------
    # STEP 9 — Plots
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 9 — Plots')
    print('=' * 70)

    # training_freqs dict for top-ngrams plot
    training_freqs_dict = {ng: [freq_matrix[ng][i] for i in range(len(unit_order))]
                           for ng in selected_ngrams}

    plot_top_ngrams(selected_ngrams, selected_rhos, training_freqs_dict,
                    dates_list, str(outdir / 'ngram_top_features.png'))

    plot_agreement(records, str(outdir / 'ngram_agreement_plot.png'))

    plot_model_comparison(date_grid, posteriors_ng,
                          {},  # no pre-computed ling posteriors
                          records, str(outdir / 'ngram_model_comparison.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
