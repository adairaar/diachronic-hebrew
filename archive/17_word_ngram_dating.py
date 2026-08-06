"""
Script 17 — Word N-gram Diachronic Dating
==========================================
Uses word-level n-grams as an independent feature set for diachronic dating.
Two complementary types are tested:

  Type A — POS-tag bigrams/trigrams
      Tokens are part-of-speech codes with verb-tense disambiguation
      (e.g., verb_wayq, prep, conj, subs).  Completely immune to topic
      because no content-word identities are used.  Sequences like
      "verb_wayq·conj·verb_wayq" (narrative wayyiqtol chain) or
      "prep·art·subs" (definite nominal phrase) can track syntactic
      register independently of vocabulary.

  Type B — Function-word lex bigrams/trigrams
      Tokens are lexeme codes (F.lex) for words drawn exclusively from
      the closed class of function words (prep, conj, art, nega, prps,
      prde, inrg).  Content words produce a None sentinel that causes
      any n-gram spanning them to be dropped.  Only adjacent function-
      word sequences survive.

Rationale
---------
Diachronic shifts in Hebrew grammar and syntax manifest not only in
individual feature rates (scripts 10–15) and character orthography
(script 16), but also in preferred collocational patterns among closed-
class items.  Word n-grams trained on the calibrated corpus provide a
third word-level signal that is:
  * topically neutral (no content lexemes in either type)
  * sensitive to word-order and collocational preferences
  * complementary to character n-grams (which track orthographic shape)

Pipeline
--------
1. Extract word sequences for every training unit and test unit.
2. Build POS-tag and function-word n-gram frequency matrices.
3. Feature selection: Spearman ρ → BH FDR (α=0.10) → LOO robustness.
4. Fit MVN Bayesian model for Type A, Type B, and combined (A+B).
5. Apply to 23 test units; compare with scripts 11–16.
6. Output comparison table and diagnostic plots.

Outputs (saved to workspace folder)
-------------------------------------
  word_ngram_typeA_features.csv    — selected POS-tag n-grams with ρ
  word_ngram_typeB_features.csv    — selected function-word n-grams
  word_ngram_training_rates.csv    — per-unit rates for all selected features
  word_ngram_dating_results.csv    — dates under Type A / B / combined
  word_ngram_typeA_top_features.png
  word_ngram_typeB_top_features.png
  word_ngram_agreement_plot.png    — MAP scatter vs. full linguistic model
  word_ngram_model_comparison.png  — posteriors for Torah/source units
"""

import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path('/sessions/relaxed-modest-dirac/mnt/Diachronic Hebrew')

# ---------------------------------------------------------------------------
# Training corpus (same as Script 16, calibrated dates)
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
    'Daniel':       [('Daniel',        [(1, 1), (8, 12)])],   # Hebrew chs only; 2–7 Aramaic
    'Ecclesiastes': [('Ecclesiastes',  [(1, 12)])],
    'Esther':       [('Esther',        [(1, 10)])],
}

# Training dates (BCE) — calibrated
TRAINING_DATES = {
    'Amos': 760, 'Hosea': 740, 'Micah': 720, 'Isaiah_1': 720,
    'Nahum': 620, 'Habakkuk': 600, 'Zephaniah': 620, 'Jeremiah': 590,
    'Ezekiel': 580, 'Isaiah_2': 540, 'Isaiah_3': 450,
    'Haggai': 520, 'Zechariah_1': 518, 'Malachi': 450,
    'Jonah': 400, 'Lamentations': 580, 'Ezra': 350,
    'Nehemiah': 350, 'Chronicles': 350, 'Daniel': 167,
    'Ecclesiastes': 250, 'Esther': 350,
}

# ---------------------------------------------------------------------------
# Test units (same 23 units as scripts 15–16)
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
NGRAM_SIZES   = [2, 3]          # bigrams and trigrams
MIN_UNIT_PREV = 8               # n-gram must appear in ≥ 8 training units
MAX_FEATURES  = 100             # hard cap on selected features per type
FDR_ALPHA     = 0.10
LOO_THRESHOLD = 0.65

# Closed-class POS tags for Type B function-word sequences
FUNCTION_POS = frozenset({'prep', 'conj', 'art', 'nega', 'prps', 'prde', 'inrg'})

# ---------------------------------------------------------------------------
# MVN model parameters
# Convention: dates stored as POSITIVE BCE values
# ---------------------------------------------------------------------------
RIDGE       = 0.20
N_GRID      = 500
DATE_HI     = 1200   # oldest end of grid
DATE_LO     =   50   # most recent end of grid
PRIOR_MU    = 600.0
PRIOR_SIGMA = 350.0


# ===========================================================================
# TOKEN EXTRACTION
# ===========================================================================

def pos_token(word, F):
    """
    Return a POS-tag token for a word.
    Verbs get tense disambiguation: verb_wayq, verb_perf, verb_impf, etc.
    All other words return F.sp (e.g., prep, conj, art, subs, nmpr …).
    """
    sp = F.sp.v(word) or 'unkn'
    if sp == 'verb':
        vt = F.vt.v(word) or 'unkn'
        return f'verb_{vt}'
    return sp


def extract_word_sequences(book_ch_pairs, F, L, T):
    """
    Extract two parallel token sequences for a text unit.

    Parameters
    ----------
    book_ch_pairs : list of (book_name, [(start_ch, end_ch), ...])

    Returns
    -------
    pos_seq  : list[str]  — POS-tag token for every word (Type A)
    fw_seq   : list[str|None]  — function-word lex for function words,
                                  None for content words (Type B)
    n_words  : int  — total word count
    """
    pos_seq = []
    fw_seq  = []
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
                sp = F.sp.v(word)
                if not sp:
                    continue
                pos_seq.append(pos_token(word, F))
                if sp in FUNCTION_POS:
                    fw_seq.append(F.lex.v(word) or sp)
                else:
                    fw_seq.append(None)   # content-word sentinel
                n_words += 1

    return pos_seq, fw_seq, n_words


# ===========================================================================
# N-GRAM COMPUTATION
# ===========================================================================

def build_ngrams(tokens, sizes=NGRAM_SIZES, skip_null=False):
    """
    Build a raw-count dict of n-gram strings from a token list.

    Parameters
    ----------
    tokens    : list[str|None]
    sizes     : list of n values
    skip_null : if True, discard any n-gram that contains a None token
                (used for Type B: keeps only all-function-word sequences)

    Returns
    -------
    dict {ngram_str: count}  where ngram_str uses '·' as token separator
    """
    counts = Counter()
    for n in sizes:
        for i in range(len(tokens) - n + 1):
            gram = tokens[i:i + n]
            if skip_null and any(t is None for t in gram):
                continue
            key = '·'.join(str(t) for t in gram)
            counts[key] += 1
    return dict(counts)


def counts_to_rates(counts, n_words, scale=1000.0):
    """Convert raw counts to rates per 1000 words."""
    if n_words == 0:
        return {}
    return {ng: cnt / n_words * scale for ng, cnt in counts.items()}


def build_freq_matrix(rates_by_unit, unit_order):
    """
    Collect all n-gram keys across units and return a unified matrix.

    Parameters
    ----------
    rates_by_unit : dict {unit: {ngram: rate}}
    unit_order    : list of unit names in desired order

    Returns
    -------
    dict {ngram_str: [rate_unit_0, rate_unit_1, ...]}
    """
    all_ng = set()
    for u in unit_order:
        all_ng.update(rates_by_unit[u].keys())
    return {ng: [rates_by_unit[u].get(ng, 0.0) for u in unit_order]
            for ng in all_ng}


# ===========================================================================
# FEATURE SELECTION  (same logic as Script 16)
# ===========================================================================

def select_features(freq_matrix, dates, label=''):
    """
    Spearman ρ → BH FDR correction → LOO robustness filter.

    Parameters
    ----------
    freq_matrix : dict {feature: [rate_per_unit]}
    dates       : list of BCE dates aligned with units
    label       : string prefix for progress messages

    Returns
    -------
    (selected_features, rhos_array)  — sorted by |ρ| descending
    """
    n         = len(dates)
    dates_arr = np.array(dates, dtype=float)
    features  = list(freq_matrix.keys())
    freqs     = np.array([freq_matrix[f] for f in features], dtype=float)

    # --- prevalence filter ---
    present   = (freqs > 0).sum(axis=1)
    mask_prev = present >= MIN_UNIT_PREV
    features  = [f for f, m in zip(features, mask_prev) if m]
    freqs     = freqs[mask_prev]
    print(f'  {label} after prevalence filter (≥{MIN_UNIT_PREV} units): {len(features):,}')

    if len(features) == 0:
        return [], np.array([])

    # --- Spearman correlation ---
    rhos, pvals = [], []
    for i in range(len(features)):
        rho, p = stats.spearmanr(dates_arr, freqs[i])
        rhos.append(rho)
        pvals.append(p)
    rhos  = np.array(rhos)
    pvals = np.array(pvals)

    # --- BH FDR correction ---
    order       = np.argsort(pvals)
    m_total     = len(pvals)
    bh_thresh   = np.arange(1, m_total + 1) / m_total * FDR_ALPHA
    bh_pass_ord = pvals[order] <= bh_thresh
    bh_critical = -1
    for k in range(m_total - 1, -1, -1):
        if bh_pass_ord[k]:
            bh_critical = k
            break
    if bh_critical < 0:
        print(f'  {label} no features survive BH FDR.')
        return [], np.array([])
    sig_mask = np.zeros(m_total, dtype=bool)
    sig_mask[order[:bh_critical + 1]] = True
    features_sig = [f for f, s in zip(features, sig_mask) if s]
    rhos_sig     = rhos[sig_mask]
    freqs_sig    = freqs[sig_mask]
    print(f'  {label} after BH FDR (α={FDR_ALPHA}): {len(features_sig):,}')

    # --- LOO robustness filter ---
    loo_frac = []
    for i in range(len(features_sig)):
        consistent = 0
        orig_sign  = np.sign(rhos_sig[i])
        for j in range(n):
            ld = np.delete(dates_arr, j)
            lf = np.delete(freqs_sig[i], j)
            if np.std(lf) < 1e-12:
                continue
            rho_j, _ = stats.spearmanr(ld, lf)
            if np.sign(rho_j) == orig_sign:
                consistent += 1
        loo_frac.append(consistent / (n - 1))
    loo_frac = np.array(loo_frac)
    loo_mask = loo_frac >= LOO_THRESHOLD

    features_final = [f for f, m in zip(features_sig, loo_mask) if m]
    rhos_final     = rhos_sig[loo_mask]
    print(f'  {label} after LOO robustness (≥{LOO_THRESHOLD:.0%}): {len(features_final)}')

    # --- sort by |ρ| descending; cap at MAX_FEATURES ---
    ord_f          = np.argsort(-np.abs(rhos_final))
    features_final = [features_final[i] for i in ord_f][:MAX_FEATURES]
    rhos_final     = rhos_final[ord_f][:MAX_FEATURES]
    print(f'  {label} final (cap={MAX_FEATURES}): {len(features_final)}')

    return features_final, rhos_final


# ===========================================================================
# MVN MODEL  (identical to scripts 11–16)
# ===========================================================================

def build_mvn_model(rates_df, dates_bce, feature_names, ridge_frac=RIDGE):
    ols_params    = {}
    residual_rows = []
    for fn in feature_names:
        y     = rates_df[fn].values.astype(float)
        x     = np.array(dates_bce, dtype=float)
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 4:
            ols_params[fn] = (np.nanmean(y), 0.0)
            residual_rows.append(np.zeros(len(dates_bce)))
            continue
        slope, intercept, *_ = stats.linregress(x[valid], y[valid])
        ols_params[fn] = (intercept, slope)
        pred  = intercept + slope * x
        resid = y - pred
        resid[~np.isfinite(resid)] = 0.0
        residual_rows.append(resid)
    R         = np.array(residual_rows).T
    Sigma     = R.T @ R / max(len(dates_bce) - 2, 1)
    K         = len(feature_names)
    lam       = ridge_frac * np.trace(Sigma) / K
    Sigma_r   = Sigma + lam * np.eye(K)
    Sigma_inv = np.linalg.inv(Sigma_r)
    return ols_params, Sigma_r, Sigma_inv


def compute_posterior(obs_vec, ols_params, Sigma_inv, feature_names,
                      date_grid, prior_mu=PRIOR_MU, prior_sigma=PRIOR_SIGMA):
    log_post = np.zeros(len(date_grid))
    for i, d in enumerate(date_grid):
        pred  = np.array([ols_params[fn][0] + ols_params[fn][1] * d
                          for fn in feature_names], dtype=float)
        diff  = obs_vec - pred
        valid = np.isfinite(diff)
        if valid.sum() < 2:
            log_post[i] = -1e9
            continue
        dv = diff[valid]
        Sv = Sigma_inv[np.ix_(valid, valid)]
        log_post[i] = -0.5 * float(dv @ Sv @ dv)
    log_prior = -0.5 * ((date_grid - prior_mu) / prior_sigma) ** 2
    log_post += log_prior
    log_post -= log_post.max()
    post  = np.exp(log_post)
    post /= post.sum()
    return post


def map_and_ci(posterior, date_grid, ci_level=0.68):
    map_date = date_grid[np.argmax(posterior)]
    cdf      = np.cumsum(posterior)
    lo_frac  = (1 - ci_level) / 2
    hi_frac  = 1 - lo_frac
    ci_lo    = date_grid[np.searchsorted(cdf, lo_frac)]
    ci_hi    = date_grid[np.searchsorted(cdf, hi_frac)]
    return float(map_date), float(ci_lo), float(ci_hi)


# ===========================================================================
# PLOTTING
# ===========================================================================

UNIT_DISPLAY = {'D_source': 'D', 'P_source': 'P', 'JE_source': 'JE'}


def display_name(u):
    return UNIT_DISPLAY.get(u, u)


def plot_top_features(features, rhos, freq_dict, dates_list, out_path, title,
                      n_show=20):
    """Scatter + OLS trend for top n features by |ρ|."""
    n_show = min(n_show, len(features))
    if n_show == 0:
        return
    order  = np.argsort(-np.abs(rhos))[:n_show]
    ncols  = 5
    nrows  = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    axes  = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()
    dates_arr = np.array(dates_list, dtype=float)
    for ax, idx in zip(axes, order):
        feat  = features[idx]
        rho   = rhos[idx]
        freqs = np.array(freq_dict[feat], dtype=float)
        valid = np.isfinite(freqs) & np.isfinite(dates_arr)
        ax.scatter(dates_arr[valid], freqs[valid], s=18,
                   color='#2c7bb6' if rho < 0 else '#d7191c', alpha=0.7)
        if valid.sum() >= 3:
            m_slope, b_int, *_ = stats.linregress(dates_arr[valid], freqs[valid])
            xs = np.linspace(dates_arr[valid].min(), dates_arr[valid].max(), 50)
            ax.plot(xs, b_int + m_slope * xs, 'k-', lw=1)
        label = feat if len(feat) <= 22 else feat[:19] + '…'
        ax.set_title(f'{label}\nρ={rho:+.2f}', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('BCE', fontsize=7)
        ax.set_ylabel('per 1k words', fontsize=7)
    for ax in axes[n_show:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


def plot_agreement_multi(records, out_path):
    """
    Three scatter panels: MAP_A, MAP_B, MAP_AB vs. full linguistic MAP.
    Blue = word n-gram dates older; red = word n-gram dates younger.
    """
    comparisons = [('map_A',  'Type A — POS-tag n-grams'),
                   ('map_B',  'Type B — function-word n-grams'),
                   ('map_AB', 'Combined A+B')]
    n_plots = sum(1 for k, _ in comparisons
                  if any(k in r and not np.isnan(r.get(k, np.nan))
                         for r in records.values()))
    if n_plots == 0:
        return
    fig, axes = plt.subplots(1, max(n_plots, 1), figsize=(6 * max(n_plots, 1), 6))
    if n_plots == 1:
        axes = [axes]
    axes = list(axes) if hasattr(axes, '__iter__') else [axes]

    ax_idx = 0
    for key, label in comparisons:
        units, x_v, y_v, noisy_v = [], [], [], []
        for u, r in records.items():
            val = r.get(key, np.nan)
            if isinstance(val, float) and np.isnan(val):
                continue
            ml = r.get('map_full', np.nan)
            if isinstance(ml, float) and np.isnan(ml):
                continue
            units.append(u)
            x_v.append(float(ml))
            y_v.append(float(val))
            noisy_v.append(bool(r.get('noisy', False)))
        if not units:
            continue
        ax = axes[ax_idx]; ax_idx += 1
        x_v = np.array(x_v); y_v = np.array(y_v)
        delta   = y_v - x_v
        noisy_a = np.array(noisy_v)
        sc = ax.scatter(x_v[~noisy_a], y_v[~noisy_a],
                        c=delta[~noisy_a], cmap='RdBu', vmin=-300, vmax=300,
                        s=60, zorder=3, label='reliable (≥1k words)')
        ax.scatter(x_v[noisy_a], y_v[noisy_a],
                   c=delta[noisy_a], cmap='RdBu', vmin=-300, vmax=300,
                   s=60, marker='^', alpha=0.5, zorder=3,
                   label='⚠ noisy (<1k words)')
        lo = min(x_v.min(), y_v.min()) - 50
        hi = max(x_v.max(), y_v.max()) + 50
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5)
        for u, xl, yn in zip(units, x_v, y_v):
            ax.annotate(display_name(u), (xl, yn),
                        textcoords='offset points', xytext=(4, 3),
                        fontsize=7, alpha=0.8)
        plt.colorbar(sc, ax=ax, label='Δ (years BCE)')
        ax.set_xlabel('Full linguistic MAP (BCE)', fontsize=9)
        ax.set_ylabel(f'{label} MAP (BCE)', fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.legend(fontsize=7)
    fig.suptitle('Word n-gram models vs. full linguistic model\n'
                 'Blue = word n-gram dates older  |  Red = dates younger',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


def plot_model_comparison(date_grid, post_A, post_B, post_AB, records, out_path):
    """Overlay posteriors (A, B, A+B) for Torah + source units."""
    MAIN   = ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
              'D_source', 'P_source', 'JE_source']
    colors = {'Genesis': '#1b9e77', 'Exodus': '#d95f02', 'Leviticus': '#7570b3',
              'Numbers': '#e7298a', 'Deuteronomy': '#66a61e',
              'D_source': '#e6ab02', 'P_source': '#a6761d', 'JE_source': '#666666'}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=False)
    axes = axes.flatten()
    for ax, unit in zip(axes, MAIN):
        col = colors.get(unit, '#888888')
        if unit in post_AB:
            ax.plot(date_grid, post_AB[unit], color=col, lw=2.0, label='A+B')
        if unit in post_A:
            ax.plot(date_grid, post_A[unit],  color=col, lw=1.2, ls='--', label='POS')
        if unit in post_B:
            ax.plot(date_grid, post_B[unit],  color=col, lw=1.2, ls=':',  label='fw-lex')
        ax.axvline(0, color='#bbbbbb', lw=0.7, ls=':')
        rec = records.get(unit, {})
        ab  = rec.get('map_AB', float('nan'))
        ml  = rec.get('map_full', float('nan'))
        ax.set_title(f"{display_name(unit)}\n"
                     f"A+B {ab:.0f}  ling {ml:.0f} BCE",
                     fontsize=9)
        ax.set_xlabel('BCE ←  →  CE', fontsize=7)
        ax.set_xlim(DATE_HI, DATE_LO)
        ax.tick_params(labelsize=7)
        if ax == axes[0]:
            ax.legend(fontsize=7)
    fig.suptitle('Torah / source units — word n-gram model (A+B) vs. linguistic model',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Word n-gram diachronic dating')
    parser.add_argument('--ling-csv', default=str(WORKSPACE / 'master_dating_results.csv'),
                        help='Master dating results CSV from scripts 11–16')
    parser.add_argument('--outdir',   default=str(WORKSPACE))
    args   = parser.parse_args()
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
    # STEP 2 — Extract word sequences for training corpus
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 2 — Extract word sequences for training corpus')
    print('=' * 70)

    unit_order     = list(TRAINING_UNITS.keys())
    typeA_rates_tr = {}   # {unit: {ngram: rate}}
    typeB_rates_tr = {}
    nwords_train   = {}

    for unit in unit_order:
        pos_seq, fw_seq, nw = extract_word_sequences(TRAINING_UNITS[unit], F, L, T)
        nwords_train[unit] = nw

        cnt_A = build_ngrams(pos_seq, sizes=NGRAM_SIZES, skip_null=False)
        cnt_B = build_ngrams(fw_seq,  sizes=NGRAM_SIZES, skip_null=True)

        typeA_rates_tr[unit] = counts_to_rates(cnt_A, nw)
        typeB_rates_tr[unit] = counts_to_rates(cnt_B, nw)

        print(f'  {unit:<20} {nw:>8,} words  '
              f'A-types: {len(cnt_A):>6,}  B-types: {len(cnt_B):>6,}')

    dates_list = [TRAINING_DATES[u] for u in unit_order]

    freq_A = build_freq_matrix(typeA_rates_tr, unit_order)
    freq_B = build_freq_matrix(typeB_rates_tr, unit_order)
    print(f'\n  Distinct Type-A n-grams: {len(freq_A):,}')
    print(f'  Distinct Type-B n-grams: {len(freq_B):,}')

    # -----------------------------------------------------------------------
    # STEP 3 — Feature selection
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 3 — Feature selection')
    print('=' * 70)

    print('\n  -- Type A (POS-tag n-grams) --')
    sel_A, rhos_A = select_features(freq_A, dates_list, label='TypeA')

    print('\n  -- Type B (function-word lex n-grams) --')
    sel_B, rhos_B = select_features(freq_B, dates_list, label='TypeB')

    if not sel_A and not sel_B:
        print('ERROR: No features selected for either type. Exiting.')
        return

    # Print top features
    def print_top(features, rhos, label, n=10):
        print(f'\n  Top-{n} {label}:')
        print(f'  {"n-gram":<32}  {"ρ":>6}  dir')
        print(f'  {"-" * 45}')
        for f, r in zip(features[:n], rhos[:n]):
            d = '↑LBH' if r > 0 else '↓LBH'
            ng_size = len(f.split('·'))
            print(f'  {f:<32}  {r:+.3f}  {d}  ({ng_size}-gram)')

    if sel_A:
        print_top(sel_A, rhos_A, 'Type A (POS-tag n-grams)')
    if sel_B:
        print_top(sel_B, rhos_B, 'Type B (function-word n-grams)')

    # Save feature CSVs
    def save_features_csv(features, rhos, path):
        if not features:
            return
        df = pd.DataFrame({
            'ngram':      features,
            'rho':        rhos,
            'direction':  ['↑ LBH' if r > 0 else '↓ LBH' for r in rhos],
            'n_gram_size': [len(f.split('·')) for f in features],
        })
        df.to_csv(path, index=False)
        print(f'  Saved: {Path(path).name}')

    save_features_csv(sel_A, rhos_A, outdir / 'word_ngram_typeA_features.csv')
    save_features_csv(sel_B, rhos_B, outdir / 'word_ngram_typeB_features.csv')

    # -----------------------------------------------------------------------
    # STEP 4 — Build MVN models
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 4 — Build MVN models')
    print('=' * 70)

    def make_rates_df(freq_matrix_full, selected, units):
        rows = {u: {f: freq_matrix_full[f][i] for f in selected}
                for i, u in enumerate(units)}
        df = pd.DataFrame(rows).T
        df.index.name = 'unit'
        return df

    ols_A = Sigma_inv_A = None
    ols_B = Sigma_inv_B = None
    ols_AB = Sigma_inv_AB = None
    sel_AB = []
    rates_A = rates_B = rates_AB = None

    if sel_A:
        rates_A = make_rates_df(freq_A, sel_A, unit_order)
        ols_A, Sigma_A, Sigma_inv_A = build_mvn_model(rates_A, dates_list, sel_A)
        print(f'  Type A: {len(sel_A)} features, cond={np.linalg.cond(Sigma_A):.0f}')

    if sel_B:
        rates_B = make_rates_df(freq_B, sel_B, unit_order)
        ols_B, Sigma_B, Sigma_inv_B = build_mvn_model(rates_B, dates_list, sel_B)
        print(f'  Type B: {len(sel_B)} features, cond={np.linalg.cond(Sigma_B):.0f}')

    if sel_A and sel_B:
        rates_AB = pd.concat([rates_A, rates_B], axis=1)
        sel_AB   = sel_A + sel_B
        ols_AB, Sigma_AB, Sigma_inv_AB = build_mvn_model(rates_AB, dates_list, sel_AB)
        print(f'  Combined A+B: {len(sel_AB)} features, '
              f'cond={np.linalg.cond(Sigma_AB):.0f}')

        # Save combined training rates
        train_path = outdir / 'word_ngram_training_rates.csv'
        out_tr = rates_AB.copy()
        out_tr.insert(0, 'date_bce', [TRAINING_DATES[u] for u in rates_AB.index])
        out_tr.to_csv(train_path)
        print(f'  Saved: {train_path.name}')
    elif sel_A:
        train_path = outdir / 'word_ngram_training_rates.csv'
        out_tr = rates_A.copy()
        out_tr.insert(0, 'date_bce', [TRAINING_DATES[u] for u in rates_A.index])
        out_tr.to_csv(train_path)
        print(f'  Saved: {train_path.name}')
    elif sel_B:
        train_path = outdir / 'word_ngram_training_rates.csv'
        out_tr = rates_B.copy()
        out_tr.insert(0, 'date_bce', [TRAINING_DATES[u] for u in rates_B.index])
        out_tr.to_csv(train_path)
        print(f'  Saved: {train_path.name}')

    # -----------------------------------------------------------------------
    # STEP 5 — Load prior linguistic results for comparison
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 5 — Load linguistic model results')
    print('=' * 70)
    ling_df = pd.read_csv(args.ling_csv)
    ling_df['unit_key'] = ling_df['unit'].replace(
        {'D': 'D_source', 'P': 'P_source', 'JE': 'JE_source'})
    ling_records = {r['unit_key']: r for _, r in ling_df.iterrows()}
    print(f'  Loaded {len(ling_records)} records from {Path(args.ling_csv).name}')

    # -----------------------------------------------------------------------
    # STEP 6 — Date test units
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 6 — Date test units with word n-gram models')
    print('=' * 70)

    date_grid = np.linspace(DATE_LO, DATE_HI, N_GRID)

    post_A_dict  = {}
    post_B_dict  = {}
    post_AB_dict = {}
    records      = {}

    print(f'\n  {"Unit":<16}  {"n_words":>8}  '
          f'{"MAP_A":>7}  {"MAP_B":>7}  {"MAP_AB":>8}  '
          f'{"MAP_ling":>9}  {"MAP_ng16":>9}  noisy')
    print('  ' + '-' * 80)

    for unit, pairs in TEST_UNITS.items():
        pos_seq, fw_seq, nw = extract_word_sequences(pairs, F, L, T)

        # Compute n-gram rates once per unit
        cnt_A_unit   = build_ngrams(pos_seq, sizes=NGRAM_SIZES, skip_null=False)
        cnt_B_unit   = build_ngrams(fw_seq,  sizes=NGRAM_SIZES, skip_null=True)
        rates_a_unit = counts_to_rates(cnt_A_unit, nw)
        rates_b_unit = counts_to_rates(cnt_B_unit, nw)

        # Type A posterior
        map_A = ci68_lo_A = ci68_hi_A = ci95_lo_A = ci95_hi_A = np.nan
        if sel_A:
            obs_A  = np.array([rates_a_unit.get(f, 0.0) for f in sel_A], dtype=float)
            post_a = compute_posterior(obs_A, ols_A, Sigma_inv_A, sel_A, date_grid)
            post_A_dict[unit] = post_a
            map_A, ci68_lo_A, ci68_hi_A = map_and_ci(post_a, date_grid)
            _,     ci95_lo_A, ci95_hi_A = map_and_ci(post_a, date_grid, ci_level=0.95)

        # Type B posterior
        map_B = ci68_lo_B = ci68_hi_B = ci95_lo_B = ci95_hi_B = np.nan
        if sel_B:
            obs_B  = np.array([rates_b_unit.get(f, 0.0) for f in sel_B], dtype=float)
            post_b = compute_posterior(obs_B, ols_B, Sigma_inv_B, sel_B, date_grid)
            post_B_dict[unit] = post_b
            map_B, ci68_lo_B, ci68_hi_B = map_and_ci(post_b, date_grid)
            _,     ci95_lo_B, ci95_hi_B = map_and_ci(post_b, date_grid, ci_level=0.95)

        # Combined A+B posterior
        map_AB = ci68_lo_AB = ci68_hi_AB = ci95_lo_AB = ci95_hi_AB = np.nan
        if sel_AB:
            obs_AB = np.concatenate([
                np.array([rates_a_unit.get(f, 0.0) for f in sel_A], dtype=float),
                np.array([rates_b_unit.get(f, 0.0) for f in sel_B], dtype=float),
            ])
            post_ab = compute_posterior(obs_AB, ols_AB, Sigma_inv_AB, sel_AB, date_grid)
            post_AB_dict[unit] = post_ab
            map_AB, ci68_lo_AB, ci68_hi_AB = map_and_ci(post_ab, date_grid)
            _,      ci95_lo_AB, ci95_hi_AB = map_and_ci(post_ab, date_grid, ci_level=0.95)

        # Pull prior results for comparison
        lr         = ling_records.get(unit, {})
        map_ling   = float(lr.get('map_full',      np.nan))
        map_resist = float(lr.get('map_resistant', np.nan))
        map_bd     = float(lr.get('map_BD',        np.nan))
        map_ng16   = float(lr.get('map_ngram',     np.nan))
        noisy_f    = nw < NOISY_THRESHOLD
        genre      = lr.get('genre', '?')

        records[unit] = dict(
            unit=unit, genre=genre, n_words=nw, noisy=noisy_f,
            # word n-gram results
            map_A=map_A,   ci68_lo_A=ci68_lo_A,   ci68_hi_A=ci68_hi_A,
            ci95_lo_A=ci95_lo_A,   ci95_hi_A=ci95_hi_A,
            map_B=map_B,   ci68_lo_B=ci68_lo_B,   ci68_hi_B=ci68_hi_B,
            ci95_lo_B=ci95_lo_B,   ci95_hi_B=ci95_hi_B,
            map_AB=map_AB, ci68_lo_AB=ci68_lo_AB,  ci68_hi_AB=ci68_hi_AB,
            ci95_lo_AB=ci95_lo_AB, ci95_hi_AB=ci95_hi_AB,
            # prior results for comparison
            map_full=map_ling, map_resistant=map_resist,
            map_BD=map_bd,     map_ngram=map_ng16,
        )

        noisy_str = '⚠' if noisy_f else ''
        print(f'  {unit:<16}  {nw:>8,}  '
              f'{map_A:>7.0f}  {map_B:>7.0f}  {map_AB:>8.0f}  '
              f'{map_ling:>9.0f}  {map_ng16:>9.0f}  {noisy_str}')

    # -----------------------------------------------------------------------
    # STEP 7 — Save results CSV
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 7 — Save results')
    print('=' * 70)

    results_df   = pd.DataFrame(list(records.values()))
    numeric_cols = [c for c in results_df.columns
                    if c not in ('unit', 'genre', 'noisy')]
    for col in numeric_cols:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce').round(1)

    out_csv = outdir / 'word_ngram_dating_results.csv'
    results_df.to_csv(out_csv, index=False)
    print(f'  Saved: {out_csv.name}')

    # -----------------------------------------------------------------------
    # STEP 8 — Print summary comparison table
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 8 — Summary comparison table (MAP in BCE)')
    print('=' * 70)
    print(f'\n  {"Unit":<16}  {"MAP_full":>9}  {"MAP_resist":>10}  '
          f'{"MAP_ng16":>9}  {"MAP_A":>7}  {"MAP_B":>7}  {"MAP_AB":>8}')
    print('  ' + '-' * 75)
    for unit in TEST_UNITS:
        r = records[unit]
        print(f'  {unit:<16}  '
              f'{r["map_full"]:>9.0f}  {r["map_resistant"]:>10.0f}  '
              f'{r["map_ngram"]:>9.0f}  '
              f'{r["map_A"]:>7.0f}  {r["map_B"]:>7.0f}  {r["map_AB"]:>8.0f}')

    # -----------------------------------------------------------------------
    # STEP 9 — Plots
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 9 — Plots')
    print('=' * 70)

    if sel_A:
        freq_A_plot = {f: [freq_A[f][i] for i in range(len(unit_order))]
                       for f in sel_A}
        plot_top_features(
            sel_A, rhos_A, freq_A_plot, dates_list,
            str(outdir / 'word_ngram_typeA_top_features.png'),
            title=f'Top 20 temporally-correlated POS-tag n-grams (Type A, {len(sel_A)} selected)')

    if sel_B:
        freq_B_plot = {f: [freq_B[f][i] for i in range(len(unit_order))]
                       for f in sel_B}
        plot_top_features(
            sel_B, rhos_B, freq_B_plot, dates_list,
            str(outdir / 'word_ngram_typeB_top_features.png'),
            title=f'Top 20 temporally-correlated function-word n-grams (Type B, {len(sel_B)} selected)')

    plot_agreement_multi(records, str(outdir / 'word_ngram_agreement_plot.png'))

    plot_model_comparison(date_grid, post_A_dict, post_B_dict, post_AB_dict,
                          records, str(outdir / 'word_ngram_model_comparison.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
