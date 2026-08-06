"""
Script 21 — Sensitivity Analysis: Restricted Training Corpus
=============================================================
Tests whether dating results are robust to the removal of training texts
that carry the highest risk of late interpolations.

The concern:  Pre-exilic prophets (especially 8th-century: Amos, Hosea,
Micah, Isaiah 1–39) are widely held to contain later additions.  If those
later additions skew what the model learns as "early" Hebrew, all date
estimates could be biased toward looking older than they are.

Three scenarios
---------------
  S0 — Full corpus (baseline, 22 training units, 760–167 BCE)
  S1 — No 8th-century prophets (drop Amos, Hosea, Micah, Isaiah_1)
         → 18 units, earliest anchor = 620 BCE (Nahum/Zephaniah)
  S2 — No pre-exilic prophets (drop Amos, Hosea, Micah, Isaiah_1,
                                 Nahum, Habakkuk, Zephaniah)
         → 15 units, earliest anchor = 590 BCE (Jeremiah)

Both char n-gram (Script 16) and word n-gram A+B (Script 17) models are
rerun for each scenario with fresh feature selection and model fitting.

Outputs (saved to workspace folder)
-------------------------------------
  sensitivity_char_ngram.csv        — MAP + CI68 per unit × scenario
  sensitivity_word_ngram.csv        — MAP + CI68 per unit × scenario
  sensitivity_comparison.csv        — combined table (both models)
  sensitivity_dotplot_char.png      — connected-dot chart (char n-gram)
  sensitivity_dotplot_word.png      — connected-dot chart (word n-gram)
  sensitivity_shift_summary.png     — bar chart: max shift across scenarios
"""

import warnings
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path('/sessions/relaxed-modest-dirac/mnt/Diachronic Hebrew')

# ---------------------------------------------------------------------------
# Training corpus  (same 22 units as Scripts 16–17)
# ---------------------------------------------------------------------------
ALL_TRAINING_UNITS = {
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
    'Daniel':       [('Daniel',        [(1, 1), (8, 12)])],
    'Ecclesiastes': [('Ecclesiastes',  [(1, 12)])],
    'Esther':       [('Esther',        [(1, 10)])],
}

ALL_TRAINING_DATES = {
    'Amos': 760, 'Hosea': 740, 'Micah': 720, 'Isaiah_1': 720,
    'Nahum': 620, 'Habakkuk': 600, 'Zephaniah': 620, 'Jeremiah': 590,
    'Ezekiel': 580, 'Isaiah_2': 540, 'Isaiah_3': 450,
    'Haggai': 520, 'Zechariah_1': 518, 'Malachi': 450,
    'Jonah': 400, 'Lamentations': 580, 'Ezra': 350,
    'Nehemiah': 350, 'Chronicles': 350, 'Daniel': 167,
    'Ecclesiastes': 250, 'Esther': 350,
}

# Units dropped in each restricted scenario
DROP_S1 = {'Amos', 'Hosea', 'Micah', 'Isaiah_1'}                         # 8th-cent prophets
DROP_S2 = DROP_S1 | {'Nahum', 'Habakkuk', 'Zephaniah'}                   # all pre-exilic

SCENARIOS = {
    'S0_Full':        set(),       # nothing dropped
    'S1_No8cent':     DROP_S1,
    'S2_NoPreExilic': DROP_S2,
}

SCENARIO_LABELS = {
    'S0_Full':        'S0: Full corpus\n(22 units, 760–167 BCE)',
    'S1_No8cent':     'S1: No 8th-cent prophets\n(18 units, 620–167 BCE)',
    'S2_NoPreExilic': 'S2: No pre-exilic prophets\n(15 units, 590–167 BCE)',
}

SCENARIO_COLORS = {
    'S0_Full':        '#2166ac',
    'S1_No8cent':     '#f4a582',
    'S2_NoPreExilic': '#d6604d',
}

# ---------------------------------------------------------------------------
# Test units  (same 23 units as Scripts 16–17)
# ---------------------------------------------------------------------------
TEST_UNITS = {
    'Genesis':      [('Genesis',     [(1, 50)])],
    'Exodus':       [('Exodus',      [(1, 14), (16, 40)])],
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
    'Judges':       [('Judges',    [(1, 4), (6, 21)])],
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

NOISY_UNITS = {'D_Song', 'Song_Sea', 'Song_Deborah'}  # < 1000 words

# Groups for the dotplot y-axis
UNIT_GROUPS = [
    ('Torah books',       ['Genesis','Exodus','Leviticus','Numbers','Deuteronomy']),
    ('Documentary sources', ['D_source','P_source','JE_source',
                             'D_Code','D_Frame','Lev_Holiness','Lev_Priestly']),
    ('Historical books',  ['Joshua','Judges','1_Samuel','2_Samuel','1_Kings','2_Kings']),
    ('Jeremiah',          ['Jer_DTR','Jer_oracle']),
    ('Ancient poems',     ['Song_Sea','Song_Deborah','D_Song']),
]

# ---------------------------------------------------------------------------
# N-gram / model parameters
# ---------------------------------------------------------------------------
# Char n-gram
CNG_SIZES        = [3, 4]
CNG_MAX_FEATURES = 120
CNG_FDR_ALPHA    = 0.10
CNG_LOO_THRESH   = 0.65
CNG_RIDGE        = 0.20
WORD_SEP         = '_'

# Word n-gram
WNG_SIZES        = [2, 3]
WNG_MAX_FEATURES = 100
WNG_FDR_ALPHA    = 0.10
WNG_LOO_THRESH   = 0.65
WNG_RIDGE        = 0.20
FUNCTION_POS     = frozenset({'prep', 'conj', 'art', 'nega', 'prps', 'prde', 'inrg'})

# MVN model (shared)
N_GRID      = 500
DATE_HI     = 1200
DATE_LO     = 50
PRIOR_MU    = 600.0
PRIOR_SIGMA = 350.0


# ===========================================================================
# TEXT EXTRACTION
# ===========================================================================

def extract_cons_text(book_ch_pairs, F, L, T):
    """Return (text_string, n_words) with WORD_SEP boundaries for char n-grams."""
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
    text = WORD_SEP + WORD_SEP.join(tokens) + WORD_SEP
    return text, n_words


def pos_token(word, F):
    sp = F.sp.v(word) or 'unkn'
    if sp == 'verb':
        vt = F.vt.v(word) or 'unkn'
        return f'verb_{vt}'
    return sp


def extract_word_sequences(book_ch_pairs, F, L, T):
    """Return (pos_seq, fw_seq, n_words) for word n-gram extraction."""
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
                fw_seq.append(F.lex.v(word) or sp if sp in FUNCTION_POS else None)
                n_words += 1
    return pos_seq, fw_seq, n_words


# ===========================================================================
# N-GRAM COMPUTATION
# ===========================================================================

def compute_cng_freqs(text, sizes=CNG_SIZES):
    """Char n-gram relative frequencies per 1000 characters."""
    if not text:
        return {}
    total = len(text)
    counts = Counter()
    for n in sizes:
        for i in range(len(text) - n + 1):
            counts[text[i:i+n]] += 1
    return {ng: cnt / total * 1000 for ng, cnt in counts.items()}


def build_word_ngrams(tokens, sizes=WNG_SIZES, skip_null=False):
    """Build raw-count dict of word n-gram strings."""
    counts = Counter()
    for n in sizes:
        for i in range(len(tokens) - n + 1):
            gram = tokens[i:i+n]
            if skip_null and any(t is None for t in gram):
                continue
            counts['·'.join(str(t) for t in gram)] += 1
    return dict(counts)


def counts_to_rates(counts, n_words, scale=1000.0):
    if n_words == 0:
        return {}
    return {ng: cnt / n_words * scale for ng, cnt in counts.items()}


def build_freq_matrix(rates_by_unit, unit_order):
    all_ng = set()
    for u in unit_order:
        all_ng.update(rates_by_unit[u].keys())
    return {ng: [rates_by_unit[u].get(ng, 0.0) for u in unit_order]
            for ng in all_ng}


# ===========================================================================
# FEATURE SELECTION
# ===========================================================================

def select_features(freq_matrix, dates, unit_names, label='',
                    fdr_alpha=0.10, loo_thresh=0.65,
                    max_features=120, min_prev=None):
    """
    Spearman ρ → BH FDR → LOO robustness filter.

    min_prev: prevalence threshold (n-gram must appear in ≥ min_prev units).
              If None, scaled from full-corpus MIN_UNIT_PREV=8/22.
    """
    n          = len(dates)
    if min_prev is None:
        min_prev = max(4, round(n * 8 / 22))
    dates_arr  = np.array(dates, dtype=float)
    features   = list(freq_matrix.keys())
    freqs      = np.array([freq_matrix[f] for f in features], dtype=float)

    # prevalence filter
    present    = (freqs > 0).sum(axis=1)
    mask_prev  = present >= min_prev
    features   = [f for f, m in zip(features, mask_prev) if m]
    freqs      = freqs[mask_prev]
    print(f'  {label} prevalence (≥{min_prev}/{n}): {len(features):,} features')
    if len(features) == 0:
        return [], np.array([])

    # Spearman correlation
    rhos, pvals = [], []
    for i in range(len(features)):
        rho, p = stats.spearmanr(dates_arr, freqs[i])
        rhos.append(rho); pvals.append(p)
    rhos  = np.array(rhos)
    pvals = np.array(pvals)

    # BH FDR correction
    order   = np.argsort(pvals)
    m_all   = len(pvals)
    bh_thresh = np.arange(1, m_all + 1) / m_all * fdr_alpha
    bh_critical = -1
    for k in range(m_all - 1, -1, -1):
        if pvals[order[k]] <= bh_thresh[k]:
            bh_critical = k; break
    if bh_critical < 0:
        print(f'  {label} No features survive BH FDR.')
        return [], np.array([])
    sig_mask = np.zeros(m_all, dtype=bool)
    sig_mask[order[:bh_critical + 1]] = True
    features_sig = [f for f, s in zip(features, sig_mask) if s]
    rhos_sig     = rhos[sig_mask]
    freqs_sig    = freqs[sig_mask]
    print(f'  {label} BH FDR (α={fdr_alpha}): {len(features_sig):,}')

    # LOO robustness
    loo_frac = []
    for i in range(len(features_sig)):
        consistent  = 0
        orig_sign   = np.sign(rhos_sig[i])
        for j in range(n):
            loo_d = np.delete(dates_arr, j)
            loo_f = np.delete(freqs_sig[i], j)
            if np.std(loo_f) < 1e-12:
                continue
            rho_j, _ = stats.spearmanr(loo_d, loo_f)
            if np.sign(rho_j) == orig_sign:
                consistent += 1
        loo_frac.append(consistent / (n - 1))
    loo_frac = np.array(loo_frac)
    loo_mask = loo_frac >= loo_thresh
    features_final = [f for f, m in zip(features_sig, loo_mask) if m]
    rhos_final     = rhos_sig[loo_mask]
    print(f'  {label} LOO (≥{loo_thresh:.0%}): {len(features_final)}')

    # sort by |ρ|, cap
    order_f = np.argsort(-np.abs(rhos_final))
    features_final = [features_final[i] for i in order_f][:max_features]
    rhos_final     = rhos_final[order_f][:max_features]
    print(f'  {label} Final (cap={max_features}): {len(features_final)}')
    return features_final, rhos_final


# ===========================================================================
# MVN MODEL
# ===========================================================================

def build_mvn_model(rates_dict, dates_bce, feature_names, ridge_frac=0.10):
    """
    rates_dict : {feature: [rate_per_training_unit]}
    dates_bce  : list aligned with training units
    Returns (ols_params, Sigma_reg, Sigma_inv)
    """
    ols_params    = {}
    residual_rows = []
    for fn in feature_names:
        y     = np.array(rates_dict[fn], dtype=float)
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
    lam       = ridge_frac * np.trace(Sigma) / K if K > 0 else 0.0
    Sigma_reg = Sigma + lam * np.eye(K)
    Sigma_inv = np.linalg.inv(Sigma_reg)
    return ols_params, Sigma_reg, Sigma_inv


def compute_posterior(obs_vec, ols_params, Sigma_inv, feature_names, date_grid):
    log_post = np.zeros(len(date_grid))
    for i, d in enumerate(date_grid):
        pred  = np.array([ols_params[fn][0] + ols_params[fn][1] * d
                          for fn in feature_names], dtype=float)
        diff  = obs_vec - pred
        valid = np.isfinite(diff)
        if valid.sum() < 2:
            log_post[i] = -1e9; continue
        d_v = diff[valid]
        S_v = Sigma_inv[np.ix_(valid, valid)]
        log_post[i] = -0.5 * float(d_v @ S_v @ d_v)
    log_prior  = -0.5 * ((date_grid - PRIOR_MU) / PRIOR_SIGMA) ** 2
    log_post  += log_prior
    log_post  -= log_post.max()
    post       = np.exp(log_post)
    post      /= post.sum()
    return post


def map_and_ci(posterior, date_grid, ci_level=0.68):
    map_date = float(date_grid[np.argmax(posterior)])
    cdf      = np.cumsum(posterior)
    lo_frac  = (1 - ci_level) / 2
    hi_frac  = 1 - lo_frac
    ci_lo    = float(date_grid[np.searchsorted(cdf, lo_frac)])
    ci_hi    = float(date_grid[np.searchsorted(cdf, hi_frac)])
    return map_date, ci_lo, ci_hi


def date_test_unit(test_rates, feature_names, ols_params, Sigma_inv, date_grid):
    """Apply fitted model to one test unit.  Returns (map, ci68_lo, ci68_hi)."""
    obs = np.array([test_rates.get(fn, 0.0) for fn in feature_names], dtype=float)
    post = compute_posterior(obs, ols_params, Sigma_inv, feature_names, date_grid)
    return map_and_ci(post, date_grid)


# ===========================================================================
# SCENARIO RUNNER
# ===========================================================================

def run_scenario_char_ngram(scenario_name, drop_set,
                             cng_train_rates, cng_test_rates,
                             date_grid, all_train_units, all_train_dates):
    """
    Run char n-gram model for one scenario.

    cng_train_rates : {unit: {ngram: rate}}  (all 22 training units)
    cng_test_rates  : {unit: {ngram: rate}}  (all 23 test units)

    Returns dict {test_unit: (map, ci68_lo, ci68_hi)}
    """
    print(f'\n--- Char n-gram: {scenario_name} ---')
    # filter training to this scenario
    train_units = [u for u in all_train_units if u not in drop_set]
    train_dates = [all_train_dates[u] for u in train_units]
    n_train     = len(train_units)
    print(f'  Training units: {n_train} | date range: '
          f'{max(train_dates)}–{min(train_dates)} BCE')

    # build frequency matrix for training units
    freq_matrix = build_freq_matrix(cng_train_rates, train_units)

    # feature selection
    selected, rhos = select_features(
        freq_matrix, train_dates, train_units,
        label=f'[{scenario_name}]',
        fdr_alpha=CNG_FDR_ALPHA, loo_thresh=CNG_LOO_THRESH,
        max_features=CNG_MAX_FEATURES)
    if not selected:
        return {u: (np.nan, np.nan, np.nan) for u in TEST_UNITS}

    # build model (use training freq matrix sliced to selected features)
    train_rates_sel = {fn: freq_matrix[fn] for fn in selected}
    ols_params, _, Sigma_inv = build_mvn_model(
        train_rates_sel, train_dates, selected, ridge_frac=CNG_RIDGE)

    # apply to test units
    results = {}
    for unit in TEST_UNITS:
        test_rates = cng_test_rates.get(unit, {})
        map_d, lo, hi = date_test_unit(test_rates, selected, ols_params, Sigma_inv, date_grid)
        results[unit] = (map_d, lo, hi)
    return results


def run_scenario_word_ngram(scenario_name, drop_set,
                             wng_train_ratesA, wng_train_ratesB,
                             wng_test_ratesA,  wng_test_ratesB,
                             date_grid, all_train_units, all_train_dates):
    """
    Run word n-gram A+B combined model for one scenario.
    Returns dict {test_unit: (map_A, lo_A, hi_A, map_B, lo_B, hi_B, map_AB, lo_AB, hi_AB)}
    """
    print(f'\n--- Word n-gram: {scenario_name} ---')
    train_units = [u for u in all_train_units if u not in drop_set]
    train_dates = [all_train_dates[u] for u in train_units]
    n_train     = len(train_units)
    print(f'  Training units: {n_train}')

    # --- Type A ---
    fmA = build_freq_matrix(wng_train_ratesA, train_units)
    selA, rhosA = select_features(
        fmA, train_dates, train_units,
        label=f'[{scenario_name}] TypeA',
        fdr_alpha=WNG_FDR_ALPHA, loo_thresh=WNG_LOO_THRESH,
        max_features=WNG_MAX_FEATURES)

    # --- Type B ---
    fmB = build_freq_matrix(wng_train_ratesB, train_units)
    selB, rhosB = select_features(
        fmB, train_dates, train_units,
        label=f'[{scenario_name}] TypeB',
        fdr_alpha=WNG_FDR_ALPHA, loo_thresh=WNG_LOO_THRESH,
        max_features=WNG_MAX_FEATURES)

    # fit models
    def fit_and_predict(selected, fm, test_rates_by_unit):
        if not selected:
            return {u: (np.nan, np.nan, np.nan) for u in TEST_UNITS}
        tr = {fn: fm[fn] for fn in selected}
        ols, _, Sinv = build_mvn_model(tr, train_dates, selected, ridge_frac=WNG_RIDGE)
        out = {}
        for unit in TEST_UNITS:
            tr2 = test_rates_by_unit.get(unit, {})
            out[unit] = date_test_unit(tr2, selected, ols, Sinv, date_grid)
        return out

    resA = fit_and_predict(selA, fmA, wng_test_ratesA)
    resB = fit_and_predict(selB, fmB, wng_test_ratesB)

    # --- Combined A+B ---
    if selA and selB:
        sel_AB   = selA + selB
        train_AB = {fn: fmA[fn] for fn in selA}
        train_AB.update({fn: fmB[fn] for fn in selB})
        ols_AB, _, Sinv_AB = build_mvn_model(
            train_AB, train_dates, sel_AB, ridge_frac=WNG_RIDGE)

        resAB = {}
        for unit in TEST_UNITS:
            trA2 = wng_test_ratesA.get(unit, {})
            trB2 = wng_test_ratesB.get(unit, {})
            obs  = np.array([trA2.get(fn, 0.0) for fn in selA] +
                            [trB2.get(fn, 0.0) for fn in selB], dtype=float)
            post = compute_posterior(obs, ols_AB, Sinv_AB, sel_AB, date_grid)
            resAB[unit] = map_and_ci(post, date_grid)
    else:
        resAB = {u: (np.nan, np.nan, np.nan) for u in TEST_UNITS}

    # combine
    combined = {}
    for unit in TEST_UNITS:
        combined[unit] = (*resA[unit], *resB[unit], *resAB[unit])
    return combined


# ===========================================================================
# PLOTTING
# ===========================================================================

def dotplot(results_by_scenario, model_label, out_path, date_grid):
    """
    Connected-dot chart: y = unit (ordered by S0 MAP), x = BCE date.
    Each unit has 3 dots (S0, S1, S2) connected by a line.
    Error bars show CI68 for S0 only (to avoid clutter).
    """
    # order units by S0 MAP for readability
    s0_maps = {u: results_by_scenario['S0_Full'][u][0] for u in TEST_UNITS
               if np.isfinite(results_by_scenario['S0_Full'][u][0])}
    ordered_units = sorted(s0_maps, key=lambda u: s0_maps[u], reverse=True)

    fig, ax = plt.subplots(figsize=(11, 10))

    yticks    = list(range(len(ordered_units)))
    yticklabs = []

    for yi, unit in enumerate(ordered_units):
        noisy = unit in NOISY_UNITS
        label = unit.replace('_', ' ')
        yticklabs.append(label + (' ⚠' if noisy else ''))

        maps_pts = []
        for sc in ['S0_Full', 'S1_No8cent', 'S2_NoPreExilic']:
            res = results_by_scenario[sc].get(unit, (np.nan, np.nan, np.nan))
            map_d, lo, hi = res[0], res[1], res[2]
            maps_pts.append(map_d)

        # draw connecting line (only where finite)
        finite_idx = [i for i, m in enumerate(maps_pts) if np.isfinite(m)]
        if len(finite_idx) >= 2:
            xs = [maps_pts[i] for i in finite_idx]
            ys = [yi] * len(finite_idx)
            ax.plot(xs, ys, color='#aaaaaa', lw=1.2, zorder=1)

        # draw dots per scenario
        for si, sc in enumerate(['S0_Full', 'S1_No8cent', 'S2_NoPreExilic']):
            res   = results_by_scenario[sc].get(unit, (np.nan, np.nan, np.nan))
            map_d = res[0]
            if not np.isfinite(map_d):
                continue
            col    = SCENARIO_COLORS[sc]
            marker = 'o' if not noisy else '^'
            size   = 60 if sc == 'S0_Full' else 45
            ax.scatter(map_d, yi, color=col, s=size, zorder=3 + si,
                       marker=marker,
                       label=SCENARIO_LABELS[sc] if yi == 0 else '')

            # CI68 error bar for S0 only
            if sc == 'S0_Full' and len(res) >= 3:
                lo, hi = res[1], res[2]
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.errorbar(map_d, yi,
                                xerr=[[abs(map_d - lo)], [abs(hi - map_d)]],
                                fmt='none', ecolor=col, elinewidth=1.0,
                                capsize=3, capthick=1.0, alpha=0.6, zorder=2)

    # group separators
    offset = 0
    for gname, gunits in UNIT_GROUPS:
        n = sum(1 for u in gunits if u in ordered_units)
        if n > 0:
            y_start = len(ordered_units) - offset - n - 0.5
            ax.axhline(y_start + n, color='#dddddd', lw=0.8, ls='-')
            ax.text(DATE_LO + 15, y_start + n / 2,
                    gname, va='center', fontsize=7.5, color='#666666',
                    fontstyle='italic')
        offset += n

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabs, fontsize=8)
    ax.set_xlabel('Date (BCE)  ←  older    |    younger  →', fontsize=10)
    ax.set_xlim(DATE_HI + 50, DATE_LO - 50)
    ax.invert_xaxis()
    ax.set_title(f'Sensitivity analysis — {model_label}\n'
                 f'Dots: MAP estimate; bars: CI68 (S0 only)',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.grid(axis='x', alpha=0.25, lw=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Dotplot saved: {Path(out_path).name}')


def shift_summary_plot(char_results, word_results, out_path):
    """
    Bar chart showing max absolute shift in MAP date between S0 and S2
    for each test unit.  Separately for char n-gram vs word n-gram.
    """
    units = [u for u in TEST_UNITS
             if np.isfinite(char_results['S0_Full'].get(u, (np.nan,))[0])]
    units.sort(key=lambda u: char_results['S0_Full'][u][0], reverse=True)

    char_shifts = []
    word_shifts = []
    for u in units:
        s0c = char_results['S0_Full'].get(u, (np.nan,))[0]
        s2c = char_results['S2_NoPreExilic'].get(u, (np.nan,))[0]
        char_shifts.append(abs(s2c - s0c) if np.isfinite(s2c) else np.nan)

        # word n-gram: use combined (index 6)
        s0w = word_results['S0_Full'].get(u, (np.nan,)*9)[6]
        s2w = word_results['S2_NoPreExilic'].get(u, (np.nan,)*9)[6]
        word_shifts.append(abs(s2w - s0w) if np.isfinite(s2w) else np.nan)

    x     = np.arange(len(units))
    width = 0.35
    labels = [u.replace('_', '\n') for u in units]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width/2, char_shifts, width, label='Char n-gram',
           color='#4dac26', alpha=0.8)
    ax.bar(x + width/2, word_shifts, width, label='Word n-gram (A+B)',
           color='#d01c8b', alpha=0.8)
    ax.axhline(50,  color='#aaaaaa', lw=0.8, ls='--', label='50-yr threshold')
    ax.axhline(100, color='#888888', lw=0.8, ls='--', label='100-yr threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('|MAP(S0) − MAP(S2)| (years)', fontsize=9)
    ax.set_title('Maximum date shift: S0 (full) vs. S2 (no pre-exilic prophets)\n'
                 'Small bars = robust; large bars = sensitive to training corpus',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(200, np.nanmax(char_shifts + word_shifts) + 30))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Shift summary saved: {Path(out_path).name}')


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    outdir = WORKSPACE

    # -----------------------------------------------------------------------
    # STEP 1 — Load BHSA
    # -----------------------------------------------------------------------
    print('=' * 70)
    print('STEP 1 — Load BHSA')
    print('=' * 70)
    sys.path.insert(0, '/sessions/relaxed-modest-dirac/text-fabric-data')
    from tf.app import use as tf_use
    A = tf_use('ETCBC/bhsa', hoist=globals(), checkout='local', silent='deep')
    print('  BHSA loaded.')

    date_grid = np.linspace(DATE_HI, DATE_LO, N_GRID)
    all_train_units = list(ALL_TRAINING_UNITS.keys())

    # -----------------------------------------------------------------------
    # STEP 2 — Extract char n-gram frequencies (training + test)
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 2 — Extract char n-gram frequencies')
    print('=' * 70)

    cng_train_rates = {}
    unit_nwords     = {}
    for unit, pairs in ALL_TRAINING_UNITS.items():
        text, nw = extract_cons_text(pairs, F, L, T)
        cng_train_rates[unit] = compute_cng_freqs(text)
        unit_nwords[unit]     = nw
        print(f'  [train] {unit:<20} {nw:>7,} words')

    cng_test_rates = {}
    test_nwords    = {}
    for unit, pairs in TEST_UNITS.items():
        text, nw = extract_cons_text(pairs, F, L, T)
        cng_test_rates[unit] = compute_cng_freqs(text)
        test_nwords[unit]    = nw
        print(f'  [test]  {unit:<20} {nw:>7,} words')

    # -----------------------------------------------------------------------
    # STEP 3 — Extract word n-gram features (training + test)
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 3 — Extract word n-gram features')
    print('=' * 70)

    wng_train_ratesA = {}
    wng_train_ratesB = {}
    for unit, pairs in ALL_TRAINING_UNITS.items():
        pos_seq, fw_seq, nw = extract_word_sequences(pairs, F, L, T)
        wng_train_ratesA[unit] = counts_to_rates(
            build_word_ngrams(pos_seq, sizes=WNG_SIZES, skip_null=False), nw)
        wng_train_ratesB[unit] = counts_to_rates(
            build_word_ngrams(fw_seq,  sizes=WNG_SIZES, skip_null=True), nw)
        print(f'  [train] {unit:<20} typeA={len(wng_train_ratesA[unit]):,}  '
              f'typeB={len(wng_train_ratesB[unit]):,}')

    wng_test_ratesA = {}
    wng_test_ratesB = {}
    for unit, pairs in TEST_UNITS.items():
        pos_seq, fw_seq, nw = extract_word_sequences(pairs, F, L, T)
        wng_test_ratesA[unit] = counts_to_rates(
            build_word_ngrams(pos_seq, sizes=WNG_SIZES, skip_null=False), nw)
        wng_test_ratesB[unit] = counts_to_rates(
            build_word_ngrams(fw_seq,  sizes=WNG_SIZES, skip_null=True), nw)
        print(f'  [test]  {unit:<20} typeA={len(wng_test_ratesA[unit]):,}  '
              f'typeB={len(wng_test_ratesB[unit]):,}')

    # -----------------------------------------------------------------------
    # STEP 4 — Run sensitivity scenarios
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 4 — Run sensitivity scenarios')
    print('=' * 70)

    char_results = {}
    word_results = {}

    for sc_name, drop_set in SCENARIOS.items():
        char_results[sc_name] = run_scenario_char_ngram(
            sc_name, drop_set,
            cng_train_rates, cng_test_rates,
            date_grid, all_train_units, ALL_TRAINING_DATES)

        word_results[sc_name] = run_scenario_word_ngram(
            sc_name, drop_set,
            wng_train_ratesA, wng_train_ratesB,
            wng_test_ratesA,  wng_test_ratesB,
            date_grid, all_train_units, ALL_TRAINING_DATES)

    # -----------------------------------------------------------------------
    # STEP 5 — Assemble comparison CSVs
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 5 — Assemble results')
    print('=' * 70)

    # --- char n-gram CSV ---
    rows_c = []
    for unit in TEST_UNITS:
        row = {'unit': unit,
               'n_words': test_nwords.get(unit, 0),
               'noisy': unit in NOISY_UNITS}
        for sc in ['S0_Full', 'S1_No8cent', 'S2_NoPreExilic']:
            res = char_results[sc].get(unit, (np.nan, np.nan, np.nan))
            row[f'map_{sc}']      = round(res[0], 1) if np.isfinite(res[0]) else ''
            row[f'ci68_lo_{sc}']  = round(res[1], 1) if np.isfinite(res[1]) else ''
            row[f'ci68_hi_{sc}']  = round(res[2], 1) if np.isfinite(res[2]) else ''
        # shift columns
        s0 = char_results['S0_Full'].get(unit, (np.nan,))[0]
        s1 = char_results['S1_No8cent'].get(unit, (np.nan,))[0]
        s2 = char_results['S2_NoPreExilic'].get(unit, (np.nan,))[0]
        row['delta_S1_vs_S0'] = round(s1 - s0, 1) if np.isfinite(s1) and np.isfinite(s0) else ''
        row['delta_S2_vs_S0'] = round(s2 - s0, 1) if np.isfinite(s2) and np.isfinite(s0) else ''
        rows_c.append(row)

    df_c = pd.DataFrame(rows_c)
    path_c = outdir / 'sensitivity_char_ngram.csv'
    df_c.to_csv(path_c, index=False)
    print(f'  Saved: {path_c.name}')

    # --- word n-gram CSV (uses combined A+B = indices 6,7,8) ---
    rows_w = []
    for unit in TEST_UNITS:
        row = {'unit': unit,
               'n_words': test_nwords.get(unit, 0),
               'noisy': unit in NOISY_UNITS}
        for sc in ['S0_Full', 'S1_No8cent', 'S2_NoPreExilic']:
            res = word_results[sc].get(unit, (np.nan,)*9)
            row[f'map_A_{sc}']       = round(res[0], 1) if np.isfinite(res[0]) else ''
            row[f'map_B_{sc}']       = round(res[3], 1) if np.isfinite(res[3]) else ''
            row[f'map_AB_{sc}']      = round(res[6], 1) if np.isfinite(res[6]) else ''
            row[f'ci68_lo_AB_{sc}']  = round(res[7], 1) if np.isfinite(res[7]) else ''
            row[f'ci68_hi_AB_{sc}']  = round(res[8], 1) if np.isfinite(res[8]) else ''
        s0 = word_results['S0_Full'].get(unit, (np.nan,)*9)[6]
        s1 = word_results['S1_No8cent'].get(unit, (np.nan,)*9)[6]
        s2 = word_results['S2_NoPreExilic'].get(unit, (np.nan,)*9)[6]
        row['delta_S1_vs_S0'] = round(s1 - s0, 1) if np.isfinite(s1) and np.isfinite(s0) else ''
        row['delta_S2_vs_S0'] = round(s2 - s0, 1) if np.isfinite(s2) and np.isfinite(s0) else ''
        rows_w.append(row)

    df_w = pd.DataFrame(rows_w)
    path_w = outdir / 'sensitivity_word_ngram.csv'
    df_w.to_csv(path_w, index=False)
    print(f'  Saved: {path_w.name}')

    # --- combined summary ---
    rows_s = []
    for unit in TEST_UNITS:
        rc = char_results['S0_Full'].get(unit, (np.nan,)*3)
        rw = word_results['S0_Full'].get(unit, (np.nan,)*9)
        dc1 = char_results['S1_No8cent'].get(unit, (np.nan,))[0]
        dc2 = char_results['S2_NoPreExilic'].get(unit, (np.nan,))[0]
        dw1 = word_results['S1_No8cent'].get(unit, (np.nan,)*9)[6]
        dw2 = word_results['S2_NoPreExilic'].get(unit, (np.nan,)*9)[6]
        rows_s.append({
            'unit':          unit,
            'n_words':       test_nwords.get(unit, 0),
            'noisy':         unit in NOISY_UNITS,
            'map_char_S0':   round(rc[0], 1) if np.isfinite(rc[0]) else '',
            'map_char_S1':   round(dc1, 1) if np.isfinite(dc1) else '',
            'map_char_S2':   round(dc2, 1) if np.isfinite(dc2) else '',
            'shift_char_S2_minus_S0': round(dc2 - rc[0], 1) if np.isfinite(dc2) and np.isfinite(rc[0]) else '',
            'map_word_S0':   round(rw[6], 1) if np.isfinite(rw[6]) else '',
            'map_word_S1':   round(dw1, 1) if np.isfinite(dw1) else '',
            'map_word_S2':   round(dw2, 1) if np.isfinite(dw2) else '',
            'shift_word_S2_minus_S0': round(dw2 - rw[6], 1) if np.isfinite(dw2) and np.isfinite(rw[6]) else '',
        })

    df_s = pd.DataFrame(rows_s)
    path_s = outdir / 'sensitivity_comparison.csv'
    df_s.to_csv(path_s, index=False)
    print(f'  Saved: {path_s.name}')

    # print quick summary table
    print('\n--- Quick summary: S0 vs S2 MAP shifts ---')
    print(f'{"Unit":<20} {"Char S0":>8} {"Char S2":>8} {"ΔChar":>7} '
          f'{"Word S0":>8} {"Word S2":>8} {"ΔWord":>7}')
    print('-' * 70)
    for _, row in df_s.iterrows():
        print(f'{row.unit:<20} {str(row.map_char_S0):>8} {str(row.map_char_S2):>8} '
              f'{str(row.shift_char_S2_minus_S0):>7} '
              f'{str(row.map_word_S0):>8} {str(row.map_word_S2):>8} '
              f'{str(row.shift_word_S2_minus_S0):>7}')

    # -----------------------------------------------------------------------
    # STEP 6 — Plots
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 6 — Plots')
    print('=' * 70)

    # connected-dot: char n-gram
    dotplot(char_results,
            'Character n-gram model',
            outdir / 'sensitivity_dotplot_char.png',
            date_grid)

    # for word n-gram dotplot, flatten to (map, ci_lo, ci_hi) from combined A+B
    word_results_flat = {}
    for sc in SCENARIOS:
        word_results_flat[sc] = {
            u: (res[6], res[7], res[8])
            for u, res in word_results[sc].items()
        }
    dotplot(word_results_flat,
            'Word n-gram model (A+B combined)',
            outdir / 'sensitivity_dotplot_word.png',
            date_grid)

    # shift summary bar chart
    shift_summary_plot(char_results, word_results,
                       outdir / 'sensitivity_shift_summary.png')

    print('\nDone. All outputs saved to workspace folder.')


if __name__ == '__main__':
    main()
