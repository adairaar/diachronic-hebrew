"""
Script 18 — Archaism, Antiquity, and Embedded-Source Diagnostic
================================================================
Tests two central hypotheses about the character vs. word n-gram split:

  H1 — SCRIBAL UPDATING (Kings/Chronicles parallel-narrative test)
  ----------------------------------------------------------------
  When a late scribe rewrites an earlier narrative, they update
  orthography (plene spelling, consonantal variants) while partly
  preserving old syntactic patterns.  Prediction:
    • Both models should date Chronicles as younger than Kings.
    • The character n-gram delta (Kings − Chronicles) should be LARGER
      than the word n-gram delta, because orthographic updating is more
      systematic than syntactic updating.
    • If the gap is equal: the Chronicler rewrote syntax just as
      aggressively as orthography.
    • If word delta > char delta: unexpected — syntax was updated more
      than orthography (would be unusual).

  H2 — EMBEDDED OLD SOURCES (ancient poems in prose contexts)
  -----------------------------------------------------------
  Archaic poems embedded in later prose books carry their original
  word-level patterns but their orthography may have been updated
  during transmission alongside the surrounding prose.  Prediction:
    • Word n-gram MAP for an embedded poem >> word n-gram MAP for the
      surrounding prose (poem shows archaic syntax patterns).
    • Character n-gram MAP for poem vs. prose may be LESS divergent
      (orthography was normalized across the whole book).
    • The gap (word_MAP_poem − char_MAP_poem) should be positive and
      larger than the same gap for the surrounding prose.

Method
------
Both models (character n-gram from script 16, word n-gram from script 17)
are reconstructed from their saved training-rate CSV files.  No BHSA
retraining is needed — only new-unit text extraction requires BHSA.

ARCHAISM INDEX  =  word_MAP_AB − char_MAP_ngram
  Positive: word patterns look older than orthographic patterns
            → evidence of archaism or orthographic updating
  Negative: orthography looks older than syntax
            → archaic orthography preserved (or modern word patterns)
  Near zero: methods agree → no differential updating signal

Outputs
-------
  archaism_diagnostic_results.csv   — all unit dates + archaism index
  archaism_quadrant_plot.png        — char_MAP vs word_MAP scatter
  archaism_kchr_comparison.png      — Kings/Chronicles delta comparison
  archaism_poem_comparison.png      — poem vs prose posterior overlays
  archaism_posterior_grid.png       — posteriors for key unit pairs
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

WORKSPACE = Path('/sessions/relaxed-modest-dirac/mnt/Diachronic Hebrew')

# ---------------------------------------------------------------------------
# Model parameters (must match scripts 16 / 17)
# ---------------------------------------------------------------------------
RIDGE_CHAR  = 0.20
RIDGE_WORD  = 0.20
N_GRID      = 500
DATE_HI     = 1200
DATE_LO     =  50
PRIOR_MU    = 600.0
PRIOR_SIGMA = 350.0

NGRAM_SIZES_CHAR = [3, 4]   # char n-gram sizes (script 16)
NGRAM_SIZES_WORD = [2, 3]   # word n-gram sizes (script 17)
WORD_SEP         = '_'
FUNCTION_POS     = frozenset({'prep', 'conj', 'art', 'nega', 'prps', 'prde', 'inrg'})

NOISY_THRESHOLD = 1000

# ---------------------------------------------------------------------------
# Test unit definitions
# ---------------------------------------------------------------------------
# Group A: Kings/Chronicles parallel narratives
# Each entry: (name, book_ch_pairs, group, notes)
# We test matched pericopes covering the same historical events in Kings
# vs. the Chronicler's rewriting of those same events.

KCHR_UNITS = {
    # --- Solomon's reign ---
    'Kings_Solomon': {
        'pairs': [('1_Kings', [(1, 11)])],
        'group': 'kings_chr',
        'match': 'Chr_Solomon',
        'notes': '1 Kgs 1–11: Solomon narrative',
    },
    'Chr_Solomon': {
        'pairs': [('2_Chronicles', [(1, 9)])],
        'group': 'kings_chr',
        'match': 'Kings_Solomon',
        'notes': '2 Chr 1–9: Chronicler\'s Solomon',
    },

    # --- Judean kings: the overlap period ---
    # Kings covers both kingdoms; we use only Judean-focused sections
    # roughly matching what Chronicles reports.
    'Kings_Judah': {
        'pairs': [('1_Kings', [(12, 22)]),
                  ('2_Kings', [(1, 20)])],
        'group': 'kings_chr',
        'match': 'Chr_Judah',
        'notes': '1 Kgs 12–22 + 2 Kgs 1–20: Judean kings to Hezekiah',
    },
    'Chr_Judah': {
        'pairs': [('2_Chronicles', [(10, 32)])],
        'group': 'kings_chr',
        'match': 'Kings_Judah',
        'notes': '2 Chr 10–32: Judean kings to Hezekiah',
    },

    # --- Hezekiah pericope: three-way test ---
    # 2 Kgs 18:13–20:19 ≈ Isaiah 36–39 (near-verbatim); Chr retells
    'Kings_Hezekiah': {
        'pairs': [('2_Kings', [(18, 20)])],
        'group': 'kchr_3way',
        'match': 'Chr_Hezekiah',
        'notes': '2 Kgs 18–20: Hezekiah narrative',
    },
    'Isa_Hezekiah': {
        'pairs': [('Isaiah', [(36, 39)])],
        'group': 'kchr_3way',
        'match': 'Chr_Hezekiah',
        'notes': 'Isa 36–39: near-verbatim parallel to 2 Kgs 18–20',
    },
    'Chr_Hezekiah': {
        'pairs': [('2_Chronicles', [(29, 32)])],
        'group': 'kchr_3way',
        'match': 'Kings_Hezekiah',
        'notes': '2 Chr 29–32: Chronicler\'s Hezekiah',
    },

    # --- Final collapse: 2 Kings 21–25 vs 2 Chronicles 33–36 ---
    'Kings_Fall': {
        'pairs': [('2_Kings', [(21, 25)])],
        'group': 'kings_chr',
        'match': 'Chr_Fall',
        'notes': '2 Kgs 21–25: Manasseh to Exile',
    },
    'Chr_Fall': {
        'pairs': [('2_Chronicles', [(33, 36)])],
        'group': 'kings_chr',
        'match': 'Kings_Fall',
        'notes': '2 Chr 33–36: Chronicler\'s Manasseh to Exile',
    },
}

# Group B: Embedded ancient poems vs. surrounding prose
# For each entry we define the poem chapter(s) and the prose remainder.
POEM_UNITS = {
    # Genesis 49: Blessing of Jacob — archaic poem in patriarchal prose
    'Gen49_poem':  {
        'pairs': [('Genesis',     [(49, 49)])],
        'group': 'poem_embed',
        'prose_unit': 'Gen49_prose',
        'notes': 'Gen 49: Blessing of Jacob (poem)',
    },
    'Gen49_prose': {
        'pairs': [('Genesis',     [(1, 48), (50, 50)])],
        'group': 'poem_embed',
        'prose_unit': None,
        'notes': 'Genesis minus ch.49',
    },

    # Deuteronomy 33: Blessing of Moses — archaic poem before death
    'Deut33_poem': {
        'pairs': [('Deuteronomy', [(33, 33)])],
        'group': 'poem_embed',
        'prose_unit': 'Deut33_prose',
        'notes': 'Deut 33: Blessing of Moses (poem)',
    },
    'Deut33_prose': {
        'pairs': [('Deuteronomy', [(1, 32), (34, 34)])],
        'group': 'poem_embed',
        'prose_unit': None,
        'notes': 'Deuteronomy minus ch.33',
    },

    # Numbers 21 embedded songs vs. surrounding Numbers prose
    # Song of the Well (21:17-18) and Taunt Song (21:27-30) are tiny;
    # use the whole of Num 21 as "poem-heavy" vs. the rest of Numbers.
    'Num21_songs': {
        'pairs': [('Numbers',     [(21, 21)])],
        'group': 'poem_embed',
        'prose_unit': 'Num_prose',
        'notes': 'Num 21: chapter with embedded ancient songs',
    },
    'Num_prose': {
        'pairs': [('Numbers',     [(1, 20), (22, 36)])],
        'group': 'poem_embed',
        'prose_unit': None,
        'notes': 'Numbers minus ch.21',
    },

    # Job: widely noted for archaic vocabulary; compare poetry vs prose frame
    'Job_poetry': {
        'pairs': [('Job',         [(3, 41)])],
        'group': 'archaic_vocab',
        'prose_unit': 'Job_frame',
        'notes': 'Job 3–41: dialogues (archaic poetic vocabulary)',
    },
    'Job_frame': {
        'pairs': [('Job',         [(1, 2), (42, 42)])],
        'group': 'archaic_vocab',
        'prose_unit': None,
        'notes': 'Job 1–2 + 42: prose frame (LBH features)',
    },
}

# Group C: Additional reference units (for context in the quadrant plot)
REF_UNITS = {
    'Psalm_29':  {'pairs': [('Psalms', [(29, 29)])],
                  'group': 'ref_psalm', 'notes': 'Ps 29: archaic (Canaanite origin?)'},
    'Psalm_68':  {'pairs': [('Psalms', [(68, 68)])],
                  'group': 'ref_psalm', 'notes': 'Ps 68: archaic vocabulary'},
    'Psalm_18':  {'pairs': [('Psalms', [(18, 18)])],
                  'group': 'ref_psalm', 'notes': 'Ps 18 ≈ 2 Sam 22'},
    '2Sam_22':   {'pairs': [('2_Samuel', [(22, 22)])],
                  'group': 'ref_psalm', 'notes': '2 Sam 22 ≈ Ps 18'},
    'Psalm_119': {'pairs': [('Psalms', [(119, 119)])],
                  'group': 'ref_psalm', 'notes': 'Ps 119: late alphabetic acrostic'},
    'Ruth':      {'pairs': [('Ruth', [(1, 4)])],
                  'group': 'ref_late',  'notes': 'Ruth: late narrative'},
    'Proverbs_early': {'pairs': [('Proverbs', [(10, 22)])],
                       'group': 'ref_mixed', 'notes': 'Prov 10–22: earlier collections'},
}

# Combine all test units
ALL_TEST_UNITS = {**KCHR_UNITS, **POEM_UNITS, **REF_UNITS}


# ===========================================================================
# TEXT EXTRACTION  (from scripts 16 and 17)
# ===========================================================================

def extract_cons_text(book_ch_pairs, F, L, T, word_sep=WORD_SEP):
    """Extract consonantal Hebrew text as a string with word-boundary markers."""
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
    text = word_sep + (word_sep).join(tokens) + word_sep
    return text, n_words


def compute_char_ngram_freqs(text, sizes=NGRAM_SIZES_CHAR):
    """Compute relative char n-gram frequencies per 1000 characters."""
    if not text:
        return {}
    total = len(text)
    counts = Counter()
    for n in sizes:
        for i in range(len(text) - n + 1):
            counts[text[i:i+n]] += 1
    return {ng: cnt / total * 1000 for ng, cnt in counts.items()}


def pos_token(word, F):
    sp = F.sp.v(word) or 'unkn'
    if sp == 'verb':
        vt = F.vt.v(word) or 'unkn'
        return f'verb_{vt}'
    return sp


def extract_word_sequences(book_ch_pairs, F, L, T):
    """Extract POS-tag and function-word sequences for word n-gram models."""
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
                    fw_seq.append(None)
                n_words += 1
    return pos_seq, fw_seq, n_words


def build_word_ngrams(tokens, sizes=NGRAM_SIZES_WORD, skip_null=False):
    counts = Counter()
    for n in sizes:
        for i in range(len(tokens) - n + 1):
            gram = tokens[i:i+n]
            if skip_null and any(t is None for t in gram):
                continue
            key = '·'.join(str(t) for t in gram)
            counts[key] += 1
    return dict(counts)


def counts_to_rates(counts, n_words, scale=1000.0):
    if n_words == 0:
        return {}
    return {ng: cnt / n_words * scale for ng, cnt in counts.items()}


# ===========================================================================
# MVN MODEL  (same as all prior scripts)
# ===========================================================================

def build_mvn_model(rates_df, dates_bce, feature_names, ridge_frac=RIDGE_CHAR):
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


def compute_posterior(obs_vec, ols_params, Sigma_inv, feature_names, date_grid):
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
    log_prior = -0.5 * ((date_grid - PRIOR_MU) / PRIOR_SIGMA) ** 2
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


def posterior_entropy(posterior):
    """Shannon entropy of the posterior — high = uncertain / bimodal."""
    p = posterior[posterior > 0]
    return float(-np.sum(p * np.log(p)))


def posterior_skewness(posterior, date_grid):
    """
    Skewness of the posterior distribution.
    Positive = tail toward older dates (right-skewed in BCE).
    """
    mu  = np.sum(date_grid * posterior)
    var = np.sum((date_grid - mu) ** 2 * posterior)
    sk  = np.sum((date_grid - mu) ** 3 * posterior) / max(var ** 1.5, 1e-12)
    return float(sk)


# ===========================================================================
# LOAD TRAINED MODELS FROM SAVED CSVs
# ===========================================================================

def load_char_model(workspace):
    """
    Reconstruct the character n-gram MVN model from script-16 CSV outputs.
    Returns (ols_params, Sigma_inv, sel_ng, dates, unit_order).
    """
    rates_path   = workspace / 'ngram_training_rates.csv'
    sel_path     = workspace / 'ngram_selected_features.csv'
    rates_df     = pd.read_csv(rates_path, index_col=0)
    sel_df       = pd.read_csv(sel_path)
    sel_ng       = sel_df['ngram'].tolist()
    dates        = rates_df['date_bce'].tolist()
    unit_order   = list(rates_df.index)
    # drop meta columns before model fitting
    meta_cols = {'date_bce', 'n_words'}
    feat_cols = [c for c in rates_df.columns if c not in meta_cols]
    # keep only selected n-grams that are present in the rate matrix
    sel_ng = [f for f in sel_ng if f in feat_cols]
    rates_sel = rates_df[sel_ng]
    ols, Sigma, Sigma_inv = build_mvn_model(rates_sel, dates, sel_ng, ridge_frac=RIDGE_CHAR)
    print(f'  Char n-gram model: {len(sel_ng)} features, '
          f'cond={np.linalg.cond(Sigma):.0f}')
    return ols, Sigma_inv, sel_ng, dates, unit_order


def load_word_model(workspace):
    """
    Reconstruct the combined word n-gram (A+B) MVN model from script-17 outputs.
    Returns (ols_A, Sigma_inv_A, sel_A,
             ols_B, Sigma_inv_B, sel_B,
             ols_AB, Sigma_inv_AB, sel_AB,
             dates, unit_order).
    """
    rates_path = workspace / 'word_ngram_training_rates.csv'
    selA_path  = workspace / 'word_ngram_typeA_features.csv'
    selB_path  = workspace / 'word_ngram_typeB_features.csv'

    rates_df  = pd.read_csv(rates_path, index_col=0)
    selA_df   = pd.read_csv(selA_path)
    selB_df   = pd.read_csv(selB_path)

    sel_A = selA_df['ngram'].tolist()
    sel_B = selB_df['ngram'].tolist()

    dates      = rates_df['date_bce'].tolist()
    unit_order = list(rates_df.index)
    meta_cols  = {'date_bce', 'n_words'}

    def keep(features, df):
        return [f for f in features if f in df.columns]

    sel_A = keep(sel_A, rates_df)
    sel_B = keep(sel_B, rates_df)
    sel_AB = sel_A + sel_B

    ols_A = ols_B = ols_AB = None
    Si_A  = Si_B  = Si_AB  = None

    if sel_A:
        ols_A, Sig_A, Si_A = build_mvn_model(
            rates_df[sel_A], dates, sel_A, ridge_frac=RIDGE_WORD)
        print(f'  Word Type-A model: {len(sel_A)} features, '
              f'cond={np.linalg.cond(Sig_A):.0f}')
    if sel_B:
        ols_B, Sig_B, Si_B = build_mvn_model(
            rates_df[sel_B], dates, sel_B, ridge_frac=RIDGE_WORD)
        print(f'  Word Type-B model: {len(sel_B)} features, '
              f'cond={np.linalg.cond(Sig_B):.0f}')
    if sel_A and sel_B:
        ols_AB, Sig_AB, Si_AB = build_mvn_model(
            rates_df[sel_AB], dates, sel_AB, ridge_frac=RIDGE_WORD)
        print(f'  Word A+B model:    {len(sel_AB)} features, '
              f'cond={np.linalg.cond(Sig_AB):.0f}')

    return (ols_A, Si_A, sel_A,
            ols_B, Si_B, sel_B,
            ols_AB, Si_AB, sel_AB,
            dates, unit_order)


# ===========================================================================
# DATE A SINGLE UNIT WITH ALL THREE MODELS
# ===========================================================================

def date_unit(unit_name, pairs, F, L, T, date_grid,
              # char model
              ols_ng, Si_ng, sel_ng,
              # word models
              ols_A, Si_A, sel_A,
              ols_B, Si_B, sel_B,
              ols_AB, Si_AB, sel_AB):
    """
    Extract text for a unit and compute MAP/CI under all models.
    Returns a dict of posteriors and a dict of scalar results.
    """
    # --- char n-gram ---
    text, n_words = extract_cons_text(pairs, F, L, T)
    ng_freqs      = compute_char_ngram_freqs(text)
    obs_ng        = np.array([ng_freqs.get(f, 0.0) for f in sel_ng], dtype=float)

    post_ng = compute_posterior(obs_ng, ols_ng, Si_ng, sel_ng, date_grid)
    map_ng, ci68_lo_ng, ci68_hi_ng = map_and_ci(post_ng, date_grid)
    _, ci95_lo_ng, ci95_hi_ng      = map_and_ci(post_ng, date_grid, 0.95)

    # --- word n-grams ---
    pos_seq, fw_seq, _ = extract_word_sequences(pairs, F, L, T)
    cnt_A = build_word_ngrams(pos_seq, NGRAM_SIZES_WORD, skip_null=False)
    cnt_B = build_word_ngrams(fw_seq,  NGRAM_SIZES_WORD, skip_null=True)
    ra    = counts_to_rates(cnt_A, n_words)
    rb    = counts_to_rates(cnt_B, n_words)

    map_A = ci68_lo_A = ci68_hi_A = np.nan
    post_A = None
    if sel_A and ols_A:
        obs_A  = np.array([ra.get(f, 0.0) for f in sel_A], dtype=float)
        post_A = compute_posterior(obs_A, ols_A, Si_A, sel_A, date_grid)
        map_A, ci68_lo_A, ci68_hi_A = map_and_ci(post_A, date_grid)

    map_B = ci68_lo_B = ci68_hi_B = np.nan
    post_B = None
    if sel_B and ols_B:
        obs_B  = np.array([rb.get(f, 0.0) for f in sel_B], dtype=float)
        post_B = compute_posterior(obs_B, ols_B, Si_B, sel_B, date_grid)
        map_B, ci68_lo_B, ci68_hi_B = map_and_ci(post_B, date_grid)

    map_AB = ci68_lo_AB = ci68_hi_AB = np.nan
    post_AB = None
    if sel_AB and ols_AB:
        obs_AB  = np.concatenate([
            np.array([ra.get(f, 0.0) for f in sel_A], dtype=float),
            np.array([rb.get(f, 0.0) for f in sel_B], dtype=float),
        ])
        post_AB = compute_posterior(obs_AB, ols_AB, Si_AB, sel_AB, date_grid)
        map_AB, ci68_lo_AB, ci68_hi_AB = map_and_ci(post_AB, date_grid)

    # Archaism index = word MAP (A+B) − char MAP
    arc_idx = (map_AB - map_ng) if not (np.isnan(map_AB) or np.isnan(map_ng)) else np.nan

    scalars = dict(
        unit=unit_name, n_words=n_words, noisy=(n_words < NOISY_THRESHOLD),
        map_ng=map_ng, ci68_lo_ng=ci68_lo_ng, ci68_hi_ng=ci68_hi_ng,
        ci95_lo_ng=ci95_lo_ng, ci95_hi_ng=ci95_hi_ng,
        map_A=map_A, ci68_lo_A=ci68_lo_A, ci68_hi_A=ci68_hi_A,
        map_B=map_B, ci68_lo_B=ci68_lo_B, ci68_hi_B=ci68_hi_B,
        map_AB=map_AB, ci68_lo_AB=ci68_lo_AB, ci68_hi_AB=ci68_hi_AB,
        archaism_index=arc_idx,
        entropy_ng=(posterior_entropy(post_ng) if post_ng is not None else np.nan),
        entropy_AB=(posterior_entropy(post_AB) if post_AB is not None else np.nan),
        skew_ng=(posterior_skewness(post_ng, date_grid) if post_ng is not None else np.nan),
        skew_AB=(posterior_skewness(post_AB, date_grid) if post_AB is not None else np.nan),
    )
    posteriors = dict(ng=post_ng, A=post_A, B=post_B, AB=post_AB)
    return scalars, posteriors


# ===========================================================================
# PLOTTING
# ===========================================================================

GROUP_COLORS = {
    'kings_chr':    '#e41a1c',   # red — Kings/Chr pairs
    'kchr_3way':    '#ff7f00',   # orange — Hezekiah 3-way
    'poem_embed':   '#4dac26',   # green — embedded poems
    'archaic_vocab':'#377eb8',   # blue — archaic vocab (Job)
    'ref_psalm':    '#984ea3',   # purple — reference psalms
    'ref_late':     '#a65628',   # brown — late reference
    'ref_mixed':    '#999999',   # grey — mixed
    'legacy':       '#cccccc',   # light grey — script-15 units
}

GROUP_LABELS = {
    'kings_chr':    'Kings/Chr parallel',
    'kchr_3way':    'Hezekiah 3-way',
    'poem_embed':   'Embedded poem',
    'archaic_vocab':'Archaic vocab (Job)',
    'ref_psalm':    'Reference psalm',
    'ref_late':     'Late reference',
    'ref_mixed':    'Mixed/wisdom',
    'legacy':       'Prior test units',
}

UNIT_DISPLAY = {'D_source': 'D', 'P_source': 'P', 'JE_source': 'JE'}


def display_name(u):
    return UNIT_DISPLAY.get(u, u)


def plot_quadrant(records, legacy_records, out_path):
    """
    Scatter: char_MAP (x) vs word_MAP_AB (y).
    Diagonal = perfect agreement.
    Top-left quadrant = archaism (word older than char).
    Bottom-right = orthographic archaism.
    """
    fig, ax = plt.subplots(figsize=(10, 9))

    # Plot legacy units in faint grey
    for unit, r in legacy_records.items():
        x = r.get('map_ngram', np.nan)
        y = r.get('map_AB_w', np.nan)   # word_AB from master CSV
        if np.isnan(x) or np.isnan(y):
            continue
        ax.scatter(x, y, color='#dddddd', s=30, zorder=2)
        ax.annotate(display_name(unit), (x, y),
                    textcoords='offset points', xytext=(3, 2),
                    fontsize=6, color='#aaaaaa', alpha=0.7)

    # Plot new test units colored by group
    group_handles = {}
    for unit, r in records.items():
        x = r.get('map_ng', np.nan)
        y = r.get('map_AB', np.nan)
        if np.isnan(x) or np.isnan(y):
            continue
        grp   = r.get('group', 'ref_late')
        color = GROUP_COLORS.get(grp, '#888888')
        noisy = r.get('noisy', False)
        mk    = '^' if noisy else 'o'
        sc    = ax.scatter(x, y, color=color, s=70, marker=mk, zorder=4,
                           edgecolors='k', linewidths=0.4)
        if grp not in group_handles:
            group_handles[grp] = sc
        ax.annotate(display_name(unit), (x, y),
                    textcoords='offset points', xytext=(5, 3),
                    fontsize=8, color=color, fontweight='bold')

    # Diagonal (perfect agreement)
    all_x = [r['map_ng']  for r in records.values()
             if not np.isnan(r.get('map_ng', np.nan))]
    all_y = [r['map_AB']  for r in records.values()
             if not np.isnan(r.get('map_AB', np.nan))]
    if all_x and all_y:
        lo = min(min(all_x), min(all_y)) - 50
        hi = max(max(all_x), max(all_y)) + 50
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.4, label='perfect agreement')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    # Quadrant labels
    mid_x = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.05
    mid_y = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.95
    ax.text(mid_x, mid_y, 'word OLDER than char\n→ archaism / orth. updating',
            fontsize=8, color='#666666', va='top', style='italic')
    ax.text(ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.60,
            ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
            'char OLDER than word\n→ archaic orth., updated syntax',
            fontsize=8, color='#666666', va='bottom', style='italic')

    ax.axline((600, 600), slope=1, color='k', lw=0.6, ls='--', alpha=0.3)
    ax.set_xlabel('Character n-gram MAP (BCE)', fontsize=11)
    ax.set_ylabel('Word n-gram A+B MAP (BCE)', fontsize=11)
    ax.set_title('Archaism diagnostic: character n-gram vs. word n-gram dating\n'
                 'Above diagonal = word model dates older (archaism indicator)',
                 fontsize=11, fontweight='bold')

    legend_handles = [plt.Line2D([0],[0], marker='o', color='w',
                                  markerfacecolor=GROUP_COLORS.get(g,'#888888'),
                                  markersize=9, markeredgecolor='k',
                                  markeredgewidth=0.4, label=GROUP_LABELS.get(g, g))
                      for g in group_handles]
    legend_handles.append(plt.Line2D([0],[0], color='k', ls='--', lw=0.8,
                                      label='perfect agreement'))
    ax.legend(handles=legend_handles, fontsize=8,
              loc='upper center', bbox_to_anchor=(0.50, 0.98),
              ncol=2, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


def plot_kchr_comparison(records, out_path):
    """
    Connected-dot plot comparing Kings vs. Chronicles matched pairs.
    Shows MAP_ng (char) and MAP_AB (word) for each pair side-by-side.
    """
    pairs = [
        ('Kings_Solomon',  'Chr_Solomon',  'Solomon narrative'),
        ('Kings_Judah',    'Chr_Judah',    'Judean kings (broad)'),
        ('Kings_Hezekiah', 'Chr_Hezekiah', 'Hezekiah pericope'),
        ('Kings_Fall',     'Chr_Fall',     'Manasseh → Exile'),
        ('Kings_Hezekiah', 'Isa_Hezekiah', 'Hezekiah: Kings vs Isaiah'),
    ]
    valid_pairs = [(k, c, label) for k, c, label in pairs
                   if k in records and c in records]
    if not valid_pairs:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax_idx, (model_key, model_label) in enumerate([('map_ng', 'Char n-gram'),
                                                        ('map_AB', 'Word n-gram A+B')]):
        ax = axes[ax_idx]
        y_pos = np.arange(len(valid_pairs))

        for i, (k_unit, c_unit, label) in enumerate(valid_pairs):
            rk = records[k_unit]
            rc = records[c_unit]
            mk = rk.get(model_key, np.nan)
            mc = rc.get(model_key, np.nan)
            if np.isnan(mk) or np.isnan(mc):
                continue
            color = '#e41a1c' if c_unit.startswith('Chr') else '#ff7f00'
            ax.plot([mk, mc], [i, i], '-', color='#888888', lw=1.2, zorder=2)
            ax.scatter(mk, i, color='#1f78b4', s=80, zorder=4,
                       label='Kings/Isaiah' if i == 0 else '')
            ax.scatter(mc, i, color=color, s=80, zorder=4, marker='s',
                       label='Chronicles' if i == 0 else '')
            delta = mk - mc
            ax.text(max(mk, mc) + 15, i, f'Δ={delta:+.0f}', va='center',
                    fontsize=8, color='#444444')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([lbl for _, _, lbl in valid_pairs], fontsize=9)
        ax.set_xlabel('MAP date (BCE)', fontsize=10)
        ax.set_title(f'{model_label}: Kings (circle) vs. Chr (square)\n'
                     f'Positive Δ = Kings dates older',
                     fontsize=9, fontweight='bold')
        ax.axvline(350, color='#bbbbbb', lw=0.6, ls=':', alpha=0.6)
        ax.axvline(550, color='#bbbbbb', lw=0.6, ls=':', alpha=0.6)
        ax.invert_xaxis()   # older dates on left

    fig.suptitle('Kings vs. Chronicles: character n-gram vs. word n-gram dating\n'
                 'Larger Δ = model more sensitive to register change',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


def plot_poem_comparison(records, posteriors_all, date_grid, out_path):
    """
    For each poem/prose pair: overlay posteriors side-by-side.
    Left column = char n-gram; right column = word n-gram A+B.
    """
    poem_pairs = [
        ('Gen49_poem',  'Gen49_prose',  'Genesis 49 (poem) vs. Genesis prose'),
        ('Deut33_poem', 'Deut33_prose', 'Deut 33 (poem) vs. Deuteronomy prose'),
        ('Num21_songs', 'Num_prose',    'Numbers 21 (song-chapter) vs. Numbers prose'),
        ('Job_poetry',  'Job_frame',    'Job dialogues vs. Job prose frame'),
    ]
    valid = [(p, pr, lbl) for p, pr, lbl in poem_pairs
             if p in records and pr in records]
    if not valid:
        return

    n_rows = len(valid)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    for row_idx, (poem_u, prose_u, label) in enumerate(valid):
        rp   = records[poem_u]
        rpr  = records[prose_u]
        post_p  = posteriors_all.get(poem_u, {})
        post_pr = posteriors_all.get(prose_u, {})

        for col_idx, (model_key_post, model_key_map, model_label) in enumerate([
            ('ng',  'map_ng',  'Char n-gram'),
            ('AB',  'map_AB',  'Word n-gram A+B'),
        ]):
            ax   = axes[row_idx, col_idx]
            pp   = post_p.get(model_key_post)
            ppr  = post_pr.get(model_key_post)
            mp   = rp.get(model_key_map, np.nan)
            mpr  = rpr.get(model_key_map, np.nan)

            if pp is not None:
                ax.plot(date_grid, pp,  color='#d73027', lw=2,
                        label=f'poem MAP={mp:.0f}')
                ax.axvline(mp,  color='#d73027', lw=1, ls='--', alpha=0.6)
            if ppr is not None:
                ax.plot(date_grid, ppr, color='#4575b4', lw=2, ls='--',
                        label=f'prose MAP={mpr:.0f}')
                ax.axvline(mpr, color='#4575b4', lw=1, ls='--', alpha=0.6)

            ax.set_xlim(DATE_HI, DATE_LO)
            ax.tick_params(labelsize=7)
            ax.set_xlabel('BCE', fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.set_title(model_label, fontsize=9, fontweight='bold')
            ax.legend(fontsize=7)

    fig.suptitle('Embedded poem vs. surrounding prose: posterior comparison\n'
                 'Red = poem; Blue = prose',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


def plot_posterior_grid(records, posteriors_all, date_grid, out_path):
    """
    Compact grid of posteriors for selected diagnostic pairs.
    Shows char (solid) and word A+B (dashed) for the same unit.
    """
    SHOW_UNITS = [
        'Kings_Hezekiah', 'Isa_Hezekiah', 'Chr_Hezekiah',
        'Kings_Solomon',  'Chr_Solomon',
        'Gen49_poem',     'Gen49_prose',
        'Deut33_poem',    'Deut33_prose',
        'Psalm_18',       '2Sam_22',
        'Job_poetry',     'Job_frame',
    ]
    show = [u for u in SHOW_UNITS if u in records]
    if not show:
        return

    ncols = 4
    nrows = (len(show) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten() if nrows > 1 else list(axes)

    for ax, unit in zip(axes, show):
        r     = records[unit]
        posts = posteriors_all.get(unit, {})
        p_ng  = posts.get('ng')
        p_ab  = posts.get('AB')
        arc   = r.get('archaism_index', np.nan)
        color = GROUP_COLORS.get(r.get('group', 'ref_late'), '#888888')

        if p_ng is not None:
            ax.plot(date_grid, p_ng, color=color, lw=2,
                    label=f'char {r.get("map_ng","?"):.0f}')
        if p_ab is not None:
            ax.plot(date_grid, p_ab, color=color, lw=1.5, ls='--',
                    label=f'word {r.get("map_AB","?"):.0f}')

        ax.set_xlim(DATE_HI, DATE_LO)
        ax.tick_params(labelsize=6)
        arc_str = f'arc={arc:+.0f}' if not np.isnan(arc) else ''
        ax.set_title(f'{display_name(unit)}\n{arc_str}', fontsize=8)
        ax.set_xlabel('BCE', fontsize=7)
        ax.legend(fontsize=6)

    for ax in axes[len(show):]:
        ax.set_visible(False)

    fig.suptitle('Posterior comparison: char n-gram (solid) vs. word n-gram A+B (dashed)\n'
                 'Archaism index = word MAP − char MAP  (positive = word appears older)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=str(WORKSPACE))
    parser.add_argument('--ling-csv', default=str(WORKSPACE / 'master_dating_results.csv'))
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
    # STEP 2 — Reconstruct trained models from saved CSV files
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 2 — Reconstruct models from saved training CSV files')
    print('=' * 70)

    ols_ng, Si_ng, sel_ng, ng_dates, ng_units = load_char_model(WORKSPACE)
    (ols_A, Si_A, sel_A,
     ols_B, Si_B, sel_B,
     ols_AB, Si_AB, sel_AB,
     w_dates, w_units) = load_word_model(WORKSPACE)

    date_grid = np.linspace(DATE_LO, DATE_HI, N_GRID)

    # -----------------------------------------------------------------------
    # STEP 3 — Load prior linguistic results for reference
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 3 — Load prior linguistic results')
    print('=' * 70)

    master_df    = pd.read_csv(args.ling_csv)
    master_df['unit_key'] = master_df['unit'].replace(
        {'D': 'D_source', 'P': 'P_source', 'JE': 'JE_source'})
    master_records = {}
    for _, row in master_df.iterrows():
        r = dict(row)
        r['map_AB_w'] = r.get('map_AB', np.nan)   # alias for quadrant plot
        master_records[r['unit_key']] = r
    # Merge word_ngram dating results if available
    wng_path = WORKSPACE / 'word_ngram_dating_results.csv'
    if wng_path.exists():
        wng_df = pd.read_csv(wng_path)
        wng_df['unit_key'] = wng_df['unit'].replace(
            {'D': 'D_source', 'P': 'P_source', 'JE': 'JE_source'})
        for _, row in wng_df.iterrows():
            uk = row['unit_key']
            if uk in master_records:
                master_records[uk]['map_AB_w'] = row.get('map_AB', np.nan)
    print(f'  Loaded {len(master_records)} legacy unit records.')

    # -----------------------------------------------------------------------
    # STEP 4 — Date all new test units
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 4 — Date new test units')
    print('=' * 70)

    print(f'\n  {"Unit":<20}  {"n_words":>8}  '
          f'{"MAP_char":>9}  {"MAP_word":>9}  {"arc_idx":>8}  '
          f'{"H_ng":>6}  {"H_AB":>6}  noisy')
    print('  ' + '-' * 82)

    records_new     = {}
    posteriors_all  = {}

    for unit, info in ALL_TEST_UNITS.items():
        pairs = info['pairs']
        group = info.get('group', 'ref_late')
        scalars, posts = date_unit(
            unit, pairs, F, L, T, date_grid,
            ols_ng, Si_ng, sel_ng,
            ols_A,  Si_A,  sel_A,
            ols_B,  Si_B,  sel_B,
            ols_AB, Si_AB, sel_AB,
        )
        scalars['group'] = group
        scalars['notes'] = info.get('notes', '')
        records_new[unit]    = scalars
        posteriors_all[unit] = posts

        noisy_str = '⚠' if scalars['noisy'] else ''
        print(f'  {unit:<20}  {scalars["n_words"]:>8,}  '
              f'{scalars["map_ng"]:>9.0f}  {scalars["map_AB"]:>9.0f}  '
              f'{scalars["archaism_index"]:>+8.0f}  '
              f'{scalars["entropy_ng"]:>6.2f}  {scalars["entropy_AB"]:>6.2f}  '
              f'{noisy_str}')

    # -----------------------------------------------------------------------
    # STEP 5 — Kings/Chronicles analysis
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 5 — Kings/Chronicles parallel analysis')
    print('=' * 70)

    kchr_pairs = [
        ('Kings_Solomon',  'Chr_Solomon',  'Solomon narrative'),
        ('Kings_Judah',    'Chr_Judah',    'Judean kings (broad)'),
        ('Kings_Hezekiah', 'Chr_Hezekiah', 'Hezekiah pericope'),
        ('Kings_Fall',     'Chr_Fall',     'Manasseh → Exile'),
        ('Kings_Hezekiah', 'Isa_Hezekiah', 'Hezekiah: Kings vs. Isaiah'),
    ]

    print(f'\n  {"Comparison":<35}  {"Δchar":>7}  {"Δword":>7}  '
          f'{"Δchar/Δword":>11}  {"interpretation"}')
    print('  ' + '-' * 90)

    for k_unit, c_unit, label in kchr_pairs:
        if k_unit not in records_new or c_unit not in records_new:
            continue
        rk = records_new[k_unit]
        rc = records_new[c_unit]
        delta_char = rk['map_ng']  - rc['map_ng']
        delta_word = rk['map_AB']  - rc['map_AB']
        ratio      = delta_char / delta_word if delta_word != 0 else float('inf')
        if abs(delta_word) < 20:
            interp = 'no word signal'
        elif delta_char > delta_word * 1.3:
            interp = 'orthographic updating > syntactic'
        elif delta_word > delta_char * 1.3:
            interp = 'syntactic updating > orthographic (?)'
        else:
            interp = 'comparable updating'
        print(f'  {label:<35}  {delta_char:>+7.0f}  {delta_word:>+7.0f}  '
              f'{ratio:>11.2f}  {interp}')

    # -----------------------------------------------------------------------
    # STEP 6 — Embedded poem analysis
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 6 — Embedded poem vs. surrounding prose')
    print('=' * 70)

    poem_pairs = [
        ('Gen49_poem',  'Gen49_prose',  'Genesis 49 poem vs. Genesis prose'),
        ('Deut33_poem', 'Deut33_prose', 'Deut 33 poem vs. Deut prose'),
        ('Num21_songs', 'Num_prose',    'Num 21 song-chapter vs. Num prose'),
        ('Job_poetry',  'Job_frame',    'Job dialogues vs. Job frame'),
    ]

    print(f'\n  {"Pair":<40}  {"char_poem":>9}  {"char_prose":>10}  '
          f'{"word_poem":>9}  {"word_prose":>10}  '
          f'{"arc_poem":>9}  {"arc_prose":>10}')
    print('  ' + '-' * 100)

    for poem_u, prose_u, label in poem_pairs:
        if poem_u not in records_new or prose_u not in records_new:
            continue
        rp  = records_new[poem_u]
        rpr = records_new[prose_u]
        print(f'  {label:<40}  '
              f'{rp["map_ng"]:>9.0f}  {rpr["map_ng"]:>10.0f}  '
              f'{rp["map_AB"]:>9.0f}  {rpr["map_AB"]:>10.0f}  '
              f'{rp["archaism_index"]:>+9.0f}  {rpr["archaism_index"]:>+10.0f}')

    # Also print Psalm 18 / 2 Sam 22 if present
    if 'Psalm_18' in records_new and '2Sam_22' in records_new:
        print('\n  -- Duplicate transmission (Psalm 18 = 2 Samuel 22) --')
        rp18  = records_new['Psalm_18']
        rs22  = records_new['2Sam_22']
        print(f'  {"Unit":<12}  {"char_MAP":>9}  {"word_MAP":>9}  {"arc_idx":>9}  '
              f'{"entropy_ng":>11}  {"entropy_AB":>11}')
        for u, r in [('Psalm_18', rp18), ('2Sam_22', rs22)]:
            print(f'  {u:<12}  {r["map_ng"]:>9.0f}  {r["map_AB"]:>9.0f}  '
                  f'{r["archaism_index"]:>+9.0f}  '
                  f'{r["entropy_ng"]:>11.2f}  {r["entropy_AB"]:>11.2f}')

    # -----------------------------------------------------------------------
    # STEP 7 — Save results CSV
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 7 — Save results')
    print('=' * 70)

    rows = []
    for unit, r in records_new.items():
        row = {k: v for k, v in r.items()}
        info = ALL_TEST_UNITS.get(unit, {})
        row['group'] = info.get('group', r.get('group', ''))
        row['notes'] = info.get('notes', r.get('notes', ''))
        rows.append(row)

    results_df   = pd.DataFrame(rows)
    numeric_cols = [c for c in results_df.columns
                    if c not in ('unit', 'group', 'notes', 'noisy')]
    for col in numeric_cols:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce').round(2)

    out_csv = outdir / 'archaism_diagnostic_results.csv'
    results_df.to_csv(out_csv, index=False)
    print(f'  Saved: {out_csv.name}')

    # -----------------------------------------------------------------------
    # STEP 8 — Plots
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 8 — Plots')
    print('=' * 70)

    plot_quadrant(records_new, master_records,
                  str(outdir / 'archaism_quadrant_plot.png'))

    plot_kchr_comparison(records_new,
                         str(outdir / 'archaism_kchr_comparison.png'))

    plot_poem_comparison(records_new, posteriors_all, date_grid,
                         str(outdir / 'archaism_poem_comparison.png'))

    plot_posterior_grid(records_new, posteriors_all, date_grid,
                        str(outdir / 'archaism_posterior_grid.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
