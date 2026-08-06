#!/usr/bin/env python3
"""
Script 11: Comprehensive Diachronic Dating Analysis
=====================================================
Synthesises all significant diachronic features from the full pipeline
(scripts 06–10) using a multivariate normal Bayesian model to date the
Pentateuch books and documentary sources (D, P, JE).

What this script does:
  1. Loads the full feature scan from script 06 (oracle Jeremiah corpus)
  2. Reports all features tested, those passing p < 0.10, and those that
     are also LOO-robust (≥ 75 % of leave-one-out resamplings).
  3. Applies the linguistic POS filter (excludes topical nouns/verbs/names).
  4. Augments the feature set with ratio features: frac_אני, frac_ש,
     rate_ūt_nouns (from the morphosyntactic catalogue in script 10).
  5. Re-extracts these features for Torah books and D/P/JE sources.
  6. Builds a multivariate normal Bayesian model with Tikhonov-regularised
     residual covariance (MVN likelihood; independent features is special case).
  7. Computes date posteriors with 68 % and 95 % credible intervals.
  8. Runs an archaism audit: per-feature LBH scores for each source.
  9. Produces summary tables (CSV) and diagnostic plots.

Usage
-----
    python 11_comprehensive_dating.py [--data-path PATH] [--outdir DIR]
                                      [--ridge FLOAT] [--n-grid INT]
"""

import sys
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict
from scipy.stats import spearmanr

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

JEREMIAH_ORACLE_CHAPTERS = set(
    list(range(1, 7)) + list(range(8, 11)) + list(range(12, 17)) +
    [19, 20] + list(range(22, 24)) + list(range(30, 32)) + list(range(46, 52))
)

# Documentary source chapter-range attributions (Friedman 2003 / Baden 2012)
DOC_SOURCES = {
    'Genesis': {
        'P':  [(1,  2), (5, 5), (6, 6), (7, 7), (9, 9), (11, 11),
               (17,17), (23,23), (25,25), (27,28), (35,36), (46,46), (49,50)],
        'JE': [(2,  4), (6, 6), (8, 8), (10,10), (12,16), (18,22),
               (24,24), (26,27), (29,34), (37,45), (47,49)],
    },
    'Exodus': {
        'P':  [(1, 2), (6, 7), (12,12), (16,16), (25,31), (35,40)],
        'JE': [(2, 5), (8,11), (13,14), (17,18), (19,24), (32,34)],  # ch.15 excluded (Song of Sea)
    },
    'Leviticus': {
        'P':  [(1, 27)],
    },
    'Numbers': {
        'P':  [(1,10), (15,15), (17,19), (25,25), (27,31), (33,36)],
        'JE': [(11,14), (16,16), (20,24), (25,25), (32,32)],
    },
    'Deuteronomy': {
        'D':  [(1, 34)],
    },
}

AGGREGATED_SOURCES = {
    'D':  [('Deuteronomy', DOC_SOURCES['Deuteronomy']['D'])],
    'P':  [('Genesis',     DOC_SOURCES['Genesis']['P']),
           ('Exodus',      DOC_SOURCES['Exodus']['P']),
           ('Leviticus',   DOC_SOURCES['Leviticus']['P']),
           ('Numbers',     DOC_SOURCES['Numbers']['P'])],
    'JE': [('Genesis',     DOC_SOURCES['Genesis']['JE']),
           ('Exodus',      DOC_SOURCES['Exodus']['JE']),
           ('Numbers',     DOC_SOURCES['Numbers']['JE'])],
}

TORAH_BOOKS = ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy']

# Chapter ranges to use instead of the full book, keyed by book name.
# Excludes embedded ancient songs so the host-book analysis is clean.
TORAH_BOOK_OVERRIDES = {
    'Exodus': [(1, 14), (16, 40)],  # exclude ch. 15 (Song of the Sea)
}

SOURCE_COLORS = {'D': '#1f77b4', 'P': '#d62728', 'JE': '#2ca02c'}

# Linguistic POS kept by the filter (grammatical/functional words only)
LINGUISTIC_POS = {
    'pronoun', 'dem-pronoun', 'prep', 'conj',
    'adverb', 'negator', 'article', 'interrogative', 'interjection',
}

# Scholarly reference date ranges (for comparison only)
SCHOLARLY_DATES = {
    'D':  (700, 621, 'Deuteronomist (traditional)'),
    'P':  (550, 400, 'Priestly source (traditional)'),
    'JE': (950, 750, 'JE combined (traditional)'),
}

# Training-range span (for judging 95 % CI width)
TRAINING_RANGE_YEARS = 760 - 167   # 593 years
CI95_WIDTH_THRESHOLD = 0.75        # flag if 95 % CI > 75 % of training range


# ---------------------------------------------------------------------------
# Feature synthesis helpers
# ---------------------------------------------------------------------------

def load_feature_scans(full_path, robust_path):
    """Load the full feature scan and the LOO-robust subset from script 06."""
    full = pd.read_csv(full_path)
    robust = pd.read_csv(robust_path)
    return full, robust


def filter_linguistic(robust_df, verbose=True):
    """Keep only grammatical/functional features (same logic as script 07)."""
    keep = []
    for _, row in robust_df.iterrows():
        feat = str(row['feature'])
        if feat.startswith(('vt::', 'vs::', 'morph::')):
            keep.append(True)
            continue
        if feat.startswith('lex::'):
            desc = str(row.get('description', ''))
            m = re.search(r'\(([^)]+)\)', desc)
            pos = m.group(1).strip() if m else ''
            keep.append(pos in LINGUISTIC_POS)
        else:
            keep.append(False)
    filtered = robust_df[keep].reset_index(drop=True)
    if verbose:
        print(f"  Linguistic filter: {len(filtered)}/{len(robust_df)} features "
              f"kept ({len(robust_df) - len(filtered)} topical features excluded)")
    return filtered


def augment_ratio_features(rates_df):
    """
    Add ratio features to the training rates DataFrame.
      frac_ani   = >NJ / (>NJ + >NKJ)
      frac_she   = C   / (C   + >CR)
      rate_ut    = sum of lexeme rates whose name ends with WT/ or WT::
                   (approximated from any 'lex::*WT/' column)

    These are derived from existing columns so no BHSA re-scan is needed
    for the training corpus.
    """
    df = rates_df.copy()
    ani  = df.get('lex::>NJ',  pd.Series(0.0, index=df.index))
    ank  = df.get('lex::>NKJ', pd.Series(0.0, index=df.index))
    she  = df.get('lex::C',    pd.Series(0.0, index=df.index))
    ash  = df.get('lex::>CR',  pd.Series(0.0, index=df.index))

    denom_ani = ani + ank
    df['frac_ani'] = np.where(denom_ani > 0, ani / denom_ani, np.nan)

    denom_she = she + ash
    df['frac_she'] = np.where(denom_she > 0, she / denom_she, np.nan)

    # Approximate -ūt noun rate: sum all lex::*WT/ columns
    ut_cols = [c for c in df.columns if re.match(r'lex::\w+WT/', c)]
    if ut_cols:
        df['rate_ut_nouns'] = df[ut_cols].sum(axis=1)
    else:
        df['rate_ut_nouns'] = 0.0

    return df


RATIO_FEATURES_META = {
    'frac_ani':     ('increase', 'Fraction אני/(אני+אנכי) — Hurvitz 1972'),
    'frac_she':     ('increase', 'Fraction ש/(ש+אשר) — Polzin 1976'),
    'rate_ut_nouns':('increase', '-ūt nominalisation rate per 1k — Hornkohl 2024'),
}

# Tier-3 features that passed p < 0.10 and LOO=1.00 in script 12
TIER3_FEATURES = ['frac_infc', 'frac_fronted', 'frac_null_subj', 'frac_wqtl_wayq']

TIER3_FEATURE_META = {
    'frac_infc':       ('+0.645 ***', 'Infinitive-construct clause fraction — script 12'),
    'frac_fronted':    ('-0.501 **',  'Fronted (non-V-initial) clause fraction — script 12'),
    'frac_null_subj':  ('-0.430 **',  'Null-subject verbal clause fraction — script 12'),
    'frac_wqtl_wayq':  ('-0.383 *',   'Waw-qatal/(wayq+wqtl) narrative ratio — script 12'),
}

# ---------------------------------------------------------------------------
# Tier-3 clause/phrase constants (mirrored from script 12)
# ---------------------------------------------------------------------------

WAYQ_TYPES  = {'Way0', 'WayX', 'WaYX'}
WQTL_TYPES  = {'WQt0', 'WQtX', 'WxQ0', 'WxQX'}
WNARR_TYPES = WAYQ_TYPES | WQTL_TYPES
FRONT_TYPES = {'xQt0', 'xYq0', 'xQtX', 'xYqX', 'xIm0',
               'WxY0', 'WxQ0', 'WxQX', 'WxYX', 'WxI0',
               'XQtl', 'XYqt'}
CPEN_TYPES  = {'CPen'}
NMCL_TYPES  = {'NmCl', 'AjCl'}
PTCP_TYPES  = {'Ptcp'}
INFC_TYPES  = {'InfC'}
VERBAL_TYPES = (WAYQ_TYPES | WQTL_TYPES | FRONT_TYPES |
                {'ZQt0', 'ZQtX', 'ZYq0', 'ZYqX', 'ZIm0',
                 'WYq0', 'WYqX', 'WIm0', 'WXYq', 'WXQt',
                 'xQt0', 'xYq0', 'xQtX', 'xYqX',
                 'Way0', 'WayX'})
SKIP_TYPES  = {'Ellp', 'Voct', 'MSyn', 'InfA'}


# ---------------------------------------------------------------------------
# BHSA loading + word iteration
# ---------------------------------------------------------------------------

def load_bhsa(data_path):
    try:
        from tf.fabric import Fabric
    except ImportError:
        sys.exit('text-fabric not installed.  Run: pip install text-fabric')
    print(f'Loading BHSA from {data_path}...')
    TF = Fabric(locations=str(data_path), modules=[''], silent=True)
    # Include typ/function/domain for Tier-3 clause/phrase features
    api = TF.load(
        'otype lex sp vs vt ps nu gn st prs_ps chapter typ function domain',
        silent=True)
    return api


def words_for_ranges(book_name, ch_ranges, F, L, T):
    """
    Yield (lex, sp, vt, vs, nu, gn, st, prs_ps) for all words in book_name
    within the given list of (start_ch, end_ch) ranges.
    """
    bn = T.nodeFromSection((book_name,))
    if bn is None:
        return
    for ch_node in L.d(bn, 'chapter'):
        ch_num = int(F.chapter.v(ch_node))
        if not any(s <= ch_num <= e for s, e in ch_ranges):
            continue
        for w in L.d(ch_node, 'word'):
            yield (F.lex.v(w), F.sp.v(w), F.vt.v(w), F.vs.v(w),
                   F.nu.v(w),  F.gn.v(w), F.st.v(w), F.prs_ps.v(w))


# ---------------------------------------------------------------------------
# Feature extraction for Torah texts
# ---------------------------------------------------------------------------

def extract_unit_features(unit_label, book_ch_pairs, F, L, T, feature_names):
    """
    Extract per-1000-word rates for all features in feature_names.

    feature_names may include:
      lex::X       → rate of lexeme X per 1k words
      vt::X        → rate of verb form X per 1k words
      vs::X        → rate of verbal stem X per 1k words
      morph::*     → specific morphological counts per 1k
      frac_ani     → computed as ratio from lex counts
      frac_she     → computed as ratio from lex counts
      rate_ut_nouns→ sum of -WT/ lexeme rates per 1k

    book_ch_pairs: list of (book_name, [(start_ch, end_ch), ...])
    """
    rows = []
    for book, ch_ranges in book_ch_pairs:
        for tup in words_for_ranges(book, ch_ranges, F, L, T):
            rows.append(tup)

    if not rows:
        return None, 0

    df = pd.DataFrame(rows, columns=['lex','sp','vt','vs','nu','gn','st','prs_ps'])
    n = len(df)

    lex_ctr  = df['lex'].value_counts()
    verb_df  = df[df['sp'] == 'verb']
    noun_df  = df[df['sp'] == 'subs']
    vt_ctr   = verb_df['vt'].value_counts()
    vs_ctr   = verb_df['vs'].value_counts()

    result = {}
    for feat in feature_names:
        if feat.startswith('lex::'):
            lex = feat[5:]
            result[feat] = lex_ctr.get(lex, 0) / n * 1000

        elif feat.startswith('vt::'):
            vt = feat[4:]
            result[feat] = vt_ctr.get(vt, 0) / n * 1000

        elif feat.startswith('vs::'):
            vs = feat[4:]
            result[feat] = vs_ctr.get(vs, 0) / n * 1000

        elif feat == 'morph::const_count':
            result[feat] = (noun_df['st'] == 'c').sum() / n * 1000
        elif feat == 'morph::abs_count':
            result[feat] = (noun_df['st'] == 'a').sum() / n * 1000
        elif feat == 'morph::prs_rate':
            result[feat] = (df['prs_ps'] != 'NA').sum() / n * 1000
        elif feat == 'morph::noun_pl_rate':
            result[feat] = (noun_df['nu'] == 'pl').sum() / n * 1000
        elif feat == 'morph::noun_f_rate':
            result[feat] = (noun_df['gn'] == 'f').sum() / n * 1000

        elif feat == 'frac_ani':
            a = lex_ctr.get('>NJ', 0)
            b = lex_ctr.get('>NKJ', 0)
            result[feat] = a / (a + b) if (a + b) > 0 else np.nan

        elif feat == 'frac_she':
            a = lex_ctr.get('C', 0)
            b = lex_ctr.get('>CR', 0)
            result[feat] = a / (a + b) if (a + b) > 0 else np.nan

        elif feat == 'rate_ut_nouns':
            ut = sum(v for k, v in lex_ctr.items()
                     if k.endswith('WT/') or k.endswith('WT'))
            result[feat] = ut / n * 1000

        else:
            result[feat] = np.nan

    return np.array([result.get(f, np.nan) for f in feature_names]), n


def extract_source(source_name, F, L, T, feature_names):
    """Aggregate features across all sections of a documentary source."""
    pairs = AGGREGATED_SOURCES.get(source_name, [])
    vecs, ns = [], []
    for book, ch_ranges in pairs:
        vec, nw = extract_unit_features(
            f'{source_name}:{book}', [(book, ch_ranges)], F, L, T, feature_names)
        if vec is not None and nw >= 50:
            vecs.append(vec)
            ns.append(nw)
    if not vecs:
        return np.full(len(feature_names), np.nan), 0
    ns_arr = np.array(ns, dtype=float)
    w = ns_arr / ns_arr.sum()
    weighted = np.nansum(np.vstack(vecs) * w[:, None], axis=0)
    all_nan = np.all(~np.isfinite(np.vstack(vecs)), axis=0)
    weighted[all_nan] = np.nan
    return weighted, int(ns_arr.sum())


def extract_torah_book(book_name, F, L, T, feature_names):
    """Extract features for an entire Torah book (all chapters)."""
    bn = T.nodeFromSection((book_name,))
    if bn is None:
        return np.full(len(feature_names), np.nan), 0
    ch_nums = [int(F.chapter.v(c)) for c in L.d(bn, 'chapter')]
    ch_ranges = [(min(ch_nums), max(ch_nums))]
    return extract_unit_features(book_name, [(book_name, ch_ranges)],
                                  F, L, T, feature_names)


def extract_tier3_for_unit(book_ch_pairs, F, L, T):
    """
    Compute Tier-3 clause/phrase features for a unit defined by book_ch_pairs.
    Pools all clauses from all sections (no averaging of fractions).
    Returns dict of TIER3_FEATURES → float (or NaN if insufficient data).
    """
    n_words = n_clauses = n_verbal = 0
    n_nmcl = n_ptcp = n_infc = n_fronted = n_cpen = 0
    n_wayq = n_wqtl = n_wnarr = 0
    n_sv = n_sv_total = n_null_subj = n_ov = n_ov_total = 0

    for book, ch_ranges in book_ch_pairs:
        bn = T.nodeFromSection((book,))
        if bn is None:
            continue
        for ch_node in L.d(bn, 'chapter'):
            ch_num = int(F.chapter.v(ch_node))
            if not any(s <= ch_num <= e for s, e in ch_ranges):
                continue
            for cl in L.d(ch_node, 'clause'):
                typ = F.typ.v(cl)
                if typ in SKIP_TYPES:
                    n_words += sum(1 for _ in L.d(cl, 'word'))
                    continue
                n_words += sum(1 for _ in L.d(cl, 'word'))
                n_clauses += 1
                if typ in NMCL_TYPES:  n_nmcl    += 1
                if typ in PTCP_TYPES:  n_ptcp    += 1
                if typ in INFC_TYPES:  n_infc    += 1
                if typ in FRONT_TYPES: n_fronted += 1
                if typ in CPEN_TYPES:  n_cpen    += 1
                if typ in WAYQ_TYPES:  n_wayq    += 1
                if typ in WQTL_TYPES:  n_wqtl    += 1
                if typ in WNARR_TYPES: n_wnarr   += 1
                if typ in VERBAL_TYPES:
                    n_verbal += 1
                    phrases   = list(L.d(cl, 'phrase'))
                    ph_funcs  = {F.function.v(ph): ph for ph in phrases}
                    has_subj  = 'Subj' in ph_funcs
                    has_pred  = 'Pred' in ph_funcs
                    has_objc  = 'Objc' in ph_funcs
                    if not has_subj:
                        n_null_subj += 1
                    if has_subj and has_pred:
                        n_sv_total += 1
                        if ph_funcs['Subj'] < ph_funcs['Pred']:
                            n_sv += 1
                    if has_objc and has_pred:
                        n_ov_total += 1
                        if ph_funcs['Objc'] < ph_funcs['Pred']:
                            n_ov += 1

    if n_clauses == 0:
        return {f: np.nan for f in TIER3_FEATURES}

    return {
        'frac_infc':      n_infc   / n_clauses,
        'frac_fronted':   n_fronted / n_clauses,
        'frac_null_subj': n_null_subj / n_verbal if n_verbal > 10 else np.nan,
        'frac_wqtl_wayq': n_wqtl / n_wnarr if n_wnarr > 5 else np.nan,
    }


# ---------------------------------------------------------------------------
# MVN model
# ---------------------------------------------------------------------------

def build_mvn_model(rates_df, dates_bce, feature_names, ridge_frac=0.10):
    """
    Fit OLS (feature ~ a + b*date) for each feature over training units.
    Returns:
      ols_params  — dict {feat: (intercept, slope, resid_std)}
      Sigma_reg   — Tikhonov-regularised residual covariance (K×K)
      Sigma_inv   — inverse of Sigma_reg
    """
    ols_params = {}
    residual_rows = []

    for fn in feature_names:
        y = rates_df[fn].values.astype(float)
        x = np.array(dates_bce, dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 4:
            ols_params[fn] = (np.nanmean(y), 0.0, np.nanstd(y) + 1e-9)
            residual_rows.append(np.zeros(len(dates_bce)))
            continue
        xv, yv = x[valid], y[valid]
        b, a = np.polyfit(xv, yv, 1)   # y = a + b*x  (polyfit returns [b, a])
        pred = a + b * x
        resid = y - pred
        resid[~valid] = 0.0
        resid_std = np.std(yv - (a + b * xv)) + 1e-9
        ols_params[fn] = (a, b, resid_std)
        residual_rows.append(resid)

    R = np.array(residual_rows).T   # (n_units, n_features)
    Sigma = np.cov(R.T)
    if Sigma.ndim == 0:
        Sigma = np.array([[float(Sigma)]])
    K = Sigma.shape[0]
    ridge = ridge_frac * np.trace(Sigma) / K
    Sigma_reg = Sigma + ridge * np.eye(K)
    Sigma_inv = np.linalg.inv(Sigma_reg)

    cond = np.linalg.cond(Sigma_reg)
    print(f"  MVN: {K} features, condition number {cond:.0f} "
          f"(ridge={ridge_frac:.2f})")
    return ols_params, Sigma_reg, Sigma_inv


def mvn_log_likelihood(x_obs, date_bce, ols_params, feature_names,
                        Sigma_inv, n_words=1):
    mu = np.array([ols_params[fn][0] + ols_params[fn][1] * date_bce
                   for fn in feature_names])
    valid = np.isfinite(x_obs)
    if valid.sum() < 2:
        return -np.inf
    diff = x_obs - mu
    diff[~valid] = 0.0
    scale = float(np.clip(n_words / 5000, 1.0, 5.0))
    return -0.5 * scale * diff @ Sigma_inv @ diff


def compute_posterior(x_obs, n_words, ols_params, feature_names, Sigma_inv,
                      date_min=1200, date_max=50, n_grid=500,
                      prior_mean=600, prior_sd=350):
    """
    Return (date_grid, posterior, map_date, ci68_lo, ci68_hi, ci95_lo, ci95_hi).
    All dates in BCE (positive = older).
    ci68 / ci95 are (earlier_date, later_date) i.e. (hi_BCE, lo_BCE).
    """
    date_grid = np.linspace(date_max, date_min, n_grid)
    log_prior = -0.5 * ((date_grid - prior_mean) / prior_sd) ** 2
    log_lik   = np.array([
        mvn_log_likelihood(x_obs, d, ols_params, feature_names, Sigma_inv, n_words)
        for d in date_grid
    ])
    log_post = log_lik + log_prior
    log_post -= log_post.max()
    post = np.exp(log_post)
    post /= post.sum()

    map_date = date_grid[post.argmax()]
    cdf = np.cumsum(post)

    def quantile(q):
        idx = np.searchsorted(cdf, q)
        return date_grid[min(idx, n_grid - 1)]

    ci68_lo = quantile(0.16)   # earlier date (higher BCE)
    ci68_hi = quantile(0.84)   # later date  (lower BCE)
    ci95_lo = quantile(0.025)
    ci95_hi = quantile(0.975)

    return date_grid, post, map_date, ci68_lo, ci68_hi, ci95_lo, ci95_hi


# ---------------------------------------------------------------------------
# Archaism audit
# ---------------------------------------------------------------------------

def compute_archaism_scores(x_obs_dict, ols_params, feature_names,
                             archaic_date=720, modern_date=250):
    """
    For each feature and each unit in x_obs_dict, compute a normalised
    LBH score: 0 = fully archaic (predicted CBH value), 1 = fully modern
    (predicted LBH value).  Values outside [0,1] are clipped and flagged.
    """
    scores = {}   # {unit: {feature: score}}
    for unit, x_obs in x_obs_dict.items():
        unit_scores = {}
        for i, fn in enumerate(feature_names):
            a, b, _ = ols_params[fn]
            cbh_val = a + b * archaic_date
            lbh_val = a + b * modern_date
            span = lbh_val - cbh_val
            if abs(span) < 1e-9 or not np.isfinite(x_obs[i]):
                unit_scores[fn] = np.nan
                continue
            score = (x_obs[i] - cbh_val) / span
            unit_scores[fn] = score
        scores[unit] = unit_scores
    return scores


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_model_comparison(date_grid, posteriors_full, posteriors_r, records,
                          output_path):
    """
    Overlay full-model and resistant-model posteriors for D, P, JE.
    This is the archaism-calibration comparison plot.
    """
    srcs = [s for s in ['D', 'P', 'JE']
            if s in posteriors_full and s in posteriors_r]
    if not srcs:
        return

    fig, axes = plt.subplots(1, len(srcs), figsize=(5 * len(srcs), 4),
                              sharey=False)
    if len(srcs) == 1:
        axes = [axes]

    for ax, src in zip(axes, srcs):
        color = SOURCE_COLORS[src]
        pf = posteriors_full[src]
        pr = posteriors_r[src]
        ax.plot(date_grid, pf / pf.max(), color=color, lw=2.5,
                label='Full model (all features)')
        ax.plot(date_grid, pr / pr.max(), color=color, lw=2.0, ls=':',
                label='Resistant model\n(Tier-3 only)')
        # MAP lines
        map_f = date_grid[pf.argmax()]
        map_r = date_grid[pr.argmax()]
        ax.axvline(map_f, color=color, lw=1.2, ls='--', alpha=0.7,
                   label=f'MAP full = {map_f:.0f} BCE')
        ax.axvline(map_r, color=color, lw=1.2, ls='dotted', alpha=0.7,
                   label=f'MAP resist = {map_r:.0f} BCE')
        ax.set_xlim(date_grid.max(), date_grid.min())
        ax.set_xlabel('Date (BCE)')
        ax.set_ylabel('Normalised posterior')
        ax.set_title(f'Source {src}')
        ax.legend(fontsize=7)

    fig.suptitle(
        'Archaism-calibration comparison:\nFull model vs. syntax-resistant model',
        fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Model-comparison plot saved: {output_path}')


def plot_posteriors(date_grid, posteriors, records, output_path):
    """Plot date posteriors for D, P, JE sources and individual Torah books."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top: sources
    ax = axes[0]
    for src in ['D', 'P', 'JE']:
        if src not in posteriors:
            continue
        post = posteriors[src]
        color = SOURCE_COLORS[src]
        ax.plot(date_grid, post, color=color, lw=2.5, label=src)
        r = next(x for x in records if x['unit'] == src)
        ax.axvline(r['map_bce'], color=color, lw=1.2, ls='--', alpha=0.7)
        ax.axvspan(r['ci68_lo'], r['ci68_hi'], alpha=0.10, color=color)
    ax.axvspan(760, 460, alpha=0.05, color='grey')
    ax.text(610, ax.get_ylim()[1] * 0.95, 'Prophetic\ntraining range',
            ha='center', va='top', fontsize=8, color='grey')
    ax.set_xlim(date_grid.max(), date_grid.min())
    ax.set_xlabel('Date (BCE)')
    ax.set_ylabel('Posterior density')
    ax.set_title('Documentary sources — MVN Bayesian date posteriors\n'
                 '(shading = 68 % CI; dashed = MAP)')
    ax.legend()

    # Bottom: individual Torah books
    ax2 = axes[1]
    book_colors = {
        'Genesis':     '#1b9e77', 'Exodus':     '#d95f02',
        'Leviticus':   '#7570b3', 'Numbers':    '#e7298a',
        'Deuteronomy': '#66a61e',
    }
    for book in TORAH_BOOKS:
        if book not in posteriors:
            continue
        post = posteriors[book]
        color = book_colors.get(book, 'black')
        ax2.plot(date_grid, post, color=color, lw=2, label=book)
        r = next(x for x in records if x['unit'] == book)
        ax2.axvline(r['map_bce'], color=color, lw=1.2, ls='--', alpha=0.7)
    ax2.axvspan(760, 460, alpha=0.05, color='grey')
    ax2.set_xlim(date_grid.max(), date_grid.min())
    ax2.set_xlabel('Date (BCE)')
    ax2.set_ylabel('Posterior density')
    ax2.set_title('Individual Torah books — date posteriors')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Posteriors plot saved: {output_path}')


def plot_archaism_heatmap(scores_dict, feature_names, output_path,
                          max_features=30):
    """
    Heatmap of LBH scores: rows = text units, columns = features.
    Blue = archaic; red = modern.
    """
    units  = list(scores_dict.keys())
    feats  = feature_names[:max_features]   # limit width

    matrix = np.full((len(units), len(feats)), np.nan)
    for i, u in enumerate(units):
        for j, f in enumerate(feats):
            matrix[i, j] = scores_dict[u].get(f, np.nan)

    fig, ax = plt.subplots(figsize=(min(2 + len(feats) * 0.35, 16),
                                    max(3, len(units) * 0.45)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r',
                   vmin=-0.3, vmax=1.3, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='LBH score (0=archaic, 1=modern)',
                 fraction=0.04, pad=0.04)

    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(
        [f.replace('lex::', '').replace('vt::', '').replace('vs::', '')
           .replace('morph::', '')
         for f in feats],
        rotation=70, ha='right', fontsize=7)
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels(units, fontsize=9)
    ax.set_title('Archaism audit — LBH scores per feature\n'
                 '(blue = archaic CBH pattern; red = modern LBH pattern)',
                 fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Archaism heatmap saved: {output_path}')


def plot_joint_dp(date_grid, posteriors, output_path):
    """2D joint posterior for D and P."""
    if 'D' not in posteriors or 'P' not in posteriors:
        return
    ll_d = np.log(posteriors['D'] + 1e-300)
    ll_p = np.log(posteriors['P'] + 1e-300)
    lj = ll_d[:, None] + ll_p[None, :]
    lj -= lj.max()
    joint = np.exp(lj)
    joint /= joint.sum()

    prob_d_older = joint[np.ix_(range(len(date_grid)), range(len(date_grid)))]
    mask = np.array([[date_grid[i] > date_grid[j]
                      for j in range(len(date_grid))]
                     for i in range(len(date_grid))])
    prob_d_earlier = float(prob_d_older[mask].sum())

    ext = [date_grid.min(), date_grid.max(), date_grid.min(), date_grid.max()]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(joint.T, extent=ext, origin='lower',
                   aspect='auto', cmap='Blues')
    plt.colorbar(im, ax=ax, label='Joint posterior')
    ax.plot(date_grid, date_grid, 'r--', lw=1, alpha=0.6, label='D = P')
    mi = np.unravel_index(joint.argmax(), joint.shape)
    ax.scatter([date_grid[mi[0]]], [date_grid[mi[1]]],
               marker='*', s=180, color='red', zorder=5, label='MAP')
    ax.set_xlabel('Date for D source (BCE)')
    ax.set_ylabel('Date for P source (BCE)')
    ax.invert_xaxis(); ax.invert_yaxis()
    ax.set_title(f'Joint posterior P(θ_D, θ_P | data)\n'
                 f'P(D earlier than P) = {prob_d_earlier:.2f}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Joint D–P posterior saved: {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-path',
        default=str(Path.home()/'text-fabric-data'/'github'/'ETCBC'/'bhsa'/'tf'/'2021'))
    parser.add_argument('--scan-full',   default='feature_scan_full.csv')
    parser.add_argument('--scan-robust', default='feature_scan_robust.csv')
    parser.add_argument('--rates-csv',   default='feature_rates_training.csv')
    parser.add_argument('--tier3-csv',   default='tier3_training_rates.csv',
        help='CSV with Tier-3 clause features from script 12 (default tier3_training_rates.csv)')
    parser.add_argument('--outdir',      default='.')
    parser.add_argument('--ridge',  type=float, default=0.10,
        help='Tikhonov ridge fraction (default 0.10)')
    parser.add_argument('--n-grid', type=int,   default=500,
        help='Date grid points (default 500)')
    args = parser.parse_args()

    outdir = Path(args.outdir)

    # ------------------------------------------------------------------
    # 1. Load feature scans
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 1 — Feature synthesis')
    print('='*70)

    full_df   = pd.read_csv(args.scan_full)
    robust_df = pd.read_csv(args.scan_robust)
    rates_df  = pd.read_csv(args.rates_csv, index_col='unit')
    dates_bce = rates_df['date_bce'].values.astype(float)
    rates_df  = rates_df.drop(columns=['date_bce', 'n_words'], errors='ignore')

    n_full    = len(full_df)
    n_p10     = (full_df['p_raw'] < 0.10).sum()
    # 'loo_robust' column is True/False or 0/1
    n_loo     = int(robust_df['loo_robust'].sum()) if 'loo_robust' in robust_df else len(robust_df)

    print(f'  Script 06 blind scan:   {n_full:>4} features tested (broad corpus, oracle Jeremiah)')
    print(f'  p < 0.10:               {n_p10:>4} features')
    print(f'  LOO-robust (≥ 75 %):    {n_loo:>4} features')

    # Theoretical features from scripts 08 / 10
    theory_features = {
        'frac_ani':      (0.042, True,  'increase', 'script 08/10 — Hurvitz 1972'),
        'frac_she':      (0.036, True,  'increase', 'script 08/10 — Polzin 1976'),
        'rate_wayyiqtol':(0.103, False, 'decrease', 'script 08 — Polzin 1976'),
        'rate_ut_nouns': (0.030, True,  'increase', 'script 10 — Hornkohl 2024'),
        'frac_anachnu':  (0.495, False, 'increase', 'script 10 — Hornkohl 2024'),
        'frac_niphal':   (0.526, False, 'increase', 'script 10'),
        'rate_pen':      (0.148, False, 'decrease', 'script 10 — Joüon §114'),
        'frac_halak_piel':(0.821,False, 'increase', 'script 10 — Hornkohl 2024'),
    }
    n_theory_sig = sum(1 for v in theory_features.values() if v[0] < 0.10)
    print(f'\n  Theoretical features (scripts 08/10):')
    print(f'    Tested:    {len(theory_features)}')
    print(f'    p < 0.10:  {n_theory_sig}  '
          f'(frac_אני, frac_ש, rate_ūt_nouns)')
    print(f'    Note: frac_אני and frac_ש are derived from lexemes already')
    print(f'    in the blind scan; rate_ūt_nouns is the sole novel addition.')

    # ------------------------------------------------------------------
    # 2. Apply linguistic filter + build working feature set
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 2 — Working feature set')
    print('='*70)

    print('  Applying linguistic POS filter to LOO-robust features...')
    ling_df = filter_linguistic(robust_df, verbose=True)

    # Build list of feature names for the model
    base_features = ling_df['feature'].tolist()

    # Add ratio features only if their components are available
    ratio_features_to_add = []
    if 'lex::>NJ' in rates_df.columns and 'lex::>NKJ' in rates_df.columns:
        ratio_features_to_add.append('frac_ani')
    if 'lex::C' in rates_df.columns and 'lex::>CR' in rates_df.columns:
        ratio_features_to_add.append('frac_she')
    ratio_features_to_add.append('rate_ut_nouns')

    # Remove duplicates (some ratio features might alias base features)
    working_features = base_features.copy()
    for rf in ratio_features_to_add:
        if rf not in working_features:
            working_features.append(rf)

    print(f'\n  Base (linguistic + LOO-robust) features: {len(base_features)}')
    print(f'  Ratio/morphosyntactic additions:          {len(ratio_features_to_add)}')

    # Augment training rates with ratio features
    rates_aug = augment_ratio_features(rates_df)

    # --- Load and merge Tier-3 training rates from script 12 ---
    tier3_path = Path(args.tier3_csv)
    tier3_added = []
    if tier3_path.exists():
        t3_df = pd.read_csv(tier3_path, index_col=0)
        # Align on training units; drop date/word-count meta columns
        meta_cols = {'date_bce', 'date_sigma', 'n_clauses', 'n_words',
                     '_n_words', '_n_clauses'}
        t3_feat_cols = [c for c in t3_df.columns
                        if c in TIER3_FEATURES and c not in meta_cols]
        # Only keep units present in both DataFrames
        common_units = rates_aug.index.intersection(t3_df.index)
        for col in t3_feat_cols:
            rates_aug.loc[common_units, col] = t3_df.loc[common_units, col]
            if col not in working_features:
                working_features.append(col)
                tier3_added.append(col)
        print(f'  Tier-3 training rates loaded from {tier3_path}')
        print(f'    Matched {len(common_units)}/{len(rates_aug)} training units')
        print(f'    Tier-3 features added to model: {tier3_added}')
    else:
        print(f'  WARNING: {tier3_path} not found — Tier-3 features will be '
              f'computed from BHSA for training corpus (slow).')
        # Mark as to-be-filled; extraction happens in Step 4 after BHSA load
        for tf in TIER3_FEATURES:
            if tf not in working_features:
                working_features.append(tf)
                tier3_added.append(tf)

    print(f'  Tier-3 features added:                    {len(tier3_added)}')
    print(f'  Total working feature set:                {len(working_features)}')

    # Check all working features are in the augmented rates
    missing = [f for f in working_features if f not in rates_aug.columns]
    if missing:
        print(f'  WARNING: {len(missing)} features not in training matrix '
              f'(will be imputed as NaN): {missing[:5]}')
        for m in missing:
            rates_aug[m] = np.nan

    # Identify feature sub-groups for the resistant model
    # "Archaism-resistant" = Tier-3 clause/phrase features (syntax is hard
    # to consciously imitate) — used later for the calibration comparison.
    resistant_features = [f for f in working_features if f in TIER3_FEATURES]

    # ------------------------------------------------------------------
    # 3. Build MVN models (full + resistant-only)
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 3 — Building MVN models')
    print('='*70)

    print('  Full model (all features):')
    ols_params, Sigma_reg, Sigma_inv = build_mvn_model(
        rates_aug[working_features], dates_bce, working_features,
        ridge_frac=args.ridge)

    ols_params_r = Sigma_inv_r = None
    if len(resistant_features) >= 2:
        print(f'\n  Resistant model (Tier-3 only — {len(resistant_features)} features):')
        # Fill NaN in training matrix with column medians for the resistant model
        r_train = rates_aug[resistant_features].copy()
        for col in resistant_features:
            med = r_train[col].median()
            r_train[col] = r_train[col].fillna(med)
        ols_params_r, _, Sigma_inv_r = build_mvn_model(
            r_train, dates_bce, resistant_features, ridge_frac=0.20)
    else:
        print('  Resistant model: insufficient Tier-3 features — skipping.')

    # ------------------------------------------------------------------
    # 4. Load BHSA + extract Torah features
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 4 — Extracting Torah features from BHSA')
    print('='*70)

    api = load_bhsa(args.data_path)
    F, L, T = api.F, api.L, api.T

    units = {}   # {label: (feature_vector, n_words)}

    # Helper: patch Tier-3 values into an already-extracted feature vector
    def patch_tier3(vec, t3_dict):
        vec = vec.copy()
        for tf in TIER3_FEATURES:
            if tf in working_features:
                idx = working_features.index(tf)
                vec[idx] = t3_dict.get(tf, np.nan)
        return vec

    # Individual Torah books
    print('\nTorah books:')
    for book in TORAH_BOOKS:
        bn = T.nodeFromSection((book,))
        ch_nums = [int(F.chapter.v(c)) for c in L.d(bn, 'chapter')] if bn else []
        if book in TORAH_BOOK_OVERRIDES:
            ch_ranges = TORAH_BOOK_OVERRIDES[book]   # song-excluded ranges
        else:
            ch_ranges = [(min(ch_nums), max(ch_nums))] if ch_nums else [(1, 999)]
        book_ch_pairs = [(book, ch_ranges)]

        vec, nw = extract_unit_features(book, book_ch_pairs, F, L, T, working_features)
        if vec is not None:
            t3 = extract_tier3_for_unit(book_ch_pairs, F, L, T)
            vec = patch_tier3(vec, t3)
        else:
            vec = np.full(len(working_features), np.nan)
        units[book] = (vec, nw)

        ani_val = vec[working_features.index('frac_ani')] \
            if 'frac_ani' in working_features else float('nan')
        infc_val = vec[working_features.index('frac_infc')] \
            if 'frac_infc' in working_features else float('nan')
        print(f'  {book:<12} {nw:>8,} words  '
              f'frac_אני={ani_val:.3f}  frac_infc={infc_val:.3f}')

    # Aggregated documentary sources
    print('\nDocumentary sources:')
    for src in ['D', 'P', 'JE']:
        # Build combined book_ch_pairs for Tier-3 extraction (pools all sections)
        src_pairs = AGGREGATED_SOURCES.get(src, [])
        flat_pairs = [(book, ch_ranges) for book, ch_ranges in src_pairs]

        vec, nw = extract_source(src, F, L, T, working_features)
        if nw > 0:
            t3 = extract_tier3_for_unit(flat_pairs, F, L, T)
            vec = patch_tier3(vec, t3)
        units[src] = (vec, nw)

        ani_val = vec[working_features.index('frac_ani')] \
            if 'frac_ani' in working_features else float('nan')
        infc_val = vec[working_features.index('frac_infc')] \
            if 'frac_infc' in working_features else float('nan')
        print(f'  {src:<6} {nw:>8,} words  '
              f'frac_אני={ani_val:.3f}  frac_infc={infc_val:.3f}')

    # ------------------------------------------------------------------
    # 5. Compute posteriors + CIs
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 5 — Date posteriors')
    print('='*70)

    posteriors  = {}   # full model posteriors
    posteriors_r = {}  # resistant-model posteriors
    records    = []

    wide_threshold = TRAINING_RANGE_YEARS * CI95_WIDTH_THRESHOLD

    header = (f'  {"Unit":<14} {"MAP(full)":>9}  {"MAP(resist)":>11}  '
              f'{"68% CI (full)":>22}  {"n_words":>8}')
    print(header)
    print('  ' + '-' * (len(header) - 2))

    dg = None
    for label, (x_obs, nw) in units.items():
        dg, post, map_d, ci68_lo, ci68_hi, ci95_lo, ci95_hi = compute_posterior(
            x_obs, nw, ols_params, working_features, Sigma_inv,
            n_grid=args.n_grid)
        posteriors[label] = post

        w95 = abs(ci95_lo - ci95_hi)
        ci95_str = (f'{ci95_lo:.0f}–{ci95_hi:.0f}'
                    if w95 < wide_threshold
                    else '(too wide)')

        # Resistant model
        map_r_val = None
        map_r_str = '     —'
        if ols_params_r is not None and Sigma_inv_r is not None:
            x_r = np.array([x_obs[working_features.index(f)]
                             if f in working_features else np.nan
                             for f in resistant_features])
            _, post_r, map_r_val, *_ = compute_posterior(
                x_r, nw, ols_params_r, resistant_features, Sigma_inv_r,
                n_grid=args.n_grid)
            posteriors_r[label] = post_r
            map_r_str = f'{map_r_val:>6.0f}'

        print(f'  {label:<14} {map_d:>9.0f}  {map_r_str:>11}  '
              f'{ci68_lo:>5.0f}–{ci68_hi:.0f} BCE          '
              f'{nw:>8,}')

        records.append({
            'unit':      label,
            'map_bce':   round(map_d, 0),
            'map_bce_resistant': round(map_r_val, 0)
                         if map_r_val is not None else None,
            'ci68_lo':   round(ci68_lo, 0),
            'ci68_hi':   round(ci68_hi, 0),
            'ci95_lo':   round(ci95_lo, 0) if w95 < wide_threshold else None,
            'ci95_hi':   round(ci95_hi, 0) if w95 < wide_threshold else None,
            'ci95_wide': w95 >= wide_threshold,
            'n_words':   nw,
        })

    if dg is not None:
        date_grid = dg

    # ------------------------------------------------------------------
    # 6. Scholarly comparison
    # ------------------------------------------------------------------
    print('\n  Scholarly reference dates for comparison:')
    for src, (early, late, label) in SCHOLARLY_DATES.items():
        r = next((x for x in records if x['unit'] == src), None)
        our_map = r['map_bce'] if r else '?'
        print(f'    {src}:  {early}–{late} BCE  ({label}) '
              f'→ our MAP: {our_map:.0f} BCE')

    # ------------------------------------------------------------------
    # 7. Archaism audit
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 6 — Archaism audit')
    print('='*70)

    x_obs_dict = {label: vec for label, (vec, _) in units.items()}
    arc_scores = compute_archaism_scores(
        x_obs_dict, ols_params, working_features)

    # Summary per unit: mean LBH score across all features
    print(f'\n  {"Unit":<14} {"Mean LBH":>9}  {"Min":>7}  {"Max":>7}  '
          f'{"n_valid":>8}  {"Classification"}')
    print('  ' + '-'*75)
    archaism_records = []
    for label in list(TORAH_BOOKS) + ['D', 'P', 'JE']:
        sc = arc_scores.get(label, {})
        vals = [v for v in sc.values() if np.isfinite(v)]
        if not vals:
            continue
        mean_lbh = np.mean(vals)
        cls = ('Archaic (CBH-like)' if mean_lbh < 0.35 else
               'Mixed/selective'    if mean_lbh < 0.65 else
               'Modern (LBH-like)')
        print(f'  {label:<14} {mean_lbh:>9.3f}  '
              f'{min(vals):>7.3f}  {max(vals):>7.3f}  '
              f'{len(vals):>8}  {cls}')
        archaism_records.append({'unit': label, 'mean_lbh': mean_lbh,
                                  'min_lbh': min(vals), 'max_lbh': max(vals),
                                  'n_valid_features': len(vals),
                                  'classification': cls})

    # Per-feature breakdown for sources
    print('\n  Feature-level archaism scores for D, P, JE:')
    feat_header = f'  {"Feature":<30}  {"D":>7}  {"P":>7}  {"JE":>7}  expected'
    print(feat_header)
    print('  ' + '-'*60)
    feat_arc_records = []
    for fn in working_features:
        scores_row = {src: arc_scores.get(src, {}).get(fn, np.nan)
                      for src in ['D', 'P', 'JE']}
        # Skip features where all sources are nan
        if all(np.isnan(v) for v in scores_row.values()):
            continue
        # Get expected direction from scan df
        row = ling_df[ling_df['feature'] == fn]
        if not row.empty:
            rho = row.iloc[0]['rho']
            exp_dir = '↑ LBH' if rho > 0 else '↓ LBH'
        elif fn in RATIO_FEATURES_META:
            exp_dir = '↑ LBH' if RATIO_FEATURES_META[fn][0] == 'increase' else '↓ LBH'
        else:
            exp_dir = '?'

        fn_short = (fn.replace('lex::', '').replace('vt::', '')
                      .replace('vs::', '').replace('morph::', ''))[:28]
        d_val  = f'{scores_row["D"]:>7.3f}'  if np.isfinite(scores_row['D'])  else '    nan'
        p_val  = f'{scores_row["P"]:>7.3f}'  if np.isfinite(scores_row['P'])  else '    nan'
        je_val = f'{scores_row["JE"]:>7.3f}' if np.isfinite(scores_row['JE']) else '    nan'
        print(f'  {fn_short:<30}  {d_val}  {p_val}  {je_val}  {exp_dir}')
        feat_arc_records.append({'feature': fn, **scores_row, 'expected': exp_dir})

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 7 — Saving outputs')
    print('='*70)

    pd.DataFrame(records).to_csv(outdir / 'dating_summary.csv', index=False)
    print('Dating summary saved: dating_summary.csv')

    pd.DataFrame(archaism_records).to_csv(outdir / 'archaism_summary.csv', index=False)
    print('Archaism summary saved: archaism_summary.csv')

    pd.DataFrame(feat_arc_records).to_csv(outdir / 'archaism_by_feature.csv', index=False)
    print('Per-feature archaism saved: archaism_by_feature.csv')

    # Feature summary (all tested + which survived each filter)
    feature_summary_rows = []
    for _, row in full_df.iterrows():
        fn = row['feature']
        feature_summary_rows.append({
            'feature':      fn,
            'rho':          row['rho'],
            'p_raw':        row['p_raw'],
            'p_lt_0.10':    row['p_raw'] < 0.10,
            'loo_robust':   row.get('loo_robust', False),
            'ling_filter':  fn in base_features,
            'in_model':     fn in working_features,
            'description':  row.get('description', ''),
        })
    # Add theoretical-only features
    for tf_name, (p, loo, direct, ref) in theory_features.items():
        if not any(r['feature'] == tf_name for r in feature_summary_rows):
            feature_summary_rows.append({
                'feature': tf_name, 'rho': np.nan, 'p_raw': p,
                'p_lt_0.10': p < 0.10, 'loo_robust': loo,
                'ling_filter': True,
                'in_model': tf_name in working_features,
                'description': ref,
            })
    pd.DataFrame(feature_summary_rows).to_csv(
        outdir / 'feature_summary_all.csv', index=False)
    print('Feature summary saved: feature_summary_all.csv')

    # ------------------------------------------------------------------
    # 8b. Archaism-calibration discussion
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 6b — Archaism-calibration summary')
    print('='*70)

    if posteriors_r:
        print('\n  Full model vs. resistant (Tier-3 syntax-only) model:')
        print(f'  {"Source":<8} {"Full MAP":>9}  {"Resist MAP":>10}  '
              f'{"Δ (full−resist)":>16}  Interpretation')
        print('  ' + '-' * 75)
        for src in ['D', 'P', 'JE']:
            r = next((x for x in records if x['unit'] == src), None)
            if r and src in posteriors_r:
                map_f  = r['map_bce']
                map_rr = r.get('map_bce_resistant')
                if map_rr is not None and np.isfinite(float(map_rr)):
                    delta = map_f - map_rr
                    interp = ('Full older: archaism inflates date'
                              if delta > 30 else
                              'Full younger: syntax earlier than lexicon'
                              if delta < -30 else
                              'Models broadly agree')
                    print(f'  {src:<8} {map_f:>9.0f}  {map_rr:>10.0f}  '
                          f'{delta:>+16.0f}  {interp}')

        print("""
  Interpretation guide:
    • If Full MAP >> Resistant MAP: the full model's lexical/pronoun features
      are dragging the date toward the past — classic archaism signal.
      The resistant (syntax-only) date may be a closer lower bound.
    • If Full MAP ≈ Resistant MAP: both feature classes agree; archaism is
      not a major confound for this source.
    • If Full MAP < Resistant MAP: the syntax features are actually more
      archaic than the lexicon (less common, but possible in liturgical prose).
    • The true composition date is likely between Full and Resistant MAPs,
      biased toward Resistant when Mean LBH (archaism score) is high.""")
    else:
        print('  (No resistant model available — Tier-3 features not loaded.)')

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 8 — Plots')
    print('='*70)

    plot_posteriors(date_grid, posteriors, records,
                    str(outdir / 'dating_posteriors.png'))
    plot_joint_dp(date_grid, posteriors,
                  str(outdir / 'dating_joint_dp.png'))

    # Heatmap: limit to features that vary meaningfully across units
    heatmap_features = [
        fn for fn in working_features
        if any(np.isfinite(arc_scores.get(u, {}).get(fn, np.nan))
               for u in arc_scores)
    ][:40]
    heatmap_units = list(TORAH_BOOKS) + ['D', 'P', 'JE']
    heatmap_scores = {u: arc_scores[u] for u in heatmap_units if u in arc_scores}
    plot_archaism_heatmap(heatmap_scores, heatmap_features,
                          str(outdir / 'archaism_heatmap.png'))

    if posteriors_r:
        plot_model_comparison(date_grid, posteriors, posteriors_r, records,
                              str(outdir / 'dating_model_comparison.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
