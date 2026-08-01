#!/usr/bin/env python3
"""
06_je_subsource_dating.py  —  Morpho dates for JE sub-units (Gen, Exo, Num)
============================================================================
Extends Script 13's extraction machinery to compute full-model morpho MAP
dates for the three JE sub-units that Script 13 did not cover:

  Gen_JE  — Genesis chapters dominated by J/E/JE material
  Exo_JE  — Exodus chapters dominated by J/E/JE material
  Num_JE  — Numbers chapters dominated by J/E/JE material

Chapter assignments follow Script 19's SOURCE_LABELS consensus map.
Chapters labeled J, E, JE, or JE+P are included; pure P, Poem, and ?
chapters are excluded.  This mirrors the approach taken for word n-gram
dating in Script 05.

NOTE — requires BHSA / text-fabric
------------------------------------
This script must be run on a machine where:
  1. text-fabric is installed:  pip install text-fabric
  2. BHSA data is available at the path passed with --data-path.
     Default (matches Script 13):
       ~/text-fabric-data/github/ETCBC/bhsa/tf/2021

The sandbox (Cowork) cannot download BHSA data due to network restrictions.
Run this script locally and copy the output CSV into the project folder.

Usage
-----
  # Using the default path (same as Script 13):
  python3 06_je_subsource_dating.py

  # Or specify a path explicitly:
  python3 06_je_subsource_dating.py --data-path ~/text-fabric-data/github/ETCBC/bhsa/tf/2021

Output
------
  je_subsource_dating.csv   — morpho MAP + CI + archaism for Gen_JE, Exo_JE, Num_JE
  (written to the directory where this script lives)

Script 05 (05_subsource_dating.py) reads this file automatically to fill in
the JE sub-unit morpho columns when it is present.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# JE chapter assignments  (from Script 19 SOURCE_LABELS consensus map)
# Include: J, E, JE, JE+P  (same set as Script 05's word n-gram JE_LABELS)
# Exclude: P, H, ?, Poem, D-code, D-frame
# ---------------------------------------------------------------------------
JE_CHAPTERS = {
    'Gen_JE': [
        ('Genesis', [
            # ch 2-4  (J)
            (2, 4),
            # ch 6-13 (JE+P, J)  — includes mixed but JE-dominated
            (6, 13),
            # ch 15-16 (JE, JE+P)
            (15, 16),
            # ch 18-22 (J, J, E, JE+P, E)
            (18, 22),
            # ch 24-35 (J, JE+P, J, J, JE+P, JE-35 mixed)
            (24, 35),
            # ch 37-48 (JE+P, J, J, JE, JE, JE, J, J, J, JE+P, JE, JE)
            (37, 48),
            # ch 50 (JE+P)
            (50, 50),
        ]),
    ],
    'Exo_JE': [
        ('Exodus', [
            # ch 1-5  (JE+P, JE, JE, JE, JE)
            (1, 5),
            # ch 7-14 (JE+P, JE+P, JE+P, JE, JE, JE+P, JE, JE+P)
            (7, 14),
            # ch 17-24 (JE, JE, JE+P, JE, JE, JE, JE, JE+P)
            (17, 24),
            # ch 32-34 (JE, JE, JE) — post-golden-calf narrative
            (32, 34),
        ]),
    ],
    'Num_JE': [
        ('Numbers', [
            # ch 11-14 (JE, JE, JE+P, JE+P)
            (11, 14),
            # ch 16    (JE+P — Korah, substantial JE)
            (16, 16),
            # ch 20-25 (JE+P, JE, JE, JE, JE, JE+P)
            (20, 25),
        ]),
    ],
}

# Tier-3 clause-level features
TIER3_FEATURES = ['frac_infc', 'frac_fronted', 'frac_null_subj', 'frac_wqtl_wayq']

# Clause-type constants (mirror of Script 13 / Script 12)
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

LINGUISTIC_POS = {
    'pronoun', 'dem-pronoun', 'prep', 'conj',
    'adverb', 'negator', 'article', 'interrogative', 'interjection',
}

# ---------------------------------------------------------------------------
# BHSA loading
# ---------------------------------------------------------------------------

def load_bhsa(data_path):
    try:
        from tf.fabric import Fabric
    except ImportError:
        sys.exit('text-fabric not installed.  Run: pip install text-fabric')

    data_path = Path(data_path)
    features  = 'otype lex sp vs vt ps nu gn st prs_ps chapter typ function domain'

    # Try multiple (location, modules) layouts — BHSA path conventions vary
    # by text-fabric version and how the corpus was downloaded.
    #   Layout A: features directly in data_path/          (modules=[''])
    #   Layout B: features in data_path/c/                 (modules=['c'])
    #   Layout C: parent dir + version subdir              (parent, ['c'])
    #   Layout D: use the high-level app loader (auto-resolves)
    attempts = [
        (str(data_path),        ['']),
        (str(data_path),        ['c']),
        (str(data_path.parent), ['c']),
        (str(data_path.parent), [data_path.name]),
    ]

    for loc, mods in attempts:
        print(f'  Trying Fabric(locations={loc!r}, modules={mods}) ...')
        TF  = Fabric(locations=loc, modules=mods, silent=True)
        api = TF.load(features, silent=True)
        if api and api is not False:
            print(f'  BHSA loaded OK  (location={loc!r}, modules={mods})')
            return api

    # --- diagnostic: walk the tree to find where otype.tf actually lives ---
    print('\nDiagnostic: searching for otype.tf under the text-fabric-data tree ...')
    import subprocess
    base_search = Path.home() / 'text-fabric-data'
    result = subprocess.run(
        ['find', str(base_search), '-name', 'otype.tf', '-maxdepth', '10'],
        capture_output=True, text=True)
    hits = result.stdout.strip().splitlines()
    if hits:
        print('  Found otype.tf at:')
        for h in hits:
            print(f'    {h}')
        print('\n  Re-run with the PARENT directory of otype.tf as --data-path')
        print(f'  e.g.  --data-path {Path(hits[0]).parent}')
    else:
        print('  No otype.tf found under ~/text-fabric-data.')
        print('  BHSA data may not be downloaded.  Run Script 13 first to')
        print('  trigger text-fabric auto-download, then re-run this script.')

    sys.exit(1)


# ---------------------------------------------------------------------------
# Feature extraction  (verbatim from Script 13)
# ---------------------------------------------------------------------------

def words_for_ranges(book_name, ch_ranges, F, L, T):
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


def extract_unit_features(book_ch_pairs, feature_names, F, L, T):
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
        if feat in TIER3_FEATURES:
            result[feat] = np.nan
            continue
        if feat.startswith('lex::'):
            result[feat] = lex_ctr.get(feat[5:], 0) / n * 1000
        elif feat.startswith('vt::'):
            result[feat] = vt_ctr.get(feat[4:], 0) / n * 1000
        elif feat.startswith('vs::'):
            result[feat] = vs_ctr.get(feat[4:], 0) / n * 1000
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
            a, b = lex_ctr.get('>NJ', 0), lex_ctr.get('>NKJ', 0)
            result[feat] = a / (a + b) if (a + b) > 0 else np.nan
        elif feat == 'frac_she':
            a, b = lex_ctr.get('C', 0), lex_ctr.get('>CR', 0)
            result[feat] = a / (a + b) if (a + b) > 0 else np.nan
        elif feat == 'rate_ut_nouns':
            ut = sum(v for k, v in lex_ctr.items()
                     if k.endswith('WT/') or k.endswith('WT'))
            result[feat] = ut / n * 1000
        else:
            result[feat] = np.nan

    return np.array([result.get(f, np.nan) for f in feature_names]), n


def extract_tier3_for_unit(book_ch_pairs, F, L, T):
    n_words = n_clauses = n_verbal = 0
    n_infc = n_fronted = 0
    n_wqtl = n_wnarr = 0
    n_null_subj = n_sv = n_sv_total = n_ov = n_ov_total = 0

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
                n_words  += sum(1 for _ in L.d(cl, 'word'))
                n_clauses += 1
                if typ in INFC_TYPES:   n_infc    += 1
                if typ in FRONT_TYPES:  n_fronted += 1
                if typ in WAYQ_TYPES:   pass
                if typ in WQTL_TYPES:   n_wqtl    += 1
                if typ in WNARR_TYPES:  n_wnarr   += 1
                if typ in VERBAL_TYPES:
                    n_verbal += 1
                    phrases  = list(L.d(cl, 'phrase'))
                    ph_funcs = {F.function.v(ph): ph for ph in phrases}
                    if 'Subj' not in ph_funcs:
                        n_null_subj += 1
                    if 'Subj' in ph_funcs and 'Pred' in ph_funcs:
                        n_sv_total += 1
                        if ph_funcs['Subj'] < ph_funcs['Pred']:
                            n_sv += 1
                    if 'Objc' in ph_funcs and 'Pred' in ph_funcs:
                        n_ov_total += 1
                        if ph_funcs['Objc'] < ph_funcs['Pred']:
                            n_ov += 1

    if n_clauses == 0:
        return {f: np.nan for f in TIER3_FEATURES}
    return {
        'frac_infc':      n_infc    / n_clauses,
        'frac_fronted':   n_fronted / n_clauses,
        'frac_null_subj': n_null_subj / n_verbal if n_verbal > 10 else np.nan,
        'frac_wqtl_wayq': n_wqtl / n_wnarr       if n_wnarr  > 5  else np.nan,
    }


def extract_multi_range(label, book_ch_pairs, feature_names, F, L, T):
    vec, nw = extract_unit_features(book_ch_pairs, feature_names, F, L, T)
    if vec is None:
        return np.full(len(feature_names), np.nan), 0
    t3 = extract_tier3_for_unit(book_ch_pairs, F, L, T)
    for tf in TIER3_FEATURES:
        if tf in feature_names:
            vec[feature_names.index(tf)] = t3.get(tf, np.nan)
    return vec, nw


# ---------------------------------------------------------------------------
# MVN model  (verbatim from Script 13)
# ---------------------------------------------------------------------------

def filter_linguistic(robust_df):
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
    return robust_df[keep].reset_index(drop=True)


def build_mvn_model(rates_df, dates_bce, feature_names, ridge_frac=0.10):
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
        b, a = np.polyfit(xv, yv, 1)
        pred = a + b * x
        resid = y - pred
        resid[~valid] = 0.0
        ols_params[fn] = (a, b, np.std(yv - (a + b * xv)) + 1e-9)
        residual_rows.append(resid)

    R = np.array(residual_rows).T
    Sigma = np.cov(R.T)
    if Sigma.ndim == 0:
        Sigma = np.array([[float(Sigma)]])
    K = Sigma.shape[0]
    ridge = ridge_frac * np.trace(Sigma) / K
    Sigma_reg = Sigma + ridge * np.eye(K)
    Sigma_inv = np.linalg.inv(Sigma_reg)
    cond = np.linalg.cond(Sigma_reg)
    print(f'    {K} features, condition number {cond:.0f} (ridge={ridge_frac:.2f})')
    return ols_params, Sigma_reg, Sigma_inv


def mvn_log_likelihood(x_obs, date_bce, ols_params, feature_names, Sigma_inv, n_words=1):
    mu    = np.array([ols_params[fn][0] + ols_params[fn][1] * date_bce
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
                      prior_mean=800, prior_sd=400):
    """Wide flat prior for JE — 800±400 BCE covers 9th–4th c. without strong pull."""
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

    return (date_grid, post, map_date,
            quantile(0.16), quantile(0.84),
            quantile(0.025), quantile(0.975))


def compute_archaism_scores(x_obs, ols_params, feature_names,
                             archaic_date=720, modern_date=250):
    scores = {}
    for i, fn in enumerate(feature_names):
        a, b, _ = ols_params[fn]
        cbh_val = a + b * archaic_date
        lbh_val = a + b * modern_date
        span = lbh_val - cbh_val
        if abs(span) < 1e-9 or not np.isfinite(x_obs[i]):
            scores[fn] = np.nan
            continue
        scores[fn] = (x_obs[i] - cbh_val) / span
    finite_vals = [v for v in scores.values() if np.isfinite(v)]
    mean_lbh = float(np.mean(finite_vals)) if finite_vals else np.nan
    return scores, mean_lbh


def arch_label(mean_lbh):
    if np.isnan(mean_lbh):
        return 'Unknown'
    if mean_lbh < 0.35:
        return 'Archaic (CBH-like)'
    if mean_lbh < 0.55:
        return 'Mixed/selective'
    return 'Modern (LBH-like)'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--data-path',
        default=str(Path.home()/'text-fabric-data'/'github'/'ETCBC'/'bhsa'/'tf'/'2021'),
        help='Path to BHSA TF data directory (default: ~/text-fabric-data/github/ETCBC/bhsa/tf/2021)')
    parser.add_argument('--scan-robust', default='feature_scan_robust.csv')
    parser.add_argument('--rates-csv',   default='feature_rates_training.csv')
    parser.add_argument('--tier3-csv',   default='tier3_training_rates.csv')
    parser.add_argument('--outdir',      default='.')
    parser.add_argument('--ridge',  type=float, default=0.10)
    parser.add_argument('--n-grid', type=int,   default=500)
    args = parser.parse_args()

    workspace = Path(__file__).parent
    outdir    = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # ── STEP 1 — Reconstruct working feature set ──────────────────────────────
    print('=' * 70)
    print('STEP 1 — Reconstruct working feature set (same as Script 13)')
    print('=' * 70)

    robust_df = pd.read_csv(workspace / args.scan_robust)
    rates_df  = pd.read_csv(workspace / args.rates_csv, index_col='unit')
    dates_bce = rates_df['date_bce'].values.astype(float)
    rates_df  = rates_df.drop(columns=['date_bce', 'n_words'], errors='ignore')

    ling_df = filter_linguistic(robust_df)
    base_features    = ling_df['feature'].tolist()
    working_features = base_features.copy()
    for rf in ['frac_ani', 'frac_she', 'rate_ut_nouns']:
        if rf not in working_features:
            working_features.append(rf)

    # Augment training rates
    df = rates_df.copy()
    ani  = df.get('lex::>NJ',  pd.Series(0.0, index=df.index))
    ank  = df.get('lex::>NKJ', pd.Series(0.0, index=df.index))
    she  = df.get('lex::C',    pd.Series(0.0, index=df.index))
    ash  = df.get('lex::>CR',  pd.Series(0.0, index=df.index))
    df['frac_ani'] = np.where((ani + ank) > 0, ani / (ani + ank), np.nan)
    df['frac_she'] = np.where((she + ash) > 0, she / (she + ash), np.nan)
    ut_cols = [c for c in df.columns if re.match(r'lex::\w+WT/', c)]
    df['rate_ut_nouns'] = df[ut_cols].sum(axis=1) if ut_cols else 0.0
    rates_aug = df

    tier3_path = workspace / args.tier3_csv
    tier3_added = []
    if tier3_path.exists():
        t3_df = pd.read_csv(tier3_path, index_col=0)
        common = rates_aug.index.intersection(t3_df.index)
        for col in TIER3_FEATURES:
            if col in t3_df.columns:
                rates_aug.loc[common, col] = t3_df.loc[common, col]
                if col not in working_features:
                    working_features.append(col)
                    tier3_added.append(col)
        print(f'  Tier-3 features loaded: {tier3_added}')
    else:
        for tf in TIER3_FEATURES:
            if tf not in working_features:
                working_features.append(tf)
        print('  tier3_training_rates.csv not found — will extract live from BHSA.')

    for f in working_features:
        if f not in rates_aug.columns:
            rates_aug[f] = np.nan

    print(f'  Working features: {len(working_features)}')

    # ── STEP 2 — Build MVN model ──────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 2 — Build MVN model')
    print('=' * 70)

    ols_params, _, Sigma_inv = build_mvn_model(
        rates_aug[working_features], dates_bce, working_features,
        ridge_frac=args.ridge)

    # ── STEP 3 — Extract JE sub-unit features from BHSA ──────────────────────
    print('\n' + '=' * 70)
    print('STEP 3 — Extract JE sub-unit features from BHSA')
    print('=' * 70)

    api  = load_bhsa(args.data_path)
    F, L, T = api.F, api.L, api.T

    units = {}
    for label, book_ch_pairs in JE_CHAPTERS.items():
        vec, nw = extract_multi_range(label, book_ch_pairs, working_features, F, L, T)
        units[label] = (vec, nw)
        ani_v  = vec[working_features.index('frac_ani')]  if 'frac_ani'  in working_features else np.nan
        infc_v = vec[working_features.index('frac_infc')] if 'frac_infc' in working_features else np.nan
        print(f'  {label:<14} {nw:>8,} words  frac_אני={ani_v:.3f}  frac_infc={infc_v:.3f}')

    # ── STEP 4 — Date posteriors ──────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('STEP 4 — Date posteriors')
    print('=' * 70)

    records = []
    header = (f'  {"Unit":<14} {"MAP(full)":>10}  {"68% CI":>18}  {"n_words":>8}  {"mean_lbh":>9}  Archaism')
    print(header)
    print('  ' + '-' * (len(header) - 2))

    for label in JE_CHAPTERS:
        x_obs, nw = units[label]
        _, _, map_d, ci68_lo, ci68_hi, ci95_lo, ci95_hi = compute_posterior(
            x_obs, nw, ols_params, working_features, Sigma_inv, n_grid=args.n_grid)

        arc_scores, mean_lbh = compute_archaism_scores(
            x_obs, ols_params, working_features)

        print(f'  {label:<14} {map_d:>10.0f}  {ci68_lo:>5.0f}–{ci68_hi:.0f} BCE  '
              f'{nw:>8,}  {mean_lbh:>9.3f}  {arch_label(mean_lbh)}')

        # Summarise which chapters were included
        ch_summary = {}
        for book, ch_ranges in JE_CHAPTERS[label]:
            ch_summary[book] = ch_ranges
        records.append({
            'unit':              label,
            'map_bce':           round(map_d, 1),
            'ci68_lo':           round(ci68_lo, 1),
            'ci68_hi':           round(ci68_hi, 1),
            'ci95_lo':           round(ci95_lo, 1),
            'ci95_hi':           round(ci95_hi, 1),
            'n_words':           nw,
            'mean_lbh':          round(mean_lbh, 4) if np.isfinite(mean_lbh) else None,
            'arch_classification': arch_label(mean_lbh),
            'je_chapters_included': str(ch_summary),
        })

    # ── STEP 5 — Save ─────────────────────────────────────────────────────────
    out_csv = outdir / 'je_subsource_dating.csv'
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f'\nSaved → {out_csv}')
    print('Copy this file to the project root folder, then re-run')
    print('05_subsource_dating.py to incorporate the morpho dates.')


if __name__ == '__main__':
    main()
