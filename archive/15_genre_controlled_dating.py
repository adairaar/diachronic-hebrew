#!/usr/bin/env python3
"""
Script 15: Genre-Controlled Dating
====================================
Implements two complementary strategies to control for genre effects when
applying a prophetic-text-trained model to legal, narrative, and poetic texts.

Strategy B — Feature-level genre discounting (Torah-internal calibration)
  The Torah books are approximately time-homogeneous (all dated to ~300–465 BCE
  by the full model, with overlapping 68 % CIs).  Any feature variation seen
  *across genres within the Torah* therefore reflects genre effects, not
  diachronic ones.

  For each feature f:
      genre_variance_f  = between-group variance (legal vs. narrative),
                          word-count-weighted, across Torah calibration units
      temporal_variance_f = variance of f across the 22 prophetic training units
      genre_ratio_f     = genre_variance_f / temporal_variance_f

  Strategy-B weight: w_f = 1 / (1 + genre_ratio_f)
    → 0 genre ratio → w = 1.0 (feature trusted fully)
    → genre_ratio = 1 → w = 0.5 (half weight)
    → genre_ratio >> 1 → w → 0 (feature nearly excluded)

  The weighted likelihood: log p ∝ -½ · scale · diffᵀ (W Σ⁻¹ W) diff
  where W = diag(w_f).

Strategy D — Per-unit σ inflation (genre-mismatch uncertainty)
  When a test unit's genre differs from the prophetic training distribution,
  inflate each feature's measurement uncertainty in proportion to its genre
  sensitivity and the degree of genre mismatch.

  Extra diagonal variance: Σ_extra[f,f] = Σ[f,f] × mismatch(g) × genre_ratio_f
  Adjusted covariance: Σ_D = Σ + diag(Σ_extra)

  Genre mismatch scores: legal=1.0, narrative=0.5, prophetic/poetry=0.0

Four models compared for every test unit:
  Raw   — standard MVN (Σ⁻¹, all weights = 1)
  B     — Strategy B only (weighted Σ⁻¹, original Σ)
  D     — Strategy D only (original weights, inflated Σ_D⁻¹)
  B+D   — both strategies combined

Usage
-----
    python 15_genre_controlled_dating.py [--data-path PATH] [--outdir DIR]
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

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Torah-internal genre calibration groups
# ---------------------------------------------------------------------------
# All drawn from the Torah, which our model places in a narrow ~300–465 BCE
# window.  Genre variation within this window = genre effect, not date effect.

CALIB_LEGAL = {
    'D_Code':       [('Deuteronomy', [(12, 26)])],
    'Lev_Holiness': [('Leviticus',   [(17, 26)])],
    'Lev_Priestly': [('Leviticus',   [(1,  16)])],
}

# JE source chapter-range attributions (Friedman 2003 / Baden 2012)
_JE_PAIRS = {
    'JE_Genesis': [('Genesis',  [(2,4),(6,6),(8,8),(10,10),(12,16),(18,22),
                                  (24,24),(26,27),(29,34),(37,45),(47,49)])],
    'JE_Exodus':  [('Exodus',   [(2,5),(8,11),(13,14),(17,18),(19,24),(32,34)])],  # ch.15 excluded
    'JE_Numbers': [('Numbers',  [(11,14),(16,16),(20,24),(25,25),(32,32)])],
}
CALIB_NARRATIVE = {
    'D_Frame':    [('Deuteronomy', [(1, 11), (27, 31), (33, 34)])],
    **_JE_PAIRS,
}

# ---------------------------------------------------------------------------
# Test units with their genres
# genre → mismatch score: legal=1.0, narrative=0.5, other=0.0
# ---------------------------------------------------------------------------
TEST_UNITS = {
    # Torah books
    'Genesis':      ('narrative', [('Genesis',     [(1, 50)])]),
    'Exodus':       ('narrative', [('Exodus',      [(1, 14), (16, 40)])]),  # ch.15 excluded (Song of Sea)
    'Leviticus':    ('legal',     [('Leviticus',   [(1, 27)])]),
    'Numbers':      ('narrative', [('Numbers',     [(1, 36)])]),
    'Deuteronomy':  ('legal',     [('Deuteronomy', [(1, 34)])]),
    # Sub-divisions
    'D_Code':       ('legal',     [('Deuteronomy', [(12, 26)])]),
    'D_Frame':      ('narrative', [('Deuteronomy', [(1,11),(27,31),(33,34)])]),
    'D_Song':       ('other',     [('Deuteronomy', [(32, 32)])]),
    'Lev_Holiness': ('legal',     [('Leviticus',   [(17, 26)])]),
    'Lev_Priestly': ('legal',     [('Leviticus',   [(1,  16)])]),
    # Documentary sources
    'D_source':     ('legal',     [('Deuteronomy', [(1, 34)])]),
    'P_source':     ('legal',     [('Genesis',     [(1,2),(5,5),(6,6),(7,7),(9,9),(11,11),
                                                     (17,17),(23,23),(25,25),(27,28),(35,36),
                                                     (46,46),(49,50)]),
                                   ('Exodus',      [(1,2),(6,7),(12,12),(16,16),(25,31),(35,40)]),
                                   ('Leviticus',   [(1, 27)]),
                                   ('Numbers',     [(1,10),(15,15),(17,19),(25,25),(27,31),(33,36)])]),
    'JE_source':    ('narrative', [('Genesis',     [(2,4),(6,6),(8,8),(10,10),(12,16),(18,22),
                                                     (24,24),(26,27),(29,34),(37,45),(47,49)]),
                                   ('Exodus',      [(2,5),(8,11),(13,14),(17,18),(19,24),(32,34)]),  # ch.15 excluded
                                   ('Numbers',     [(11,14),(16,16),(20,24),(25,25),(32,32)])]),
    # Dtr History
    'Joshua':       ('narrative', [('Joshua',    [(1, 24)])]),
    'Judges':       ('narrative', [('Judges',    [(1, 4), (6, 21)])]),  # ch.5 excluded (Song of Deborah)
    '1_Samuel':     ('narrative', [('1_Samuel',  [(1, 31)])]),
    '2_Samuel':     ('narrative', [('2_Samuel',  [(1, 24)])]),
    '1_Kings':      ('narrative', [('1_Kings',   [(1, 22)])]),
    '2_Kings':      ('narrative', [('2_Kings',   [(1, 25)])]),
    # Jeremiah halves
    'Jer_DTR':      ('narrative', [('Jeremiah',  [(7,7),(11,11),(17,18),(21,21),
                                                   (24,29),(32,45),(52,52)])]),
    'Jer_oracle':   ('other',     [('Jeremiah',  [(1,6),(8,10),(12,16),(19,20),
                                                   (22,23),(30,31),(46,51)])]),
    # Ancient songs (small — noisy)
    'Song_Sea':     ('other',     [('Exodus',    [(15, 15)])]),
    'Song_Deborah': ('other',     [('Judges',    [(5,  5)])]),
}

GENRE_MISMATCH = {'legal': 1.0, 'narrative': 0.5, 'other': 0.0}

# Tier-3 features
TIER3_FEATURES = ['frac_infc', 'frac_fronted', 'frac_null_subj', 'frac_wqtl_wayq']

LINGUISTIC_POS = {
    'pronoun','dem-pronoun','prep','conj',
    'adverb','negator','article','interrogative','interjection',
}

# Clause-type constants
WAYQ_TYPES   = {'Way0','WayX','WaYX'}
WQTL_TYPES   = {'WQt0','WQtX','WxQ0','WxQX'}
WNARR_TYPES  = WAYQ_TYPES | WQTL_TYPES
FRONT_TYPES  = {'xQt0','xYq0','xQtX','xYqX','xIm0',
                'WxY0','WxQ0','WxQX','WxYX','WxI0','XQtl','XYqt'}
CPEN_TYPES   = {'CPen'}
NMCL_TYPES   = {'NmCl','AjCl'}
PTCP_TYPES   = {'Ptcp'}
INFC_TYPES   = {'InfC'}
VERBAL_TYPES = (WAYQ_TYPES | WQTL_TYPES | FRONT_TYPES |
                {'ZQt0','ZQtX','ZYq0','ZYqX','ZIm0',
                 'WYq0','WYqX','WIm0','WXYq','WXQt',
                 'xQt0','xYq0','xQtX','xYqX','Way0','WayX'})
SKIP_TYPES   = {'Ellp','Voct','MSyn','InfA'}

# ---------------------------------------------------------------------------
# BHSA
# ---------------------------------------------------------------------------
def load_bhsa(data_path):
    try:
        from tf.fabric import Fabric
    except ImportError:
        sys.exit('text-fabric not installed.')
    print(f'Loading BHSA from {data_path}...')
    TF = Fabric(locations=str(data_path), modules=[''], silent=True)
    return TF.load(
        'otype lex sp vs vt ps nu gn st prs_ps chapter typ function domain',
        silent=True)

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def words_for_ranges(book, ch_ranges, F, L, T):
    bn = T.nodeFromSection((book,))
    if bn is None: return
    for ch_node in L.d(bn, 'chapter'):
        ch = int(F.chapter.v(ch_node))
        if any(s <= ch <= e for s, e in ch_ranges):
            for w in L.d(ch_node, 'word'):
                yield (F.lex.v(w), F.sp.v(w), F.vt.v(w), F.vs.v(w),
                       F.nu.v(w),  F.gn.v(w), F.st.v(w), F.prs_ps.v(w))


def extract_word_features(pairs, feature_names, F, L, T):
    rows = []
    for book, ch_ranges in pairs:
        rows.extend(words_for_ranges(book, ch_ranges, F, L, T))
    if not rows: return None, 0
    df  = pd.DataFrame(rows, columns=['lex','sp','vt','vs','nu','gn','st','prs_ps'])
    n   = len(df)
    lc  = df['lex'].value_counts()
    vb  = df[df['sp']=='verb']; nb = df[df['sp']=='subs']
    vtc = vb['vt'].value_counts(); vsc = vb['vs'].value_counts()
    result = {}
    for feat in feature_names:
        if feat in TIER3_FEATURES: result[feat] = np.nan; continue
        if   feat.startswith('lex::'):        result[feat] = lc.get(feat[5:],0)/n*1000
        elif feat.startswith('vt::'):         result[feat] = vtc.get(feat[4:],0)/n*1000
        elif feat.startswith('vs::'):         result[feat] = vsc.get(feat[4:],0)/n*1000
        elif feat=='morph::const_count':      result[feat] = (nb['st']=='c').sum()/n*1000
        elif feat=='morph::abs_count':        result[feat] = (nb['st']=='a').sum()/n*1000
        elif feat=='morph::prs_rate':         result[feat] = (df['prs_ps']!='NA').sum()/n*1000
        elif feat=='morph::noun_pl_rate':     result[feat] = (nb['nu']=='pl').sum()/n*1000
        elif feat=='morph::noun_f_rate':      result[feat] = (nb['gn']=='f').sum()/n*1000
        elif feat=='frac_ani':
            a,b = lc.get('>NJ',0),lc.get('>NKJ',0)
            result[feat] = a/(a+b) if a+b>0 else np.nan
        elif feat=='frac_she':
            a,b = lc.get('C',0),lc.get('>CR',0)
            result[feat] = a/(a+b) if a+b>0 else np.nan
        elif feat=='rate_ut_nouns':
            ut = sum(v for k,v in lc.items() if k.endswith('WT/') or k.endswith('WT'))
            result[feat] = ut/n*1000
        else: result[feat] = np.nan
    return np.array([result.get(f,np.nan) for f in feature_names]), n


def extract_tier3(pairs, F, L, T):
    nc=nv=ni=nfr=nwq=nwn=nwy=nns=0; nclause=0
    for book, ch_ranges in pairs:
        bn = T.nodeFromSection((book,))
        if bn is None: continue
        for ch_node in L.d(bn, 'chapter'):
            ch = int(F.chapter.v(ch_node))
            if not any(s<=ch<=e for s,e in ch_ranges): continue
            for cl in L.d(ch_node, 'clause'):
                typ = F.typ.v(cl)
                if typ in SKIP_TYPES: continue
                nclause += 1
                if typ in INFC_TYPES:  ni  += 1
                if typ in FRONT_TYPES: nfr += 1
                if typ in WAYQ_TYPES:  nwy += 1
                if typ in WQTL_TYPES:  nwq += 1
                if typ in WNARR_TYPES: nwn += 1
                if typ in VERBAL_TYPES:
                    nv += 1
                    phs = list(L.d(cl,'phrase'))
                    pf  = {F.function.v(ph):ph for ph in phs}
                    if 'Subj' not in pf: nns += 1
    if nclause == 0:
        return {f: np.nan for f in TIER3_FEATURES}
    return {
        'frac_infc':      ni  / nclause,
        'frac_fronted':   nfr / nclause,
        'frac_null_subj': nns / nv  if nv  > 10 else np.nan,
        'frac_wqtl_wayq': nwq / nwn if nwn >  5 else np.nan,
    }


def extract_unit(pairs, feature_names, F, L, T):
    vec, nw = extract_word_features(pairs, feature_names, F, L, T)
    if vec is None: return np.full(len(feature_names), np.nan), 0
    t3 = extract_tier3(pairs, F, L, T)
    for tf in TIER3_FEATURES:
        if tf in feature_names:
            vec[feature_names.index(tf)] = t3.get(tf, np.nan)
    return vec, nw

# ---------------------------------------------------------------------------
# Training setup (identical to scripts 11–14)
# ---------------------------------------------------------------------------
def build_training(robust_csv, rates_csv, tier3_csv, ridge=0.10):
    robust_df = pd.read_csv(robust_csv)
    rates_df  = pd.read_csv(rates_csv, index_col='unit')
    dates_bce = rates_df['date_bce'].values.astype(float)
    rates_df  = rates_df.drop(columns=['date_bce','n_words'], errors='ignore')

    # Linguistic filter
    keep = []
    for _, row in robust_df.iterrows():
        feat = str(row['feature'])
        if feat.startswith(('vt::','vs::','morph::')):
            keep.append(True); continue
        if feat.startswith('lex::'):
            desc = str(row.get('description',''))
            m = re.search(r'\(([^)]+)\)', desc)
            pos = m.group(1).strip() if m else ''
            keep.append(pos in LINGUISTIC_POS)
        else: keep.append(False)
    ling_df = robust_df[keep].reset_index(drop=True)
    wf = ling_df['feature'].tolist()
    for rf in ['frac_ani','frac_she','rate_ut_nouns']:
        if rf not in wf: wf.append(rf)

    df = rates_df.copy()
    ani = df.get('lex::>NJ', pd.Series(0., index=df.index))
    ank = df.get('lex::>NKJ', pd.Series(0., index=df.index))
    she = df.get('lex::C',    pd.Series(0., index=df.index))
    ash = df.get('lex::>CR',  pd.Series(0., index=df.index))
    df['frac_ani'] = np.where(ani+ank>0, ani/(ani+ank), np.nan)
    df['frac_she'] = np.where(she+ash>0, she/(she+ash), np.nan)
    ut_cols = [c for c in df.columns if re.match(r'lex::\w+WT/', c)]
    df['rate_ut_nouns'] = df[ut_cols].sum(axis=1) if ut_cols else 0.
    rates_aug = df

    tier3_path = Path(tier3_csv)
    if tier3_path.exists():
        t3 = pd.read_csv(tier3_path, index_col=0)
        for col in TIER3_FEATURES:
            if col in t3.columns:
                common = rates_aug.index.intersection(t3.index)
                rates_aug.loc[common, col] = t3.loc[common, col]
                if col not in wf: wf.append(col)

    for f in wf:
        if f not in rates_aug.columns: rates_aug[f] = np.nan

    # Fit OLS per feature + build residual covariance
    ols_params, residual_rows = {}, []
    for fn in wf:
        y = rates_aug[fn].values.astype(float)
        x = np.array(dates_bce, dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 4:
            ols_params[fn] = (np.nanmean(y), 0.0, np.nanstd(y)+1e-9)
            residual_rows.append(np.zeros(len(dates_bce)))
            continue
        xv, yv = x[valid], y[valid]
        b, a = np.polyfit(xv, yv, 1)
        pred = a + b * x; resid = y - pred; resid[~valid] = 0.
        ols_params[fn] = (a, b, np.std(yv-(a+b*xv))+1e-9)
        residual_rows.append(resid)

    R = np.array(residual_rows).T
    Sigma = np.cov(R.T)
    if Sigma.ndim == 0: Sigma = np.array([[float(Sigma)]])
    K = Sigma.shape[0]
    ridge_val = ridge * np.trace(Sigma) / K
    Sigma_reg = Sigma + ridge_val * np.eye(K)
    Sigma_inv = np.linalg.inv(Sigma_reg)

    # Temporal variance per feature (for genre ratio denominator)
    temporal_var = {fn: float(np.nanvar(rates_aug[fn].values.astype(float)))
                    for fn in wf}

    print(f'  Training: {len(wf)} features, {len(dates_bce)} units, '
          f'cond={np.linalg.cond(Sigma_reg):.0f}')
    return wf, ols_params, Sigma_reg, Sigma_inv, temporal_var, rates_aug, dates_bce

# ---------------------------------------------------------------------------
# Genre calibration: compute genre_ratio per feature
# ---------------------------------------------------------------------------
def compute_genre_ratios(calib_vecs, working_features, temporal_var):
    """
    calib_vecs: {'legal': {unit: (vec, nw)}, 'narrative': {unit: (vec, nw)}}
    Returns dict {feature: genre_ratio}
    """
    genre_ratios = {}
    for fi, fn in enumerate(working_features):
        group_means, group_ns = {}, {}
        for genre, units in calib_vecs.items():
            vals, wts = [], []
            for uname, (vec, nw) in units.items():
                if np.isfinite(vec[fi]) and nw > 0:
                    vals.append(vec[fi]); wts.append(float(nw))
            if len(vals) >= 1:
                wts_a = np.array(wts); wts_a /= wts_a.sum()
                group_means[genre] = float(np.average(vals, weights=wts_a))
                group_ns[genre]    = sum(wts)

        if len(group_means) < 2:
            genre_ratios[fn] = 0.0; continue

        total_n = sum(group_ns.values())
        overall = sum(group_ns[g]/total_n * group_means[g] for g in group_means)
        between_var = sum(group_ns[g]/total_n * (group_means[g]-overall)**2
                          for g in group_means)

        tvar = max(temporal_var.get(fn, 1e-12), 1e-12)
        genre_ratios[fn] = between_var / tvar

    return genre_ratios

# ---------------------------------------------------------------------------
# Build the four model variants
# ---------------------------------------------------------------------------
def make_model_variants(Sigma_reg, Sigma_inv, genre_ratios, working_features,
                        mismatch_score):
    """
    Returns dict variant_name → (Sigma_inv_eff, weight_vec)
    """
    K  = len(working_features)
    gv = np.array([genre_ratios.get(fn, 0.0) for fn in working_features])

    # Strategy-B weight vector
    w_B = 1.0 / (1.0 + gv)          # in [0, 1]
    W_B = np.diag(w_B)

    # Strategy-D inflated covariance
    extra_diag = np.diag(Sigma_reg) * mismatch_score * gv
    Sigma_D = Sigma_reg.copy()
    np.fill_diagonal(Sigma_D, np.diag(Sigma_D) + extra_diag)
    Sigma_D_inv = np.linalg.inv(Sigma_D)

    variants = {
        'Raw':  (Sigma_inv,                     np.ones(K)),
        'B':    (W_B @ Sigma_inv    @ W_B,       w_B),
        'D':    (Sigma_D_inv,                    np.ones(K)),
        'B+D':  (W_B @ Sigma_D_inv  @ W_B,       w_B),
    }
    return variants

# ---------------------------------------------------------------------------
# Posterior computation
# ---------------------------------------------------------------------------
def mvn_log_lik(x_obs, date, ols_params, feature_names, Sigma_inv_eff, nw):
    mu    = np.array([ols_params[fn][0]+ols_params[fn][1]*date for fn in feature_names])
    valid = np.isfinite(x_obs)
    if valid.sum() < 2: return -np.inf
    diff = x_obs - mu; diff[~valid] = 0.
    scale = float(np.clip(nw/5000, 1.0, 5.0))
    return -0.5 * scale * diff @ Sigma_inv_eff @ diff


def compute_posterior(x_obs, nw, ols_params, feature_names, Sigma_inv_eff,
                      n_grid=500, prior_mean=600, prior_sd=350):
    dg   = np.linspace(50, 1200, n_grid)
    lp   = -0.5 * ((dg - prior_mean)/prior_sd)**2
    ll   = np.array([mvn_log_lik(x_obs, d, ols_params, feature_names,
                                  Sigma_inv_eff, nw) for d in dg])
    lpo  = lp + ll; lpo -= lpo.max()
    post = np.exp(lpo); post /= post.sum()
    cdf  = np.cumsum(post)
    def q(p): return dg[min(np.searchsorted(cdf, p), n_grid-1)]
    return dg, post, dg[post.argmax()], q(0.16), q(0.84)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
VARIANT_STYLE = {
    'Raw':  ('#888888', '-',  2.0),
    'B':    ('#1f77b4', '--', 2.0),
    'D':    ('#d62728', ':',  2.0),
    'B+D':  ('#2ca02c', '-',  2.5),
}
VARIANT_ORDER = ['Raw', 'B', 'D', 'B+D']


def plot_genre_ratios(genre_ratios, working_features, output_path):
    """Bar chart: genre_ratio per feature, sorted descending."""
    ratios = np.array([genre_ratios.get(fn, 0.) for fn in working_features])
    order  = np.argsort(ratios)[::-1]
    labels = [working_features[i].replace('lex::','').replace('vt::','')
              .replace('vs::','').replace('morph::','') for i in order]

    # colour by feature type
    def fcol(fn):
        if fn in TIER3_FEATURES: return '#9467bd'
        if fn.startswith('lex::'):    return '#1f77b4'
        if fn.startswith(('vt::','vs::')): return '#d62728'
        if fn.startswith('morph::'):  return '#ff7f0e'
        return '#2ca02c'   # ratio features
    colors = [fcol(working_features[i]) for i in order]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(ratios)), ratios[order], color=colors, edgecolor='none')
    ax.axhline(1.0, color='black', lw=1, ls='--', alpha=0.4,
               label='genre = temporal variance')
    ax.axhline(0.5, color='grey', lw=0.8, ls=':', alpha=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=75, ha='right', fontsize=7)
    ax.set_ylabel('Genre ratio (between-group / temporal variance)')
    ax.set_title('Feature genre sensitivity — Torah-internal calibration\n'
                 'Features above dashed line are more genre-driven than time-driven')
    # Legend patches
    import matplotlib.patches as mpatches
    ax.legend(handles=[
        mpatches.Patch(color='#1f77b4', label='lex::'),
        mpatches.Patch(color='#d62728', label='vt:: / vs::'),
        mpatches.Patch(color='#ff7f0e', label='morph::'),
        mpatches.Patch(color='#2ca02c', label='ratio features'),
        mpatches.Patch(color='#9467bd', label='Tier-3'),
    ], fontsize=7, loc='upper right')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Genre-ratio plot saved: {output_path}')


def plot_posterior_grid(date_grid, unit_posteriors, selected_units,
                        title, output_path):
    """
    For each selected unit, overlay the 4 variant posteriors.
    Layout: 2 columns × N/2 rows.
    """
    su   = [u for u in selected_units if u in unit_posteriors]
    ncol = 2; nrow = (len(su)+1)//2
    fig, axes = plt.subplots(nrow, ncol, figsize=(10, 3.2*nrow), squeeze=False)
    axes_flat = axes.flatten()

    for ax, unit in zip(axes_flat, su):
        genre = TEST_UNITS.get(unit, ('?',))[0]
        for var in VARIANT_ORDER:
            post = unit_posteriors[unit].get(var)
            if post is None: continue
            col, ls, lw = VARIANT_STYLE[var]
            ax.plot(date_grid, post/post.max(), color=col, ls=ls, lw=lw,
                    label=var)
            ax.axvline(date_grid[post.argmax()], color=col, lw=0.8,
                       ls=':', alpha=0.5)
        ax.set_xlim(date_grid.max(), date_grid.min())
        ax.set_xlabel('Date (BCE)', fontsize=8)
        ax.set_ylabel('Norm. posterior', fontsize=8)
        ax.set_title(f'{unit.replace("_"," ")} [{genre}]', fontsize=9)
        ax.legend(fontsize=7)

    for ax in axes_flat[len(su):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Posterior grid saved: {output_path}')


def plot_summary_comparison(records, output_path):
    """
    Grouped horizontal bar chart: Raw vs B+D MAP for all units, by genre.
    Units sorted within group by Raw MAP.
    """
    genre_order = ['legal', 'narrative', 'other']
    genre_col   = {'legal':'#d62728','narrative':'#1f77b4','other':'#2ca02c'}

    # Group and sort
    grouped = {g: [] for g in genre_order}
    for r in records:
        g = r['genre']
        if g in grouped: grouped[g].append(r)
    for g in grouped:
        grouped[g].sort(key=lambda r: r['Raw'])  # sort by raw MAP

    all_rows = []
    separators = []
    for g in genre_order:
        if grouped[g]:
            separators.append(len(all_rows))
        all_rows.extend(grouped[g])

    fig, ax = plt.subplots(figsize=(8, max(5, len(all_rows)*0.38)))
    ys     = np.arange(len(all_rows))
    height = 0.3

    for i, r in enumerate(all_rows):
        gc = genre_col[r['genre']]
        # Raw
        ax.barh(i+height/2, r['Raw'], height, color=gc, alpha=0.85,
                label=f'Raw ({r["genre"]})' if i==0 else '')
        ax.errorbar(r['Raw'], i+height/2,
                    xerr=[[abs(r['Raw']-r['ci68_lo_raw'])],
                           [abs(r['ci68_hi_raw']-r['Raw'])]],
                    fmt='none', color='black', capsize=2, lw=0.8)
        # B+D
        ax.barh(i-height/2, r['B+D'], height, color=gc, alpha=0.40,
                hatch='//', label='B+D' if i==0 else '')
        ax.errorbar(r['B+D'], i-height/2,
                    xerr=[[abs(r['B+D']-r['ci68_lo_bd'])],
                           [abs(r['ci68_hi_bd']-r['B+D'])]],
                    fmt='none', color='black', capsize=2, lw=0.8)

    # Genre separator lines
    for sep in separators[1:]:
        ax.axhline(sep - 0.5, color='grey', lw=1.0)

    ax.set_yticks(ys)
    ax.set_yticklabels([r['unit'].replace('_',' ') for r in all_rows], fontsize=8)
    ax.invert_xaxis()
    ax.set_xlabel('Date MAP (BCE, left = older)')
    ax.set_title('Raw (solid) vs. B+D genre-corrected (hatched) date estimates\n'
                 '± 68 % CI whiskers', fontsize=10)

    from matplotlib.patches import Patch
    handles = [
        Patch(color=genre_col['legal'],     alpha=0.85, label='Legal (solid=Raw, hatched=B+D)'),
        Patch(color=genre_col['narrative'], alpha=0.85, label='Narrative'),
        Patch(color=genre_col['other'],     alpha=0.85, label='Prophetic/Poetry'),
    ]
    ax.legend(handles=handles, fontsize=7, loc='lower left')
    ax.axvline(760, color='grey', lw=0.8, ls='--', alpha=0.4)
    ax.axvline(167, color='grey', lw=0.8, ls='--', alpha=0.4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Summary comparison saved: {output_path}')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-path',
        default=str(Path.home()/'text-fabric-data'/'github'/'ETCBC'/'bhsa'/'tf'/'2021'))
    parser.add_argument('--scan-robust', default='feature_scan_robust.csv')
    parser.add_argument('--rates-csv',   default='feature_rates_training.csv')
    parser.add_argument('--tier3-csv',   default='tier3_training_rates.csv')
    parser.add_argument('--outdir',      default='.')
    parser.add_argument('--ridge',       type=float, default=0.10)
    parser.add_argument('--n-grid',      type=int,   default=500)
    args = parser.parse_args()

    outdir = Path(args.outdir)

    # ------------------------------------------------------------------
    # 1. Training model
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 1 — Build training model')
    print('='*70)

    wf, ols_params, Sigma_reg, Sigma_inv, temporal_var, _, _ = build_training(
        args.scan_robust, args.rates_csv, args.tier3_csv, args.ridge)

    # ------------------------------------------------------------------
    # 2. Load BHSA
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 2 — Load BHSA')
    print('='*70)
    api = load_bhsa(args.data_path)
    F, L, T = api.F, api.L, api.T

    # ------------------------------------------------------------------
    # 3. Extract Torah-internal calibration vectors
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 3 — Torah-internal genre calibration')
    print('='*70)

    calib_vecs = {'legal': {}, 'narrative': {}}
    print('  Legal calibration units:')
    for uname, pairs in CALIB_LEGAL.items():
        vec, nw = extract_unit(pairs, wf, F, L, T)
        calib_vecs['legal'][uname] = (vec, nw)
        print(f'    {uname:<16} {nw:>7,} words')
    print('  Narrative calibration units:')
    for uname, pairs in CALIB_NARRATIVE.items():
        vec, nw = extract_unit(pairs, wf, F, L, T)
        calib_vecs['narrative'][uname] = (vec, nw)
        print(f'    {uname:<16} {nw:>7,} words')

    # ------------------------------------------------------------------
    # 4. Compute genre ratios
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 4 — Genre ratios per feature')
    print('='*70)

    genre_ratios = compute_genre_ratios(calib_vecs, wf, temporal_var)

    # Report most genre-sensitive and most stable features
    sorted_gr = sorted(genre_ratios.items(), key=lambda x: -x[1])
    print(f'\n  {"Feature":<35}  {"genre_ratio":>12}  {"B_weight":>9}  note')
    print('  ' + '-'*65)
    for fn, gr in sorted_gr:
        w  = 1.0/(1.0+gr)
        fn_s = fn.replace('lex::','').replace('vt::','').replace('vs::','').replace('morph::','')
        note = ('*** highly genre-sensitive' if gr > 2.0 else
                '**  genre-sensitive'        if gr > 1.0 else
                '*   mildly genre-sensitive' if gr > 0.5 else '')
        print(f'  {fn_s:<35}  {gr:>12.3f}  {w:>9.3f}  {note}')

    n_high = sum(1 for gr in genre_ratios.values() if gr > 1.0)
    print(f'\n  Features with genre_ratio > 1.0 (genre > temporal variance): {n_high}/{len(wf)}')

    # ------------------------------------------------------------------
    # 5. Extract features for all test units
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 5 — Extract test unit features')
    print('='*70)

    unit_vecs = {}
    for label, (genre, pairs) in TEST_UNITS.items():
        vec, nw = extract_unit(pairs, wf, F, L, T)
        unit_vecs[label] = (vec, nw)
        nflag = ' ⚠' if nw < 1000 else ''
        print(f'  {label:<18} [{genre:<9}] {nw:>8,} words{nflag}')

    # ------------------------------------------------------------------
    # 6. Compute posteriors under all 4 model variants
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 6 — Date posteriors (4 variants)')
    print('='*70)

    unit_posteriors = {}   # {unit: {variant: post_array}}
    records = []
    date_grid = None

    # Header
    print(f'  {"Unit":<18} {"Genre":<9} | '
          f'{"Raw":>6} {"B":>6} {"D":>6} {"B+D":>6} | '
          f'{"ΔB":>5} {"ΔD":>5} {"ΔB+D":>5} | {"n_words":>8}')
    print('  ' + '-'*80)

    for label, (genre, pairs) in TEST_UNITS.items():
        x_obs, nw = unit_vecs[label]
        mismatch  = GENRE_MISMATCH[genre]

        variants = make_model_variants(Sigma_reg, Sigma_inv, genre_ratios,
                                        wf, mismatch)
        unit_posteriors[label] = {}
        maps = {}
        ci68 = {}

        for var_name, (S_inv_eff, _) in variants.items():
            dg, post, map_d, ci68_lo, ci68_hi = compute_posterior(
                x_obs, nw, ols_params, wf, S_inv_eff, n_grid=args.n_grid)
            unit_posteriors[label][var_name] = post
            maps[var_name] = map_d
            ci68[var_name] = (ci68_lo, ci68_hi)
            if date_grid is None:
                date_grid = dg

        delta_b  = maps['B']   - maps['Raw']
        delta_d  = maps['D']   - maps['Raw']
        delta_bd = maps['B+D'] - maps['Raw']
        flag = ' ⚠' if nw < 1000 else ''

        print(f'  {label:<18} {genre:<9} | '
              f'{maps["Raw"]:>6.0f} {maps["B"]:>6.0f} '
              f'{maps["D"]:>6.0f} {maps["B+D"]:>6.0f} | '
              f'{delta_b:>+5.0f} {delta_d:>+5.0f} {delta_bd:>+5.0f} | '
              f'{nw:>8,}{flag}')

        records.append({
            'unit': label, 'genre': genre, 'n_words': nw,
            'Raw':  round(maps['Raw'], 0), 'B':   round(maps['B'],   0),
            'D':    round(maps['D'],   0), 'B+D': round(maps['B+D'], 0),
            'delta_B':   round(delta_b,  0),
            'delta_D':   round(delta_d,  0),
            'delta_BD':  round(delta_bd, 0),
            'ci68_lo_raw': round(ci68['Raw'][0], 0),
            'ci68_hi_raw': round(ci68['Raw'][1], 0),
            'ci68_lo_bd':  round(ci68['B+D'][0], 0),
            'ci68_hi_bd':  round(ci68['B+D'][1], 0),
        })

    # ------------------------------------------------------------------
    # 7. Print genre-grouped summary
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 7 — Genre-grouped summary')
    print('='*70)
    for genre in ['legal', 'narrative', 'other']:
        grp = [r for r in records if r['genre'] == genre]
        if not grp: continue
        print(f'\n  {genre.upper()} texts:')
        print(f'  {"Unit":<18} | {"Raw":>6} {"B":>6} {"D":>6} {"B+D":>6} | '
              f'{"ΔB":>5} {"ΔD":>5} {"ΔB+D":>5}')
        print('  ' + '-'*60)
        for r in grp:
            print(f'  {r["unit"]:<18} | '
                  f'{r["Raw"]:>6.0f} {r["B"]:>6.0f} '
                  f'{r["D"]:>6.0f} {r["B+D"]:>6.0f} | '
                  f'{r["delta_B"]:>+5.0f} {r["delta_D"]:>+5.0f} '
                  f'{r["delta_BD"]:>+5.0f}')
        # Average delta by genre
        avg_bd = np.mean([r['delta_BD'] for r in grp if r['n_words']>=500])
        print(f'  Mean Δ(B+D) for {genre}: {avg_bd:+.0f} years')

    print('\n  Interpretation guide:')
    print('    Positive Δ = genre correction pushes date OLDER (higher BCE)')
    print('    Negative Δ = genre correction pushes date YOUNGER')
    print('    Legal texts expected to shift older (archaic syntax was discounted;')
    print('      law-register lexical features were genre-noise not date signal).')
    print('    Prophetic/other texts should barely move (minimal genre mismatch).')

    # ------------------------------------------------------------------
    # 8. Save outputs
    # ------------------------------------------------------------------
    pd.DataFrame(records).to_csv(outdir / 'genre_controlled_dating.csv', index=False)

    # Genre ratios CSV
    gr_rows = [{'feature': fn, 'genre_ratio': gr,
                'b_weight': 1.0/(1.0+gr),
                'type': ('tier3' if fn in TIER3_FEATURES else
                         'lex'   if fn.startswith('lex::') else
                         'vbt'   if fn.startswith(('vt::','vs::')) else
                         'morph' if fn.startswith('morph::') else 'ratio')}
               for fn, gr in sorted_gr]
    pd.DataFrame(gr_rows).to_csv(outdir / 'genre_ratios.csv', index=False)
    print('\nOutputs: genre_controlled_dating.csv  genre_ratios.csv')

    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 8 — Plots')
    print('='*70)

    # Feature genre-sensitivity chart
    plot_genre_ratios(genre_ratios, wf, str(outdir/'genre_ratios.png'))

    # Posterior grids for representative units
    legal_sel  = ['D_Code','Lev_Holiness','Lev_Priestly','D_Song']
    narr_sel   = ['D_Frame','JE_source','2_Samuel','Jer_DTR']
    other_sel  = ['Jer_oracle','Song_Sea','Song_Deborah']

    if date_grid is not None:
        plot_posterior_grid(date_grid, unit_posteriors, legal_sel,
                            'Legal texts — Raw / B / D / B+D posteriors',
                            str(outdir/'genre_posteriors_legal.png'))
        plot_posterior_grid(date_grid, unit_posteriors, narr_sel,
                            'Narrative texts — Raw / B / D / B+D posteriors',
                            str(outdir/'genre_posteriors_narrative.png'))
        plot_posterior_grid(date_grid, unit_posteriors, other_sel,
                            'Prophetic/poetic texts — Raw / B / D / B+D posteriors\n'
                            '(should barely change — validation check)',
                            str(outdir/'genre_posteriors_other.png'))

    # Summary comparison chart (all units)
    plot_summary_comparison(records, str(outdir/'genre_summary_comparison.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
