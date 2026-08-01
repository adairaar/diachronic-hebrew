#!/usr/bin/env python3
"""
Script 13: Sub-source and Deuteronomistic History Analysis
===========================================================
Applies the comprehensive MVN dating model (script 11) to finer-grained
text units that have scholarly significance:

  D sub-components:
    D_Code   — Deuteronomic Code, chapters 12–26
    D_Frame  — Framing speeches/narratives, chapters 1–11 + 27–31 + 33–34
    D_Song   — Song of Moses, chapter 32 (ancient poem)

  Deuteronomistic History (book by book):
    Joshua, Judges, 1_Samuel, 2_Samuel, 1_Kings, 2_Kings

  Leviticus sub-components:
    Lev_Holiness — Holiness Code, chapters 17–26
    Lev_Priestly — Priestly ritual laws, chapters 1–16

For each unit the script computes:
  • Full-model MAP date + 68 % CI (all 36 features)
  • Resistant-model MAP date (Tier-3 syntax-only, 4 features)
  • Archaism audit (mean LBH score + per-feature breakdown)

This allows testing whether the anomalous D archaism profile originates
in the law-code register (D_Code) or the parenetic framing (D_Frame),
and whether the Dtr History books share D's linguistic fingerprint.

Usage
-----
    python 13_subsource_analysis.py [--data-path PATH] [--outdir DIR]
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
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Unit definitions
# ---------------------------------------------------------------------------

# D sub-components
D_UNITS = {
    'D_Code':  [('Deuteronomy', [(12, 26)])],
    'D_Frame': [('Deuteronomy', [(1, 11), (27, 31), (33, 34)])],
    'D_Song':  [('Deuteronomy', [(32, 32)])],
    'D_full':  [('Deuteronomy', [(1, 34)])],   # whole book (reference)
}

# Deuteronomistic History, book by book
DTR_BOOKS = ['Joshua', 'Judges', '1_Samuel', '2_Samuel', '1_Kings', '2_Kings']
# Display labels (drop underscore)
DTR_LABELS = {b: b.replace('_', ' ') for b in DTR_BOOKS}

# Chapter-range overrides for books containing embedded ancient songs.
DTR_BOOK_OVERRIDES = {
    'Judges': [('Judges', [(1, 4), (6, 21)])],  # exclude ch. 5 (Song of Deborah)
}

# Leviticus sub-components
LEV_UNITS = {
    'Lev_Holiness': [('Leviticus', [(17, 26)])],
    'Lev_Priestly': [('Leviticus', [(1, 16)])],
    'Lev_full':     [('Leviticus', [(1, 27)])],  # whole book (reference)
}

# Tier-3 features (same 4 that passed in script 12)
TIER3_FEATURES = ['frac_infc', 'frac_fronted', 'frac_null_subj', 'frac_wqtl_wayq']

# Linguistic POS kept by the filter
LINGUISTIC_POS = {
    'pronoun', 'dem-pronoun', 'prep', 'conj',
    'adverb', 'negator', 'article', 'interrogative', 'interjection',
}

# ---------------------------------------------------------------------------
# Clause-type constants (mirrored from scripts 12/11)
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
# BHSA loading
# ---------------------------------------------------------------------------

def load_bhsa(data_path):
    try:
        from tf.fabric import Fabric
    except ImportError:
        sys.exit('text-fabric not installed.  Run: pip install text-fabric')
    print(f'Loading BHSA from {data_path}...')
    TF = Fabric(locations=str(data_path), modules=[''], silent=True)
    api = TF.load(
        'otype lex sp vs vt ps nu gn st prs_ps chapter typ function domain',
        silent=True)
    return api


# ---------------------------------------------------------------------------
# Feature extraction helpers (same logic as script 11)
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
    """Extract per-1000-word rates for word-level features."""
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
            result[feat] = np.nan   # filled later by extract_tier3_for_unit
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
    """Pool all clauses in book_ch_pairs and compute Tier-3 features."""
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


def extract_full_book(book_name, feature_names, F, L, T):
    """Convenience wrapper for an entire book."""
    bn = T.nodeFromSection((book_name,))
    if bn is None:
        return np.full(len(feature_names), np.nan), 0
    ch_nums = [int(F.chapter.v(c)) for c in L.d(bn, 'chapter')]
    pairs   = [(book_name, [(min(ch_nums), max(ch_nums))])]
    vec, nw = extract_unit_features(pairs, feature_names, F, L, T)
    if vec is not None:
        t3 = extract_tier3_for_unit(pairs, F, L, T)
        for tf in TIER3_FEATURES:
            if tf in feature_names:
                vec[feature_names.index(tf)] = t3.get(tf, np.nan)
    return (vec if vec is not None else np.full(len(feature_names), np.nan)), nw


def extract_multi_range(label, book_ch_pairs, feature_names, F, L, T):
    """Extract features for a unit spanning multiple (book, ranges) pairs."""
    vec, nw = extract_unit_features(book_ch_pairs, feature_names, F, L, T)
    if vec is None:
        return np.full(len(feature_names), np.nan), 0
    t3 = extract_tier3_for_unit(book_ch_pairs, F, L, T)
    for tf in TIER3_FEATURES:
        if tf in feature_names:
            vec[feature_names.index(tf)] = t3.get(tf, np.nan)
    return vec, nw


# ---------------------------------------------------------------------------
# MVN model (verbatim from script 11)
# ---------------------------------------------------------------------------

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
        resid_std = np.std(yv - (a + b * xv)) + 1e-9
        ols_params[fn] = (a, b, resid_std)
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
    print(f"    {K} features, condition number {cond:.0f} (ridge={ridge_frac:.2f})")
    return ols_params, Sigma_reg, Sigma_inv


def mvn_log_likelihood(x_obs, date_bce, ols_params, feature_names,
                        Sigma_inv, n_words=1):
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
                      prior_mean=600, prior_sd=350):
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


def compute_archaism_scores(x_obs_dict, ols_params, feature_names,
                             archaic_date=720, modern_date=250):
    scores = {}
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
            unit_scores[fn] = (x_obs[i] - cbh_val) / span
        scores[unit] = unit_scores
    return scores


# ---------------------------------------------------------------------------
# Linguistic filter (same as script 11)
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


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

UNIT_GROUPS = {
    'D sub-components':   ['D_full', 'D_Code', 'D_Frame', 'D_Song'],
    'Dtr History':        ['Joshua', 'Judges', '1_Samuel', '2_Samuel',
                           '1_Kings', '2_Kings'],
    'Lev sub-components': ['Lev_full', 'Lev_Holiness', 'Lev_Priestly'],
}

GROUP_COLORS = {
    'D sub-components':   ['#1f77b4', '#aec7e8', '#6baed6', '#c6dbef'],
    'Dtr History':        ['#d62728', '#ff9896', '#e6550d', '#fd8d3c',
                           '#843c39', '#e7ba52'],
    'Lev sub-components': ['#2ca02c', '#98df8a', '#c7e9c0'],
}


def plot_comparison_bars(records, output_path):
    """
    Grouped bar chart: full-model MAP vs resistant-model MAP for all units,
    coloured by group, with 68 % CI whiskers.
    """
    groups   = list(UNIT_GROUPS.keys())
    fig, axes = plt.subplots(1, len(groups),
                              figsize=(5.5 * len(groups), 5),
                              sharey=False)

    for ax, grp in zip(axes, groups):
        unit_list = [u for u in UNIT_GROUPS[grp]
                     if any(r['unit'] == u for r in records)]
        colors = GROUP_COLORS.get(grp, ['grey'] * len(unit_list))

        x      = np.arange(len(unit_list))
        bar_w  = 0.35

        for i, (unit, col) in enumerate(zip(unit_list, colors)):
            r = next(x_ for x_ in records if x_['unit'] == unit)
            # Full model
            ax.bar(i - bar_w/2, r['map_bce'], bar_w,
                   color=col, alpha=0.85, label='Full' if i == 0 else '')
            ci_lo, ci_hi = r['ci68_lo'], r['ci68_hi']
            ax.errorbar(i - bar_w/2, r['map_bce'],
                        yerr=[[abs(r['map_bce'] - ci_lo)],
                               [abs(ci_hi - r['map_bce'])]],
                        fmt='none', color='black', capsize=3, lw=1)
            # Resistant model
            if r.get('map_bce_resistant') is not None:
                ax.bar(i + bar_w/2, r['map_bce_resistant'], bar_w,
                       color=col, alpha=0.40, hatch='//',
                       label='Resistant' if i == 0 else '')

        ax.set_xticks(range(len(unit_list)))
        ax.set_xticklabels([u.replace('_', '\n') for u in unit_list],
                           fontsize=8)
        ax.invert_yaxis()   # larger BCE = older = top of chart
        ax.set_ylabel('Date (BCE, top = older)')
        ax.set_title(grp, fontsize=10)
        ax.axhspan(760, 460, alpha=0.05, color='grey')
        ax.text(ax.get_xlim()[1], 610, 'Prophetic\ntraining', fontsize=6,
                color='grey', va='center')
        if i == 0:
            ax.legend(fontsize=7)

    fig.suptitle('Dating estimates: full model (solid) vs resistant/syntax-only (hatched)\n'
                 'Error bars = 68 % CI', fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Comparison bar chart saved: {output_path}')


def plot_archaism_comparison(scores_dict, feature_names, unit_groups,
                              output_path):
    """
    Side-by-side archaism heatmaps for each unit group.
    Rows = units, columns = features, colour = LBH score.
    """
    # Identify features with any finite scores
    active_feats = [f for f in feature_names
                    if any(np.isfinite(scores_dict.get(u, {}).get(f, np.nan))
                           for units in unit_groups.values() for u in units)][:35]

    all_units = [u for grp in unit_groups.values() for u in grp
                 if u in scores_dict]

    matrix = np.full((len(all_units), len(active_feats)), np.nan)
    for i, u in enumerate(all_units):
        for j, f in enumerate(active_feats):
            matrix[i, j] = scores_dict[u].get(f, np.nan)

    fig, ax = plt.subplots(figsize=(min(2 + len(active_feats) * 0.32, 16),
                                    max(4, len(all_units) * 0.45)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r',
                   vmin=-0.5, vmax=1.5, interpolation='nearest')
    plt.colorbar(im, ax=ax, label='LBH score (0=archaic, 1=modern)',
                 fraction=0.04, pad=0.04)

    ax.set_xticks(range(len(active_feats)))
    ax.set_xticklabels(
        [f.replace('lex::', '').replace('vt::', '').replace('vs::', '')
          .replace('morph::', '')
         for f in active_feats],
        rotation=70, ha='right', fontsize=6.5)
    ax.set_yticks(range(len(all_units)))
    ax.set_yticklabels([u.replace('_', ' ') for u in all_units], fontsize=8)

    # Draw separator lines between groups
    y = 0
    for grp, units in unit_groups.items():
        present = [u for u in units if u in scores_dict]
        y += len(present)
        if y < len(all_units):
            ax.axhline(y - 0.5, color='black', lw=1.5)

    ax.set_title('Archaism audit — LBH scores per feature\n'
                 '(blue = archaic CBH; red = modern LBH)',
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Archaism heatmap saved: {output_path}')


def plot_posteriors_grid(date_grid, posteriors, records, unit_groups,
                          output_path):
    """
    Posterior density curves arranged in one row per group.
    Solid = full model; dotted = resistant model (if available).
    """
    nrows = len(unit_groups)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.5 * nrows))
    if nrows == 1:
        axes = [axes]

    for ax, (grp, unit_list) in zip(axes, unit_groups.items()):
        present = [u for u in unit_list if u in posteriors]
        palette = plt.cm.tab10(np.linspace(0, 0.9, max(len(present), 1)))

        for unit, col in zip(present, palette):
            post = posteriors[unit]
            ax.plot(date_grid, post / post.max(), color=col, lw=2,
                    label=unit.replace('_', ' '))
            map_d = date_grid[post.argmax()]
            ax.axvline(map_d, color=col, lw=1, ls='--', alpha=0.6)

        ax.set_xlim(date_grid.max(), date_grid.min())
        ax.set_xlabel('Date (BCE)')
        ax.set_ylabel('Normalised posterior')
        ax.set_title(grp)
        ax.legend(fontsize=7, ncol=3)
        ax.axvspan(760, 460, alpha=0.04, color='grey')

    fig.suptitle('Date posteriors — full MVN model\n(dashed = MAP)', fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Posterior grid saved: {output_path}')


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
    parser.add_argument('--tier3-csv',   default='tier3_training_rates.csv')
    parser.add_argument('--outdir',      default='.')
    parser.add_argument('--ridge',  type=float, default=0.10)
    parser.add_argument('--n-grid', type=int,   default=500)
    args = parser.parse_args()

    outdir = Path(args.outdir)

    # ------------------------------------------------------------------
    # 1. Load feature scans + build working feature set (same as script 11)
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 1 — Load feature scans and build working feature set')
    print('='*70)

    robust_df = pd.read_csv(args.scan_robust)
    rates_df  = pd.read_csv(args.rates_csv, index_col='unit')
    dates_bce = rates_df['date_bce'].values.astype(float)
    rates_df  = rates_df.drop(columns=['date_bce', 'n_words'], errors='ignore')

    ling_df = filter_linguistic(robust_df)
    base_features = ling_df['feature'].tolist()

    working_features = base_features.copy()
    for rf in ['frac_ani', 'frac_she', 'rate_ut_nouns']:
        if rf not in working_features:
            working_features.append(rf)

    # Augment training rates with ratio features
    df = rates_df.copy()
    ani  = df.get('lex::>NJ',  pd.Series(0.0, index=df.index))
    ank  = df.get('lex::>NKJ', pd.Series(0.0, index=df.index))
    she  = df.get('lex::C',    pd.Series(0.0, index=df.index))
    ash  = df.get('lex::>CR',  pd.Series(0.0, index=df.index))
    denom_ani = ani + ank
    df['frac_ani'] = np.where(denom_ani > 0, ani / denom_ani, np.nan)
    denom_she = she + ash
    df['frac_she'] = np.where(denom_she > 0, she / denom_she, np.nan)
    ut_cols = [c for c in df.columns if re.match(r'lex::\w+WT/', c)]
    df['rate_ut_nouns'] = df[ut_cols].sum(axis=1) if ut_cols else 0.0
    rates_aug = df

    # Load Tier-3 training rates
    tier3_path = Path(args.tier3_csv)
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
                tier3_added.append(tf)
        print(f'  tier3_training_rates.csv not found — will extract live.')

    # Fill any missing columns with NaN
    for f in working_features:
        if f not in rates_aug.columns:
            rates_aug[f] = np.nan

    resistant_features = [f for f in working_features if f in TIER3_FEATURES]
    print(f'  Total working features:  {len(working_features)}')
    print(f'  Resistant features:      {len(resistant_features)}')

    # ------------------------------------------------------------------
    # 2. Build MVN models
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 2 — Build MVN models')
    print('='*70)

    print('  Full model:')
    ols_params, _, Sigma_inv = build_mvn_model(
        rates_aug[working_features], dates_bce, working_features,
        ridge_frac=args.ridge)

    ols_params_r = Sigma_inv_r = None
    if len(resistant_features) >= 2:
        print('  Resistant model (Tier-3 only):')
        r_train = rates_aug[resistant_features].copy()
        for col in resistant_features:
            r_train[col] = r_train[col].fillna(r_train[col].median())
        ols_params_r, _, Sigma_inv_r = build_mvn_model(
            r_train, dates_bce, resistant_features, ridge_frac=0.20)

    # ------------------------------------------------------------------
    # 3. Load BHSA + extract features for all analysis units
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 3 — Extracting features from BHSA')
    print('='*70)

    api  = load_bhsa(args.data_path)
    F, L, T = api.F, api.L, api.T

    units = {}  # {label: (feature_vector, n_words)}

    # ---- D sub-components ----
    print('\nD sub-components:')
    for label, pairs in D_UNITS.items():
        vec, nw = extract_multi_range(label, pairs, working_features, F, L, T)
        units[label] = (vec, nw)
        ani_v  = vec[working_features.index('frac_ani')]  if 'frac_ani'  in working_features else np.nan
        infc_v = vec[working_features.index('frac_infc')] if 'frac_infc' in working_features else np.nan
        print(f'  {label:<14} {nw:>8,} words  '
              f'frac_אני={ani_v:.3f}  frac_infc={infc_v:.3f}')

    # ---- Dtr History, book by book ----
    print('\nDeuteronomistic History:')
    for book in DTR_BOOKS:
        if book in DTR_BOOK_OVERRIDES:
            vec, nw = extract_multi_range(book, DTR_BOOK_OVERRIDES[book],
                                          working_features, F, L, T)
        else:
            vec, nw = extract_full_book(book, working_features, F, L, T)
        units[book] = (vec, nw)
        ani_v  = vec[working_features.index('frac_ani')]  if 'frac_ani'  in working_features else np.nan
        infc_v = vec[working_features.index('frac_infc')] if 'frac_infc' in working_features else np.nan
        print(f'  {book:<14} {nw:>8,} words  '
              f'frac_אני={ani_v:.3f}  frac_infc={infc_v:.3f}')

    # ---- Leviticus sub-components ----
    print('\nLeviticus sub-components:')
    for label, pairs in LEV_UNITS.items():
        vec, nw = extract_multi_range(label, pairs, working_features, F, L, T)
        units[label] = (vec, nw)
        ani_v  = vec[working_features.index('frac_ani')]  if 'frac_ani'  in working_features else np.nan
        infc_v = vec[working_features.index('frac_infc')] if 'frac_infc' in working_features else np.nan
        print(f'  {label:<14} {nw:>8,} words  '
              f'frac_אני={ani_v:.3f}  frac_infc={infc_v:.3f}')

    # ------------------------------------------------------------------
    # 4. Date posteriors
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 4 — Date posteriors')
    print('='*70)

    TRAINING_RANGE = 760 - 167
    WIDE_THRESH    = TRAINING_RANGE * 0.75
    posteriors     = {}
    records        = []
    date_grid      = None

    header = (f'  {"Unit":<16} {"MAP(full)":>9}  {"MAP(resist)":>11}  '
              f'{"68% CI (full)":>22}  {"n_words":>8}')
    print(header)
    print('  ' + '-' * (len(header) - 2))

    ordered_units = (list(D_UNITS.keys()) + DTR_BOOKS + list(LEV_UNITS.keys()))

    for label in ordered_units:
        if label not in units:
            continue
        x_obs, nw = units[label]

        dg, post, map_d, ci68_lo, ci68_hi, ci95_lo, ci95_hi = compute_posterior(
            x_obs, nw, ols_params, working_features, Sigma_inv,
            n_grid=args.n_grid)
        posteriors[label] = post
        date_grid = dg

        w95 = abs(ci95_lo - ci95_hi)

        map_r_val = None
        if ols_params_r is not None:
            x_r = np.array([x_obs[working_features.index(f)]
                             if f in working_features else np.nan
                             for f in resistant_features])
            _, _, map_r_val, *_ = compute_posterior(
                x_r, nw, ols_params_r, resistant_features, Sigma_inv_r,
                n_grid=args.n_grid)

        map_r_str = f'{map_r_val:>6.0f}' if map_r_val is not None else '     —'

        print(f'  {label:<16} {map_d:>9.0f}  {map_r_str:>11}  '
              f'{ci68_lo:>5.0f}–{ci68_hi:.0f} BCE          '
              f'{nw:>8,}')

        records.append({
            'unit':              label,
            'map_bce':           round(map_d, 0),
            'map_bce_resistant': round(map_r_val, 0) if map_r_val else None,
            'ci68_lo':           round(ci68_lo, 0),
            'ci68_hi':           round(ci68_hi, 0),
            'ci95_lo':           round(ci95_lo, 0) if w95 < WIDE_THRESH else None,
            'ci95_hi':           round(ci95_hi, 0) if w95 < WIDE_THRESH else None,
            'ci95_wide':         w95 >= WIDE_THRESH,
            'n_words':           nw,
        })

    # ------------------------------------------------------------------
    # 5. Archaism audit
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 5 — Archaism audit')
    print('='*70)

    x_obs_dict = {lbl: vec for lbl, (vec, _) in units.items()}
    arc_scores = compute_archaism_scores(x_obs_dict, ols_params, working_features)

    print(f'\n  {"Unit":<16} {"Mean LBH":>9}  {"Min":>7}  {"Max":>7}  '
          f'{"n_valid":>8}  {"Classification"}')
    print('  ' + '-'*75)

    archaism_records = []
    for label in ordered_units:
        sc = arc_scores.get(label, {})
        vals = [v for v in sc.values() if np.isfinite(v)]
        if not vals:
            continue
        mean_lbh = np.mean(vals)
        cls = ('Archaic (CBH-like)' if mean_lbh < 0.35 else
               'Mixed/selective'    if mean_lbh < 0.65 else
               'Modern (LBH-like)')
        print(f'  {label:<16} {mean_lbh:>9.3f}  '
              f'{min(vals):>7.3f}  {max(vals):>7.3f}  '
              f'{len(vals):>8}  {cls}')
        archaism_records.append({
            'unit': label, 'mean_lbh': mean_lbh,
            'min_lbh': min(vals), 'max_lbh': max(vals),
            'n_valid': len(vals), 'classification': cls,
        })

    # Tier-3 archaism scores for all units
    print('\n  Tier-3 feature archaism scores:')
    print(f'  {"Unit":<16}  {"frac_infc":>9}  {"frac_fronted":>12}  '
          f'{"frac_null_s":>11}  {"wqtl_wayq":>9}')
    print('  ' + '-'*70)
    for label in ordered_units:
        sc = arc_scores.get(label, {})
        vals = [f'{sc.get(tf, float("nan")):>9.3f}'
                if np.isfinite(sc.get(tf, float('nan'))) else '      nan'
                for tf in TIER3_FEATURES]
        print(f'  {label:<16}  {"  ".join(vals)}')

    # Model gap table
    print('\n  Full vs. resistant MAP comparison:')
    print(f'  {"Unit":<16}  {"Full":>6}  {"Resist":>7}  {"Δ":>5}  '
          f'Interpretation')
    print('  ' + '-'*65)
    for r in records:
        if r.get('map_bce_resistant') is None:
            continue
        delta = r['map_bce'] - r['map_bce_resistant']
        interp = ('syntax archaic vs lexicon' if delta < -50 else
                  'lexicon archaic vs syntax' if delta >  50 else
                  'models agree')
        print(f'  {r["unit"]:<16}  {r["map_bce"]:>6.0f}  '
              f'{r["map_bce_resistant"]:>7.0f}  {delta:>+5.0f}  {interp}')

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 6 — Save outputs')
    print('='*70)

    pd.DataFrame(records).to_csv(outdir / 'subsource_dating.csv', index=False)
    print('Dating results saved: subsource_dating.csv')

    pd.DataFrame(archaism_records).to_csv(outdir / 'subsource_archaism.csv',
                                           index=False)
    print('Archaism summary saved: subsource_archaism.csv')

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 7 — Plots')
    print('='*70)

    if date_grid is not None:
        # Which units belong to which group (only those we have posteriors for)
        plot_groups = {
            grp: [u for u in units_list if u in posteriors]
            for grp, units_list in UNIT_GROUPS.items()
        }

        plot_posteriors_grid(date_grid, posteriors, records, plot_groups,
                             str(outdir / 'subsource_posteriors.png'))

    heatmap_groups = {
        grp: [u for u in units_list if u in arc_scores]
        for grp, units_list in UNIT_GROUPS.items()
    }
    plot_archaism_comparison(
        arc_scores, working_features, heatmap_groups,
        str(outdir / 'subsource_archaism_heatmap.png'))

    plot_comparison_bars(records, str(outdir / 'subsource_comparison_bars.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
