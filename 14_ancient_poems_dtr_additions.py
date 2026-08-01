#!/usr/bin/env python3
"""
Script 14: Ancient Poems and DTR Additions Analysis
====================================================
Three focused investigations:

  1. DTR prose additions to Jeremiah
       Jer_DTR — chapters 7, 11, 17–18, 21, 24–29, 32–45, 52
       (the chapters excluded from the oracle-only Jeremiah in our training
       corpus, attributed to the Deuteronomistic school / Baruch's narrative)
       Compared with Dtr History books from script 13.

  2. Traditionally "ancient" songs
       Song_Sea     — Exodus 15          (433 words)
       Song_Deborah — Judges 5           (462 words)
       Compared with D_Song (Deut 32, 782 words) loaded from script 13 CSV.
       ⚠  CAUTION: results for sub-500-word units are noisy.

  3. Host books before and after removing the songs
       Exod_full / Exod_no_song  (Exodus minus ch. 15)
       Judg_full / Judg_no_song  (Judges minus ch. 5)
       Song is < 2 % (Exodus) / 3 % (Judges) of the host book, so the
       host-book dates are expected to barely move.

Usage
-----
    python 14_ancient_poems_dtr_additions.py [--data-path PATH] [--outdir DIR]
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
from pathlib import Path

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Analysis unit definitions
# ---------------------------------------------------------------------------

# DTR prose chapters of Jeremiah (inverse of oracle filter used in training)
JEREMIAH_ORACLE_CHAPTERS = set(
    list(range(1, 7)) + list(range(8, 11)) + list(range(12, 17)) +
    [19, 20] + list(range(22, 24)) + list(range(30, 32)) + list(range(46, 52))
)
JER_DTR_RANGES  = [(7,7), (11,11), (17,18), (21,21),
                   (24,29), (32,45), (52,52)]
JER_ORC_RANGES  = [(1,6), (8,10), (12,16), (19,20),
                   (22,23), (30,31), (46,51)]

NEW_UNITS = {
    # --- DTR Jeremiah ---
    'Jer_DTR':      [('Jeremiah', JER_DTR_RANGES)],
    'Jer_oracle':   [('Jeremiah', JER_ORC_RANGES)],   # reference
    # --- Ancient songs ---
    'Song_Sea':     [('Exodus',  [(15, 15)])],
    'Song_Deborah': [('Judges',  [(5,  5)])],
    # --- Host books with / without songs ---
    'Exod_full':    [('Exodus',  [(1, 40)])],
    'Exod_no_song': [('Exodus',  [(1, 14), (16, 40)])],
    'Judg_full':    [('Judges',  [(1, 21)])],
    'Judg_no_song': [('Judges',  [(1,  4), (6, 21)])],
}

# Word-count thresholds
MIN_WORDS_FULL   = 1000   # reliable for full model
MIN_WORDS_TIER3  = 200    # minimum for Tier-3 clause features

# Tier-3 features
TIER3_FEATURES = ['frac_infc', 'frac_fronted', 'frac_null_subj', 'frac_wqtl_wayq']

LINGUISTIC_POS = {
    'pronoun', 'dem-pronoun', 'prep', 'conj',
    'adverb', 'negator', 'article', 'interrogative', 'interjection',
}

# ---------------------------------------------------------------------------
# Clause-type constants
# ---------------------------------------------------------------------------
WAYQ_TYPES   = {'Way0', 'WayX', 'WaYX'}
WQTL_TYPES   = {'WQt0', 'WQtX', 'WxQ0', 'WxQX'}
WNARR_TYPES  = WAYQ_TYPES | WQTL_TYPES
FRONT_TYPES  = {'xQt0', 'xYq0', 'xQtX', 'xYqX', 'xIm0',
                'WxY0', 'WxQ0', 'WxQX', 'WxYX', 'WxI0',
                'XQtl', 'XYqt'}
CPEN_TYPES   = {'CPen'}
NMCL_TYPES   = {'NmCl', 'AjCl'}
PTCP_TYPES   = {'Ptcp'}
INFC_TYPES   = {'InfC'}
VERBAL_TYPES = (WAYQ_TYPES | WQTL_TYPES | FRONT_TYPES |
                {'ZQt0', 'ZQtX', 'ZYq0', 'ZYqX', 'ZIm0',
                 'WYq0', 'WYqX', 'WIm0', 'WXYq', 'WXQt',
                 'xQt0', 'xYq0', 'xQtX', 'xYqX', 'Way0', 'WayX'})
SKIP_TYPES   = {'Ellp', 'Voct', 'MSyn', 'InfA'}

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
# Feature extraction (same as scripts 11–13)
# ---------------------------------------------------------------------------
def words_for_ranges(book, ch_ranges, F, L, T):
    bn = T.nodeFromSection((book,))
    if bn is None:
        return
    for ch_node in L.d(bn, 'chapter'):
        ch = int(F.chapter.v(ch_node))
        if any(s <= ch <= e for s, e in ch_ranges):
            for w in L.d(ch_node, 'word'):
                yield (F.lex.v(w), F.sp.v(w), F.vt.v(w), F.vs.v(w),
                       F.nu.v(w),  F.gn.v(w), F.st.v(w), F.prs_ps.v(w))


def extract_word_features(book_ch_pairs, feature_names, F, L, T):
    rows = []
    for book, ch_ranges in book_ch_pairs:
        rows.extend(words_for_ranges(book, ch_ranges, F, L, T))
    if not rows:
        return None, 0
    df  = pd.DataFrame(rows, columns=['lex','sp','vt','vs','nu','gn','st','prs_ps'])
    n   = len(df)
    lc  = df['lex'].value_counts()
    vb  = df[df['sp'] == 'verb']
    nb  = df[df['sp'] == 'subs']
    vtc = vb['vt'].value_counts()
    vsc = vb['vs'].value_counts()

    result = {}
    for feat in feature_names:
        if feat in TIER3_FEATURES:
            result[feat] = np.nan
            continue
        if   feat.startswith('lex::'):   result[feat] = lc.get(feat[5:], 0) / n * 1000
        elif feat.startswith('vt::'):    result[feat] = vtc.get(feat[4:], 0) / n * 1000
        elif feat.startswith('vs::'):    result[feat] = vsc.get(feat[4:], 0) / n * 1000
        elif feat == 'morph::const_count':  result[feat] = (nb['st']=='c').sum() / n * 1000
        elif feat == 'morph::abs_count':    result[feat] = (nb['st']=='a').sum() / n * 1000
        elif feat == 'morph::prs_rate':     result[feat] = (df['prs_ps']!='NA').sum() / n * 1000
        elif feat == 'morph::noun_pl_rate': result[feat] = (nb['nu']=='pl').sum() / n * 1000
        elif feat == 'morph::noun_f_rate':  result[feat] = (nb['gn']=='f').sum() / n * 1000
        elif feat == 'frac_ani':
            a, b = lc.get('>NJ',0), lc.get('>NKJ',0)
            result[feat] = a/(a+b) if a+b > 0 else np.nan
        elif feat == 'frac_she':
            a, b = lc.get('C',0), lc.get('>CR',0)
            result[feat] = a/(a+b) if a+b > 0 else np.nan
        elif feat == 'rate_ut_nouns':
            ut = sum(v for k,v in lc.items() if k.endswith('WT/') or k.endswith('WT'))
            result[feat] = ut / n * 1000
        else:
            result[feat] = np.nan
    return np.array([result.get(f, np.nan) for f in feature_names]), n


def extract_tier3(book_ch_pairs, F, L, T):
    n_words = n_clauses = n_verbal = 0
    n_infc = n_fronted = n_cpen = 0
    n_wayq = n_wqtl = n_wnarr = 0
    n_sv = n_sv_total = n_null_subj = n_ov = n_ov_total = 0

    for book, ch_ranges in book_ch_pairs:
        bn = T.nodeFromSection((book,))
        if bn is None: continue
        for ch_node in L.d(bn, 'chapter'):
            ch = int(F.chapter.v(ch_node))
            if not any(s <= ch <= e for s, e in ch_ranges): continue
            for cl in L.d(ch_node, 'clause'):
                typ = F.typ.v(cl)
                if typ in SKIP_TYPES:
                    n_words += sum(1 for _ in L.d(cl, 'word'))
                    continue
                n_words   += sum(1 for _ in L.d(cl, 'word'))
                n_clauses += 1
                if typ in INFC_TYPES:  n_infc    += 1
                if typ in FRONT_TYPES: n_fronted += 1
                if typ in CPEN_TYPES:  n_cpen    += 1
                if typ in WAYQ_TYPES:  n_wayq    += 1
                if typ in WQTL_TYPES:  n_wqtl    += 1
                if typ in WNARR_TYPES: n_wnarr   += 1
                if typ in VERBAL_TYPES:
                    n_verbal += 1
                    phs = list(L.d(cl, 'phrase'))
                    pf  = {F.function.v(ph): ph for ph in phs}
                    if 'Subj' not in pf:  n_null_subj += 1
                    if 'Subj' in pf and 'Pred' in pf:
                        n_sv_total += 1
                        if pf['Subj'] < pf['Pred']: n_sv += 1
                    if 'Objc' in pf and 'Pred' in pf:
                        n_ov_total += 1
                        if pf['Objc'] < pf['Pred']: n_ov += 1

    if n_clauses == 0:
        return {f: np.nan for f in TIER3_FEATURES}
    return {
        'frac_infc':      n_infc    / n_clauses,
        'frac_fronted':   n_fronted / n_clauses,
        'frac_null_subj': n_null_subj / n_verbal if n_verbal > 10 else np.nan,
        'frac_wqtl_wayq': n_wqtl / n_wnarr       if n_wnarr  >  5 else np.nan,
    }


def extract_unit(label, book_ch_pairs, feature_names, F, L, T):
    vec, nw = extract_word_features(book_ch_pairs, feature_names, F, L, T)
    if vec is None:
        return np.full(len(feature_names), np.nan), 0
    t3 = extract_tier3(book_ch_pairs, F, L, T)
    for tf in TIER3_FEATURES:
        if tf in feature_names:
            vec[feature_names.index(tf)] = t3.get(tf, np.nan)
    return vec, nw

# ---------------------------------------------------------------------------
# MVN model
# ---------------------------------------------------------------------------
def build_mvn_model(rates_df, dates_bce, feature_names, ridge_frac=0.10):
    ols_params, residual_rows = {}, []
    for fn in feature_names:
        y = rates_df[fn].values.astype(float)
        x = np.array(dates_bce, dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 4:
            ols_params[fn] = (np.nanmean(y), 0.0, np.nanstd(y)+1e-9)
            residual_rows.append(np.zeros(len(dates_bce)))
            continue
        xv, yv = x[valid], y[valid]
        b, a = np.polyfit(xv, yv, 1)
        pred = a + b * x
        resid = y - pred; resid[~valid] = 0.0
        ols_params[fn] = (a, b, np.std(yv-(a+b*xv))+1e-9)
        residual_rows.append(resid)
    R = np.array(residual_rows).T
    Sigma = np.cov(R.T)
    if Sigma.ndim == 0: Sigma = np.array([[float(Sigma)]])
    K = Sigma.shape[0]
    ridge = ridge_frac * np.trace(Sigma) / K
    Sigma_reg = Sigma + ridge * np.eye(K)
    Sigma_inv = np.linalg.inv(Sigma_reg)
    print(f"    {K} features, cond={np.linalg.cond(Sigma_reg):.0f}")
    return ols_params, Sigma_inv


def mvn_log_lik(x_obs, date, ols_params, feature_names, Sigma_inv, nw=1):
    mu = np.array([ols_params[fn][0]+ols_params[fn][1]*date for fn in feature_names])
    v  = np.isfinite(x_obs)
    if v.sum() < 2: return -np.inf
    d = x_obs - mu; d[~v] = 0.0
    scale = float(np.clip(nw/5000, 1.0, 5.0))
    return -0.5 * scale * d @ Sigma_inv @ d


def compute_posterior(x_obs, nw, ols_params, feature_names, Sigma_inv,
                      date_min=1200, date_max=50, n_grid=500,
                      prior_mean=600, prior_sd=350):
    dg = np.linspace(date_max, date_min, n_grid)
    lp = -0.5 * ((dg - prior_mean)/prior_sd)**2
    ll = np.array([mvn_log_lik(x_obs, d, ols_params, feature_names, Sigma_inv, nw)
                   for d in dg])
    lpost = lp + ll; lpost -= lpost.max()
    post = np.exp(lpost); post /= post.sum()
    cdf  = np.cumsum(post)
    def q(p): return dg[min(np.searchsorted(cdf, p), n_grid-1)]
    return dg, post, dg[post.argmax()], q(0.16), q(0.84), q(0.025), q(0.975)


def compute_archaism(x_obs_dict, ols_params, feature_names,
                     archaic_date=720, modern_date=250):
    out = {}
    for unit, xo in x_obs_dict.items():
        d = {}
        for i, fn in enumerate(feature_names):
            a, b, _ = ols_params[fn]
            cbh = a + b*archaic_date; lbh = a + b*modern_date
            span = lbh - cbh
            if abs(span) < 1e-9 or not np.isfinite(xo[i]):
                d[fn] = np.nan
            else:
                d[fn] = (xo[i] - cbh) / span
        out[unit] = d
    return out


def filter_linguistic(robust_df):
    keep = []
    for _, row in robust_df.iterrows():
        feat = str(row['feature'])
        if feat.startswith(('vt::', 'vs::', 'morph::')):
            keep.append(True); continue
        if feat.startswith('lex::'):
            desc = str(row.get('description', ''))
            m    = re.search(r'\(([^)]+)\)', desc)
            pos  = m.group(1).strip() if m else ''
            keep.append(pos in LINGUISTIC_POS)
        else:
            keep.append(False)
    return robust_df[keep].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_songs_comparison(date_grid, posteriors, records,
                           prior_records, output_path):
    """
    Overlay posteriors for the three songs + D_Song reference.
    Also show a bar chart of MAP dates with 68% CI whiskers.
    """
    song_labels  = ['Song_Sea', 'Song_Deborah']
    ref_label    = 'D_Song'   # loaded from prior CSV

    all_labels = song_labels + [ref_label]
    colors = {'Song_Sea': '#e6550d', 'Song_Deborah': '#31a354', 'D_Song': '#756bb1'}
    linestyles = {'Song_Sea': '-', 'Song_Deborah': '-', 'D_Song': '--'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: posteriors
    ax = axes[0]
    for lbl in all_labels:
        if lbl in posteriors:
            post = posteriors[lbl]
            ax.plot(date_grid, post / post.max(),
                    color=colors[lbl], lw=2.2,
                    ls=linestyles[lbl], label=lbl.replace('_', ' '))
            ax.axvline(date_grid[post.argmax()], color=colors[lbl],
                       lw=1, ls=':', alpha=0.6)
        elif lbl in prior_records:   # D_Song from CSV — draw as vertical band
            pr = prior_records[lbl]
            ax.axvline(pr['map_bce'], color=colors[lbl], lw=2,
                       ls='--', label=f"D_Song (script 13 MAP={pr['map_bce']:.0f})")
            ax.axvspan(pr['ci68_lo'], pr['ci68_hi'],
                       alpha=0.12, color=colors[lbl])

    ax.set_xlim(date_grid.max(), date_grid.min())
    ax.set_xlabel('Date (BCE)')
    ax.set_ylabel('Normalised posterior')
    ax.set_title('Ancient songs — date posteriors\n'
                 '(⚠ Song of Sea / Deborah ≈ 433–462 words; results noisy)')
    ax.axvspan(760, 460, alpha=0.04, color='grey')
    ax.text(610, 0.95, 'Training\nrange', ha='center', va='top',
            fontsize=7, color='grey', transform=ax.get_xaxis_transform())
    ax.legend(fontsize=8)

    # Right: bar chart of MAP + 68 CI
    ax2 = axes[1]
    x   = np.arange(len(all_labels))
    for i, lbl in enumerate(all_labels):
        r = next((r for r in records if r['unit'] == lbl), None)
        if r is None and lbl in prior_records:
            r = prior_records[lbl]
        if r is None:
            continue
        col = colors[lbl]
        ax2.bar(i, r['map_bce'], 0.5, color=col, alpha=0.8)
        ax2.errorbar(i, r['map_bce'],
                     yerr=[[abs(r['map_bce']-r['ci68_lo'])],
                            [abs(r['ci68_hi']-r['map_bce'])]],
                     fmt='none', color='black', capsize=4)
        ax2.text(i, r['map_bce'] + 10, f"{r['map_bce']:.0f}",
                 ha='center', va='bottom', fontsize=8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([l.replace('_', '\n') for l in all_labels], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_ylabel('Date MAP (BCE, top = older)')
    ax2.set_title('MAP dates ± 68 % CI')
    ax2.axhline(760, color='grey', lw=0.8, ls='--', alpha=0.4)
    ax2.axhline(167, color='grey', lw=0.8, ls='--', alpha=0.4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Songs comparison plot saved: {output_path}')


def plot_dtr_comparison(date_grid, posteriors, records,
                         prior_records, output_path):
    """
    Jer_DTR vs. Jer_oracle alongside Dtr History books from script 13.
    """
    dtr_labels = ['Jer_DTR', 'Jer_oracle',
                  'Joshua', 'Judges', '1_Samuel', '2_Samuel', '1_Kings', '2_Kings']
    colors_map = {
        'Jer_DTR':    '#d62728', 'Jer_oracle': '#ff7f0e',
        'Joshua':     '#1f77b4', 'Judges':     '#9467bd',
        '1_Samuel':   '#2ca02c', '2_Samuel':   '#8c564b',
        '1_Kings':    '#e377c2', '2_Kings':    '#7f7f7f',
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for lbl in dtr_labels:
        col = colors_map.get(lbl, 'black')
        if lbl in posteriors:
            post = posteriors[lbl]
            ax.plot(date_grid, post / post.max(), color=col, lw=2,
                    label=lbl.replace('_', ' '))
            ax.axvline(date_grid[post.argmax()], color=col,
                       lw=0.9, ls=':', alpha=0.5)
        elif lbl in prior_records:
            pr = prior_records[lbl]
            ax.axvline(pr['map_bce'], color=col, lw=2, ls='--',
                       label=f"{lbl.replace('_',' ')} ({pr['map_bce']:.0f})")
            ax.axvspan(pr['ci68_lo'], pr['ci68_hi'], alpha=0.08, color=col)

    ax.set_xlim(date_grid.max(), date_grid.min())
    ax.set_xlabel('Date (BCE)')
    ax.set_ylabel('Normalised posterior')
    ax.set_title('Jer_DTR / Jer_oracle vs. Dtr History books')
    ax.axvspan(760, 460, alpha=0.04, color='grey')
    ax.legend(fontsize=7, ncol=2)

    # Bar chart
    ax2 = axes[1]
    all_records = {r['unit']: r for r in records}
    all_records.update(prior_records)

    xs, maps_full, maps_r, yerr_lo, yerr_hi, xlabs = [], [], [], [], [], []
    for i, lbl in enumerate(dtr_labels):
        r = all_records.get(lbl)
        if r is None: continue
        xs.append(i)
        maps_full.append(r['map_bce'])
        maps_r.append(r.get('map_bce_resistant') or r['map_bce'])
        yerr_lo.append(abs(r['map_bce'] - r['ci68_lo']))
        yerr_hi.append(abs(r['ci68_hi'] - r['map_bce']))
        xlabs.append(lbl.replace('_', '\n'))
        col = colors_map.get(lbl, 'black')
        ax2.bar(i-0.2, r['map_bce'], 0.35, color=col, alpha=0.85)
        ax2.errorbar(i-0.2, r['map_bce'],
                     yerr=[[abs(r['map_bce']-r['ci68_lo'])],
                            [abs(r['ci68_hi']-r['map_bce'])]],
                     fmt='none', color='black', capsize=3)
        mr = r.get('map_bce_resistant')
        if mr is not None:
            ax2.bar(i+0.2, mr, 0.35, color=col, alpha=0.35, hatch='//')

    ax2.set_xticks(range(len(dtr_labels)))
    ax2.set_xticklabels([l.replace('_','\n') for l in dtr_labels], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_ylabel('Date (BCE, top = older)')
    ax2.set_title('Full (solid) vs. resistant (hatched) MAPs')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'DTR comparison plot saved: {output_path}')


def plot_host_book_delta(records, prior_records, output_path):
    """
    Show how much Exodus/Judges dates shift when the ancient song is removed.
    """
    comparisons = [
        ('Exod_full', 'Exod_no_song', 'Exodus', '#e6550d'),
        ('Judg_full', 'Judg_no_song', 'Judges', '#31a354'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    all_rec = {r['unit']: r for r in records}
    all_rec.update(prior_records)

    for ax, (full_lbl, nosong_lbl, title, col) in zip(axes, comparisons):
        rf = all_rec.get(full_lbl)
        rn = all_rec.get(nosong_lbl)
        if rf is None or rn is None:
            ax.set_title(f'{title} — data missing')
            continue

        labels  = [f'{title}\n(full)', f'{title}\n(no song)']
        maps    = [rf['map_bce'], rn['map_bce']]
        ci_lo   = [rf['ci68_lo'], rn['ci68_lo']]
        ci_hi   = [rf['ci68_hi'], rn['ci68_hi']]

        ax.bar([0], [maps[0]], 0.4, color=col, alpha=0.85)
        ax.bar([1], [maps[1]], 0.4, color=col, alpha=0.45, hatch='//')
        for xi, (m, lo, hi) in enumerate(zip(maps, ci_lo, ci_hi)):
            ax.errorbar(xi, m,
                        yerr=[[abs(m-lo)], [abs(hi-m)]],
                        fmt='none', color='black', capsize=5)
            ax.text(xi, m+8, f'{m:.0f}', ha='center', fontsize=9, fontweight='bold')

        delta = rn['map_bce'] - rf['map_bce']
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_ylabel('Date MAP (BCE)')
        ax.set_title(f'{title}: Δ MAP = {delta:+.0f} yr after removing song')
        ax.axhspan(760, 460, alpha=0.04, color='grey')

    fig.suptitle('Effect of removing ancient song on host-book date estimate',
                 fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Host-book delta plot saved: {output_path}')


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
    parser.add_argument('--prior-csv',   default='subsource_dating.csv',
        help='Script 13 dating CSV (for D_Song and Dtr History reference)')
    parser.add_argument('--outdir',  default='.')
    parser.add_argument('--ridge',   type=float, default=0.10)
    parser.add_argument('--n-grid',  type=int,   default=500)
    args = parser.parse_args()

    outdir = Path(args.outdir)

    # ------------------------------------------------------------------
    # 1. Build working feature set (identical to scripts 11–13)
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 1 — Build working feature set and MVN models')
    print('='*70)

    robust_df = pd.read_csv(args.scan_robust)
    rates_df  = pd.read_csv(args.rates_csv, index_col='unit')
    dates_bce = rates_df['date_bce'].values.astype(float)
    rates_df  = rates_df.drop(columns=['date_bce','n_words'], errors='ignore')

    ling_df = filter_linguistic(robust_df)
    working_features = ling_df['feature'].tolist()
    for rf in ['frac_ani', 'frac_she', 'rate_ut_nouns']:
        if rf not in working_features:
            working_features.append(rf)

    df = rates_df.copy()
    ani = df.get('lex::>NJ', pd.Series(0., index=df.index))
    ank = df.get('lex::>NKJ', pd.Series(0., index=df.index))
    she = df.get('lex::C',    pd.Series(0., index=df.index))
    ash = df.get('lex::>CR',  pd.Series(0., index=df.index))
    df['frac_ani'] = np.where(ani+ank > 0, ani/(ani+ank), np.nan)
    df['frac_she'] = np.where(she+ash > 0, she/(she+ash), np.nan)
    ut_cols = [c for c in df.columns if re.match(r'lex::\w+WT/', c)]
    df['rate_ut_nouns'] = df[ut_cols].sum(axis=1) if ut_cols else 0.
    rates_aug = df

    tier3_path = Path(args.tier3_csv)
    if tier3_path.exists():
        t3 = pd.read_csv(tier3_path, index_col=0)
        for col in TIER3_FEATURES:
            if col in t3.columns:
                common = rates_aug.index.intersection(t3.index)
                rates_aug.loc[common, col] = t3.loc[common, col]
                if col not in working_features:
                    working_features.append(col)

    for f in working_features:
        if f not in rates_aug.columns:
            rates_aug[f] = np.nan

    resistant_features = [f for f in working_features if f in TIER3_FEATURES]
    print(f'  Working features: {len(working_features)}, Tier-3 resistant: {len(resistant_features)}')

    print('  Full model:')
    ols_params, Sigma_inv = build_mvn_model(
        rates_aug[working_features], dates_bce, working_features, args.ridge)

    ols_params_r = Sigma_inv_r = None
    if len(resistant_features) >= 2:
        print('  Resistant model:')
        rt = rates_aug[resistant_features].copy()
        for col in resistant_features:
            rt[col] = rt[col].fillna(rt[col].median())
        ols_params_r, Sigma_inv_r = build_mvn_model(
            rt, dates_bce, resistant_features, 0.20)

    # ------------------------------------------------------------------
    # 2. Load prior results for reference
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 2 — Load script 13 reference results')
    print('='*70)

    prior_records = {}
    prior_path = Path(args.prior_csv)
    if prior_path.exists():
        prior_df = pd.read_csv(prior_path)
        for _, row in prior_df.iterrows():
            prior_records[row['unit']] = row.to_dict()
        print(f'  Loaded {len(prior_records)} units from {prior_path.name}')
        ref_units = ['D_Song', 'Joshua', 'Judges', '1_Samuel',
                     '2_Samuel', '1_Kings', '2_Kings']
        for u in ref_units:
            if u in prior_records:
                pr = prior_records[u]
                print(f'    {u:<14} MAP={pr["map_bce"]:.0f}  '
                      f'CI68={pr["ci68_lo"]:.0f}–{pr["ci68_hi"]:.0f} BCE')
    else:
        print(f'  WARNING: {prior_path} not found — D_Song and Dtr History '
              f'references will be missing from plots.')

    # ------------------------------------------------------------------
    # 3. Load BHSA + extract features
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 3 — Extracting features from BHSA')
    print('='*70)

    api = load_bhsa(args.data_path)
    F, L, T = api.F, api.L, api.T

    units = {}
    TRAINING_RANGE = 760 - 167
    WIDE_THRESH    = TRAINING_RANGE * 0.75

    for label, pairs in NEW_UNITS.items():
        vec, nw = extract_unit(label, pairs, working_features, F, L, T)
        units[label] = (vec, nw)
        ani_v  = vec[working_features.index('frac_ani')]  if 'frac_ani'  in working_features else np.nan
        infc_v = vec[working_features.index('frac_infc')] if 'frac_infc' in working_features else np.nan
        flag   = ' ⚠ (< 500 words — noisy)' if nw < MIN_WORDS_FULL else ''
        print(f'  {label:<16} {nw:>7,} words  '
              f'frac_אני={ani_v:.3f}  frac_infc={infc_v:.3f}{flag}')

    # ------------------------------------------------------------------
    # 4. Date posteriors
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 4 — Date posteriors')
    print('='*70)

    posteriors  = {}
    records     = []
    date_grid   = None

    print(f'  {"Unit":<16} {"MAP(full)":>9}  {"MAP(resist)":>11}  '
          f'{"68% CI":>20}  {"n_words":>8}  note')
    print('  ' + '-'*85)

    for label, (x_obs, nw) in units.items():
        dg, post, map_d, ci68_lo, ci68_hi, ci95_lo, ci95_hi = compute_posterior(
            x_obs, nw, ols_params, working_features, Sigma_inv, n_grid=args.n_grid)
        posteriors[label] = post
        date_grid = dg

        map_r_val = None
        if ols_params_r is not None:
            x_r = np.array([x_obs[working_features.index(f)]
                             if f in working_features else np.nan
                             for f in resistant_features])
            _, _, map_r_val, *_ = compute_posterior(
                x_r, nw, ols_params_r, resistant_features, Sigma_inv_r,
                n_grid=args.n_grid)

        map_r_str = f'{map_r_val:>6.0f}' if map_r_val is not None else '     —'
        note = '⚠ noisy' if nw < MIN_WORDS_FULL else ''

        print(f'  {label:<16} {map_d:>9.0f}  {map_r_str:>11}  '
              f'{ci68_lo:>5.0f}–{ci68_hi:.0f} BCE          '
              f'{nw:>8,}  {note}')

        records.append({
            'unit':              label,
            'map_bce':           round(map_d,    0),
            'map_bce_resistant': round(map_r_val,0) if map_r_val else None,
            'ci68_lo':           round(ci68_lo,  0),
            'ci68_hi':           round(ci68_hi,  0),
            'n_words':           nw,
            'noisy':             nw < MIN_WORDS_FULL,
        })

    # ------------------------------------------------------------------
    # 5. Archaism audit
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 5 — Archaism audit')
    print('='*70)

    x_obs_dict = {lbl: vec for lbl, (vec, _) in units.items()}
    arc_scores  = compute_archaism(x_obs_dict, ols_params, working_features)

    print(f'\n  {"Unit":<16} {"Mean LBH":>9}  {"Min":>7}  {"Max":>7}  '
          f'{"n_valid":>8}  {"Classification"}  note')
    print('  ' + '-'*85)

    arc_records = []
    for label in NEW_UNITS:
        sc   = arc_scores.get(label, {})
        vals = [v for v in sc.values() if np.isfinite(v)]
        if not vals: continue
        mean_lbh = np.mean(vals)
        cls = ('Archaic (CBH-like)' if mean_lbh < 0.35 else
               'Mixed/selective'    if mean_lbh < 0.65 else
               'Modern (LBH-like)')
        nw   = units[label][1]
        note = '⚠ noisy' if nw < MIN_WORDS_FULL else ''
        print(f'  {label:<16} {mean_lbh:>9.3f}  '
              f'{min(vals):>7.3f}  {max(vals):>7.3f}  '
              f'{len(vals):>8}  {cls:<20} {note}')
        arc_records.append({'unit': label, 'mean_lbh': mean_lbh,
                             'classification': cls, 'n_words': nw})

    # Tier-3 scores
    print('\n  Tier-3 archaism scores:')
    print(f'  {"Unit":<16}  {"frac_infc":>9}  {"frac_fronted":>12}  '
          f'{"frac_null_s":>11}  {"wqtl_wayq":>9}')
    print('  ' + '-'*65)
    for label in NEW_UNITS:
        sc   = arc_scores.get(label, {})
        nw   = units[label][1]
        flag = ' ⚠' if nw < MIN_WORDS_FULL else ''
        vals = [f'{sc.get(tf, float("nan")):>9.3f}'
                if np.isfinite(sc.get(tf, float('nan'))) else '      nan'
                for tf in TIER3_FEATURES]
        print(f'  {label:<16}  {"  ".join(vals)}{flag}')

    # Model gap
    print('\n  Full vs. resistant MAP gap:')
    print(f'  {"Unit":<16}  {"Full":>6}  {"Resist":>7}  {"Δ":>6}  Interpretation')
    print('  ' + '-'*65)
    for r in records:
        mr = r.get('map_bce_resistant')
        if mr is None: continue
        delta  = r['map_bce'] - mr
        interp = ('syntax archaic vs lexicon' if delta < -50 else
                  'lexicon archaic vs syntax' if delta >  50 else
                  'models agree')
        flag = ' ⚠' if r.get('noisy') else ''
        print(f'  {r["unit"]:<16}  {r["map_bce"]:>6.0f}  '
              f'{mr:>7.0f}  {delta:>+6.0f}  {interp}{flag}')

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 6 — Save outputs')
    print('='*70)

    pd.DataFrame(records).to_csv(outdir / 'poem_dtr_dating.csv', index=False)
    pd.DataFrame(arc_records).to_csv(outdir / 'poem_dtr_archaism.csv', index=False)
    print('Results saved: poem_dtr_dating.csv  poem_dtr_archaism.csv')

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    print('\n' + '='*70)
    print('STEP 7 — Plots')
    print('='*70)

    if date_grid is not None:
        plot_songs_comparison(date_grid, posteriors, records,
                              prior_records,
                              str(outdir / 'song_comparison.png'))
        plot_dtr_comparison(date_grid, posteriors, records,
                            prior_records,
                            str(outdir / 'dtr_jer_comparison.png'))

    plot_host_book_delta(records, prior_records,
                         str(outdir / 'host_book_delta.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
