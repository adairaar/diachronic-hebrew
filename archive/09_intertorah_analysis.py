#!/usr/bin/env python3
"""
Inter-Torah Source Analysis and Archaism Audit
================================================
Two questions:

1. INTER-TORAH CLUSTERING
   Do the hypothetical documentary sources (D, P, JE) produce linguistically
   coherent groups when analysed with diachronic features?  If scholarly
   source attributions are right, D and P should cluster near different parts
   of the CBH–LBH spectrum.  If a "source" is internally inconsistent, its
   feature profile will be scattered.

2. ARCHAISM AUDIT
   For each source, and especially for Deuteronomy, we ask: which features
   look archaic (CBH-like) and which look modern (LBH-like)?  A source that
   deliberately archaizes would show high archaism on some well-known markers
   (e.g. אנכי pronoun) but normal/modern values on others (e.g. preposition
   patterns, conjunction rates) — features that are harder to consciously
   manipulate.  This produces a "mixed archaism fingerprint."

Method
------
For each feature with a temporal trend in the training corpus, we compute a
"LBH-ness score" for each documentary source:

    LBH-ness(source, feature) =
        (obs_value - archaic_end) / (modern_end - archaic_end)

where archaic_end = value at the MOST ARCHAIC training unit for that feature
      modern_end  = value at the MOST MODERN training unit for that feature

  score = 0   → as archaic as the most archaic training text
  score = 1   → as modern as the most modern training text
  score < 0   → MORE archaic than any training text (extrapolated)
  score > 1   → MORE modern than any training text (extrapolated)

A source with uniform archaism would have all scores near 0 or below.
A source with SELECTIVE archaism would have some scores near/below 0
(the features it archaizes on) and other scores at 0.5–1 or above
(the features it doesn't bother to archaize).

Documentary sources used
------------------------
Based on broad consensus; chapter-level attributions are approximate.
See Friedman (2003) and Baden (2012) for the specific assignments used.

Outputs
-------
source_feature_profiles.csv   — LBH-ness scores for all sources × features
archaism_audit.png             — forest plot of LBH-ness scores per source
source_pca.png                 — PCA of sources + training corpus
cluster_heatmap.png            — feature heatmap for sources + training
source_stats.csv               — summary statistics per source
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from pathlib import Path

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Training corpus definitions (same as scripts 06/08)
# ---------------------------------------------------------------------------

# Jeremiah oracle chapters (non-DTR: excludes 7,11,17-18,21,24-29,32-45)
JEREMIAH_ORACLE_CHAPTERS = set(
    list(range(1, 7)) + list(range(8, 11)) + list(range(12, 17)) +
    [19, 20] + list(range(22, 24)) + list(range(30, 32)) + list(range(46, 52))
)

PROPHETIC_SPECS = [
    ('Amos',        ['Amos'],        None,       760, 15),
    ('Hosea',       ['Hosea'],       None,       725, 20),
    ('Micah',       ['Micah'],       None,       720, 20),
    ('Isaiah_1',    ['Isaiah'],      (1,  39),   700, 15),
    ('Zephaniah',   ['Zephaniah'],   None,       630, 15),
    ('Nahum',       ['Nahum'],       None,       620, 20),
    ('Habakkuk',    ['Habakkuk'],    None,       605, 20),
    ('Jeremiah',    ['Jeremiah'],    'oracle',   590, 15),
    ('Lamentations',['Lamentations'],None,       586, 20),
    ('Ezekiel',     ['Ezekiel'],     None,       570, 15),
    ('Isaiah_2',    ['Isaiah'],      (40, 55),   550, 20),
    ('Haggai',      ['Haggai'],      None,       520,  5),
    ('Zechariah_1', ['Zechariah'],   (1,   8),   518,  5),
    ('Isaiah_3',    ['Isaiah'],      (56, 66),   460, 100),
    ('Malachi',     ['Malachi'],     None,       460, 20),
]

BROAD_SPECS = PROPHETIC_SPECS + [
    ('Jonah',       ['Jonah'],                       None,  400, 50),
    ('Ezra',        ['Ezra'],                        None,  400, 75),
    ('Nehemiah',    ['Nehemiah'],                    None,  420, 75),
    ('Chronicles',  ['1_Chronicles', '2_Chronicles'],None,  350, 50),
    ('Esther',      ['Esther'],                      None,  350, 50),
    ('Ecclesiastes',['Ecclesiastes'],                None,  330, 80),
    ('Daniel',      ['Daniel'],                      None,  167, 10),
]

DEFAULT_BHSA_PATH = (
    Path.home() / 'text-fabric-data' / 'github' / 'ETCBC' / 'bhsa' / 'tf' / '2021'
)

TF_FEATURES = 'otype oslots otext lex sp vt vs nu gn st prs_ps chapter'

# ---------------------------------------------------------------------------
# Documentary source chapter-range attributions
# ---------------------------------------------------------------------------
# Based on Friedman (2003) "Who Wrote the Bible?" and Baden (2012).
# Chapter ranges are approximate; the Documentary Hypothesis is contested and
# any individual verse assignment is uncertain.  We use whole-chapter blocks.
#
# Format: book → { source_label: [(start_ch, end_ch), ...], ... }

DOC_SOURCES = {
    'Genesis': {
        'P':  [(1,  2), (5, 5), (6, 6), (7, 7), (9, 9), (11, 11),
               (17,17), (23,23), (25,25), (27,28), (35,36), (46,46), (49,50)],
        'JE': [(2,  4), (6, 6), (8, 8), (10,10), (12,16), (18,22),
               (24,24), (26,27), (29,34), (37,45), (47,49)],
    },
    'Exodus': {
        'P':  [(1, 2), (6, 7), (12,12), (16,16), (25,31), (35,40)],
        'JE': [(2, 5), (8,11), (13,15), (17,18), (19,24), (32,34)],
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

# Aggregated sources: each element is a label and a list of (book, ch_ranges)
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

SOURCE_COLORS = {
    'D':       '#1f77b4',   # blue
    'P':       '#d62728',   # red
    'JE':      '#2ca02c',   # green
    'Gen:P':   '#ff7f0e',
    'Gen:JE':  '#9467bd',
    'Exo:P':   '#8c564b',
    'Exo:JE':  '#e377c2',
    'Lev:P':   '#d62728',
    'Num:P':   '#bcbd22',
    'Num:JE':  '#17becf',
    'Deu:D':   '#1f77b4',
}

# ---------------------------------------------------------------------------
# Feature catalogue (same as script 08 — theoretically motivated features)
# ---------------------------------------------------------------------------

FEATURE_CATALOGUE = [
    ('frac_ani',        'Fraction אני/(אני+אנכי)',    'fraction_lex',  ['>NJ',  '>NKJ'],   'increase'),
    ('rate_anochi',     'אנכי rate per 1k',           'rate_lex',      ['>NKJ'],            'decrease'),
    ('rate_ani',        'אני rate per 1k',            'rate_lex',      ['>NJ'],             'increase'),
    ('frac_she',        'Fraction ש/(ש+אשר)',         'fraction_lex',  ['C',    '>CR'],     'increase'),
    ('rate_asher',      'אשר rate per 1k',            'rate_lex',      ['>CR'],             'decrease'),
    ('frac_ein',        'Fraction אין/(אין+לא)',      'fraction_lex',  ['>JN/', 'L>'],      'increase'),
    ('rate_neg_al',     'אל negator rate per 1k',     'rate_nega',     ['>L'],              'decrease'),
    ('frac_neg_al',     'Fraction אל/(אל+לא)',        'fraction_nega', ['>L',  'L>'],       'decrease'),
    ('rate_neg_lo',     'לא negator rate per 1k',     'rate_nega',     ['L>'],              'unknown'),
    ('rate_ki',         'כי conj rate per 1k',        'rate_lex_sp',   ['KJ', 'conj'],      'unknown'),
    ('rate_gam',        'גם rate per 1k',             'rate_lex',      ['GM'],              'unknown'),
    ('rate_lakhen',     'לכן rate per 1k',            'rate_lex',      ['LKN'],             'increase'),
    ('rate_hinne',      'הנה rate per 1k',            'rate_lex',      ['HNH/'],            'unknown'),
    ('rate_atta',       'עתה rate per 1k',            'rate_lex',      ['<TH/'],            'unknown'),
    ('rate_az',         'אז rate per 1k',             'rate_lex',      ['>Z'],              'decrease'),
    ('rate_af',         'אף rate per 1k',             'rate_lex',      ['>P/'],             'unknown'),
    ('rate_wayyiqtol',  'Wayyiqtol rate per 1k',      'rate_vt',       ['wayq'],            'decrease'),
    ('rate_qatal',      'Qatal rate per 1k',          'rate_vt',       ['perf'],            'unknown'),
    ('rate_yiqtol',     'Yiqtol rate per 1k',         'rate_vt',       ['impf'],            'unknown'),
    ('rate_ptca',       'Active participle per 1k',   'rate_vt',       ['ptca'],            'increase'),
    ('rate_inf_abs',    'Inf. absolute per 1k',       'rate_vt',       ['infa'],            'decrease'),
    ('rate_inf_con',    'Inf. construct per 1k',      'rate_vt',       ['infc'],            'unknown'),
    ('rate_impv',       'Imperative per 1k',          'rate_vt',       ['impv'],            'unknown'),
    ('rate_qal',        'Qal stem rate per 1k',       'rate_vs',       ['qal'],             'unknown'),
    ('rate_hiphil',     'Hiphil stem rate per 1k',    'rate_vs',       ['hif'],             'unknown'),
    ('rate_piel',       'Piel stem rate per 1k',      'rate_vs',       ['piel'],            'unknown'),
    ('rate_niphal',     'Niphal stem rate per 1k',    'rate_vs',       ['nif'],             'unknown'),
    ('rate_hithpael',   'Hithpael stem rate per 1k',  'rate_vs',       ['hit'],             'unknown'),
    ('frac_baqash',     'Fraction בקש/(בקש+שאל)',     'fraction_lex',  ['BQC[', 'C>L['],   'increase'),
    ('rate_baqash',     'בקש rate per 1k',            'rate_lex',      ['BQC['],            'increase'),
    ('rate_shaal',      'שאל rate per 1k',            'rate_lex',      ['C>L['],            'decrease'),
    ('frac_qahal',      'Fraction קהל/(קהל+עדה)',     'fraction_lex',  ['QHL/', '<DH/'],    'increase'),
    ('rate_hayah',      'היה rate per 1k',            'rate_lex',      ['HJH['],            'unknown'),
    ('rate_amar',       'אמר rate per 1k',            'rate_lex',      ['>MR['],            'unknown'),
    ('rate_halak',      'הלך rate per 1k',            'rate_lex',      ['HLK['],            'unknown'),
    ('rate_const',      'Construct state per 1k',     'morph_const',   [],                  'unknown'),
    ('rate_prs',        'Pron. suffix rate per 1k',   'morph_prs',     [],                  'unknown'),
    ('rate_pl_noun',    'Plural noun rate per 1k',    'morph_pl_noun', [],                  'unknown'),
    ('rate_f_noun',     'Feminine noun rate per 1k',  'morph_f_noun',  [],                  'unknown'),
]


# ---------------------------------------------------------------------------
# BHSA loading and feature extraction
# ---------------------------------------------------------------------------

def load_bhsa(data_path):
    try:
        from tf.fabric import Fabric
    except ImportError:
        print("ERROR: pip install text-fabric"); sys.exit(1)
    p = Path(data_path).expanduser()
    if not p.exists():
        print(f"ERROR: BHSA not found at {p}"); sys.exit(1)
    TF  = Fabric(locations=str(p), silent=True)
    api = TF.load(TF_FEATURES, silent=True)
    return api.F, api.L, api.T


def words_for_ranges(book, ch_ranges, F, L, T):
    """Return a list of word rows (lex,sp,vt,vs,nu,gn,st,prs_ps) for a
    set of chapter ranges within a single book."""
    bn = T.nodeFromSection((book,))
    if bn is None:
        return []
    rows = []
    for ch in L.d(bn, 'chapter'):
        ch_num = F.chapter.v(ch)
        if not any(s <= ch_num <= e for s, e in ch_ranges):
            continue
        for w in L.d(ch, 'word'):
            rows.append((
                F.lex.v(w), F.sp.v(w), F.vt.v(w), F.vs.v(w),
                F.nu.v(w),  F.gn.v(w), F.st.v(w), F.prs_ps.v(w),
            ))
    return rows


def extract_features_from_rows(rows):
    """Compute all FEATURE_CATALOGUE entries from a list of word rows."""
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['lex','sp','vt','vs','nu','gn','st','prs_ps'])
    n  = len(df)
    per = 1000.0 / max(n, 1)

    verb_df = df[df['sp'] == 'verb']
    noun_df = df[df['sp'] == 'subs']
    nega_df = df[df['sp'] == 'nega']

    out = {'n_words': n}
    for (name, desc, ext, params, direction) in FEATURE_CATALOGUE:
        if ext == 'rate_lex':
            out[name] = df['lex'].isin(params).sum() * per
        elif ext == 'rate_lex_sp':
            lx, sp = params[:-1], params[-1]
            out[name] = (df['lex'].isin(lx) & (df['sp'] == sp)).sum() * per
        elif ext == 'fraction_lex':
            num = (df['lex'] == params[0]).sum()
            den = (df['lex'] == params[1]).sum()
            out[name] = num / (num + den) if (num + den) > 0 else float('nan')
        elif ext == 'rate_nega':
            out[name] = nega_df['lex'].isin(params).sum() * per
        elif ext == 'fraction_nega':
            num = (nega_df['lex'] == params[0]).sum()
            den = (nega_df['lex'] == params[1]).sum()
            out[name] = num / (num + den) if (num + den) > 0 else float('nan')
        elif ext == 'rate_vt':
            out[name] = verb_df['vt'].isin(params).sum() * per
        elif ext == 'rate_vs':
            out[name] = verb_df['vs'].isin(params).sum() * per
        elif ext == 'morph_const':
            out[name] = (noun_df['st'] == 'c').sum() * per
        elif ext == 'morph_prs':
            out[name] = (df['prs_ps'] != 'NA').sum() * per
        elif ext == 'morph_pl_noun':
            out[name] = (noun_df['nu'] == 'pl').sum() * per
        elif ext == 'morph_f_noun':
            out[name] = (noun_df['gn'] == 'f').sum() * per
    return out


def load_training_corpus(specs, F, L, T):
    """Extract features for every unit in specs. Returns (meta_df, rates_df)."""
    meta_rows, rate_rows = [], []
    # Contiguous-range representation of oracle chapters for Jeremiah
    JEREMIAH_ORACLE_RANGES = [(1,6),(8,10),(12,16),(19,20),(22,23),(30,31),(46,51)]

    for (name, books, chap_range, date, sigma) in specs:
        all_rows = []
        for bk in books:
            if chap_range == 'oracle':
                ch_ranges = JEREMIAH_ORACLE_RANGES
            elif chap_range:
                ch_ranges = [chap_range]
            else:
                bn = T.nodeFromSection((bk,))
                if bn is None:
                    continue
                ch_nums = [F.chapter.v(c) for c in L.d(bn, 'chapter')]
                ch_ranges = [(min(ch_nums), max(ch_nums))]
            all_rows.extend(words_for_ranges(bk, ch_ranges, F, L, T))
        feat = extract_features_from_rows(all_rows)
        if feat is None:
            print(f"  WARNING: no words for {name}")
            continue
        meta_rows.append({'unit': name, 'date_bce': date, 'date_sigma': sigma})
        rate_rows.append({'unit': name, **feat})

    meta_df  = pd.DataFrame(meta_rows).set_index('unit')
    rates_df = pd.DataFrame(rate_rows).set_index('unit')
    return meta_df, rates_df


def load_doc_sources(F, L, T):
    """
    Extract features for each individual source section and for the three
    aggregated sources (D, P, JE).
    Returns two dicts: individual_rates, aggregated_rates
    Each maps label → feature_dict.
    """
    individual_rates = {}
    for book, sources in DOC_SOURCES.items():
        for src_label, ch_ranges in sources.items():
            label = f"{book[:3]}:{src_label}"
            rows  = words_for_ranges(book, ch_ranges, F, L, T)
            feat  = extract_features_from_rows(rows)
            if feat:
                individual_rates[label] = feat

    aggregated_rates = {}
    for src_label, parts in AGGREGATED_SOURCES.items():
        all_rows = []
        for book, ch_ranges in parts:
            all_rows.extend(words_for_ranges(book, ch_ranges, F, L, T))
        feat = extract_features_from_rows(all_rows)
        if feat:
            aggregated_rates[src_label] = feat

    return individual_rates, aggregated_rates


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_correlations(rates_df, dates):
    """Spearman ρ vs date for each feature."""
    results = {}
    for col in rates_df.columns:
        if col == 'n_words':
            continue
        vals  = rates_df[col].values.astype(float)
        valid = np.isfinite(vals)
        if valid.sum() < 5 or vals[valid].std() < 1e-9:
            continue
        r, p = spearmanr(-dates[valid], vals[valid])
        results[col] = {'rho': r, 'p_raw': p}
    return results


def compute_lbh_scores(source_feats, rates_df, dates, corr, min_rho=0.25):
    """
    For each feature with |ρ| ≥ min_rho, compute the LBH-ness score for a
    source relative to the training corpus range.

    LBH-ness = (obs - archaic_end) / (modern_end - archaic_end)

    where archaic_end = value at most archaic training unit (highest date_bce)
          modern_end  = value at most modern training unit (lowest date_bce)

    For features that DECREASE with time (rho < 0): archaic_end = max, modern_end = min
    For features that INCREASE with time (rho > 0): archaic_end = min, modern_end = max

    Returns dict: feature → LBH-ness score
    """
    # Sort training by date (oldest first)
    date_order = np.argsort(-dates)   # descending date_bce = oldest first
    sorted_df   = rates_df.iloc[date_order]

    scores = {}
    for col, stats in corr.items():
        if abs(stats['rho']) < min_rho:
            continue
        if col not in source_feats or col not in sorted_df.columns:
            continue
        obs = source_feats[col]
        if not np.isfinite(obs):
            continue

        train_vals = sorted_df[col].values.astype(float)
        valid = np.isfinite(train_vals)
        if valid.sum() < 3:
            continue

        tv = train_vals[valid]
        if stats['rho'] > 0:
            # increasing → archaic_end = min value, modern_end = max value
            archaic_end = tv.min()
            modern_end  = tv.max()
        else:
            # decreasing → archaic_end = max value, modern_end = min value
            archaic_end = tv.max()
            modern_end  = tv.min()

        denom = modern_end - archaic_end
        if abs(denom) < 1e-9:
            continue

        scores[col] = (obs - archaic_end) / denom

    return scores


def feature_display_name(feat_name):
    """Map feature name to short human-readable label."""
    for (name, desc, *_) in FEATURE_CATALOGUE:
        if name == feat_name:
            return desc
    return feat_name


# ---------------------------------------------------------------------------
# Plot 1: Archaism audit — LBH-ness forest plot
# ---------------------------------------------------------------------------

def plot_archaism_audit(scores_by_source, source_colors, title_suffix, outpath,
                        direction_map, min_rho=0.25):
    """
    Forest plot (dot plot) showing LBH-ness scores for each source on each
    feature.

    Layout: one row per feature; x-axis = LBH-ness score.
    Background bands:
      gray   = [0, 1]  — within the training range
      left   = < 0     — more archaic than any training text
      right  = > 1     — more modern than any training text
    """
    # Collect all features that appear in at least one source
    all_feats = set()
    for sc in scores_by_source.values():
        all_feats.update(sc.keys())

    # Sort features: first by expected direction (increase → top), then by
    # median LBH-ness score (so archaic features cluster at left)
    def sort_key(f):
        vals = [sc[f] for sc in scores_by_source.values() if f in sc]
        return np.nanmedian(vals) if vals else 0
    sorted_feats = sorted(all_feats, key=sort_key)

    n_feats   = len(sorted_feats)
    n_sources = len(scores_by_source)
    fig_h     = max(6, n_feats * 0.45 + 2)
    fig, ax   = plt.subplots(figsize=(12, fig_h))

    # Background shading
    ax.axvspan(-0.5, 0, alpha=0.08, color='steelblue',  zorder=0)
    ax.axvspan( 1,   1.5, alpha=0.08, color='firebrick', zorder=0)
    ax.axvline(0, color='steelblue', linewidth=1.2, linestyle='--', alpha=0.7,
               label='Most archaic training text')
    ax.axvline(1, color='firebrick', linewidth=1.2, linestyle='--', alpha=0.7,
               label='Most modern training text')
    ax.axvline(0.5, color='gray', linewidth=0.6, linestyle=':', alpha=0.5)

    # One row per feature
    y_positions = np.arange(n_feats)
    source_list = list(scores_by_source.keys())
    offsets     = np.linspace(-0.18, 0.18, len(source_list)) if len(source_list) > 1 else [0]

    for si, src in enumerate(source_list):
        sc    = scores_by_source[src]
        color = source_colors.get(src, 'gray')
        xs, ys = [], []
        for fi, feat in enumerate(sorted_feats):
            if feat in sc and np.isfinite(sc[feat]):
                xs.append(sc[feat])
                ys.append(y_positions[fi] + offsets[si])
        ax.scatter(xs, ys, color=color, s=55, zorder=3,
                   label=src, alpha=0.85, edgecolors='white', linewidths=0.4)

    # Connect dots across sources for each feature
    for fi, feat in enumerate(sorted_feats):
        vals = [scores_by_source[src].get(feat, np.nan) for src in source_list]
        valid_vals = [v for v in vals if np.isfinite(v)]
        if len(valid_vals) > 1:
            valid_y = [y_positions[fi] + offsets[si]
                       for si, src in enumerate(source_list)
                       if np.isfinite(scores_by_source[src].get(feat, np.nan))]
            ax.plot(valid_vals, valid_y, color='gray', linewidth=0.5,
                    alpha=0.4, zorder=2)

    # Feature labels on y-axis
    ax.set_yticks(y_positions)
    ax.set_yticklabels([feature_display_name(f) for f in sorted_feats],
                       fontsize=8)

    # Expected-direction indicators
    for fi, feat in enumerate(sorted_feats):
        exp = direction_map.get(feat, 'unknown')
        marker = '↑' if exp == 'increase' else ('↓' if exp == 'decrease' else '·')
        color  = '#2ca02c' if exp == 'increase' else ('#d62728' if exp == 'decrease' else 'gray')
        ax.text(-0.52, y_positions[fi], marker, fontsize=8,
                color=color, va='center', ha='center', fontweight='bold')

    ax.set_xlim(-0.55, 1.55)
    ax.set_xlabel('LBH-ness score\n'
                  '(0 = as archaic as most archaic training text; '
                  '1 = as modern as most modern training text;\n'
                  '<0 = more archaic than any training text; '
                  '>1 = more modern)', fontsize=9)
    ax.set_title(f'Archaism Audit: LBH-ness scores per feature  {title_suffix}\n'
                 '↑/↓ = expected LBH direction from theory',
                 fontsize=10)

    # Add a legend label for the background zones
    patch_arch  = mpatches.Patch(color='steelblue', alpha=0.2,
                                 label='More archaic than training range')
    patch_mod   = mpatches.Patch(color='firebrick', alpha=0.2,
                                 label='More modern than training range')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [patch_arch, patch_mod],
              labels  + ['More archaic than training range',
                          'More modern than training range'],
              loc='lower right', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(str(outpath), dpi=150)
    plt.close()
    print(f"Archaism audit plot saved: {outpath.name}")


# ---------------------------------------------------------------------------
# Plot 2: PCA of sources + training corpus
# ---------------------------------------------------------------------------

def plot_pca(rates_df, meta_df, source_rates_dict, source_colors, outpath,
             n_features=15):
    """
    PCA using the top n_features (by variance) from the training corpus.
    Training units are plotted as circles colored by date.
    Documentary sources are plotted as diamond markers with labels.
    """
    # Select features: most variance in training
    feat_cols = [c for c in rates_df.columns
                 if c != 'n_words' and rates_df[c].std() > 0.01]
    if len(feat_cols) > n_features:
        variances  = rates_df[feat_cols].std()
        feat_cols  = variances.nlargest(n_features).index.tolist()

    # Standardize training data
    train_mat  = rates_df[feat_cols].values.astype(float)
    means      = np.nanmean(train_mat, axis=0)
    stds       = np.nanstd(train_mat, axis=0)
    stds[stds < 1e-9] = 1.0
    train_std  = (train_mat - means) / stds

    # Replace NaNs with column means (for PCA)
    for j in range(train_std.shape[1]):
        mask = ~np.isfinite(train_std[:, j])
        train_std[mask, j] = 0.0

    # PCA via eigendecomposition of covariance
    cov      = np.cov(train_std.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order    = np.argsort(eigvals)[::-1]
    eigvecs  = eigvecs[:, order]
    pcs      = train_std @ eigvecs
    var_exp  = eigvals[order] / eigvals.sum() * 100

    fig, ax  = plt.subplots(figsize=(10, 8))

    # Training corpus: color by date
    dates    = meta_df['date_bce'].values
    cmap     = plt.cm.RdYlBu_r
    norm     = plt.Normalize(dates.min(), dates.max())
    sc       = ax.scatter(pcs[:, 0], pcs[:, 1], c=dates, cmap=cmap,
                          norm=norm, s=80, zorder=3, edgecolors='k',
                          linewidths=0.5, alpha=0.85)
    for i, name in enumerate(meta_df.index):
        ax.annotate(name[:9], (pcs[i, 0], pcs[i, 1]),
                    xytext=(3, 3), textcoords='offset points', fontsize=6)
    plt.colorbar(sc, ax=ax, label='Date BCE (darker = more recent)', shrink=0.7)

    # Documentary sources
    for src_label, feats in source_rates_dict.items():
        row = np.array([feats.get(f, np.nan) for f in feat_cols], dtype=float)
        row = np.where(np.isfinite(row), row, means)
        row_std = (row - means) / stds
        proj    = row_std @ eigvecs
        color   = source_colors.get(src_label, 'gray')
        ax.scatter(proj[0], proj[1], marker='D', s=200, color=color,
                   zorder=5, edgecolors='black', linewidths=1.2)
        ax.annotate(src_label, (proj[0], proj[1]),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=color)

    ax.set_xlabel(f'PC1 ({var_exp[0]:.1f}% variance)', fontsize=10)
    ax.set_ylabel(f'PC2 ({var_exp[1]:.1f}% variance)', fontsize=10)
    ax.set_title('PCA of training corpus + documentary sources\n'
                 'Circles = training prophets (color = date); '
                 'Diamonds = Torah sources', fontsize=10)
    plt.tight_layout()
    plt.savefig(str(outpath), dpi=150)
    plt.close()
    print(f"PCA plot saved: {outpath.name}")


# ---------------------------------------------------------------------------
# Plot 3: Feature heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(rates_df, meta_df, source_rates_dict, source_colors,
                 corr, outpath, min_rho=0.20):
    """
    Heatmap: columns = units (training + sources), rows = features with
    |ρ| > min_rho.  Values are row-normalized (z-scored) so CBH-modern
    gradient is visible.
    """
    # Select features
    feats = [col for col, st in corr.items() if abs(st['rho']) >= min_rho]
    if not feats:
        feats = list(corr.keys())[:20]
    feats = sorted(feats, key=lambda f: corr[f]['rho'])  # archaic→modern

    # Build matrix: training units ordered by date, then sources
    train_order = meta_df['date_bce'].sort_values(ascending=False).index.tolist()
    source_order = ['D', 'P', 'JE']

    all_labels  = train_order + [s for s in source_order if s in source_rates_dict]
    matrix = []
    for unit in all_labels:
        if unit in rates_df.index:
            row = [rates_df.loc[unit, f] if f in rates_df.columns else np.nan
                   for f in feats]
        elif unit in source_rates_dict:
            row = [source_rates_dict[unit].get(f, np.nan) for f in feats]
        else:
            row = [np.nan] * len(feats)
        matrix.append(row)
    mat = np.array(matrix, dtype=float)

    # Z-score normalize by row (feature) across training units only
    n_train = len(train_order)
    for j in range(mat.shape[1]):
        col = mat[:n_train, j]
        valid = col[np.isfinite(col)]
        if len(valid) < 2:
            continue
        mu = valid.mean(); sigma = valid.std()
        if sigma < 1e-9:
            continue
        mat[:, j] = (mat[:, j] - mu) / sigma

    fig, ax = plt.subplots(figsize=(max(10, len(all_labels) * 0.55),
                                     max(8, len(feats) * 0.4)))
    im = ax.imshow(mat.T, aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5)
    plt.colorbar(im, ax=ax, label='Z-score (training normalised; blue=archaic, red=modern)',
                 shrink=0.6)

    ax.set_xticks(range(len(all_labels)))
    xlabels = all_labels.copy()
    # Add date annotations for training units
    for i, u in enumerate(train_order):
        d = int(meta_df.loc[u, 'date_bce'])
        xlabels[i] = f"{u}\n({d})"
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=7)

    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([feature_display_name(f) for f in feats], fontsize=7)

    # Vertical separator between training and sources
    ax.axvline(n_train - 0.5, color='black', linewidth=2)
    ax.text(n_train - 0.5, -0.8, 'Training corpus | Torah sources',
            ha='center', fontsize=8, style='italic')

    ax.set_title('Feature heatmap: training corpus + documentary sources\n'
                 '(blue = more archaic/CBH-like; red = more modern/LBH-like)',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(str(outpath), dpi=150)
    plt.close()
    print(f"Heatmap saved: {outpath.name}")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(scores_by_source):
    """
    For each source, summarise:
    - mean LBH-ness across all features
    - fraction of features where score < 0  (more archaic than all training)
    - fraction of features where score > 1  (more modern than all training)
    - variance (how scattered = mixed archaism fingerprint)
    """
    rows = []
    for src, scores in scores_by_source.items():
        vals = np.array([v for v in scores.values() if np.isfinite(v)])
        if len(vals) == 0:
            continue
        rows.append({
            'source':            src,
            'n_features':        len(vals),
            'mean_lbh_score':    round(vals.mean(), 3),
            'std_lbh_score':     round(vals.std(), 3),
            'frac_below_0':      round((vals < 0).mean(), 3),
            'frac_above_1':      round((vals > 1).mean(), 3),
            'frac_within_range': round(((vals >= 0) & (vals <= 1)).mean(), 3),
            'interpretation':    (
                'Uniformly archaic'     if vals.mean() < 0.3 and vals.std() < 0.25 else
                'Uniformly modern'      if vals.mean() > 0.7 and vals.std() < 0.25 else
                'Mixed/selective'       if vals.std() >= 0.25 else
                'Intermediate'
            ),
        })
    return pd.DataFrame(rows).sort_values('mean_lbh_score')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-path', default=str(DEFAULT_BHSA_PATH),
                        help='Path to BHSA tf/2021 directory')
    parser.add_argument('--corpus', choices=['prophetic', 'broad'],
                        default='broad',
                        help='"prophetic" or "broad" (default: broad, adds late '
                             'narrative prose through 167 BCE)')
    parser.add_argument('--min-rho', type=float, default=0.25,
                        help='Minimum |ρ| for features to appear in audit (default: 0.25)')
    parser.add_argument('--outdir', default='.', help='Output directory')
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    specs = BROAD_SPECS if args.corpus == 'broad' else PROPHETIC_SPECS
    print(f"Using {'BROAD' if args.corpus == 'broad' else 'PROPHETIC'} corpus "
          f"({len(specs)} training units)")

    # Load BHSA
    print(f"\nLoading BHSA from {args.data_path}...")
    F, L, T = load_bhsa(args.data_path)

    # Extract training corpus features
    print(f"\nExtracting features for {len(specs)} training units...")
    meta_df, rates_df = load_training_corpus(specs, F, L, T)
    print(f"  Done: {len(rates_df)} units × {len(rates_df.columns)} features")

    # Extract documentary source features
    print("\nExtracting features for documentary sources...")
    individual_rates, aggregated_rates = load_doc_sources(F, L, T)
    for label, feat in sorted(individual_rates.items()):
        n = feat['n_words']
        an = feat.get('rate_anochi', 0) * n / 1000
        ai = feat.get('rate_ani', 0)    * n / 1000
        tot = an + ai
        frac = f"{an/tot:.2f}" if tot > 0 else "n/a"
        print(f"  {label:12}  {n:7,} words  frac_אנכי={frac}")

    # Correlations in training corpus
    print("\nComputing feature-date correlations in training corpus...")
    dates = meta_df['date_bce'].values
    corr  = compute_correlations(rates_df.drop(columns=['n_words'], errors='ignore'),
                                  dates)
    n_sig = sum(1 for st in corr.values() if st['p_raw'] < 0.05)
    print(f"  {len(corr)} features; {n_sig} with p < 0.05")

    direction_map = {row[0]: row[4] for row in FEATURE_CATALOGUE}

    # LBH-ness scores for aggregated sources
    print("\nComputing LBH-ness scores...")
    agg_scores = {}
    for src, feats in aggregated_rates.items():
        agg_scores[src] = compute_lbh_scores(feats, rates_df, dates, corr,
                                              min_rho=args.min_rho)

    ind_scores = {}
    for src, feats in individual_rates.items():
        ind_scores[src] = compute_lbh_scores(feats, rates_df, dates, corr,
                                              min_rho=args.min_rho)

    # Summary statistics
    stats_df = compute_summary_stats(agg_scores)
    stats_path = outdir / 'source_stats.csv'
    stats_df.to_csv(str(stats_path), index=False)
    print(f"\n{'='*70}")
    print("DOCUMENTARY SOURCE SUMMARY (aggregated D, P, JE)")
    print('='*70)
    print(stats_df.to_string(index=False))
    print(f"\nMean LBH-ness = 0: matches most archaic training text")
    print(f"Mean LBH-ness = 1: matches most modern training text")
    print(f"Score < 0: MORE archaic than any training text")
    print(f"Score > 1: MORE modern than any training text")
    print(f"High std = mixed/selective archaism profile")

    # Print per-feature breakdown for D and P
    print(f"\n{'='*70}")
    print("FEATURE-BY-FEATURE: DEUTERONOMY (D)")
    print('='*70)
    d_scores = agg_scores.get('D', {})
    sorted_d = sorted(d_scores.items(), key=lambda x: x[1])
    for feat, score in sorted_d:
        desc = feature_display_name(feat)
        exp  = direction_map.get(feat, '?')
        tag  = (' ← ARCHAIC' if score < 0 else
                ' ← MODERN'  if score > 1 else '')
        print(f"  {desc:35}  LBH={score:+.2f}  (expected: {exp}){tag}")

    print(f"\n{'='*70}")
    print("FEATURE-BY-FEATURE: P SOURCE (aggregated)")
    print('='*70)
    p_scores = agg_scores.get('P', {})
    sorted_p = sorted(p_scores.items(), key=lambda x: x[1])
    for feat, score in sorted_p:
        desc = feature_display_name(feat)
        exp  = direction_map.get(feat, '?')
        tag  = (' ← ARCHAIC' if score < 0 else
                ' ← MODERN'  if score > 1 else '')
        print(f"  {desc:35}  LBH={score:+.2f}  (expected: {exp}){tag}")

    # Save profiles
    all_feats = sorted(set().union(*[sc.keys() for sc in agg_scores.values()]))
    profile_rows = []
    for src, sc in agg_scores.items():
        row = {'source': src}
        for f in all_feats:
            row[f] = round(sc.get(f, np.nan), 4)
        profile_rows.append(row)
    profile_df = pd.DataFrame(profile_rows)
    profile_path = outdir / 'source_feature_profiles.csv'
    profile_df.to_csv(str(profile_path), index=False)
    print(f"\nProfiles saved: {profile_path.name}")

    # === Plots ===
    # 1. Archaism audit — aggregated D, P, JE
    plot_archaism_audit(
        agg_scores,
        SOURCE_COLORS,
        '(D, P, JE aggregated)',
        outdir / 'archaism_audit_aggregated.png',
        direction_map, min_rho=args.min_rho,
    )

    # 2. Archaism audit — individual book-level sources
    plot_archaism_audit(
        ind_scores,
        SOURCE_COLORS,
        '(individual book-source sections)',
        outdir / 'archaism_audit_individual.png',
        direction_map, min_rho=args.min_rho,
    )

    # 3. PCA
    # Combine individual rates for PCA (include all sources)
    all_source_rates = {**individual_rates, **aggregated_rates}
    plot_pca(
        rates_df.drop(columns=['n_words'], errors='ignore'),
        meta_df,
        all_source_rates,
        SOURCE_COLORS,
        outdir / 'source_pca.png',
    )

    # 4. Feature heatmap
    plot_heatmap(
        rates_df.drop(columns=['n_words'], errors='ignore'),
        meta_df,
        aggregated_rates,
        SOURCE_COLORS,
        corr,
        outdir / 'source_heatmap.png',
        min_rho=args.min_rho,
    )

    print("\nDone.")


if __name__ == '__main__':
    main()
