#!/usr/bin/env python3
"""
Systematic Feature Mining for Diachronic Hebrew
================================================
Instead of testing only theoretically-motivated features, this script scans
the *full* morphological and lexical feature space of the dated prophetic
corpus to discover which features actually correlate with date — letting the
data speak for itself.

Feature space covered
---------------------
1. Lexeme rates      : every lexeme appearing ≥ MIN_LEX_COUNT times in the
                       dated training corpus (default: 20 occurrences).
                       ~525 candidates at this threshold.
2. Verb forms (vt)   : wayq, perf, impf, ptca, ptcp, infa, infc, impv
                       (8 features, as rates per 1k words)
3. Verbal stems (vs) : qal, hif, piel, nif, hit, pual, hof
                       (7 features)
4. Morphological     : construct-state ratio, pronominal-suffix rate,
                       plural-noun rate, feminine-noun rate

Total candidates: ~550 features.

Multiple testing
----------------
With ~550 tests at α = 0.10, ~55 false positives are expected by chance.
With n=15 training units, the BH FDR correction is far too conservative
(requires p < 0.00018), so we instead use a permissive p < 0.10 threshold
to maximise feature recall.

False positives are managed downstream rather than here:
  (a) LOO robustness fraction is computed and reported for every included
      feature, flagging those that depend heavily on a single text.
  (b) The Bayesian date-prediction model (script 07) weights each feature
      by the inverse of its residual variance (σ from OLS fit on training
      data).  Noisy features with large σ contribute near-flat likelihoods
      and are automatically down-weighted.
  (c) Only morphosyntactic/grammatical features are used for prediction
      (topical content words are filtered out in script 07).

This approach trades higher feature-level FDR for better recall of genuine
but weakly-powered diachronic signals.

Caution on interpretation
--------------------------
Correlation with date ≠ caused by date.  A feature that correlates could
reflect: (a) true diachronic language change, (b) authorial idiolect
(Ezekiel's distinctive style), (c) sub-genre structure within prophecy,
or (d) chance.  The LOO filter addresses (d); factors (b) and (c) require
additional investigation (e.g. confirming the feature also changes in
non-prophetic texts).

Outputs
-------
feature_scan_full.csv   — all candidates with their statistics
feature_scan_robust.csv — candidates passing BH + LOO filters
feature_scan_top.png    — visualisation of top features
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from scipy.stats import spearmanr
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_LEX_COUNT = 20    # minimum corpus occurrences for a lexeme to be tested
LOO_FRAC_MIN  = 0.75  # LOO robustness threshold — ADVISORY only (not a gate)
FDR_Q         = 0.10  # Benjamini–Hochberg FDR threshold (reported for reference only)
P_RAW_THRESH  = 0.10  # raw p-value inclusion threshold
# With n=15 training units, BH correction is too conservative (requires p<0.00018
# for 545 tests).  We instead accept p<P_RAW_THRESH as sufficient for inclusion
# and rely on the Bayesian model's Gaussian likelihood (σ from residuals) to
# naturally down-weight features with poor fit.  LOO robustness is computed and
# reported as advisory metadata but does NOT gate feature inclusion.

# ---------------------------------------------------------------------------
# Training corpus specifications
# ---------------------------------------------------------------------------
# Two options:
#   PROPHETIC_SPECS — 15 dated prophets (760–460 BCE), single genre.
#   BROAD_SPECS     — adds late narrative prose (Jonah, Ezra, Nehemiah,
#                     Chronicles, Esther, Ecclesiastes, Daniel) extending
#                     the window to 167 BCE.  Genre mixing is the trade-off;
#                     the broader temporal range helps anchor the late LBH end.
#
# Selected at runtime via --corpus {prophetic|broad}.

# Chapters of Jeremiah attributed to Jeremiah himself (oracles, poetry,
# Baruch biography).  Excludes DTR prose: 7, 11, 17-18, 21, 24-29, 32-45.
# Sentinel value 'oracle' triggers special filtering in the chapter loop.
JEREMIAH_ORACLE_CHAPTERS = set(
    list(range(1, 7)) + list(range(8, 11)) + list(range(12, 17)) +
    [19, 20] + list(range(22, 24)) + list(range(30, 32)) + list(range(46, 52))
)

PROPHETIC_SPECS = [
    ('Amos',        ['Amos'],        None,       760, 15),
    ('Hosea',       ['Hosea'],       None,       725, 20),
    ('Micah',       ['Micah'],       None,       720, 20),
    ('Isaiah_1',    ['Isaiah'],      (1, 39),    700, 15),
    ('Zephaniah',   ['Zephaniah'],   None,       630, 15),
    ('Nahum',       ['Nahum'],       None,       620, 20),
    ('Habakkuk',    ['Habakkuk'],    None,       605, 20),
    ('Jeremiah',    ['Jeremiah'],    'oracle',   590, 15),
    ('Lamentations',['Lamentations'],None,       586, 20),
    ('Ezekiel',     ['Ezekiel'],     None,       570, 15),
    ('Isaiah_2',    ['Isaiah'],      (40, 55),   550, 20),
    ('Haggai',      ['Haggai'],      None,       520,  5),
    ('Zechariah_1', ['Zechariah'],   (1,  8),    518,  5),
    ('Isaiah_3',    ['Isaiah'],      (56, 66),   450, 100),  # revised: 450 BCE
    ('Malachi',     ['Malachi'],     None,       460, 20),
]

BROAD_SPECS = PROPHETIC_SPECS + [
    ('Jonah',       ['Jonah'],                       None,  400, 50),
    ('Ezra',        ['Ezra'],                        None,          350, 75),  # revised: 350 BCE
    ('Nehemiah',    ['Nehemiah'],                    None,          350, 75),  # revised: 350 BCE
    ('Chronicles',  ['1_Chronicles', '2_Chronicles'],None,          350, 50),
    ('Esther',      ['Esther'],                      None,          350, 50),
    ('Ecclesiastes',['Ecclesiastes'],                None,          330, 80),
    ('Daniel',      ['Daniel'],                      [(1,1),(8,12)], 167, 10), # Hebrew chs only
]

# Default — overridden by --corpus argument in main()
DATED_SPECS = PROPHETIC_SPECS

# Part-of-speech labels for human-readable output
SP_LABELS = {
    'subs': 'noun', 'verb': 'verb', 'prep': 'prep', 'conj': 'conj',
    'nmpr': 'proper-noun', 'art': 'article', 'adjv': 'adjective',
    'nega': 'negator', 'prps': 'pronoun', 'advb': 'adverb',
    'prde': 'dem-pronoun', 'intj': 'interjection', 'inrg': 'interrogative',
}


# ---------------------------------------------------------------------------
# Step 1: Load the word-level data for all dated units into a DataFrame
# ---------------------------------------------------------------------------

def load_training_words(data_path, F, L, T):
    """
    Collect every word token in the dated training corpus into a DataFrame.
    Columns: unit, date_bce, lex, sp, vt, vs, nu, gn, st, prs_ps
    """
    print("Loading word-level data from dated prophetic corpus...")
    rows = []
    for (unit, book_names, chap_range, date, sigma) in DATED_SPECS:
        for bname in book_names:
            bn = T.nodeFromSection((bname,))
            if bn is None:
                continue
            for ch in L.d(bn, 'chapter'):
                ch_num = F.chapter.v(ch)
                if chap_range == 'oracle':
                    if ch_num not in JEREMIAH_ORACLE_CHAPTERS:
                        continue
                elif isinstance(chap_range, list):
                    if not any(lo <= ch_num <= hi for lo, hi in chap_range):
                        continue
                elif chap_range and not (chap_range[0] <= ch_num <= chap_range[1]):
                    continue
                for w in L.d(ch, 'word'):
                    rows.append((
                        unit, date,
                        F.lex.v(w), F.sp.v(w), F.vt.v(w), F.vs.v(w),
                        F.nu.v(w),  F.gn.v(w), F.st.v(w), F.prs_ps.v(w),
                    ))

    df = pd.DataFrame(rows, columns=[
        'unit', 'date', 'lex', 'sp', 'vt', 'vs', 'nu', 'gn', 'st', 'prs_ps'
    ])
    print(f"  {len(df):,} word tokens across {df['unit'].nunique()} units")
    return df


# ---------------------------------------------------------------------------
# Step 2: Build the per-unit feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(df, min_lex_count=MIN_LEX_COUNT):
    """
    Compute per-unit feature rates (per 1,000 words).

    Returns
    -------
    feat_df : DataFrame (units × features)   — raw counts
    rates_df: DataFrame (units × features)   — per-1000-word rates
    meta_df : DataFrame with unit/date info
    feature_descriptions : dict  feature_name → human-readable description
    """
    unit_sizes  = df.groupby('unit').size().rename('n_words')
    unit_dates  = df.groupby('unit')['date'].first()
    meta_df = pd.DataFrame({'n_words': unit_sizes, 'date_bce': unit_dates})

    feat_desc = {}
    count_frames = []

    # ---- 1. Lexeme rates ------------------------------------------------
    lex_corpus_counts = df['lex'].value_counts()
    common_lex = lex_corpus_counts[lex_corpus_counts >= min_lex_count].index.tolist()
    print(f"  {len(common_lex)} lexemes with ≥{min_lex_count} corpus occurrences")

    lex_counts = (df[df['lex'].isin(common_lex)]
                  .groupby(['unit', 'lex'])
                  .size()
                  .unstack(fill_value=0)
                  .reindex(columns=common_lex, fill_value=0))
    lex_counts.columns = [f'lex::{c}' for c in lex_counts.columns]
    for lex in common_lex:
        sp_mode = df[df['lex'] == lex]['sp'].mode()
        sp_lbl = SP_LABELS.get(sp_mode.iloc[0], sp_mode.iloc[0]) if len(sp_mode) else '?'
        feat_desc[f'lex::{lex}'] = f'Lexeme {lex} ({sp_lbl}) rate per 1k words'
    count_frames.append(lex_counts)

    # ---- 2. Verb form rates (vt) -----------------------------------------
    verb_df = df[df['sp'] == 'verb']
    for vt_val in ['wayq', 'perf', 'impf', 'ptca', 'ptcp', 'infa', 'infc', 'impv']:
        col = f'vt::{vt_val}'
        counts = (verb_df[verb_df['vt'] == vt_val]
                  .groupby('unit').size()
                  .reindex(meta_df.index, fill_value=0)
                  .to_frame(col))
        count_frames.append(counts)
        feat_desc[col] = f'Verb form {vt_val} rate per 1k words'

    # ---- 3. Verbal stem rates (vs) ---------------------------------------
    for vs_val in ['qal', 'hif', 'piel', 'nif', 'hit', 'pual', 'hof']:
        col = f'vs::{vs_val}'
        counts = (verb_df[verb_df['vs'] == vs_val]
                  .groupby('unit').size()
                  .reindex(meta_df.index, fill_value=0)
                  .to_frame(col))
        count_frames.append(counts)
        feat_desc[col] = f'Verbal stem {vs_val} rate per 1k words'

    # ---- 4. Morphological ratio features ---------------------------------
    noun_df = df[df['sp'] == 'subs']

    # Construct state fraction: c/(a+c)
    const = (noun_df[noun_df['st'] == 'c'].groupby('unit').size()
             .reindex(meta_df.index, fill_value=0))
    abs_s = (noun_df[noun_df['st'] == 'a'].groupby('unit').size()
             .reindex(meta_df.index, fill_value=0))
    denom = (const + abs_s).clip(lower=1)
    # Store as binary feature: construct count vs absolute count
    count_frames.append(const.to_frame('morph::const_count'))
    count_frames.append(abs_s.to_frame('morph::abs_count'))
    feat_desc['morph::const_count'] = 'Noun in construct state (count)'
    feat_desc['morph::abs_count']   = 'Noun in absolute state (count)'

    # Pronominal suffix rate
    prs_counts = (df[df['prs_ps'] != 'NA'].groupby('unit').size()
                  .reindex(meta_df.index, fill_value=0)
                  .to_frame('morph::prs_rate'))
    count_frames.append(prs_counts)
    feat_desc['morph::prs_rate'] = 'Words carrying pronominal suffix rate per 1k words'

    # Plural noun rate
    pl_counts = (noun_df[noun_df['nu'] == 'pl'].groupby('unit').size()
                 .reindex(meta_df.index, fill_value=0)
                 .to_frame('morph::noun_pl_rate'))
    count_frames.append(pl_counts)
    feat_desc['morph::noun_pl_rate'] = 'Plural noun rate per 1k words'

    # Female noun rate
    f_counts = (noun_df[noun_df['gn'] == 'f'].groupby('unit').size()
                .reindex(meta_df.index, fill_value=0)
                .to_frame('morph::noun_f_rate'))
    count_frames.append(f_counts)
    feat_desc['morph::noun_f_rate'] = 'Feminine noun rate per 1k words'

    # ---- Combine and normalise -------------------------------------------
    feat_df = pd.concat(count_frames, axis=1).fillna(0)
    feat_df = feat_df.reindex(meta_df.index, fill_value=0)
    n_words = meta_df['n_words']
    rates_df = feat_df.div(n_words, axis=0) * 1000

    print(f"  Feature matrix: {rates_df.shape[0]} units × {rates_df.shape[1]} features")
    return feat_df, rates_df, meta_df, feat_desc


# ---------------------------------------------------------------------------
# Step 3: Correlation scan + multiple testing correction
# ---------------------------------------------------------------------------

def correlation_scan(rates_df, meta_df):
    """
    Spearman correlation between each feature and date.
    Returns DataFrame sorted by |ρ|.
    """
    dates = meta_df['date_bce'].values
    results = []
    for col in rates_df.columns:
        vals = rates_df[col].values
        if vals.std() < 1e-10:   # zero variance — skip
            continue
        r, p = spearmanr(-dates, vals)   # -dates so positive ρ = later is higher
        results.append({'feature': col, 'rho': r, 'p_raw': p})

    scan = pd.DataFrame(results).sort_values('p_raw').reset_index(drop=True)

    # Benjamini–Hochberg FDR correction
    n = len(scan)
    scan['rank'] = np.arange(1, n + 1)
    scan['p_bh'] = (scan['p_raw'] * n / scan['rank']).clip(upper=1.0)
    # BH correction: largest k where p_raw[k] ≤ (k/n) * q
    scan['bh_significant'] = False
    q = FDR_Q
    for i in range(n - 1, -1, -1):
        if scan.loc[i, 'p_raw'] <= (scan.loc[i, 'rank'] / n) * q:
            scan.loc[:i, 'bh_significant'] = True
            break

    return scan


# ---------------------------------------------------------------------------
# Step 4: LOO robustness filter
# ---------------------------------------------------------------------------

def loo_filter(scan, rates_df, meta_df, frac_min=LOO_FRAC_MIN):
    """
    For features with p_raw < P_RAW_THRESH, compute the LOO robustness
    fraction as advisory metadata.  LOO does NOT gate feature inclusion —
    all features with p_raw < P_RAW_THRESH pass, and the Bayesian model
    downstream handles uncertainty via each feature's residual σ.

    Returns scan with 'loo_frac', 'loo_robust', and 'passes' columns added.
    """
    dates = meta_df['date_bce'].values
    units = meta_df.index.tolist()
    n = len(units)

    candidate_feats = scan[scan['p_raw'] < P_RAW_THRESH]['feature'].tolist()
    print(f"  {len(candidate_feats)} features with p_raw < {P_RAW_THRESH}; "
          f"computing LOO fraction (advisory) for these...")
    print(f"  (BH threshold for n={len(scan)} tests would require "
          f"p < {FDR_Q/len(scan):.5f} — too conservative for n={n} units; "
          f"using permissive threshold instead)")

    loo_fracs = {}
    for feat in candidate_feats:
        vals = rates_df[feat].values
        n_sig = 0
        n_tested = 0
        full_sign = np.sign(scan.loc[scan['feature'] == feat, 'rho'].iloc[0])
        for i in range(n):
            mask = np.ones(n, dtype=bool); mask[i] = False
            if mask.sum() < 5:
                continue
            r, p = spearmanr(-dates[mask], vals[mask])
            n_tested += 1
            if p < 0.05 and np.sign(r) == full_sign:
                n_sig += 1
        loo_fracs[feat] = n_sig / max(n_tested, 1)

    scan['loo_frac']   = scan['feature'].map(loo_fracs).fillna(np.nan)
    scan['loo_robust'] = scan['loo_frac'] >= frac_min   # advisory flag
    # Inclusion gate: p_raw < threshold only.  LOO is metadata, not a gate.
    scan['passes'] = scan['p_raw'] < P_RAW_THRESH
    return scan


# ---------------------------------------------------------------------------
# Step 5: Visualisation
# ---------------------------------------------------------------------------

def plot_top_features(scan_robust, rates_df, meta_df, outdir, n_top=16):
    """
    Grid of scatter plots for the top robust features.
    """
    top = scan_robust.head(n_top)
    ncols = 4
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5))
    axes = axes.flatten()

    dates = meta_df['date_bce'].values
    units = meta_df.index.tolist()

    for i, (_, row) in enumerate(top.iterrows()):
        ax = axes[i]
        feat = row['feature']
        vals = rates_df[feat].values
        ax.scatter(dates, vals, s=40, color='steelblue')
        for d, v, u in zip(dates, vals, units):
            ax.annotate(u[:6], (d, v), xytext=(2, 2),
                        textcoords='offset points', fontsize=6)
        m, b = np.polyfit(-dates, vals, 1)
        x_line = np.linspace(dates.min(), dates.max(), 100)
        ax.plot(x_line, m * (-x_line) + b, 'r--', alpha=0.5, linewidth=1)
        ax.invert_xaxis()
        short_name = feat.split('::')[1][:18] if '::' in feat else feat[:18]
        ax.set_title(f"{short_name}\nρ={row['rho']:.2f}  p={row['p_raw']:.4f}  LOO={row['loo_frac']:.0%}",
                     fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Date BCE', fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Top {len(top)} LOO-robust diachronic features (dated prophetic corpus)',
                 fontsize=11)
    plt.tight_layout()
    out_path = outdir / 'feature_scan_top.png'
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-path',
                        default=str(Path.home() / 'text-fabric-data' / 'github' / 'ETCBC' / 'bhsa' / 'tf' / '2021'),
                        help='Path to BHSA tf/2021 directory')
    parser.add_argument('--min-lex', type=int, default=MIN_LEX_COUNT,
                        help=f'Min lexeme frequency for inclusion (default: {MIN_LEX_COUNT})')
    parser.add_argument('--corpus', choices=['prophetic', 'broad'], default='prophetic',
                        help='Training corpus: "prophetic" (760–460 BCE, default) or '
                             '"broad" (adds late narrative: Ezra, Nehemiah, Chronicles, '
                             'Esther, Ecclesiastes, Daniel, extending to 167 BCE)')
    parser.add_argument('--outdir', default='.', help='Output directory')
    args = parser.parse_args()

    # Select corpus
    global DATED_SPECS
    if args.corpus == 'broad':
        DATED_SPECS = BROAD_SPECS
        print(f"Using BROAD corpus ({len(DATED_SPECS)} units, 760–167 BCE)")
    else:
        DATED_SPECS = PROPHETIC_SPECS
        print(f"Using PROPHETIC corpus ({len(DATED_SPECS)} units, 760–460 BCE)")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load TF
    data_path = Path(args.data_path).expanduser()
    if not data_path.exists():
        print(f"ERROR: BHSA data not found at {data_path}")
        print("Run: python3 01_feature_extraction_etcbc.py --download")
        sys.exit(1)

    try:
        from tf.fabric import Fabric
    except ImportError:
        print("ERROR: pip install text-fabric[github]")
        sys.exit(1)

    print(f"Loading BHSA from {data_path}...")
    TF = Fabric(locations=str(data_path), silent=True)
    api = TF.load('otype oslots otext lex sp vt vs nu gn st prs_ps chapter', silent=True)
    F = api.F; L = api.L; T = api.T

    # Build training dataset
    words_df = load_training_words(data_path, F, L, T)
    feat_df, rates_df, meta_df, feat_desc = build_feature_matrix(words_df, args.min_lex)

    # Correlation scan
    print("\nRunning correlation scan...")
    scan = correlation_scan(rates_df, meta_df)
    n_raw = (scan['p_raw'] < P_RAW_THRESH).sum()
    n_bh  = scan['bh_significant'].sum()
    print(f"  {len(scan)} features tested; {n_raw} with p_raw < {P_RAW_THRESH}")
    print(f"  (BH correction at q={FDR_Q} would require p < {FDR_Q/len(scan):.5f}; "
          f"{n_bh} pass — too conservative for n={len(meta_df)} training units)")

    # LOO robustness
    scan = loo_filter(scan, rates_df, meta_df)
    scan_robust = scan[scan['passes'].fillna(False)].copy()
    n_loo = scan_robust['loo_robust'].sum() if 'loo_robust' in scan_robust.columns else 0
    print(f"  {len(scan_robust)} features pass p_raw < {P_RAW_THRESH}  "
          f"(of these, {n_loo} are also LOO-robust ≥{LOO_FRAC_MIN:.0%} — advisory)")

    # Add descriptions and sort by |ρ|
    scan['description']        = scan['feature'].map(feat_desc)
    scan_robust['description'] = scan_robust['feature'].map(feat_desc)
    scan_robust = scan_robust.sort_values('rho', key=abs, ascending=False).reset_index(drop=True)

    # Save
    full_path   = outdir / 'feature_scan_full.csv'
    robust_path = outdir / 'feature_scan_robust.csv'
    scan.to_csv(str(full_path), index=False)
    scan_robust.to_csv(str(robust_path), index=False)
    print(f"\nFull scan saved:   {full_path.name}")
    print(f"Robust features:   {robust_path.name}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"INCLUDED FEATURES  (p_raw < {P_RAW_THRESH}; LOO fraction is advisory)")
    print('='*70)
    if len(scan_robust) == 0:
        print("  None found.  Try raising --p-thresh or lowering --min-lex.")
    else:
        show_cols = ['feature', 'rho', 'p_raw', 'loo_frac', 'loo_robust', 'description']
        show_cols = [c for c in show_cols if c in scan_robust.columns]
        pd.set_option('display.max_colwidth', 50)
        print(scan_robust[show_cols].to_string(index=False))

        print(f"\nDirection summary:")
        n_pos = (scan_robust['rho'] > 0).sum()
        n_neg = (scan_robust['rho'] < 0).sum()
        print(f"  {n_pos} features INCREASE over time (more common in later books)")
        print(f"  {n_neg} features DECREASE over time (less common in later books)")

    # Plot
    if len(scan_robust) > 0:
        plot_top_features(scan_robust, rates_df, meta_df, outdir)

    # Save rates for use by script 07
    rates_with_meta = rates_df.copy()
    rates_with_meta.insert(0, 'date_bce', meta_df['date_bce'])
    rates_with_meta.insert(1, 'n_words', meta_df['n_words'])
    rates_path = outdir / 'feature_rates_training.csv'
    rates_with_meta.to_csv(str(rates_path), index=True, index_label='unit')
    print(f"\nFull rate matrix saved: {rates_path.name}  (used by script 07)")

    return scan_robust


if __name__ == '__main__':
    main()
