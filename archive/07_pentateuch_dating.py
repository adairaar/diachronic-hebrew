#!/usr/bin/env python3
"""
Pentateuch Temporal Placement via Diachronic Features
======================================================
Using the diachronic features discovered in scripts 05 and 06, this script
asks: if we treat each book of the Torah as an unknown text and apply our
dating model — trained on the dated prophetic corpus — where does the model
place it on the temporal axis?

Two analytical approaches
--------------------------
1. **Nearest-neighbour**: find which dated prophetic book is most similar to
   each Pentateuch book in the feature space of robust diachronic markers.
   This is assumption-free and interpretable.

2. **Bayesian regression prediction**: fit a simple Gaussian model
   (feature value ~ date) on the training corpus, then invert it to
   predict P(date | feature value observed) for each Torah book.
   Individual feature posteriors are combined by multiplication,
   giving an overall date posterior.

Important caveats
-----------------
• The training corpus is prophetic texts (~760–460 BCE).  We are
  *extrapolating* to a genre (law/narrative) that may have its own
  stylistic conventions independent of date.  A feature that changes with
  date in prophecy may not change the same way in other genres.

• The Torah was almost certainly composed and edited over a long period and
  by multiple authors; treating each book as a stylistically uniform unit is
  a simplification.

• This analysis is exploratory and hypothesis-generating.  Its results should
  be compared with independent evidence before drawing historical conclusions.

Documentary source analysis
----------------------------
The five Torah books vary considerably in how homogeneous they are with
respect to the Documentary Hypothesis:
  • Deuteronomy  ≈ entirely D source
  • Leviticus    ≈ entirely P source
  • Numbers      ≈ mix of P and JE
  • Genesis/Exodus: complex P + JE mixture

The script analyses all five books as units, then provides a focused
comparison of Deuteronomy (D) vs. Leviticus (P) — the two clearest
"pure" source cases — which directly bears on the relative dating of D and P.

For full source-level analysis, pass --source-file pointing to a JSON file
mapping {book: {source: [[start_ch, end_ch], ...]}} with chapter ranges per
source.  A template is printed with --print-source-template.

Usage
-----
    python 07_pentateuch_dating.py --data-path ~/text-fabric-data/...
                                   --robust-features feature_scan_robust.csv
                                   --training-rates feature_rates_training.csv
"""

import sys
import re
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, norm as scipy_norm
from pathlib import Path

# ---------------------------------------------------------------------------
# Jeremiah oracle chapters (non-DTR: excludes 7,11,17-18,21,24-29,32-45)
# ---------------------------------------------------------------------------
JEREMIAH_ORACLE_CHAPTERS = set(
    list(range(1, 7)) + list(range(8, 11)) + list(range(12, 17)) +
    [19, 20] + list(range(22, 24)) + list(range(30, 32)) + list(range(46, 52))
)

# ---------------------------------------------------------------------------
# Pentateuch book definitions
# ---------------------------------------------------------------------------
TORAH_BOOKS = [
    ('Genesis',      'Genesis',       None),
    ('Exodus',       'Exodus',        None),
    ('Leviticus',    'Leviticus',     None),
    ('Numbers',      'Numbers',       None),
    ('Deuteronomy',  'Deuteronomy',   None),
]

# Simplified documentary source chapter ranges
# Based on broad consensus in the field (cf. Friedman 2003; Baden 2012).
# These are necessarily approximate; chapters marked as one source often
# contain verses from another.
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
        'P':  [(1, 27)],   # essentially the whole book
    },
    'Numbers': {
        'P':  [(1,10), (15,15), (17,19), (25,25), (27,31), (33,36)],
        'JE': [(11,14), (16,16), (20,24), (25,25), (32,32)],
    },
    'Deuteronomy': {
        'D':  [(1, 30), (31,32), (34,34)],   # Deut is predominantly D
    },
}

# Broad scholarly date ranges (BCE) for the major sources, for comparison:
SCHOLARLY_DATES = {
    'J':  (950, 850, 'Yahwist (traditional)'),
    'E':  (900, 750, 'Elohist (traditional)'),
    'JE': (950, 750, 'JE combined (traditional)'),
    'D':  (700, 621, 'Deuteronomist'),
    'P':  (550, 400, 'Priestly source'),
}


# ---------------------------------------------------------------------------
# Linguistic feature filter
# ---------------------------------------------------------------------------
# Part-of-speech labels that indicate a grammatical/functional word rather
# than a content word (noun, proper noun, or lexical verb).  Features in
# these categories are more likely to reflect genuine linguistic register
# change rather than topical shifts in subject matter.

LINGUISTIC_POS = {
    'pronoun', 'dem-pronoun', 'prep', 'conj',
    'adverb', 'negator', 'article', 'interrogative', 'interjection',
}


def filter_linguistic_features(robust_features, verbose=True):
    """
    Return only morphosyntactic/grammatical features, discarding topical
    content words that correlate with date for historical-subject-matter
    reasons rather than linguistic-register reasons.

    Kept:
      - All vt:: (verb form) features — grammatical paradigm
      - All vs:: (verbal stem) features — grammatical paradigm
      - All morph:: (morphological) features — grammatical paradigm
      - lex:: features whose POS is in LINGUISTIC_POS (pronouns, preps,
        conjunctions, adverbs, negators, articles, interrogatives)

    Excluded:
      - lex:: features whose POS is 'noun', 'proper-noun', 'verb', etc.
        These often correlate with date because subject matter changed
        (e.g. 'Samaria' is rare after 722 BCE for historical, not
        linguistic, reasons).
    """
    keep = []
    for _, row in robust_features.iterrows():
        feat = row['feature']
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

    filtered = robust_features[keep].reset_index(drop=True)
    if verbose:
        n_total = len(robust_features)
        n_kept  = len(filtered)
        print(f"  Linguistic filter: {n_kept}/{n_total} features kept "
              f"({n_total - n_kept} topical nouns/verbs/proper-nouns excluded)")
        if n_kept > 0:
            print(f"  Features used: {', '.join(filtered['feature'].tolist())}")
    return filtered


# ---------------------------------------------------------------------------
# Feature extraction from any book/chapter range
# ---------------------------------------------------------------------------

def extract_features_for_unit(unit_name, book_names, chap_range, F, L, T,
                               target_lexemes, target_vt, target_vs):
    """
    Extract the same feature rates used in training, for a new text unit.

    Parameters
    ----------
    target_lexemes : list of str  — lexemes to count (from training feature matrix)
    target_vt, target_vs : list   — verb-form and verb-stem values to count

    Returns
    -------
    dict mapping feature_name → per-1000-word rate
    """
    word_rows = []
    for bname in book_names:
        bn = T.nodeFromSection((bname,))
        if bn is None:
            print(f"  WARNING: '{bname}' not found")
            continue
        for ch in L.d(bn, 'chapter'):
            ch_num = F.chapter.v(ch)
            if chap_range == 'oracle':
                if ch_num not in JEREMIAH_ORACLE_CHAPTERS:
                    continue
            elif chap_range and not (chap_range[0] <= ch_num <= chap_range[1]):
                continue
            for w in L.d(ch, 'word'):
                word_rows.append((
                    F.lex.v(w), F.sp.v(w), F.vt.v(w), F.vs.v(w),
                    F.nu.v(w),  F.gn.v(w), F.st.v(w), F.prs_ps.v(w),
                ))

    if not word_rows:
        return None

    df = pd.DataFrame(word_rows, columns=['lex', 'sp', 'vt', 'vs', 'nu', 'gn', 'st', 'prs_ps'])
    n = len(df)
    rates = {'unit': unit_name, 'n_words': n}

    # Lexeme rates
    lex_ctr = df['lex'].value_counts()
    for lex in target_lexemes:
        rates[f'lex::{lex}'] = lex_ctr.get(lex, 0) / n * 1000

    # Verb form rates
    verb_df = df[df['sp'] == 'verb']
    vt_ctr = verb_df['vt'].value_counts()
    for vt in target_vt:
        rates[f'vt::{vt}'] = vt_ctr.get(vt, 0) / n * 1000

    # Verbal stem rates
    vs_ctr = verb_df['vs'].value_counts()
    for vs in target_vs:
        rates[f'vs::{vs}'] = vs_ctr.get(vs, 0) / n * 1000

    # Morphological features
    noun_df = df[df['sp'] == 'subs']
    const_n = (noun_df['st'] == 'c').sum()
    abs_n   = (noun_df['st'] == 'a').sum()
    rates['morph::const_count'] = const_n / n * 1000
    rates['morph::abs_count']   = abs_n   / n * 1000
    rates['morph::prs_rate']    = (df['prs_ps'] != 'NA').sum() / n * 1000
    rates['morph::noun_pl_rate']= (noun_df['nu'] == 'pl').sum() / n * 1000
    rates['morph::noun_f_rate'] = (noun_df['gn'] == 'f').sum()  / n * 1000

    return rates


# ---------------------------------------------------------------------------
# Bayesian date prediction using multiple features
# ---------------------------------------------------------------------------

def predict_date_bayesian(unit_rates, robust_features, training_rates,
                           date_grid=None, verbose=False):
    """
    For each robust feature, fit a Gaussian likelihood (obs ~ N(μ(date), σ))
    using OLS on the training data, then compute the posterior P(date | obs)
    and multiply across features.

    Parameters
    ----------
    unit_rates       : dict  feature → rate for the unit being dated
    robust_features  : DataFrame  with 'feature' and 'rho' columns
    training_rates   : DataFrame  with columns = features, index = units,
                       plus a 'date_bce' column

    Returns
    -------
    date_grid : array of dates tested
    posterior : normalized probability density over date_grid
    summary   : dict with MAP estimate, 68% and 95% credible intervals
    """
    if date_grid is None:
        date_grid = np.linspace(1000, 100, 500)  # 1000 BCE to 100 BCE

    log_posterior = np.zeros(len(date_grid))
    n_used = 0

    for _, row in robust_features.iterrows():
        feat = row['feature']
        if feat not in unit_rates or feat not in training_rates.columns:
            continue
        obs_val = unit_rates[feat]

        # OLS: feature_rate ~ a + b * date
        train_dates = training_rates['date_bce'].values
        train_vals  = training_rates[feat].values
        mask = np.isfinite(train_vals) & np.isfinite(train_dates)
        if mask.sum() < 4:
            continue

        b, a = np.polyfit(train_dates[mask], train_vals[mask], 1)
        residuals = train_vals[mask] - (a + b * train_dates[mask])
        sigma = residuals.std()
        if sigma < 1e-9:
            continue

        # Gaussian likelihood: P(obs | date) ~ N(a + b*date, sigma)
        predicted = a + b * date_grid
        log_lik = -0.5 * ((obs_val - predicted) / sigma) ** 2
        log_posterior += log_lik
        n_used += 1

        if verbose:
            print(f"  {feat[:40]:40} obs={obs_val:.3f}  slope={b:.5f}  σ={sigma:.3f}")

    if n_used == 0:
        return date_grid, np.ones(len(date_grid)) / len(date_grid), {}

    # Normalise
    log_posterior -= log_posterior.max()
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum() * (date_grid[0] - date_grid[-1]) * (-1)

    # Summary statistics
    cdf = np.cumsum(posterior) / posterior.sum()
    map_date = date_grid[np.argmax(posterior)]
    lo68 = date_grid[np.searchsorted(cdf, 0.16)]
    hi68 = date_grid[np.searchsorted(cdf, 0.84)]
    lo95 = date_grid[np.searchsorted(cdf, 0.025)]
    hi95 = date_grid[np.searchsorted(cdf, 0.975)]

    summary = {
        'MAP_date_BCE': round(map_date),
        'CI68_lo_BCE':  round(hi68),   # hi in date = earlier (older), lo = later
        'CI68_hi_BCE':  round(lo68),
        'CI95_lo_BCE':  round(hi95),
        'CI95_hi_BCE':  round(lo95),
        'n_features':   n_used,
    }
    return date_grid, posterior, summary


# ---------------------------------------------------------------------------
# Nearest-neighbour comparison
# ---------------------------------------------------------------------------

def nearest_neighbour(unit_rates, robust_features, training_rates):
    """
    Find the most similar training unit(s) by cosine similarity on robust features.
    """
    feats = [f for f in robust_features['feature']
             if f in unit_rates and f in training_rates.columns]
    if not feats:
        return []

    query = np.array([unit_rates[f] for f in feats])
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return []

    sims = []
    for idx, trow in training_rates.iterrows():
        ref = np.array([trow[f] for f in feats])
        ref_norm = np.linalg.norm(ref)
        if ref_norm == 0:
            continue
        cos = np.dot(query, ref) / (query_norm * ref_norm)
        sims.append({'unit': idx, 'date_bce': trow['date_bce'], 'cosine_sim': cos})

    return sorted(sims, key=lambda x: -x['cosine_sim'])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_pentateuch_results(books_summary, date_posteriors, date_grid,
                             training_rates, robust_features, outdir):
    """
    Three-panel figure:
      Left:  Posterior date distributions for each Torah book
      Middle: Feature profile heatmap (Torah vs. training)
      Right: Deuteronomy vs. Leviticus direct comparison
    """
    n_books = len(books_summary)
    fig = plt.figure(figsize=(18, max(8, n_books * 1.5 + 4)))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ---- Panel 1: Posterior distributions ---------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.tab10
    colors = [cmap(i) for i in range(n_books)]
    for i, (bname, summary) in enumerate(books_summary.items()):
        if bname not in date_posteriors:
            continue
        post = date_posteriors[bname]
        ax1.plot(date_grid, post / post.max(), color=colors[i],
                 label=bname, linewidth=2)
        map_d = summary.get('MAP_date_BCE', np.nan)
        if not np.isnan(map_d):
            ax1.axvline(map_d, color=colors[i], linewidth=0.8, alpha=0.5, linestyle='--')
    ax1.invert_xaxis()
    ax1.set_xlabel('Date (BCE)')
    ax1.set_ylabel('Normalised posterior')
    ax1.set_title('Posterior date distributions\n(Torah books, naive model)')
    ax1.legend(fontsize=8, loc='upper left')
    # Mark training corpus range
    train_dates = training_rates['date_bce'].values
    ax1.axvspan(train_dates.min(), train_dates.max(), alpha=0.07, color='gray',
                label='Training range')

    # ---- Panel 2: Summary table -------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    table_data = []
    for bname, summ in books_summary.items():
        if 'MAP_date_BCE' in summ:
            table_data.append([
                bname,
                f"{summ['MAP_date_BCE']} BCE",
                f"{summ['CI68_lo_BCE']}–{summ['CI68_hi_BCE']} BCE",
                str(summ.get('top_neighbor', '—')),
            ])
    if table_data:
        tbl = ax2.table(
            cellText=table_data,
            colLabels=['Book', 'MAP date', '68% CI', 'Nearest prophet'],
            loc='center', cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.2, 1.6)
    ax2.set_title('Bayesian date estimates\n(trained on dated prophets)', fontsize=9)

    # ---- Panel 3: D vs P direct feature comparison ------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    top_feats = robust_features.head(10)['feature'].tolist()
    d_vals, p_vals, feat_labels = [], [], []
    for feat in top_feats:
        if feat in books_summary.get('Deuteronomy', {}).get('feature_rates', {}):
            d_rate = books_summary['Deuteronomy']['feature_rates'].get(feat, np.nan)
            p_rate = books_summary.get('Leviticus', {}).get('feature_rates', {}).get(feat, np.nan)
            if not np.isnan(d_rate) and not np.isnan(p_rate):
                d_vals.append(d_rate); p_vals.append(p_rate)
                feat_labels.append(feat.split('::')[-1][:14])
    if d_vals:
        y = np.arange(len(d_vals))
        ax3.barh(y - 0.2, d_vals, 0.35, label='Deuteronomy (D)', color='steelblue', alpha=0.8)
        ax3.barh(y + 0.2, p_vals, 0.35, label='Leviticus (P)',   color='tomato',    alpha=0.8)
        ax3.set_yticks(y); ax3.set_yticklabels(feat_labels, fontsize=7)
        ax3.set_xlabel('Rate per 1,000 words')
        ax3.set_title('Deuteronomy (D) vs. Leviticus (P)\nTop-10 diachronic features', fontsize=9)
        ax3.legend(fontsize=8)

    # ---- Panel 4: Torah books on the temporal feature gradient ------------
    ax4 = fig.add_subplot(gs[1, :])
    # Use the single most reliable feature (אנכי/אני) if available
    key_feat = 'lex::>NKJ'   # אנכי; feature with known temporal meaning
    alt_feat = 'lex::>NJ'    # אני
    if key_feat in training_rates.columns and alt_feat in training_rates.columns:
        train_d = training_rates['date_bce'].values
        # Fraction of (אנכי + אני) that is אנכי — should decline over time
        anochi = training_rates[key_feat].values
        ani    = training_rates[alt_feat].values
        total  = anochi + ani
        frac_anochi = np.where(total > 0, anochi / total, np.nan)

        ax4.scatter(train_d, frac_anochi, s=60, color='steelblue',
                    label='Dated prophets', zorder=3)
        for i, u in enumerate(training_rates.index):
            if np.isfinite(frac_anochi[i]):
                ax4.annotate(u[:8], (train_d[i], frac_anochi[i]),
                             xytext=(2, 4), textcoords='offset points', fontsize=6)

        # Torah books
        torah_colors = {'Genesis': 'green', 'Exodus': 'orange', 'Leviticus': 'red',
                        'Numbers': 'purple', 'Deuteronomy': 'brown'}
        for bname, summ in books_summary.items():
            fr = summ.get('feature_rates', {})
            an = fr.get(key_feat, np.nan)
            ai = fr.get(alt_feat, np.nan)
            tot = an + ai
            if tot > 0:
                frac = an / tot
                # Place on x-axis using MAP date
                map_d = summ.get('MAP_date_BCE', 600)
                ax4.scatter(map_d, frac, s=120, marker='D', zorder=4,
                            color=torah_colors.get(bname, 'gray'), label=bname)
                ax4.annotate(bname, (map_d, frac),
                             xytext=(4, 4), textcoords='offset points',
                             fontsize=8, fontweight='bold',
                             color=torah_colors.get(bname, 'gray'))

        ax4.invert_xaxis()
        ax4.set_xlabel('Date (BCE) — Torah books placed at MAP estimate')
        ax4.set_ylabel('Fraction אנכי/(אנכי+אני)\n(higher = more archaic)')
        ax4.set_title('Pronoun archaism indicator: dated prophets + Torah books')
        ax4.legend(fontsize=7, loc='upper right', ncol=2)

    plt.suptitle('Pentateuch Temporal Analysis  |  Trained on Dated Prophetic Corpus',
                 fontsize=12, y=1.01)
    out_path = outdir / 'pentateuch_dating.png'
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Source-level analysis
# ---------------------------------------------------------------------------

def analyse_sources(source_map, F, L, T,
                    target_lexemes, target_vt, target_vs,
                    robust_features, training_rates, date_grid, outdir):
    """
    Extract features for each documentary source sub-corpus and predict dates.
    source_map: {book: {source: [(start_ch, end_ch), ...]}}
    """
    print("\n--- Documentary Source Analysis ---")
    source_results = {}
    for book, sources in source_map.items():
        for source_label, ch_ranges in sources.items():
            unit_name = f"{book[:3]}:{source_label}"
            book_tf_name = book  # assumes BHSA uses plain book names
            rates = extract_features_for_unit(
                unit_name, [book_tf_name],
                None,   # we handle chapter filtering manually below
                F, L, T, target_lexemes, target_vt, target_vs,
            )
            # Re-extract restricted to chapter ranges only
            word_rows = []
            bn = T.nodeFromSection((book,))
            if bn is None:
                continue
            for ch in L.d(bn, 'chapter'):
                ch_num = F.chapter.v(ch)
                in_range = any(s <= ch_num <= e for s, e in ch_ranges)
                if not in_range:
                    continue
                for w in L.d(ch, 'word'):
                    word_rows.append(F.lex.v(w))

            if not word_rows:
                continue

            n = len(word_rows)
            # Recompute just the key pronoun feature
            anochi = word_rows.count('>NKJ')
            ani    = word_rows.count('>NJ')
            total  = anochi + ani
            frac   = anochi / total if total > 0 else np.nan

            _, posterior, summary = predict_date_bayesian(
                rates or {}, robust_features, training_rates, date_grid)
            summary['pronoun_anochi'] = anochi
            summary['pronoun_ani']    = ani
            summary['frac_anochi']    = round(frac, 3) if not np.isnan(frac) else 'NA'
            summary['n_words']        = n
            source_results[unit_name] = summary
            print(f"  {unit_name:15}: {n:6,} words  "
                  f"אנכי={anochi:3d}  אני={ani:3d}  "
                  f"MAP≈{summary.get('MAP_date_BCE','?')} BCE")

    return source_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-path',
                        default=str(Path.home() / 'text-fabric-data' / 'github' / 'ETCBC' / 'bhsa' / 'tf' / '2021'))
    parser.add_argument('--robust-features', default='feature_scan_robust.csv',
                        help='Output of script 06 (feature_scan_robust.csv)')
    parser.add_argument('--training-rates', default='feature_rates_training.csv',
                        help='Output of script 06 (feature_rates_training.csv)')
    parser.add_argument('--source-file', default=None,
                        help='Optional JSON with chapter-level source attributions')
    parser.add_argument('--print-source-template', action='store_true',
                        help='Print built-in source attribution and exit')
    parser.add_argument('--outdir', default='.')
    args = parser.parse_args()

    if args.print_source_template:
        print(json.dumps(DOC_SOURCES, indent=2))
        return

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load training data
    robust_path = Path(args.robust_features)
    rates_path  = Path(args.training_rates)
    if not robust_path.exists() or not rates_path.exists():
        print("ERROR: Run script 06 first to generate feature_scan_robust.csv "
              "and feature_rates_training.csv")
        sys.exit(1)

    robust_features  = pd.read_csv(robust_path)
    training_rates   = pd.read_csv(rates_path, index_col='unit')
    print(f"Loaded {len(robust_features)} robust features from {robust_path.name}")
    print(f"Training corpus: {len(training_rates)} units")

    # Apply linguistic filter: keep only morphosyntactic/grammatical features.
    # Topical content words (nouns, proper nouns, lexical verbs) are excluded
    # because they correlate with date due to historical subject-matter shifts,
    # not genuine linguistic register change.
    print("\nApplying linguistic feature filter...")
    robust_features = filter_linguistic_features(robust_features, verbose=True)
    if len(robust_features) == 0:
        print("WARNING: No linguistic features remain after filtering.  "
              "The Bayesian model will fall back to pronoun data only.")

    # Identify which lexemes/forms are needed
    target_lex = [f.replace('lex::', '') for f in robust_features['feature']
                  if f.startswith('lex::')]
    target_vt  = [f.replace('vt::', '')  for f in robust_features['feature']
                  if f.startswith('vt::')]
    target_vs  = [f.replace('vs::', '')  for f in robust_features['feature']
                  if f.startswith('vs::')]
    # Always include the pronoun pair even if not in robust set
    for lex in ['>NKJ', '>NJ']:
        if lex not in target_lex:
            target_lex.append(lex)

    # Load TF
    data_path = Path(args.data_path).expanduser()
    try:
        from tf.fabric import Fabric
    except ImportError:
        print("ERROR: pip install text-fabric[github]"); sys.exit(1)
    if not data_path.exists():
        print(f"ERROR: BHSA data not found at {data_path}"); sys.exit(1)

    print(f"\nLoading BHSA from {data_path}...")
    TF = Fabric(locations=str(data_path), silent=True)
    api = TF.load('otype oslots otext lex sp vt vs nu gn st prs_ps chapter', silent=True)
    F = api.F; L = api.L; T = api.T

    date_grid = np.linspace(1100, 50, 600)

    # --- Analyse each Torah book ---
    print("\n=== Whole-book Pentateuch analysis ===")
    books_summary   = {}
    date_posteriors = {}

    for (display_name, tf_book, chap_range) in TORAH_BOOKS:
        print(f"\n  {display_name}...")
        rates = extract_features_for_unit(
            display_name, [tf_book], chap_range,
            F, L, T, target_lex, target_vt, target_vs)
        if rates is None:
            print(f"  WARNING: no words extracted for {display_name}")
            continue

        n = rates['n_words']
        an = rates.get('lex::>NKJ', 0) * n / 1000
        ai = rates.get('lex::>NJ',  0) * n / 1000
        tot = an + ai
        frac = an / tot if tot > 0 else float('nan')
        print(f"    {n:7,} words  אנכי={int(an):4d}  אני={int(ai):4d}  "
              f"frac_anochi={frac:.3f}" if tot > 0 else f"    {n:7,} words  (no 1sg pronouns)")

        dg, posterior, summary = predict_date_bayesian(
            rates, robust_features, training_rates, date_grid, verbose=False)

        # Nearest neighbour
        nn = nearest_neighbour(rates, robust_features, training_rates)
        summary['top_neighbor']   = nn[0]['unit']    if nn else '?'
        summary['top_neighbor_2'] = nn[1]['unit']    if len(nn) > 1 else '?'
        summary['feature_rates']  = rates

        books_summary[display_name]   = summary
        date_posteriors[display_name] = posterior

        print(f"    MAP date ≈ {summary.get('MAP_date_BCE','?')} BCE  "
              f"68% CI: {summary.get('CI68_lo_BCE','?')}–{summary.get('CI68_hi_BCE','?')} BCE")
        print(f"    Nearest dated prophet: {summary['top_neighbor']}  "
              f"(2nd: {summary['top_neighbor_2']})")

    # --- Summary table ---
    print(f"\n{'='*70}")
    print("PENTATEUCH DATING SUMMARY")
    print('='*70)
    rows = []
    for bname, summ in books_summary.items():
        an  = summ['feature_rates'].get('lex::>NKJ', 0)
        ai  = summ['feature_rates'].get('lex::>NJ', 0)
        n   = summ['feature_rates']['n_words']
        an_c = int(an * n / 1000); ai_c = int(ai * n / 1000)
        tot = an_c + ai_c
        rows.append({
            'Book':       bname,
            'n_words':    n,
            'אנכי':       an_c,
            'אני':        ai_c,
            'frac_אנכי':  f"{an_c/tot:.3f}" if tot > 0 else 'NA',
            'MAP_BCE':    summ.get('MAP_date_BCE', '?'),
            '68%_CI':     f"{summ.get('CI68_lo_BCE','?')}–{summ.get('CI68_hi_BCE','?')}",
            'nearest':    summ.get('top_neighbor', '?'),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    print(f"\nScholarly reference dates for comparison:")
    for src, (lo, hi, label) in SCHOLARLY_DATES.items():
        print(f"  {src}: {hi}–{lo} BCE  ({label})")

    # --- Source-level analysis ---
    print(f"\n=== Documentary source analysis ===")
    source_map = DOC_SOURCES
    if args.source_file:
        with open(args.source_file) as f:
            source_map = json.load(f)

    source_results = analyse_sources(
        source_map, F, L, T, target_lex, target_vt, target_vs,
        robust_features, training_rates, date_grid, outdir)

    # Save source results
    if source_results:
        src_rows = [{'unit': k, **{kk: vv for kk, vv in v.items()
                                   if kk != 'feature_rates'}}
                    for k, v in source_results.items()]
        src_df = pd.DataFrame(src_rows)
        src_path = outdir / 'source_dating.csv'
        src_df.to_csv(str(src_path), index=False)
        print(f"\nSource dating saved: {src_path.name}")

    # --- Plot ---
    plot_pentateuch_results(books_summary, date_posteriors, date_grid,
                            training_rates, robust_features, outdir)

    # --- Interpretation caution ---
    print(f"\n{'='*70}")
    print("INTERPRETATION NOTES")
    print('='*70)
    print("""
The MAP dates above are posterior modes given the prophetic training corpus.
They should be read as: "the dated prophet with the most similar feature
profile to this Torah book was active around X BCE."

Key caveats:
1. Genre: The training data is prophetic poetry/prose.  Torah texts include
   law codes, genealogies, and narrative — genres with their own conventions.
   A 'late-looking' feature in Torah may reflect genre rather than date.

2. Composite authorship: Each Torah book mixes material of different origin.
   Whole-book predictions average across this mixture.  The D/P source
   comparison (Deuteronomy vs. Leviticus) is more interpretable because
   those books are relatively compositionally homogeneous.

3. Extrapolation: The training range is ~760–460 BCE.  Predictions well
   outside this range (e.g., < 460 BCE or > 760 BCE) involve extrapolation
   with high uncertainty.

4. The pronoun (אנכי/אני) feature is our single LOO-robust marker.
   The Bayesian model uses all robust features, but in a small corpus
   the result will be dominated by whichever features are most informative.
""")


if __name__ == '__main__':
    main()
