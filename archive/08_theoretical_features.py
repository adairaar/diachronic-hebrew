#!/usr/bin/env python3
"""
Theoretical CBH/LBH Feature Analysis
======================================
Tests theoretically motivated features from the scholarly literature on
Classical Biblical Hebrew (CBH) vs. Late Biblical Hebrew (LBH).

Unlike script 06 (data-mining the full lexical space), this script starts
from features that scholars have *independently* identified as diachronic
markers — then asks whether those features show temporal trends in our
dated prophetic corpus.

Feature groups covered
----------------------
1.  Personal pronoun pairs          (אנכי/אני and others)
2.  Relative particle               (אשר/ש)
3.  Negation system                 (לא/אל prohibition, אין/לא existential)
4.  Conjunctions & discourse-level  (כי, גם, לכן, הנה, עתה, אז)
5.  Verb form paradigm              (wayyiqtol, participle, jussive,
                                     cohortative, infinitives)
6.  Verbal stem ratios              (Qal, Hiphil, Piel, Niphal, Hithpael)
7.  LBH lexical substitutions       (בקש/שאל, קהל/עדה)
8.  High-frequency functional verbs (היה, אמר, נתן)
9.  Morphological indices           (construct state, pronominal-suffix,
                                     plural-noun, feminine-noun rates)

Literature
----------
Hurvitz, A. (1972). A Linguistic Study of the Relationship Between the
  Priestly Source and the Book of Ezekiel.  Jerusalem: Magnes.
Polzin, R. (1976). Late Biblical Hebrew: Toward an Historical Typology
  of Biblical Hebrew Prose.  Missoula: Scholars Press.
Rooker, M.F. (1990). Biblical Hebrew in Transition.  Sheffield: JSOT.
Young, I., Rezetko, R., Ehrensvärd, M. (2008). Linguistic Dating of
  Biblical Texts.  London: Equinox.
Hendel, R. & Joosten, J. (2018). How Old Is the Hebrew Bible?
  New Haven: Yale UP.
Joüon, P. & Muraoka, T. (2006). A Grammar of Biblical Hebrew. Rome: PIB.

Output
------
theoretical_features_training.csv  — feature rates for the training corpus
theoretical_features_scan.csv      — Spearman correlations + statistics
theoretical_features_plot.png      — visualisation of top features
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import spearmanr
from pathlib import Path

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Constants
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

DATED_SPECS = PROPHETIC_SPECS  # default; overridden by --corpus broad

DEFAULT_BHSA_PATH = (
    Path.home() / 'text-fabric-data' / 'github' / 'ETCBC' / 'bhsa' / 'tf' / '2021'
)

# Features to load from BHSA
TF_FEATURES = 'otype oslots otext lex sp vt vs nu gn st prs_ps chapter'


# ---------------------------------------------------------------------------
# Feature catalogue with literature references
# ---------------------------------------------------------------------------
#
# Each entry:
#   feature_name : str          — output column name
#   description  : str          — human-readable label
#   extract_type : str          — extraction method (see extract_features())
#   params       : list         — arguments for the extraction method
#   direction    : str          — 'increase' (LBH > CBH) | 'decrease' |
#                                 'unknown'
#   reference    : str          — key literature citations

FEATURE_CATALOGUE = [

    # --- Group 1: Personal pronouns -----------------------------------------
    # 1sg: אנכי (>NKJ) is the archaic/CBH form; אני (>NJ) is the modern/LBH form.
    # The fraction of אני is expected to increase from CBH to LBH.
    ('frac_ani',        'Fraction אני/(אני+אנכי) — 1sg pronoun [LBH↑]',
     'fraction_lex',   ['>NJ', '>NKJ'],   'increase',
     'Hurvitz 1972; Rezetko & Young 2014'),

    ('rate_anochi',     'אנכי rate per 1k words [CBH marker↓]',
     'rate_lex',        ['>NKJ'],          'decrease',
     'Hurvitz 1972'),

    ('rate_ani',        'אני rate per 1k words [LBH marker↑]',
     'rate_lex',        ['>NJ'],           'increase',
     'Hurvitz 1972'),

    # --- Group 2: Relative particle ------------------------------------------
    # אשר (>CR) is the standard CBH relative; ש (C, no trailing mark) is the
    # Mishnaic/LBH colloquial form.
    ('frac_she',        'Fraction ש/(ש+אשר) — relative particle [LBH↑]',
     'fraction_lex',   ['C', '>CR'],       'increase',
     'Polzin 1976; Rooker 1990'),

    ('rate_asher',      'אשר rate per 1k words',
     'rate_lex',        ['>CR'],           'decrease',
     'General CBH/LBH research'),

    # --- Group 3: Negation ---------------------------------------------------
    # Existential negation: אין (>JN/) is the LBH preference over לא (L>).
    ('frac_ein',        'Fraction אין/(אין+לא) — existential neg [LBH↑]',
     'fraction_lex',   ['>JN/', 'L>'],    'increase',
     'Hurvitz 1972'),

    # Prohibition negation: אל (>L as NEGA) is the traditional "jussive"
    # negator (modal prohibition); לא is the indicative negator.
    # In CBH prose the two are kept more distinct; LBH allows לא to encroach.
    # So the rate of אל (prohibition-specific negation) should decline in LBH.
    ('rate_neg_al',     'אל prohibition-negator rate per 1k [CBH↓]',
     'rate_nega',       ['>L'],            'decrease',
     'Joüon-Muraoka §114; Hendel & Joosten 2018'),

    ('frac_neg_al',     'Fraction אל/(אל+לא) negators [CBH form↓]',
     'fraction_nega',  ['>L', 'L>'],      'decrease',
     'Hendel & Joosten 2018'),

    ('rate_neg_lo',     'לא negator rate per 1k words',
     'rate_nega',       ['L>'],            'unknown',
     'General research'),

    # --- Group 4: Conjunctions & discourse particles -------------------------
    # כי (KJ): multifunctional subordinator (because / that / when / for).
    # Its total frequency and the distribution of its usages shifts in LBH.
    ('rate_ki',         'כי conjunction rate per 1k words',
     'rate_lex_sp',    ['KJ', 'conj'],    'unknown',
     'General CBH/LBH research'),

    # גם (GM): additive conjunction "also/even".  Frequency shifts in LBH.
    ('rate_gam',        'גם "also/even" rate per 1k words',
     'rate_lex',        ['GM'],            'unknown',
     'Young et al. 2008'),

    # לכן (LKN): "therefore/thus" — increases in later texts.
    ('rate_lakhen',     'לכן "therefore" rate per 1k [LBH↑]',
     'rate_lex',        ['LKN'],           'increase',
     'Polzin 1976'),

    # הנה (HNH/): presentative particle "behold/look".
    ('rate_hinne',      'הנה "behold" particle rate per 1k',
     'rate_lex',        ['HNH/'],          'unknown',
     'General research'),

    # עתה (<TH/): adverb "now" — may increase in later prose.
    ('rate_atta',       'עתה "now" adverb rate per 1k',
     'rate_lex',        ['<TH/'],          'unknown',
     'General LBH research'),

    # אז (>Z): adverb "then" — tends to decrease in LBH.
    ('rate_az',         'אז "then" adverb rate per 1k [CBH↓]',
     'rate_lex',        ['>Z'],            'decrease',
     'Hendel & Joosten 2018'),

    # אף (<P/? or >P/): "also/even/indeed" — distribution shifts.
    # NB: ETCBC ayin = <, so אף = >P (aleph + pe)
    ('rate_af',         'אף "also/indeed" particle rate per 1k',
     'rate_lex',        ['>P/'],           'unknown',
     'General research'),

    # --- Group 5: Verb forms -------------------------------------------------
    # The classical wayyiqtol (narrative past) declines in LBH, where the
    # Perfect tends to usurp its function.
    ('rate_wayyiqtol',  'Wayyiqtol rate per 1k words [CBH↓]',
     'rate_vt',         ['wayq'],          'decrease',
     'Polzin 1976; Hendel & Joosten 2018'),

    ('rate_qatal',      'Qatal (perfect) rate per 1k words',
     'rate_vt',         ['perf'],          'unknown',
     'General morphology'),

    ('rate_yiqtol',     'Yiqtol (imperfect) rate per 1k words',
     'rate_vt',         ['impf'],          'unknown',
     'General morphology'),

    # Active participle increases as a main predicate in LBH.
    ('rate_ptca',       'Active participle rate per 1k [LBH↑]',
     'rate_vt',         ['ptca'],          'increase',
     'Polzin 1976; Rooker 1990'),

    ('rate_ptcp',       'Passive participle rate per 1k',
     'rate_vt',         ['ptcp'],          'unknown',
     'General morphology'),

    # Infinitive absolute (emphatic/supplementary) declines sharply in LBH.
    ('rate_inf_abs',    'Infinitive absolute rate per 1k [CBH↓]',
     'rate_vt',         ['infa'],          'decrease',
     'Joüon-Muraoka §123; Rooker 1990'),

    ('rate_inf_con',    'Infinitive construct rate per 1k',
     'rate_vt',         ['infc'],          'unknown',
     'General morphology'),

    ('rate_impv',       'Imperative rate per 1k words',
     'rate_vt',         ['impv'],          'unknown',
     'General morphology'),

    # Jussive (short imperfect, 3rd/2nd person volitional) and cohortative
    # (1sg/pl volitional).  Both are morphologically distinct in BHSA.
    ('rate_jussive',    'Jussive rate per 1k words',
     'rate_vt',         ['jus'],           'unknown',
     'Joüon-Muraoka §114'),

    ('rate_cohort',     'Cohortative rate per 1k words',
     'rate_vt',         ['coho'],          'unknown',
     'Joüon-Muraoka §114'),

    # --- Group 6: Verbal stems -----------------------------------------------
    # Piel (intensive/factitive) and Hiphil (causative) rates may shift.
    # The overall verb/non-verb ratio also changes.
    ('rate_qal',        'Qal stem rate per 1k words (base stem)',
     'rate_vs',         ['qal'],           'unknown',
     'General morphology'),

    ('rate_hiphil',     'Hiphil stem rate per 1k words (causative)',
     'rate_vs',         ['hif'],           'unknown',
     'General morphology'),

    ('rate_piel',       'Piel stem rate per 1k words (intensive)',
     'rate_vs',         ['piel'],          'unknown',
     'General morphology'),

    ('rate_niphal',     'Niphal stem rate per 1k words (passive/reflexive)',
     'rate_vs',         ['nif'],           'unknown',
     'General morphology'),

    ('rate_hithpael',   'Hithpael stem rate per 1k words (reflexive)',
     'rate_vs',         ['hit'],           'unknown',
     'General morphology'),

    ('rate_hophal',     'Hophal stem rate per 1k words (passive causative)',
     'rate_vs',         ['hof'],           'unknown',
     'General morphology'),

    # --- Group 7: LBH lexical substitution pairs ----------------------------
    # בקש (BQC[, Piel "seek/request") increasingly replaces שאל (C>L[,
    # Qal "ask/enquire") in later texts.
    ('frac_baqash',     'Fraction בקש/(בקש+שאל) — "seek" [LBH↑]',
     'fraction_lex',   ['BQC[', 'C>L['],  'increase',
     'Hurvitz 1972'),

    ('rate_baqash',     'בקש "seek" rate per 1k words',
     'rate_lex',        ['BQC['],          'increase',
     'Hurvitz 1972'),

    ('rate_shaal',      'שאל "ask/seek" rate per 1k words',
     'rate_lex',        ['C>L['],          'decrease',
     'Hurvitz 1972'),

    # קהל (QHL/) "assembly" (general) vs עדה (<DH/) "congregation" (Priestly).
    # D prefers עדה; later texts prefer קהל.
    ('frac_qahal',      'Fraction קהל/(קהל+עדה) — "assembly" [LBH↑]',
     'fraction_lex',   ['QHL/', '<DH/'],   'increase',
     'Hurvitz 1972'),

    # --- Group 8: High-frequency functional verbs ---------------------------
    # These are extremely common and may show register shifts independent of topic.
    ('rate_hayah',      'היה "to be" rate per 1k words',
     'rate_lex',        ['HJH['],          'unknown',
     'General research'),

    ('rate_amar',       'אמר "to say" rate per 1k words',
     'rate_lex',        ['>MR['],          'unknown',
     'General research'),

    ('rate_natan',      'נתן "to give" rate per 1k words',
     'rate_lex',        ['NTN/'],          'unknown',
     'General research'),

    ('rate_halak',      'הלך "to go/walk" rate per 1k words',
     'rate_lex',        ['HLK['],          'unknown',
     'General research'),

    # --- Group 9: Morphological indices ------------------------------------
    # Construct state (smixut) rate: proportion of nouns in construct form.
    ('rate_const',      'Construct-state noun rate per 1k words',
     'morph_const',     [],                'unknown',
     'General morphology'),

    # Pronominal suffix rate: proportion of words bearing a nominal/verbal suffix.
    ('rate_prs',        'Pronominal-suffix rate per 1k words',
     'morph_prs',       [],                'unknown',
     'General morphology'),

    # Plural noun rate.
    ('rate_pl_noun',    'Plural noun rate per 1k words',
     'morph_pl_noun',   [],                'unknown',
     'General morphology'),

    # Feminine noun rate.
    ('rate_f_noun',     'Feminine noun rate per 1k words',
     'morph_f_noun',    [],                'unknown',
     'General morphology'),
]


# ---------------------------------------------------------------------------
# Feature extraction from a DataFrame of word rows
# ---------------------------------------------------------------------------

def extract_all_features(df, n):
    """
    Extract all theoretical features from a word-level DataFrame.

    Parameters
    ----------
    df : DataFrame with columns lex, sp, vt, vs, nu, gn, st, prs_ps
    n  : int — total word count (for rate normalisation)

    Returns
    -------
    dict : feature_name → value
    """
    rates = {}
    per = 1000.0 / max(n, 1)

    verb_df = df[df['sp'] == 'verb']
    noun_df = df[df['sp'] == 'subs']
    nega_df = df[df['sp'] == 'nega']

    for (name, desc, ext_type, params, direction, ref) in FEATURE_CATALOGUE:

        if ext_type == 'rate_lex':
            # Raw count of specific lexeme(s)
            cnt = df['lex'].isin(params).sum()
            rates[name] = cnt * per

        elif ext_type == 'rate_lex_sp':
            # Count of lexeme(s) where sp also matches
            lex_codes, sp_code = params[:-1], params[-1]
            cnt = (df['lex'].isin(lex_codes) & (df['sp'] == sp_code)).sum()
            rates[name] = cnt * per

        elif ext_type == 'fraction_lex':
            # numerator / (numerator + denominator)
            num_lex, den_lex = params[0], params[1]
            num = (df['lex'] == num_lex).sum()
            den = (df['lex'] == den_lex).sum()
            total = num + den
            rates[name] = num / total if total >= 1 else float('nan')

        elif ext_type == 'rate_nega':
            # Rate of specific negator lexeme(s)
            cnt = nega_df['lex'].isin(params).sum()
            rates[name] = cnt * per

        elif ext_type == 'fraction_nega':
            # Fraction of (negator A) / (negator A + negator B)
            num_lex, den_lex = params[0], params[1]
            num = (nega_df['lex'] == num_lex).sum()
            den = (nega_df['lex'] == den_lex).sum()
            total = num + den
            rates[name] = num / total if total >= 1 else float('nan')

        elif ext_type == 'rate_vt':
            # Verb-form rate per 1k total words
            cnt = verb_df['vt'].isin(params).sum()
            rates[name] = cnt * per

        elif ext_type == 'rate_vs':
            # Verbal-stem rate per 1k total words
            cnt = verb_df['vs'].isin(params).sum()
            rates[name] = cnt * per

        elif ext_type == 'morph_const':
            cnt = (noun_df['st'] == 'c').sum()
            rates[name] = cnt * per

        elif ext_type == 'morph_prs':
            cnt = (df['prs_ps'] != 'NA').sum()
            rates[name] = cnt * per

        elif ext_type == 'morph_pl_noun':
            cnt = (noun_df['nu'] == 'pl').sum()
            rates[name] = cnt * per

        elif ext_type == 'morph_f_noun':
            cnt = (noun_df['gn'] == 'f').sum()
            rates[name] = cnt * per

    return rates


# ---------------------------------------------------------------------------
# Load word-level data for dated units
# ---------------------------------------------------------------------------

def load_corpus(data_path):
    """
    Load BHSA and return TF API objects.
    """
    try:
        from tf.fabric import Fabric
    except ImportError:
        print("ERROR: pip install text-fabric"); sys.exit(1)

    data_path = Path(data_path).expanduser()
    if not data_path.exists():
        print(f"ERROR: BHSA not found at {data_path}"); sys.exit(1)

    TF  = Fabric(locations=str(data_path), silent=True)
    api = TF.load(TF_FEATURES, silent=True)
    return api.F, api.L, api.T


def extract_unit(unit_name, book_names, chap_range, F, L, T):
    """
    Return a DataFrame of word-level features for one text unit.
    """
    rows = []
    for bname in book_names:
        bn = T.nodeFromSection((bname,))
        if bn is None:
            print(f"  WARNING: '{bname}' not found — skipping.")
            continue
        for ch in L.d(bn, 'chapter'):
            ch_num = F.chapter.v(ch)
            if chap_range == 'oracle':
                if ch_num not in JEREMIAH_ORACLE_CHAPTERS:
                    continue
            elif chap_range and not (chap_range[0] <= ch_num <= chap_range[1]):
                continue
            for w in L.d(ch, 'word'):
                rows.append((
                    F.lex.v(w), F.sp.v(w), F.vt.v(w), F.vs.v(w),
                    F.nu.v(w),  F.gn.v(w), F.st.v(w), F.prs_ps.v(w),
                ))
    if not rows:
        return None
    return pd.DataFrame(rows, columns=['lex', 'sp', 'vt', 'vs', 'nu', 'gn', 'st', 'prs_ps'])


# ---------------------------------------------------------------------------
# Correlation scan
# ---------------------------------------------------------------------------

def scan_features(rates_df, dates):
    """
    Spearman ρ between each feature and date (negative = decreasing over time).
    Returns a DataFrame sorted by |ρ|.
    """
    results = []
    for col in rates_df.columns:
        vals = rates_df[col].values
        valid = np.isfinite(vals)
        if valid.sum() < 5 or vals[valid].std() < 1e-10:
            continue
        r, p = spearmanr(-dates[valid], vals[valid])
        results.append({
            'feature': col, 'rho': r, 'p_raw': p,
            'n_valid': int(valid.sum()),
        })

    scan = pd.DataFrame(results).sort_values('p_raw').reset_index(drop=True)

    # Attach descriptions and expected-direction from catalogue
    desc_map      = {row[0]: row[1] for row in FEATURE_CATALOGUE}
    direction_map = {row[0]: row[4] for row in FEATURE_CATALOGUE}
    ref_map       = {row[0]: row[5] for row in FEATURE_CATALOGUE}
    scan['description'] = scan['feature'].map(desc_map)
    scan['expected_dir']= scan['feature'].map(direction_map)
    scan['reference']   = scan['feature'].map(ref_map)

    # Flag consistency with expected direction
    def consistent(row):
        exp = row['expected_dir']
        if exp == 'increase': return row['rho'] > 0
        if exp == 'decrease': return row['rho'] < 0
        return None   # 'unknown'
    scan['dir_consistent'] = scan.apply(consistent, axis=1)

    return scan


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_theoretical_features(scan, rates_df, dates, unit_names, outdir, n_top=20):
    """
    Grid of scatter plots for the top-N features by |ρ|.
    """
    top = scan.dropna(subset=['p_raw']).head(n_top)
    ncols = 4
    nrows = int(np.ceil(len(top) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

    for i, (_, row) in enumerate(top.iterrows()):
        ax  = axes[i]
        col = row['feature']
        if col not in rates_df.columns:
            ax.axis('off'); continue

        vals = rates_df[col].values
        valid = np.isfinite(vals)
        x = dates[valid]; y = vals[valid]

        # Colour by period
        colors = ['steelblue' if d >= 580 else
                  ('darkorange' if d >= 520 else 'crimson')
                  for d in x]
        ax.scatter(x, y, c=colors, s=40, zorder=3, alpha=0.8)
        for j, u in enumerate(np.array(unit_names)[valid]):
            ax.annotate(u[:7], (x[j], y[j]),
                        xytext=(2, 2), textcoords='offset points', fontsize=5)

        # Trend line
        if len(x) >= 4:
            z = np.polyfit(x, y, 1)
            xr = np.linspace(x.min(), x.max(), 100)
            ax.plot(xr, np.polyval(z, xr), 'k--', linewidth=1, alpha=0.5)

        ax.invert_xaxis()
        star = ('**' if row['p_raw'] < 0.01 else
                '*'  if row['p_raw'] < 0.05 else
                '(.)' if row['p_raw'] < 0.10 else '')
        consistent = '✓' if row['dir_consistent'] else ('✗' if row['dir_consistent'] is False else '?')
        ax.set_title(f"{col}\nρ={row['rho']:.2f} p={row['p_raw']:.3f}{star}  {consistent}",
                     fontsize=7)
        ax.set_xlabel('Date BCE', fontsize=6)
        ax.set_ylabel(col.split('_', 1)[-1][:18], fontsize=6)
        ax.tick_params(labelsize=6)

    # Hide unused panels
    for j in range(len(top), len(axes)):
        axes[j].axis('off')

    # Legend for period colours
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue',   label='pre-exilic (>580 BCE)'),
        Patch(facecolor='darkorange',  label='exilic (520–580 BCE)'),
        Patch(facecolor='crimson',     label='post-exilic (<520 BCE)'),
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=8,
               bbox_to_anchor=(0.98, 0.01))

    fig.suptitle(f'Theoretical CBH/LBH Features — Top {len(top)} by |ρ|  '
                 f'(★ p<0.05; ✓ = direction consistent with theory)',
                 fontsize=11)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_path = outdir / 'theoretical_features_plot.png'
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Plot saved: {out_path.name}")


def plot_summary_bars(scan, outdir):
    """
    Horizontal bar chart: all features, coloured by significance and
    direction consistency.
    """
    df = scan.dropna(subset=['p_raw']).copy()
    df = df.sort_values('rho')

    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.35)))
    colors = []
    for _, row in df.iterrows():
        if row['p_raw'] < 0.05 and row['dir_consistent'] is True:
            colors.append('steelblue')
        elif row['p_raw'] < 0.05 and row['dir_consistent'] is False:
            colors.append('tomato')
        elif row['p_raw'] < 0.10:
            colors.append('goldenrod')
        else:
            colors.append('lightgray')

    y = np.arange(len(df))
    ax.barh(y, df['rho'].values, color=colors, edgecolor='white', linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(df['feature'].values, fontsize=7)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Spearman ρ  (positive = increases over time / toward LBH)')
    ax.set_title('Theoretical CBH/LBH Features: Spearman correlations with date\n'
                 'Blue = p<0.05 direction-consistent  '
                 'Red = p<0.05 opposite direction  '
                 'Gold = p<0.10  Gray = n.s.')
    plt.tight_layout()
    out_path = outdir / 'theoretical_features_bars.png'
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Summary bar chart saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-path', default=str(DEFAULT_BHSA_PATH),
                        help='Path to BHSA tf/2021 directory')
    parser.add_argument('--outdir', default='.',
                        help='Directory for output files (default: .)')
    parser.add_argument('--corpus', choices=['prophetic', 'broad'], default='prophetic',
                        help='"prophetic" (760–460 BCE, default) or "broad" (adds late '
                             'narrative to 167 BCE)')
    parser.add_argument('--append-to-training', action='store_true',
                        help='Append theoretical features to feature_rates_training.csv '
                             'so script 07 can use them')
    args = parser.parse_args()

    global DATED_SPECS
    DATED_SPECS = BROAD_SPECS if args.corpus == 'broad' else PROPHETIC_SPECS
    print(f"Using {'BROAD' if args.corpus == 'broad' else 'PROPHETIC'} corpus "
          f"({len(DATED_SPECS)} units)")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load corpus
    print(f"Loading BHSA from {args.data_path}...")
    F, L, T = load_corpus(args.data_path)

    # Extract features for each dated unit
    print(f"\nExtracting theoretical features for {len(DATED_SPECS)} dated units...")
    meta_rows  = []
    rates_rows = []
    for (unit_name, book_names, chap_range, date, sigma) in DATED_SPECS:
        df = extract_unit(unit_name, book_names, chap_range, F, L, T)
        if df is None:
            print(f"  WARNING: no words for {unit_name}")
            continue
        n = len(df)
        feat = extract_all_features(df, n)
        meta_rows.append({'unit': unit_name, 'date_bce': date,
                          'date_sigma': sigma, 'n_words': n})
        rates_rows.append({'unit': unit_name, **feat})
        print(f"  {unit_name:<16}  {n:>7,} words  "
              f"frac_ani={feat.get('frac_ani', float('nan')):.3f}  "
              f"rate_wayyiqtol={feat.get('rate_wayyiqtol', float('nan')):.1f}  "
              f"rate_jus={feat.get('rate_jussive', float('nan')):.2f}")

    meta_df  = pd.DataFrame(meta_rows).set_index('unit')
    rates_df = pd.DataFrame(rates_rows).set_index('unit')

    # Save training rates
    out_rates = outdir / 'theoretical_features_training.csv'
    rates_df.to_csv(str(out_rates), index=True, index_label='unit')
    print(f"\nTraining rate matrix saved: {out_rates.name}  "
          f"({len(rates_df)} units × {len(rates_df.columns)} features)")

    # Correlation scan
    dates = meta_df['date_bce'].values
    unit_names = list(meta_df.index)
    print("\nRunning Spearman correlation scan...")
    scan = scan_features(rates_df, dates)

    n_sig    = (scan['p_raw'] < 0.05).sum()
    n_trend  = (scan['p_raw'] < 0.10).sum()
    n_consist= ((scan['p_raw'] < 0.05) & (scan['dir_consistent'] == True)).sum()
    n_oppose = ((scan['p_raw'] < 0.05) & (scan['dir_consistent'] == False)).sum()
    print(f"  {len(scan)} features scanned")
    print(f"  {n_sig} with p < 0.05  (of these {n_consist} direction-consistent, "
          f"{n_oppose} opposite to theory)")
    print(f"  {n_trend} with p < 0.10")

    # Save scan
    out_scan = outdir / 'theoretical_features_scan.csv'
    scan.to_csv(str(out_scan), index=False)
    print(f"Scan results saved: {out_scan.name}")

    # Print top results
    print(f"\n{'='*80}")
    print("TOP THEORETICAL FEATURES BY |ρ|")
    print('='*80)
    show = scan.head(20)[['feature', 'rho', 'p_raw', 'n_valid',
                           'expected_dir', 'dir_consistent', 'reference']]
    pd.set_option('display.max_colwidth', 40)
    print(show.to_string(index=False))

    print(f"\n{'='*80}")
    print("DIRECTION-CONSISTENT SIGNIFICANT FEATURES  (p<0.05 + matches theory)")
    print('='*80)
    good = scan[(scan['p_raw'] < 0.05) & (scan['dir_consistent'] == True)]
    if len(good) == 0:
        print("  None at p<0.05.  Showing p<0.10:")
        good = scan[(scan['p_raw'] < 0.10) & (scan['dir_consistent'] == True)]
    if len(good) > 0:
        print(good[['feature', 'rho', 'p_raw', 'expected_dir', 'reference']].to_string(index=False))
    else:
        print("  None found.")

    print(f"\n{'='*80}")
    print("DIRECTION-INCONSISTENT SIGNIFICANT FEATURES  (p<0.05 but opposite to theory)")
    print('='*80)
    bad = scan[(scan['p_raw'] < 0.05) & (scan['dir_consistent'] == False)]
    if len(bad) > 0:
        print(bad[['feature', 'rho', 'p_raw', 'expected_dir', 'reference']].to_string(index=False))
        print("\n  NOTE: Opposite-direction results do not necessarily invalidate the")
        print("  theory.  In prophetic texts, a feature may be genre-constrained")
        print("  and not show the same diachronic trajectory as in narrative prose.")
    else:
        print("  None.")

    # Optionally append to script-06/07 training rates
    if args.append_to_training:
        existing_path = outdir / 'feature_rates_training.csv'
        if existing_path.exists():
            existing = pd.read_csv(existing_path, index_col='unit')
            # Merge, keeping only the new theoretical features not already present
            new_cols = [c for c in rates_df.columns if c not in existing.columns]
            if new_cols:
                merged = existing.join(rates_df[new_cols], how='left')
                merged.to_csv(str(existing_path), index=True, index_label='unit')
                print(f"\nAppended {len(new_cols)} new features to {existing_path.name}")
            else:
                print("\nNo new features to append (all already present).")
        else:
            print(f"\nWARNING: {existing_path} not found; run script 06 first.")

    # Plots
    plot_theoretical_features(scan, rates_df, dates, unit_names, outdir, n_top=20)
    plot_summary_bars(scan, outdir)

    print("\nDone.")


if __name__ == '__main__':
    main()
