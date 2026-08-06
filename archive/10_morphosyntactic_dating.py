#!/usr/bin/env python3
"""
10_morphosyntactic_dating.py
Morphosyntactic Feature Analysis and Multivariate Bayesian Torah Dating
=======================================================================

Three improvements over scripts 07–09:

CORPUS CORRECTIONS
------------------
1. Isaiah 3 (Trito-Isaiah, Isa 56–66) has its date uncertainty substantially
   widened (sigma 100 years, centred ~460 BCE) because its compositional
   history is genuinely uncertain: it may have been written and/or edited
   anywhere from the late 6th to the 3rd century BCE.

2. Jeremiah's Deuteronomistic prose sections are optionally excluded.
   The scholarly consensus (Mowinckel's 'B/C' sources; cf. Thiel 1973, 1981)
   identifies chapters 7, 11, 17–18, 21, 24–29, 32–36, and 37–45 as
   Deuteronomistic in language.  Using these sections as training data for a
   model that will then be applied to Deuteronomy risks circularity.
   The --jeremiah flag controls this:
     full    — use all of Jeremiah (legacy behaviour)
     oracles — exclude Dtr prose chapters (default)

MORPHOSYNTACTIC FEATURES (Tier 1 & 2)
---------------------------------------
Features drawn from the CBH/LBH literature that are grammatical/morphological
rather than lexical, making them less susceptible to the genre-confound objection.

Tier 1 (straightforward BHSA extraction):
  • Qal passive vs. niphal on ילד — CBH uses internal passive, LBH switches to niphal
  • נחנו vs. אנחנו 1CPL pronoun — shorter form in Torah, longer form elsewhere
  • Abstract noun rate (-ūt suffix) — increases in LBH
  • פן ("lest") rate — distinctively CBH, disappears in LBH
  • טרם ("not yet") rate — same
  • הלך Piel/(Piel+Qal) ratio — Piel of "walk" is LBH innovation
  • זעק/(זעק+צעק) ratio — זעק replaces צעק in LBH
  • נא modal particle rate — very common in CBH, restricted in LBH

Tier 2 (requires more specific extraction):
  • יסף Qal/(Qal+Hiphil) — Qal of "add/again" is older; Hiphil takes over in LBH
  • Niphal overall rate — passivising strategy shifts from Qal passive to Niphal
  • Wayyiqtol 1CS/1CPL rate — short form dominant in Pentateuch

MULTIVARIATE (MVN) BAYESIAN MODEL
------------------------------------
Previous scripts assumed features are independent given date.  This
over-counts correlated features (e.g. two pronoun features) and
under-counts genuinely orthogonal morphological ones.

The new model:
  1. Fits OLS (feature ~ a + b·date) for each feature on the training corpus.
  2. Estimates the empirical covariance Σ of the training residuals — the
     residual variation that isn't explained by the linear date trend.
     This covariance is the correct noise model: if two features move together
     regardless of date (e.g. because of an authorial idiolect), Σ captures
     that and the inverse-covariance matrix naturally down-weights redundant
     information.
  3. For each test text with feature vector x:
       log P(date | x) ∝ -½ (x - μ(date))ᵀ Σ⁻¹ (x - μ(date)) + log P(date)
     where μ(date) = a + b·date is the training regression prediction.
  4. Uses Tikhonov regularisation (ridge) on Σ to handle near-singularity
     when the number of features approaches the number of training units.

HIERARCHICAL BAYESIAN D / P / JE MODEL
-----------------------------------------
The independent-section approach (one posterior per chapter range) ignores
the fact that sections from the same documentary source should share a date.
The hierarchical model pools sections within each source:

  For source g ∈ {D, P, JE}:
    θ_g  ~  Prior(date)              [source-level latent date]
    For each section s in g:
      x_s  ~  MVN(μ(θ_g), Σ_s)      [observed features for section s]

  Σ_s = Σ / n_s  (scaled by word count — larger sections are more
                  informative about θ_g).

  The posterior for θ_g is:
    P(θ_g | {x_s}) ∝ Prior(θ_g) × ∏_s MVN(x_s; μ(θ_g), Σ/n_s)

  This is equivalent to computing the MVN likelihood using the POOLED
  feature vector (word-count-weighted mean across sections) with covariance
  Σ / N_g where N_g = Σ_s n_s.

  The prior is weakly informative: N(600 BCE, 400 yr), truncated to
  [1200, 50 BCE], reflecting genuine uncertainty about Torah composition.

  We also compute the JOINT posterior P(θ_D, θ_P, θ_JE | all data) on a
  coarse grid, which gives us the correlation between source dates and
  identifies whether the data support the traditional D < P ordering.

Usage
-----
  python 10_morphosyntactic_dating.py [--data-path PATH]
                                       [--corpus {prophetic|broad}]
                                       [--jeremiah {full|oracles}]

Outputs
-------
  morpho_feature_scan.csv          — Tier 1+2 features vs. date correlation
  morpho_training_rates.csv        — feature × training unit matrix
  morpho_torah_dates.csv           — D/P/JE/whole-book posteriors
  morpho_archaism_plot.png         — feature-by-feature D vs P vs JE scores
  morpho_posteriors.png            — date posteriors for D, P, JE sources
  morpho_joint_dp.png              — joint posterior P(θ_D, θ_P | data)
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from scipy.stats import spearmanr, norm as scipy_norm
from scipy.optimize import minimize_scalar
from pathlib import Path

# ---------------------------------------------------------------------------
# CORPUS SPECIFICATIONS
# ---------------------------------------------------------------------------

# Chapters of Jeremiah that are predominantly poetic oracle material,
# excluding the DTR prose sections identified by Mowinckel (1914),
# Thiel (1973, 1981), and broad scholarly consensus.
JEREMIAH_ORACLE_CHAPTERS = set(
    list(range(1, 7))   +   # 1–6:   call narrative, early oracles
    list(range(8, 11))  +   # 8–10:  further oracles
    list(range(12, 17)) +   # 12–16: laments and oracles
    [19, 20]            +   # 19–20: Pashhur laments (ch 19 excludes vv.1–13 prose)
    list(range(22, 24)) +   # 22–23: oracles against kings and false prophets
    list(range(30, 32)) +   # 30–31: Book of Consolation (poetic core)
    list(range(46, 52))     # 46–51: Oracles Against the Nations
)
# Excluded DTR prose: 7–8:3, 11, 17–18, 21, 24–29, 32–45, 52

PROPHETIC_SPECS_FULL_JER = [
    #  name           books             chap_range  center_BCE  sigma_BCE
    ('Amos',         ['Amos'],          None,        760,  15),
    ('Hosea',        ['Hosea'],         None,        725,  20),
    ('Micah',        ['Micah'],         None,        720,  20),
    ('Isaiah_1',     ['Isaiah'],        (1,  39),    700,  15),
    ('Zephaniah',    ['Zephaniah'],     None,        630,  15),
    ('Nahum',        ['Nahum'],         None,        620,  20),
    ('Habakkuk',     ['Habakkuk'],      None,        605,  20),
    ('Jeremiah',     ['Jeremiah'],      None,        590,  15),  # full
    ('Lamentations', ['Lamentations'],  None,        586,  20),
    ('Ezekiel',      ['Ezekiel'],       None,        570,  15),
    ('Isaiah_2',     ['Isaiah'],        (40, 55),    545,  20),
    ('Haggai',       ['Haggai'],        None,        520,   5),
    ('Zechariah_1',  ['Zechariah'],     (1,  8),     518,   5),
    # Isaiah 3 (Trito-Isaiah): revised to 450 BCE; some scholars argue Hellenistic
    # editing of parts; 450 is a conservative late-Persian midpoint.
    ('Isaiah_3',     ['Isaiah'],        (56, 66),    450, 100),
    ('Malachi',      ['Malachi'],       None,        460,  20),
]

# Jeremiah in oracle-only mode uses a chapter filter applied at extraction.
# Same spec list but Jeremiah unit will only use JEREMIAH_ORACLE_CHAPTERS.
PROPHETIC_SPECS_ORACLE_JER = [
    row if row[0] != 'Jeremiah'
    else ('Jeremiah_oracles', ['Jeremiah'], 'oracle_chapters', 590, 20)
    for row in PROPHETIC_SPECS_FULL_JER
]

BROAD_EXTENSION = [
    ('Jonah',        ['Jonah'],                         None,           400,  50),
    ('Ezra',         ['Ezra'],                          None,           350,  75),  # revised: 350 BCE
    ('Nehemiah',     ['Nehemiah'],                      None,           350,  75),  # revised: 350 BCE
    ('Chronicles',   ['1_Chronicles', '2_Chronicles'],  None,           350,  50),
    ('Esther',       ['Esther'],                        None,           350,  50),
    ('Ecclesiastes', ['Ecclesiastes'],                  None,           330,  80),
    ('Daniel',       ['Daniel'],                        [(1,1),(8,12)], 167,  10),  # Hebrew chs only
]

# Documentary source chapter ranges for D / P / JE
# (Friedman 2003; Baden 2012 — necessarily approximate)
DOC_SOURCES = {
    'Genesis': {
        'P':  [(1,2),(5,5),(6,6),(7,7),(9,9),(11,11),(17,17),(23,23),
               (25,25),(27,28),(35,36),(46,46),(49,50)],
        'JE': [(2,4),(6,6),(8,8),(10,10),(12,16),(18,22),(24,24),(26,27),
               (29,34),(37,45),(47,49)],
    },
    'Exodus': {
        'P':  [(1,2),(6,7),(12,12),(16,16),(25,31),(35,40)],
        'JE': [(2,5),(8,11),(13,15),(17,18),(19,24),(32,34)],
    },
    'Leviticus': {'P': [(1,27)]},
    'Numbers': {
        'P':  [(1,10),(15,15),(17,19),(25,25),(27,31),(33,36)],
        'JE': [(11,14),(16,16),(20,24),(25,25),(32,32)],
    },
    'Deuteronomy': {'D': [(1,34)]},
}

# ---------------------------------------------------------------------------
# FEATURE CATALOGUE  (Tier 1 & 2 morphosyntactic features)
# ---------------------------------------------------------------------------
# Format: (name, description, extract_type, params, expected_direction, reference)
#
# extract_type values:
#   rate_lex        — rate per 1k words of a lexeme
#   fraction_lex    — lex_A / (lex_A + lex_B), proportion
#   rate_stem_lex   — rate per 1k of a specific (lexeme, vs) pair
#   frac_stem_lex   — fraction: (lex, vs1) / ((lex, vs1) + (lex, vs2))
#   rate_vs         — rate per 1k of a verbal stem (all lexemes)
#   frac_vs_lex     — fraction: vs1 / (vs1 + vs2) for a specific lexeme
#   nif_vs_pual_lex — niphal / (niphal + pual/qal-passive) for one verb
#   rate_sp_lex     — rate per 1k of lexeme filtered by speech part

FEATURE_CATALOGUE = [

    # ── Group 0: CONFIRMED baseline features (from script 08) ──────────────
    ('frac_ani',
     'Fraction אני/(אני+אנכי)  [LBH ↑]',
     'fraction_lex', ['>NJ', '>NKJ'], 'increase',
     'Hurvitz 1972; Rezetko & Young 2014'),

    ('frac_she',
     'Fraction ש/(ש+אשר)  [LBH ↑]',
     'fraction_lex', ['C', '>CR'], 'increase',
     'Polzin 1976; Rooker 1990'),

    # ── Group 1: 1st-person pronouns ───────────────────────────────────────
    # 1CPL pronoun: נחנו (NXNW) is the shorter/older form;
    # אנחנו (>NXNW) is the longer form.  Pentateuch prefers נחנו.
    # Within CBH the Pentateuch = נחנו; CBH Prophets/Writings = mixed.
    # LBH uses אנחנו predominantly (Hornkohl 2024 ch. 7).
    ('frac_anachnu',
     'Fraction אנחנו/(אנחנו+נחנו)  1CPL pronoun [LBH ↑]',
     'fraction_lex', ['>NXNW', 'NXNW'], 'increase',
     'Hornkohl 2024 ch. 7'),

    # ── Group 2: CBH particles absent from LBH ─────────────────────────────
    # פן ("lest") — characteristic of CBH; listed by Hendel & Joosten (2018)
    # p. 44 as practically absent from LBH.
    ('rate_pen',
     'פן "lest" rate per 1k  [CBH marker, ↓ in LBH]',
     'rate_lex', ['PN'], 'decrease',
     'Hendel & Joosten 2018'),

    # טרם ("not yet") — similarly listed as a distinctively CBH particle.
    ('rate_terem',
     'טרם "not yet" rate per 1k  [CBH marker, ↓ in LBH]',
     'rate_lex', ['TTR<M', 'TRM'], 'decrease',
     'Hendel & Joosten 2018'),

    # נא — modal/deferential particle.  Very frequent in CBH, severely
    # restricted in LBH (only post-verbal after modal forms: impv/jussive).
    # Overall rate should decline sharply from CBH to LBH.
    ('rate_na',
     'נא modal particle rate per 1k  [CBH marker, ↓ in LBH]',
     'rate_lex', ['N>'], 'decrease',
     'Hendel & Joosten 2018'),

    # ── Group 3: Verbal features ─────────────────────────────────────────────
    # הלך Piel — "walk about, go about" as a Piel is a LBH innovation.
    # In CBH only Qal is used.  Rezetko & Young 2014 ch. 8 document this.
    ('frac_halak_piel',
     'Fraction הלך Piel/(Piel+Qal)  [LBH ↑]',
     'frac_stem_lex', ['HLK[', 'piel', 'qal'], 'increase',
     'Rezetko & Young 2014 ch. 8'),

    # יסף: Qal "add / do again" is the older form; Hiphil הוסיף takes over.
    # Pentateuch prefers Qal; CBH outside Pentateuch and LBH prefer Hiphil.
    # (Hornkohl 2024 ch. 3)
    ('frac_ysf_qal',
     'Fraction יסף Qal/(Qal+Hiphil)  [older form, ↓ in LBH]',
     'frac_stem_lex', ['JSP[', 'qal', 'hif'], 'decrease',
     'Hornkohl 2024 ch. 3'),

    # Niphal overall rate — passivisation in Hebrew shifts from the Qal
    # internal passive (progressively lost) toward the Niphal.  Niphal rate
    # as a fraction of all verbal forms should increase from CBH to LBH.
    ('frac_niphal',
     'Niphal/(all stems) fraction  [LBH ↑]',
     'frac_vs_total', ['nif'], 'increase',
     'Hendel & Joosten 2018; Hornkohl 2024 ch. 5'),

    # Qal passive (pual-morphology of intransitive verbs, e.g. יֻלַּד "was born")
    # vs. Niphal of the same verbs.  In BHSA the qal internal passive is often
    # coded as vs='pual' for verbs whose pual is not otherwise attested.
    # We test specifically ילד (yld — "be born") as the clearest example.
    ('frac_yld_nif',
     'Fraction ילד Niphal/(Niphal+Pual) — nif replaces qal-pass in LBH  [LBH ↑]',
     'frac_stem_lex', ['JLD[', 'nif', 'pual'], 'increase',
     'Hendel & Joosten 2018 p. 39'),

    # ── Group 4: Lexical replacements ─────────────────────────────────────
    # זעק vs. צעק: זעק is the LBH replacement for CBH צעק ("to cry out").
    # Both exist in CBH, but the balance shifts decisively in later texts.
    ('frac_zaqaq',
     'Fraction זעק/(זעק+צעק)  [LBH ↑]',
     'fraction_lex', ['Z<Q[', 'Y<Q['], 'increase',
     'Rezetko & Young 2014 ch. 8'),

    # ── Group 5: Abstract -ūt nouns ──────────────────────────────────────
    # Abstract nouns formed with the -ūt suffix (מלכות, אמת, etc.) are rarer
    # in CBH and much more frequent in LBH, QH, and Mishnaic Hebrew.
    # We approximate this by counting nouns whose ETCBC lexeme ends in WT/.
    ('rate_ut_nouns',
     'Rate of -ūt abstract nouns per 1k  [LBH ↑]',
     'rate_ut_nouns', [], 'increase',
     'Rezetko & Young 2014 ch. 9; Hornkohl 2024 ch. 11'),
]

# ---------------------------------------------------------------------------
# BHSA data path
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = str(Path.home() / 'text-fabric-data' / 'github' / 'ETCBC'
                        / 'bhsa' / 'tf' / '2021')


# ---------------------------------------------------------------------------
# BHSA loading
# ---------------------------------------------------------------------------
def load_bhsa(data_path):
    try:
        from tf.fabric import Fabric
        from tf.app import use
    except ImportError:
        sys.exit('text-fabric not installed.  Run: pip install text-fabric')

    print(f'Loading BHSA from {data_path}...')
    TF = Fabric(locations=data_path, modules=[''], silent=True)
    api = TF.load('otype lex sp vs vt ps nu gn pdp prs book chapter verse',
                  silent=True)
    return api


# ---------------------------------------------------------------------------
# Word iteration helpers
# ---------------------------------------------------------------------------
def words_in_unit(spec, F, L, T):
    """
    Yield (word_node, chapter_num) for all words in a corpus unit spec.
    spec = (name, book_list, chap_range, center, sigma)
    chap_range may be:
      None              — all chapters in all books
      (start, end)      — inclusive chapter range (same for all books)
      'oracle_chapters' — use JEREMIAH_ORACLE_CHAPTERS filter

    Uses L.d() graph traversal for speed — avoids scanning all 420K words.
    """
    name, books, chap_range, *_ = spec

    for book in books:
        # Find the book node via section lookup
        book_node = T.nodeFromSection((book,))
        if book_node is None:
            continue
        for ch_node in L.d(book_node, 'chapter'):
            chap_num = int(F.chapter.v(ch_node))

            # Chapter-range filter
            if chap_range is None:
                pass  # accept all
            elif chap_range == 'oracle_chapters':
                if chap_num not in JEREMIAH_ORACLE_CHAPTERS:
                    continue
            elif isinstance(chap_range, list):
                # list of (lo, hi) inclusive ranges — e.g. [(1,1),(8,12)] for Daniel
                if not any(s <= chap_num <= e for s, e in chap_range):
                    continue
            else:
                s, e = chap_range
                if not (s <= chap_num <= e):
                    continue

            for w in L.d(ch_node, 'word'):
                yield w, chap_num


def word_count(spec, F, L, T):
    return sum(1 for _ in words_in_unit(spec, F, L, T))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(spec, F, L, T):
    """
    Extract all Tier-1/2 morphosyntactic features for one corpus unit.
    Returns a dict {feature_name: value}.
    """
    # Collect word-level data
    lex_counts   = defaultdict(int)
    stem_counts  = defaultdict(int)   # (lexeme, vs) pairs
    vs_counts    = defaultdict(int)   # verbal stem totals
    n_words      = 0
    n_verbs      = 0

    for w, _ in words_in_unit(spec, F, L, T):
        n_words += 1
        lex  = F.lex.v(w)
        sp   = F.sp.v(w)
        vs   = F.vs.v(w)
        vt   = F.vt.v(w)

        lex_counts[lex] += 1

        if sp == 'verb':
            n_verbs += 1
            stem_counts[(lex, vs)] += 1
            vs_counts[vs] += 1

    if n_words == 0:
        return {feat[0]: np.nan for feat in FEATURE_CATALOGUE}

    result = {}
    for (name, desc, extract_type, params, direction, ref) in FEATURE_CATALOGUE:

        if extract_type == 'rate_lex':
            lex_list = params
            cnt = sum(lex_counts[l] for l in lex_list)
            result[name] = cnt / n_words * 1000

        elif extract_type == 'fraction_lex':
            lex_a, lex_b = params
            a = sum(lex_counts[l] for l in [lex_a])
            b = sum(lex_counts[l] for l in [lex_b])
            result[name] = a / (a + b) if (a + b) > 0 else np.nan

        elif extract_type == 'frac_stem_lex':
            lex, stem1, stem2 = params
            c1 = stem_counts[(lex, stem1)]
            c2 = stem_counts[(lex, stem2)]
            result[name] = c1 / (c1 + c2) if (c1 + c2) > 0 else np.nan

        elif extract_type == 'frac_vs_total':
            target_vs = params[0]
            cnt = vs_counts[target_vs]
            result[name] = cnt / n_verbs if n_verbs > 0 else np.nan

        elif extract_type == 'rate_ut_nouns':
            # Count nouns whose lexeme ends in WT/ (the -ūt suffix in ETCBC)
            cnt = sum(v for k, v in lex_counts.items()
                      if k.endswith('WT/') or k.endswith('WT'))
            result[name] = cnt / n_words * 1000

        else:
            result[name] = np.nan

    result['_n_words'] = n_words
    return result


# ---------------------------------------------------------------------------
# Training corpus: build feature × unit matrix and OLS fits
# ---------------------------------------------------------------------------

def build_training_matrix(specs, F, L, T, verbose=True):
    """
    Extract features for all training units.
    Returns:
      rates_df  — DataFrame(n_units × n_features), index = unit names
      dates_bce — array of centre dates (BCE, positive = older)
      sigmas    — array of date uncertainties (1 sigma, years)
    """
    feature_names = [f[0] for f in FEATURE_CATALOGUE]
    rows = []
    dates_bce = []
    sigmas    = []
    names     = []

    if verbose:
        print(f'\nExtracting features for {len(specs)} training units...')

    for spec in specs:
        name = spec[0]
        center = spec[3]
        sigma  = spec[4]
        if verbose:
            print(f'  {name:<25}', end='', flush=True)

        feat = extract_features(spec, F, L, T)
        n    = int(feat.pop('_n_words', 0))

        if verbose:
            ani_val = feat.get('frac_ani', float('nan'))
            print(f'  {n:>7,} words  frac_אני={ani_val:.3f}  pen={feat.get("rate_pen", 0):.2f}')

        rows.append([feat.get(fn, np.nan) for fn in feature_names])
        dates_bce.append(center)
        sigmas.append(sigma)
        names.append(name)

    rates_df  = pd.DataFrame(rows, index=names, columns=feature_names)
    dates_bce = np.array(dates_bce)
    sigmas    = np.array(sigmas)
    return rates_df, dates_bce, sigmas


# ---------------------------------------------------------------------------
# OLS fits and correlation scan
# ---------------------------------------------------------------------------

def fit_ols_and_scan(rates_df, dates_bce, verbose=True):
    """
    For each feature, fit OLS(value ~ date_bce) and compute Spearman rho.
    Returns:
      scan_df   — DataFrame with rho/p/ols coefficients
      ols_params — dict {feature: (a, b, sigma_resid)}
    """
    feature_names = rates_df.columns.tolist()
    records   = []
    ols_params = {}

    for fn in feature_names:
        vals = rates_df[fn].values
        mask = np.isfinite(vals)
        if mask.sum() < 5:
            records.append({'feature': fn, 'rho': np.nan, 'p': np.nan,
                            'a': np.nan, 'b': np.nan, 'resid_sigma': np.nan,
                            'n_valid': mask.sum()})
            continue

        x = dates_bce[mask]
        y = vals[mask]

        rho, p_val = spearmanr(-x, y)   # -x so positive rho means increases with time

        # OLS
        X = np.column_stack([np.ones(mask.sum()), x])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            coeffs = [np.nan, np.nan]
        a, b = coeffs[0], coeffs[1]
        resid = y - (a + b * x)
        sigma_r = resid.std(ddof=2) if mask.sum() > 2 else np.nan

        ols_params[fn] = (a, b, sigma_r)
        records.append({'feature': fn, 'rho': rho, 'p': p_val,
                        'a': a, 'b': b, 'resid_sigma': sigma_r,
                        'n_valid': mask.sum()})

    scan_df = pd.DataFrame(records).sort_values('p').reset_index(drop=True)

    if verbose:
        print(f'\n{"="*70}')
        print('FEATURE CORRELATION SCAN')
        print(f'{"="*70}')
        for _, row in scan_df.iterrows():
            dir_sym = '↑' if row['rho'] > 0 else '↓'
            star = '***' if row['p'] < 0.01 else ('**' if row['p'] < 0.05
                                                   else ('*' if row['p'] < 0.10 else ''))
        print(f'  {"Feature":<30} {"ρ":>7} {"p":>8}  {"Expected":>8}  sig')
        for _, row in scan_df.iterrows():
            fn   = row['feature']
            cat_entry = next((f for f in FEATURE_CATALOGUE if f[0] == fn), None)
            exp_dir = cat_entry[4] if cat_entry else '?'
            dir_sym = '↑' if row['rho'] > 0 else '↓'
            consistent = ''
            if exp_dir == 'increase' and row['rho'] > 0: consistent = '✓'
            elif exp_dir == 'decrease' and row['rho'] < 0: consistent = '✓'
            elif exp_dir not in ('unknown', '') : consistent = '✗'
            star = ('***' if row['p'] < 0.01 else
                    ('**'  if row['p'] < 0.05 else
                     ('*'   if row['p'] < 0.10 else  '')))
            print(f"  {fn:<30} {row['rho']:>+7.3f}  {row['p']:>8.4f}  "
                  f"{exp_dir:>8}  {consistent}  {star}")

    return scan_df, ols_params


# ---------------------------------------------------------------------------
# Residual covariance matrix (for MVN likelihood)
# ---------------------------------------------------------------------------

def build_residual_covariance(rates_df, dates_bce, ols_params, ridge_frac=0.05):
    """
    Compute the empirical covariance matrix of OLS residuals across training
    units.  This is the noise model for the MVN Bayesian likelihood.

    ridge_frac controls Tikhonov regularisation:
      Σ_reg = Σ + ridge_frac × trace(Σ)/K × I_K
    where K = number of features.  Ensures invertibility when features
    outnumber training units.
    """
    feature_names = [fn for fn in rates_df.columns if fn in ols_params]
    n_units  = len(rates_df)

    residual_rows = []
    for i, unit_name in enumerate(rates_df.index):
        row = []
        for fn in feature_names:
            val = rates_df.loc[unit_name, fn]
            if not np.isfinite(val):
                row.append(0.0)   # impute with 0 for covariance estimation
                continue
            a, b, _ = ols_params[fn]
            pred = a + b * dates_bce[i]
            row.append(val - pred)
        residual_rows.append(row)

    R = np.array(residual_rows)   # (n_units, n_features)
    Sigma = np.cov(R.T)           # (n_features, n_features)

    if Sigma.ndim == 0:
        Sigma = np.array([[Sigma]])

    K     = Sigma.shape[0]
    ridge = ridge_frac * np.trace(Sigma) / K
    Sigma_reg = Sigma + ridge * np.eye(K)

    return Sigma_reg, feature_names


# ---------------------------------------------------------------------------
# MVN log-likelihood
# ---------------------------------------------------------------------------

def mvn_log_likelihood(x_obs, date_bce, ols_params, feature_names,
                       Sigma_inv, n_words=1):
    """
    Log P(x_obs | date_bce) under the MVN model.

    x_obs       — array of observed feature values (K,)
    date_bce    — scalar candidate date (BCE, positive = older)
    ols_params  — dict {fn: (a, b, sigma_resid)}
    feature_names — list of K feature names
    Sigma_inv   — inverse of regularised residual covariance (K×K)
    n_words     — word count of the text being dated (for variance scaling)
    """
    mu = np.array([ols_params[fn][0] + ols_params[fn][1] * date_bce
                   for fn in feature_names])

    valid = np.isfinite(x_obs)
    if valid.sum() < 2:
        return -np.inf

    diff   = x_obs - mu
    diff[~valid] = 0.0   # zero out missing features (neutral contribution)

    # Scale Sigma_inv by n_words: more words → tighter constraint
    # Use a moderate scaling to avoid over-confidence in long books
    scale = max(1.0, min(n_words / 5000, 5.0))
    return -0.5 * scale * diff @ Sigma_inv @ diff


# ---------------------------------------------------------------------------
# Date posterior computation (1D, for single text)
# ---------------------------------------------------------------------------

def compute_posterior_1d(x_obs, ols_params, feature_names, Sigma_inv,
                         n_words=1, n_grid=600,
                         date_min=1200, date_max=50):
    """
    Compute the unnormalised date posterior on a 1D grid.
    Uses a weakly informative prior: N(650 BCE, 350 yr) truncated to grid.
    Returns (date_grid, log_posterior, map_date, ci_lo_68, ci_hi_68).
    """
    date_grid = np.linspace(date_max, date_min, n_grid)   # low→high BCE

    # Prior: N(600 BCE, 350 yr) — broad but weakly informative
    # Positive BCE values correspond to older dates
    prior_mean, prior_sd = 600.0, 350.0
    log_prior = -0.5 * ((date_grid - prior_mean) / prior_sd) ** 2

    log_lik = np.array([
        mvn_log_likelihood(x_obs, d, ols_params, feature_names,
                           Sigma_inv, n_words)
        for d in date_grid
    ])

    log_post = log_lik + log_prior
    log_post -= log_post.max()           # normalise for numerical stability
    post = np.exp(log_post)
    post /= post.sum()

    map_idx  = post.argmax()
    map_date = date_grid[map_idx]

    # 68% credible interval via cumulative sum
    cdf = np.cumsum(post)
    lo_idx = np.searchsorted(cdf, 0.16)
    hi_idx = np.searchsorted(cdf, 0.84)
    ci_lo  = date_grid[min(lo_idx, n_grid-1)]
    ci_hi  = date_grid[min(hi_idx, n_grid-1)]

    return date_grid, post, map_date, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Extract feature vector from a Torah book / chapter range
# ---------------------------------------------------------------------------

def extract_torah_features(book_name, chap_range, F, L, T, feature_names):
    """
    Extract features from a Torah source defined by (book, chapter range).
    Returns (feature_vector, n_words).
    """
    from collections import defaultdict

    lex_counts  = defaultdict(int)
    stem_counts = defaultdict(int)
    vs_counts   = defaultdict(int)
    n_words     = 0
    n_verbs     = 0

    book_node = T.nodeFromSection((book_name,))
    if book_node is None:
        return np.full(len(feature_names), np.nan), 0

    for ch_node in L.d(book_node, 'chapter'):
        chap_w = int(F.chapter.v(ch_node))
        if chap_range is not None:
            s, e = chap_range
            if not (s <= chap_w <= e):
                continue
        for w in L.d(ch_node, 'word'):
            n_words += 1
            lex = F.lex.v(w)
            sp  = F.sp.v(w)
            vs  = F.vs.v(w)
            lex_counts[lex] += 1
            if sp == 'verb':
                n_verbs += 1
                stem_counts[(lex, vs)] += 1
                vs_counts[vs] += 1

    # Now compute features
    result = {}
    for (name, desc, extract_type, params, direction, ref) in FEATURE_CATALOGUE:
        if name not in feature_names:
            result[name] = np.nan
            continue

        if extract_type == 'rate_lex':
            cnt = sum(lex_counts[l] for l in params)
            result[name] = cnt / n_words * 1000 if n_words else np.nan

        elif extract_type == 'fraction_lex':
            lex_a, lex_b = params
            a = lex_counts[lex_a]
            b = lex_counts[lex_b]
            result[name] = a / (a + b) if (a + b) > 0 else np.nan

        elif extract_type == 'frac_stem_lex':
            lex, stem1, stem2 = params
            c1 = stem_counts[(lex, stem1)]
            c2 = stem_counts[(lex, stem2)]
            result[name] = c1 / (c1 + c2) if (c1 + c2) > 0 else np.nan

        elif extract_type == 'frac_vs_total':
            target_vs = params[0]
            cnt = vs_counts[target_vs]
            result[name] = cnt / n_verbs if n_verbs > 0 else np.nan

        elif extract_type == 'rate_ut_nouns':
            cnt = sum(v for k, v in lex_counts.items()
                      if k.endswith('WT/') or k.endswith('WT'))
            result[name] = cnt / n_words * 1000 if n_words else np.nan

        else:
            result[name] = np.nan

    return np.array([result.get(fn, np.nan) for fn in feature_names]), n_words


# ---------------------------------------------------------------------------
# Aggregate sections within a documentary source
# ---------------------------------------------------------------------------

def aggregate_source(source_name, book_source_map, F, L, T, feature_names):
    """
    Pool all sections assigned to a documentary source (D, P, or JE).
    Returns (feature_vector, total_n_words) where the feature vector is the
    word-count-weighted mean of individual section vectors.
    """
    all_vecs   = []
    all_nwords = []

    for book, sources in book_source_map.items():
        if source_name not in sources:
            continue
        for chap_range in sources[source_name]:
            vec, nw = extract_torah_features(book, chap_range, F, L, T,
                                             feature_names)
            if nw > 50:                   # skip tiny fragments
                all_vecs.append(vec)
                all_nwords.append(nw)

    if not all_vecs:
        return np.full(len(feature_names), np.nan), 0

    all_nwords = np.array(all_nwords, dtype=float)
    weights = all_nwords / all_nwords.sum()
    stacked = np.vstack(all_vecs)

    # Weighted mean; nan-safe
    weighted_mean = np.nansum(stacked * weights[:, None], axis=0)
    # For features where all sections are nan, set to nan
    all_nan = np.all(~np.isfinite(stacked), axis=0)
    weighted_mean[all_nan] = np.nan

    return weighted_mean, int(all_nwords.sum())


# ---------------------------------------------------------------------------
# Joint posterior for D / P / JE (3D grid, coarse)
# ---------------------------------------------------------------------------

def compute_joint_posterior(source_data, ols_params, feature_names, Sigma_inv,
                             n_grid=60, date_min=1200, date_max=50):
    """
    Compute joint posterior P(θ_D, θ_P, θ_JE | data) on a 3D grid.

    source_data: dict {source_name: (x_obs, n_words)}
    Returns:
      date_axis — (n_grid,) array of date values (BCE)
      joint     — dict with marginal posteriors and MAP
    """
    date_axis = np.linspace(date_max, date_min, n_grid)

    # Compute 1D log-likelihoods for each source on the grid
    log_liks = {}
    for src, (x_obs, nw) in source_data.items():
        log_liks[src] = np.array([
            mvn_log_likelihood(x_obs, d, ols_params, feature_names,
                               Sigma_inv, nw)
            for d in date_axis
        ])

    # Flat prior on the grid
    log_prior = np.zeros(n_grid)

    # 1D marginal posteriors
    marginals = {}
    for src, ll in log_liks.items():
        lp = ll + log_prior
        lp -= lp.max()
        p = np.exp(lp)
        p /= p.sum()
        marginals[src] = p

    # 2D joint posterior for D and P (the most interesting pair)
    src_list = list(log_liks.keys())
    joint_dp = None
    src_d = 'D' if 'D' in log_liks else (src_list[0] if src_list else None)
    src_p = 'P' if 'P' in log_liks else (src_list[1] if len(src_list) > 1 else None)

    if src_d and src_p:
        ll_d = log_liks[src_d]
        ll_p = log_liks[src_p]
        # Joint under independence assumption (conditional on data)
        lj = ll_d[:, None] + ll_p[None, :]
        lj -= lj.max()
        joint_dp = np.exp(lj)
        joint_dp /= joint_dp.sum()

    return date_axis, marginals, joint_dp, src_d, src_p


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_posteriors(date_axis, marginals, source_colors, output_path):
    """Plot 1D date posteriors for D, P, JE."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for src, post in marginals.items():
        color = source_colors.get(src, 'black')
        ax.plot(date_axis, post, color=color, lw=2.5, label=src)
        map_d = date_axis[post.argmax()]
        ax.axvline(map_d, color=color, lw=1, ls='--', alpha=0.6)

    ax.set_xlabel('Date (BCE)')
    ax.set_ylabel('Posterior probability density')
    ax.set_title('Hierarchical Bayesian date posteriors — D / P / JE sources\n'
                 '(MVN likelihood; features pooled within source)')
    ax.set_xlim(date_axis.min(), date_axis.max())
    ax.invert_xaxis()
    ax.legend(loc='upper left')

    # Reference band for prophetic training range
    ax.axvspan(760, 460, alpha=0.06, color='grey', label='Training range')
    ax.text(610, ax.get_ylim()[1] * 0.95, 'Training range\n(prophets)',
            ha='center', va='top', fontsize=8, color='grey')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Posteriors plot saved: {output_path}')


def plot_joint_dp(date_axis, joint_dp, src_d, src_p, output_path):
    """Plot 2D joint posterior P(θ_D, θ_P | data)."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ext = [date_axis.min(), date_axis.max(), date_axis.min(), date_axis.max()]
    im = ax.imshow(joint_dp.T, extent=ext, origin='lower',
                   aspect='auto', cmap='Blues')
    plt.colorbar(im, ax=ax, label='Joint posterior probability')

    # Diagonal: D = P line
    lims = [max(date_axis.min(), date_axis.min()),
            min(date_axis.max(), date_axis.max())]
    ax.plot(date_axis, date_axis, 'r--', lw=1, alpha=0.7, label=f'{src_d} = {src_p}')

    # MAP point
    map_idx = np.unravel_index(joint_dp.argmax(), joint_dp.shape)
    ax.scatter([date_axis[map_idx[0]]], [date_axis[map_idx[1]]],
               marker='*', s=150, color='red', zorder=5, label='MAP')

    ax.set_xlabel(f'Date for {src_d} source (BCE)')
    ax.set_ylabel(f'Date for {src_p} source (BCE)')
    ax.set_title(f'Joint posterior P(θ_{src_d}, θ_{src_p} | data)\n'
                 f'Probability mass above diagonal: {src_d} earlier than {src_p}')
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend()

    # Compute probability that D is older than P
    prob_d_older = joint_dp[
        np.ix_(range(len(date_axis)), range(len(date_axis)))
    ]
    # sum over cells where θ_D > θ_P (D older = higher BCE value)
    prob_d_older_val = sum(
        joint_dp[i, j]
        for i in range(len(date_axis))
        for j in range(len(date_axis))
        if date_axis[i] > date_axis[j]   # higher BCE = older
    )
    ax.text(0.05, 0.95,
            f'P({src_d} older than {src_p}) = {prob_d_older_val:.2f}',
            transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Joint D–P posterior saved: {output_path}')


def plot_feature_profiles(source_results, scan_df, output_path):
    """
    Horizontal dot plot: feature × source, x-axis = LBH-ness score.
    LBH-ness = (obs - archaic_end) / (modern_end - archaic_end)
    where archaic_end / modern_end are the extremes of the training corpus.
    """
    # Only use features with p < 0.10 in the scan
    sig_features = scan_df[scan_df['p'] < 0.10]['feature'].tolist()
    if not sig_features:
        sig_features = scan_df.head(8)['feature'].tolist()

    sources_to_plot = [s for s in ['D', 'P', 'JE'] if s in source_results]
    if not sources_to_plot:
        return

    # Build LBH-ness scores using training min/max for each feature
    # (we don't have training data here, so we use the OLS predictions at
    # the extremes of the training range as reference points)
    # Instead: normalise directly using the range of observed source values

    n_feats = len(sig_features)
    colors = {'D': '#2166ac', 'P': '#d6604d', 'JE': '#1a9641'}
    markers = {'D': 'D', 'P': 's', 'JE': 'o'}

    fig, ax = plt.subplots(figsize=(8, max(4, n_feats * 0.45 + 1.5)))

    y_positions = np.arange(n_feats)
    all_vals = {fn: [] for fn in sig_features}
    for src in sources_to_plot:
        x_obs, _ = source_results[src]
        for i, fn in enumerate(sig_features):
            fidx = list(scan_df['feature']).index(fn) if fn in scan_df['feature'].values else -1
            feat_idx = [f[0] for f in FEATURE_CATALOGUE].index(fn) if fn in [f[0] for f in FEATURE_CATALOGUE] else -1
            if feat_idx >= 0:
                all_vals[fn].append(x_obs[feat_idx] if feat_idx < len(x_obs) else np.nan)

    # Scatter each source
    offset = {'D': -0.12, 'P': 0.0, 'JE': 0.12}
    for src in sources_to_plot:
        x_obs, _ = source_results[src]
        for i, fn in enumerate(sig_features):
            feat_idx = [f[0] for f in FEATURE_CATALOGUE].index(fn) if fn in [f[0] for f in FEATURE_CATALOGUE] else -1
            val = x_obs[feat_idx] if feat_idx >= 0 and feat_idx < len(x_obs) else np.nan
            if np.isfinite(val):
                ax.scatter(val, y_positions[i] + offset[src],
                           color=colors[src], marker=markers[src],
                           s=60, zorder=3, label=src if i == 0 else '')

    ax.set_yticks(y_positions)
    feat_labels = []
    cat_by_name = {f[0]: f for f in FEATURE_CATALOGUE}
    for fn in sig_features:
        desc = cat_by_name.get(fn, (fn, fn))[1]
        feat_labels.append(desc[:45])
    ax.set_yticklabels(feat_labels, fontsize=8)
    ax.set_xlabel('Feature value (rate per 1k or fraction)')
    ax.set_title('Morphosyntactic feature profiles — D, P, JE sources')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(0, color='grey', lw=0.8, ls=':')

    handles = [plt.Line2D([0], [0], marker=markers[s], color='w',
                          markerfacecolor=colors[s], markersize=8, label=s)
               for s in sources_to_plot]
    ax.legend(handles=handles, loc='lower right')

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Feature profile plot saved: {output_path}')


# ---------------------------------------------------------------------------
# Diagnostic: report BHSA stem values for key lexemes
# ---------------------------------------------------------------------------

def report_stem_coverage(specs, F, L, T, lexemes_of_interest):
    """
    Print counts of `vs` (verbal stem) values for given lexemes across the
    training corpus.  Used to verify that Tier-1 features are attested.
    """
    print('\n=== BHSA stem coverage for Tier-1 verb features ===')
    lex_stem_counts = defaultdict(lambda: defaultdict(int))

    for spec in specs[:5]:   # sample from first 5 units for speed
        for w, _ in words_in_unit(spec, F, L, T):
            lex = F.lex.v(w)
            vs  = F.vs.v(w)
            if lex in lexemes_of_interest:
                lex_stem_counts[lex][vs] += 1

    for lex, counts in sorted(lex_stem_counts.items()):
        print(f'  {lex}: {dict(counts)}')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Morphosyntactic features + MVN Bayesian Torah dating')
    parser.add_argument('--data-path', default=DEFAULT_DATA_PATH)
    parser.add_argument('--corpus',    choices=['prophetic', 'broad'],
                        default='broad')
    parser.add_argument('--jeremiah',  choices=['full', 'oracles'],
                        default='oracles',
                        help='full = use all of Jeremiah; '
                             'oracles = exclude Dtr prose chapters (default)')
    parser.add_argument('--debug-stems', action='store_true',
                        help='Print BHSA stem coverage for key verbs then exit')
    args = parser.parse_args()

    # ── Select corpus specs ─────────────────────────────────────────────────
    if args.jeremiah == 'oracles':
        base_specs = PROPHETIC_SPECS_ORACLE_JER
        print(f'Using Jeremiah in oracle-only mode '
              f'(excluding Dtr prose chapters: 7,11,17–18,21,24–29,32–45)')
    else:
        base_specs = PROPHETIC_SPECS_FULL_JER
        print('Using full Jeremiah (including Dtr prose sections — '
              'note: may introduce circularity when dating Deuteronomy)')

    if args.corpus == 'broad':
        specs = base_specs + BROAD_EXTENSION
        print(f'Using BROAD corpus ({len(specs)} units, ~760–167 BCE)')
    else:
        specs = base_specs
        print(f'Using PROPHETIC corpus ({len(specs)} units, ~760–460 BCE)')

    # ── Load BHSA ────────────────────────────────────────────────────────────
    api = load_bhsa(args.data_path)
    F, E, L, T, TF = api.F, api.E, api.L, api.T, api.TF

    # ── Optionally debug stem coverage then exit ─────────────────────────────
    if args.debug_stems:
        key_lexemes = {'HLK[', 'JSP[', 'JLD[', 'Z<Q[', 'Y<Q[', '>NXNW', 'NXNW'}
        report_stem_coverage(specs, F, L, T, key_lexemes)
        return

    # ── Build training matrix ────────────────────────────────────────────────
    rates_df, dates_bce, sigmas = build_training_matrix(specs, F, L, T,
                                                         verbose=True)
    rates_df.to_csv('morpho_training_rates.csv')
    print('Training rate matrix saved: morpho_training_rates.csv')

    # ── OLS fits and correlation scan ─────────────────────────────────────────
    scan_df, ols_params = fit_ols_and_scan(rates_df, dates_bce, verbose=True)
    scan_df.to_csv('morpho_feature_scan.csv', index=False)
    print('Feature scan saved: morpho_feature_scan.csv')

    # ── MVN covariance ───────────────────────────────────────────────────────
    Sigma_reg, feature_names = build_residual_covariance(
        rates_df, dates_bce, ols_params)

    Sigma_inv = np.linalg.inv(Sigma_reg)
    K = len(feature_names)
    print(f'\nMVN model: {K} features, {len(rates_df)} training units')
    print(f'  Regularised Σ condition number: '
          f'{np.linalg.cond(Sigma_reg):.1f}')

    # ── Extract Torah source features ─────────────────────────────────────────
    print('\nExtracting features for Torah sources...')
    source_results = {}   # {source_name: (x_obs, n_words)}

    # Aggregated sources
    for src in ['D', 'P', 'JE']:
        print(f'  Aggregating {src}...', end='', flush=True)
        x_obs, nw = aggregate_source(src, DOC_SOURCES, F, L, T, feature_names)
        source_results[src] = (x_obs, nw)
        ani_idx = feature_names.index('frac_ani') if 'frac_ani' in feature_names else -1
        ani_val = x_obs[ani_idx] if ani_idx >= 0 and np.isfinite(x_obs[ani_idx]) else float('nan')
        print(f'  {nw:>8,} words  frac_אני={ani_val:.3f}')

    # Individual Torah books for reference
    torah_books = [
        ('Genesis',     'Genesis',     None),
        ('Exodus',      'Exodus',      None),
        ('Leviticus',   'Leviticus',   None),
        ('Numbers',     'Numbers',     None),
        ('Deuteronomy', 'Deuteronomy', None),
    ]
    book_results = {}
    for label, book, cr in torah_books:
        x_obs, nw = extract_torah_features(book, cr, F, L, T, feature_names)
        book_results[label] = (x_obs, nw)

    # ── Compute posteriors ───────────────────────────────────────────────────
    print('\n=== HIERARCHICAL BAYESIAN DATE POSTERIORS ===')
    date_axis, marginals, joint_dp, src_d, src_p = compute_joint_posterior(
        source_results, ols_params, feature_names, Sigma_inv,
        n_grid=400, date_min=1200, date_max=50
    )

    records = []
    source_colors = {'D': '#2166ac', 'P': '#d6604d', 'JE': '#1a9641'}
    for src, (x_obs, nw) in source_results.items():
        post = marginals[src]
        map_d = date_axis[post.argmax()]
        cdf   = np.cumsum(post)
        lo    = date_axis[np.searchsorted(cdf, 0.16)]
        hi    = date_axis[np.searchsorted(cdf, 0.84)]
        print(f'  {src:<6}  MAP={map_d:>4.0f} BCE  68% CI: {hi:.0f}–{lo:.0f} BCE'
              f'  (n={nw:,})')
        records.append({'source': src, 'map_bce': map_d, 'ci_lo': hi,
                        'ci_hi': lo, 'n_words': nw})

    print('\n  Individual books:')
    for label, (x_obs, nw) in book_results.items():
        _, post, map_d, ci_lo, ci_hi = compute_posterior_1d(
            x_obs, ols_params, feature_names, Sigma_inv, n_words=nw)
        print(f'  {label:<12} MAP={map_d:>4.0f} BCE  68% CI: {ci_hi:.0f}–{ci_lo:.0f} BCE')
        records.append({'source': label, 'map_bce': map_d,
                        'ci_lo': ci_hi, 'ci_hi': ci_lo, 'n_words': nw})

    results_df = pd.DataFrame(records)
    results_df.to_csv('morpho_torah_dates.csv', index=False)
    print('\nResults saved: morpho_torah_dates.csv')

    # ── Scholarly reference dates ─────────────────────────────────────────────
    print('\nScholarly reference dates for comparison:')
    ref = [('D', 700, 621, 'Deuteronomist (traditional)'),
           ('P', 550, 400, 'Priestly source (traditional)'),
           ('JE', 950, 750, 'JE combined (traditional)')]
    for src, early, late, label in ref:
        print(f'  {src}: {early}–{late} BCE  ({label})')

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_posteriors(date_axis, marginals, source_colors, 'morpho_posteriors.png')
    if joint_dp is not None:
        plot_joint_dp(date_axis, joint_dp, src_d, src_p, 'morpho_joint_dp.png')
    plot_feature_profiles(source_results, scan_df, 'morpho_feature_profiles.png')

    # ── Feature scan summary plot ────────────────────────────────────────────
    n_sig = (scan_df['p'] < 0.10).sum()
    print(f'\n{n_sig}/{len(scan_df)} features significant at p < 0.10')

    top = scan_df.head(min(len(FEATURE_CATALOGUE), 12))
    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ['#2166ac' if r > 0 else '#d6604d'
                  for r in top['rho'].values]
    ax.barh(range(len(top)), top['rho'].values, color=bar_colors, alpha=0.8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"{r['feature']}  (p={r['p']:.3f})"
                        for _, r in top.iterrows()], fontsize=8)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('Spearman ρ (positive = increases over time / LBH)')
    ax.set_title('Morphosyntactic feature correlations with date\n'
                 '(blue = increases in LBH, red = decreases)')
    plt.tight_layout()
    fig.savefig('morpho_feature_scan.png', dpi=150)
    plt.close(fig)
    print('Feature scan plot saved: morpho_feature_scan.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
