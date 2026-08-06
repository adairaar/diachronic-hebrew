"""
Script 22 — Monte Carlo Interpolation Sensitivity Analysis
===========================================================
Tests how much date estimates change when passages with scholarly suspicion
of being later interpolations are randomly removed from the training corpus.

Design
------
Rather than removing entire books (Script 21), this script targets specific
passages within the 7 pre-exilic training units that scholars have flagged as
potentially secondary.  Each passage is tagged with a tier:

  LIKELY  (~75% probability of removal per iteration)
    — widely agreed among critical scholars (70%+ of major commentators)
  POSSIBLE (~30% probability of removal per iteration)
    — debated or minority-to-substantial-minority scholarly position

For each of 200 Monte Carlo iterations:
  1. For every flagged passage, draw random ∈ [0,1]; remove if < tier threshold.
  2. Recompute training unit rates from remaining verses.
  3. Refit MVN Bayesian model using the S0 pre-selected feature set.
  4. Apply model to all 23 test units; record MAP dates.

Using the S0 feature set (rather than re-running feature selection each iteration)
makes the analysis computationally tractable and answers: "given the same
diachronically-informative features, how stable are the model's date predictions
when the training data has up to ~30% of potentially-interpolated text removed?"

Scholarly sources for the interpolation list
---------------------------------------------
Amos: Wolff (1977), Paul (1991), Nogalski (2011)
Hosea: Andersen & Freedman (1980), Yee (1987), Macintosh (1997)
Micah: Wolff (1990), Sweeney (FOTL), Hillers (1984)
Isaiah 1-39: Blenkinsopp (2000), Childs (2001), Sweeney (1996), Kaiser (1983)
Nahum: Roberts (1991), Spronk (1997)
Habakkuk: Roberts (1991), Andersen (2001)
Zephaniah: Sweeney (FOTL), Nogalski (2011), Ben Zvi (1991)

Outputs
-------
  mc_sensitivity_char_results.csv   — per-unit distribution (mean, std, pct)
  mc_sensitivity_word_results.csv   — same for word n-gram model
  mc_sensitivity_summary.csv        — combined comparison vs S0 baseline
  mc_sensitivity_violin_char.png    — violin/box plots per unit
  mc_sensitivity_violin_word.png    — same for word n-gram
  mc_sensitivity_shift_dist.png     — distribution of max shift magnitude
"""

import sys
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

WORKSPACE = Path('/sessions/relaxed-modest-dirac/mnt/Diachronic Hebrew')
RNG_SEED  = 42

# ===========================================================================
# INTERPOLATION CATALOGUE
# Each entry: (book, start_ch, start_vs, end_ch, end_vs, tier, note)
# Versification follows Masoretic / BHSA standard.
# Tier probabilities: LIKELY=0.75, POSSIBLE=0.30
# ===========================================================================
TIER_PROB = {'LIKELY': 0.75, 'POSSIBLE': 0.30}

INTERPOLATIONS = [
    # ---- AMOS ---------------------------------------------------------------
    # Doxologies: near-universal scholarly consensus (Wolff, Paul, Crenshaw)
    ('Amos',    4, 13,  4, 13, 'LIKELY',   'Hymnic doxology; cosmological language, breaks poetic context'),
    ('Amos',    5,  8,  5,  9, 'LIKELY',   'Hymnic doxology; interrupts 5:7 to 5:10; creation imagery'),
    ('Amos',    9,  5,  9,  6, 'LIKELY',   'Third doxology; interrupts destruction oracle'),
    # Eschatological conclusion
    ('Amos',    9, 11,  9, 15, 'POSSIBLE', 'Davidic restoration; abrupt shift from judgment; many date post-587'),
    # Oracles against Tyre and Edom: lack closing formula, Deuteronomic language
    ('Amos',    1,  9,  1, 12, 'POSSIBLE', 'Tyre/Edom oracles; redactional expansion; lack distinctive Amos formula'),
    # Oracle against Judah: Deuteronomic vocabulary
    ('Amos',    2,  4,  2,  5, 'POSSIBLE', 'Judah oracle; Deuteronomic language; breaks Israelite focus'),

    # ---- HOSEA --------------------------------------------------------------
    # Wisdom epilogue: sapiential language foreign to prophetic corpus
    ('Hosea',  14, 10, 14, 10, 'LIKELY',   'Wisdom epilogue; sapiential vocabulary (wisdom, understanding, paths)'),
    # Judah reference: post-fall-of-Samaria pro-Judean insertion
    ('Hosea',   1,  7,  1,  7, 'POSSIBLE', 'Judah exception clause; breaks Israel focus; post-Samaria perspective'),
    # Restoration oracles: tonal/theological shift
    ('Hosea',   2, 14,  2, 23, 'POSSIBLE', 'Restoration section; reversal of judgment names; debated origin'),
    ('Hosea',   3,  1,  3,  5, 'POSSIBLE', 'Redemption narrative; textual and exegetical difficulties; debated'),
    ('Hosea',  11,  8, 11, 11, 'POSSIBLE', 'Divine compassion oracle; theological shift; minority sees as exilic'),
    ('Hosea',  14,  2,  14, 9, 'POSSIBLE', 'Call to repentance and promise; wisdom-influenced language'),

    # ---- MICAH --------------------------------------------------------------
    # Swords-to-plowshares: identical to Isa 2:2-4; one is secondary
    ('Micah',   4,  1,  4,  5, 'LIKELY',   'Parallel to Isa 2:2-4; widely secondary in Micah or Isaiah; post-exilic framing'),
    # Exilic/post-exilic restoration blocks
    ('Micah',   2, 12,  2, 13, 'POSSIBLE', 'Shepherd-flock oracle; radical tonal shift; Stade excluded as secondary'),
    ('Micah',   4,  9,  5, 15, 'POSSIBLE', 'Eschatological block (Sweeney); late exilic to post-exilic; salvation against judgment'),
    ('Micah',   7,  8,  7, 20, 'POSSIBLE', 'Psalm of trust and restoration; liturgical nature; last post-Micah addition'),

    # ---- ISAIAH 1-39 --------------------------------------------------------
    # Isaiah Apocalypse: most scholars date to Persian period or later
    ('Isaiah', 24,  1, 27, 13, 'LIKELY',   'Isaiah Apocalypse; cosmic/eschatological vision; resurrection language; Blenkinsopp: post-539'),
    # "Little Apocalypse": Deutero-Isaianic influence
    ('Isaiah', 34,  1, 35, 10, 'LIKELY',   'Little Apocalypse; Deutero-Isaianic language and theology; 6th cent or later'),
    # Historical appendix = 2 Kings 18-20 with variations
    ('Isaiah', 36,  1, 39,  8, 'LIKELY',   'Prose appendix paralleling 2 Kgs 18-20; editorial fatigue evidence; later insertion'),
    # Doxological conclusion to Isaiah 1-11
    ('Isaiah', 12,  1, 12,  6, 'POSSIBLE', 'Doxological epilogue to first section; lyrical frame; widely seen as redactional'),
    # Babylon oracle and expansions
    ('Isaiah', 13,  1, 14, 27, 'POSSIBLE', 'Babylon oracles; redactional expansion; 14:1-2 and 14:26-27 exilic additions'),
    # Moab additions
    ('Isaiah', 16,  1, 16,  5, 'POSSIBLE', 'Moab oracle expansions; redactional supplements within oracles-against-nations'),
    # Egypt material
    ('Isaiah', 19, 16, 19, 25, 'POSSIBLE', 'Pro-Egyptian salvation oracles; distinctly non-Isaianic perspective'),
    # Tyre oracle expansion
    ('Isaiah', 23,  8, 23, 18, 'POSSIBLE', 'Expansions within Tyre oracle; programmatic late-exilic additions'),
    # Women of Jerusalem pericope (disputed)
    ('Isaiah', 32,  9, 32, 20, 'POSSIBLE', 'Women of Jerusalem oracle; some dispute Isaianic origin'),

    # ---- NAHUM --------------------------------------------------------------
    # Acrostic hymn introduction: universally recognized as distinct composition
    ('Nahum',   1,  2,  1, 10, 'LIKELY',   'Alphabetic acrostic theophany psalm; compositionally distinct from rest of book'),

    # ---- HABAKKUK -----------------------------------------------------------
    # Psalm appendix: separate superscript, distinct genre
    ('Habakkuk', 3,  1,  3, 19, 'LIKELY',  'Psalm with separate title/subscript; hymnic prayer; often seen as appended'),
    # Chaldean oracle (minority position)
    ('Habakkuk', 1,  5,  1, 11, 'POSSIBLE', 'Chaldean oracle; earlier scholarship excised; divergent manuscript interpretation'),

    # ---- ZEPHANIAH ----------------------------------------------------------
    # Salvation epilogue: near-universal scholarly consensus
    ('Zephaniah', 3, 14, 3, 20, 'LIKELY',  'Salvation epilogue; abrupt shift from judgment; lacks moral prerequisites; later addition'),
    # Redactional layers within the book
    ('Zephaniah', 2,  3,  2,  3, 'POSSIBLE', 'Humility appeal; restoration-of-remnant redaction layer'),
    ('Zephaniah', 2,  7,  2,  7, 'POSSIBLE', 'Remnant restoration promise; secondary to core judgment material'),
    ('Zephaniah', 2,  8,  2, 11, 'POSSIBLE', 'Moab/Ammon oracle with salvation-of-nations ending; redactional expansion'),
    ('Zephaniah', 3,  9,  3, 13, 'POSSIBLE', 'Universal conversion and remnant promises; salvation-of-nations redaction'),
]

# Map passage to training unit name
BOOK_TO_UNIT = {
    'Amos': 'Amos', 'Hosea': 'Hosea', 'Micah': 'Micah',
    'Isaiah': 'Isaiah_1',  # only chs 1-39 are in Isaiah_1
    'Nahum': 'Nahum', 'Habakkuk': 'Habakkuk', 'Zephaniah': 'Zephaniah',
}

# Training units that appear in the interpolation catalogue
FLAGGED_UNITS = {'Amos', 'Hosea', 'Micah', 'Isaiah_1', 'Nahum', 'Habakkuk', 'Zephaniah'}

# ===========================================================================
# TRAINING CORPUS  (same as Scripts 16-17)
# ===========================================================================
ALL_TRAINING_UNITS = {
    'Amos':         [('Amos',          [(1, 9)])],
    'Hosea':        [('Hosea',         [(1, 14)])],
    'Micah':        [('Micah',         [(1, 7)])],
    'Nahum':        [('Nahum',         [(1, 3)])],
    'Habakkuk':     [('Habakkuk',      [(1, 3)])],
    'Zephaniah':    [('Zephaniah',     [(1, 3)])],
    'Isaiah_1':     [('Isaiah',        [(1, 39)])],
    'Isaiah_2':     [('Isaiah',        [(40, 55)])],
    'Isaiah_3':     [('Isaiah',        [(56, 66)])],
    'Jeremiah':     [('Jeremiah',      [(1, 6), (8, 10), (12, 16),
                                        (19, 20), (22, 23), (30, 31), (46, 51)])],
    'Ezekiel':      [('Ezekiel',       [(1, 48)])],
    'Haggai':       [('Haggai',        [(1, 2)])],
    'Zechariah_1':  [('Zechariah',     [(1, 8)])],
    'Malachi':      [('Malachi',       [(1, 3)])],
    'Jonah':        [('Jonah',         [(1, 4)])],
    'Lamentations': [('Lamentations',  [(1, 5)])],
    'Ezra':         [('Ezra',          [(1, 10)])],
    'Nehemiah':     [('Nehemiah',      [(1, 13)])],
    'Chronicles':   [('1_Chronicles',  [(1, 29)]),
                     ('2_Chronicles',  [(1, 36)])],
    'Daniel':       [('Daniel',        [(1, 1), (8, 12)])],
    'Ecclesiastes': [('Ecclesiastes',  [(1, 12)])],
    'Esther':       [('Esther',        [(1, 10)])],
}
ALL_TRAINING_DATES = {
    'Amos': 760, 'Hosea': 740, 'Micah': 720, 'Isaiah_1': 720,
    'Nahum': 620, 'Habakkuk': 600, 'Zephaniah': 620, 'Jeremiah': 590,
    'Ezekiel': 580, 'Isaiah_2': 540, 'Isaiah_3': 450,
    'Haggai': 520, 'Zechariah_1': 518, 'Malachi': 450,
    'Jonah': 400, 'Lamentations': 580, 'Ezra': 350,
    'Nehemiah': 350, 'Chronicles': 350, 'Daniel': 167,
    'Ecclesiastes': 250, 'Esther': 350,
}
UNIT_ORDER = list(ALL_TRAINING_UNITS.keys())
TRAIN_DATES = [ALL_TRAINING_DATES[u] for u in UNIT_ORDER]

# Test units
TEST_UNITS = {
    'Genesis':      [('Genesis',     [(1, 50)])],
    'Exodus':       [('Exodus',      [(1, 14), (16, 40)])],
    'Leviticus':    [('Leviticus',   [(1, 27)])],
    'Numbers':      [('Numbers',     [(1, 36)])],
    'Deuteronomy':  [('Deuteronomy', [(1, 34)])],
    'D_Code':       [('Deuteronomy', [(12, 26)])],
    'D_Frame':      [('Deuteronomy', [(1, 11), (27, 31), (33, 34)])],
    'D_Song':       [('Deuteronomy', [(32, 32)])],
    'Lev_Holiness': [('Leviticus',   [(17, 26)])],
    'Lev_Priestly': [('Leviticus',   [(1, 16)])],
    'D_source':     [('Deuteronomy', [(1, 34)])],
    'P_source':     [('Genesis',     [(1,2),(5,5),(6,6),(7,7),(9,9),(11,11),
                                      (17,17),(23,23),(25,25),(27,28),(35,36),(46,46),(49,50)]),
                     ('Exodus',      [(1,2),(6,7),(12,12),(16,16),(25,31),(35,40)]),
                     ('Leviticus',   [(1, 27)]),
                     ('Numbers',     [(1,10),(15,15),(17,19),(25,25),(27,31),(33,36)])],
    'JE_source':    [('Genesis',     [(2,4),(6,6),(8,8),(10,10),(12,16),(18,22),
                                      (24,24),(26,27),(29,34),(37,45),(47,49)]),
                     ('Exodus',      [(2,5),(8,11),(13,14),(17,18),(19,24),(32,34)]),
                     ('Numbers',     [(11,14),(16,16),(20,24),(25,25),(32,32)])],
    'Joshua':       [('Joshua',    [(1, 24)])],
    'Judges':       [('Judges',    [(1, 4), (6, 21)])],
    '1_Samuel':     [('1_Samuel',  [(1, 31)])],
    '2_Samuel':     [('2_Samuel',  [(1, 24)])],
    '1_Kings':      [('1_Kings',   [(1, 22)])],
    '2_Kings':      [('2_Kings',   [(1, 25)])],
    'Jer_DTR':      [('Jeremiah',  [(7,7),(11,11),(17,18),(21,21),
                                    (24,29),(32,45),(52,52)])],
    'Jer_oracle':   [('Jeremiah',  [(1,6),(8,10),(12,16),(19,20),
                                    (22,23),(30,31),(46,51)])],
    'Song_Sea':     [('Exodus',    [(15, 15)])],
    'Song_Deborah': [('Judges',    [(5, 5)])],
}
NOISY_UNITS = {'D_Song', 'Song_Sea', 'Song_Deborah'}

# ===========================================================================
# N-GRAM PARAMETERS
# ===========================================================================
CNG_SIZES  = [3, 4]
WNG_SIZES  = [2, 3]
WORD_SEP   = '_'
FUNCTION_POS = frozenset({'prep', 'conj', 'art', 'nega', 'prps', 'prde', 'inrg'})

# MVN parameters
N_GRID      = 500
DATE_HI     = 1200
DATE_LO     = 50
PRIOR_MU    = 600.0
PRIOR_SIGMA = 350.0
RIDGE       = 0.20

# Monte Carlo parameters
N_ITERATIONS = 200


# ===========================================================================
# TEXT / FEATURE EXTRACTION  (verse-level)
# ===========================================================================

def in_range(ch, vs, start_ch, start_vs, end_ch, end_vs):
    """True if (ch:vs) is within the range [start_ch:start_vs, end_ch:end_vs]."""
    start = (start_ch, start_vs)
    end   = (end_ch,   end_vs)
    point = (ch, vs)
    return start <= point <= end


def verse_is_flagged(book, ch, vs):
    """Return (tier, prob) if this verse belongs to a flagged passage, else None."""
    for entry in INTERPOLATIONS:
        b, sch, svs, ech, evs, tier, _ = entry
        if b == book and in_range(ch, vs, sch, svs, ech, evs):
            return tier
    return None


def extract_verse_level_cng(book_ch_pairs, F, L, T):
    """
    Extract char n-gram raw counts per verse for a training unit.

    Returns:
        verse_counts: {(ch, vs): {ngram: count}}
        verse_chars:  {(ch, vs): int}   # character length
    """
    verse_counts = {}
    verse_chars  = {}
    for book, ch_ranges in book_ch_pairs:
        bn = T.nodeFromSection((book,))
        if bn is None:
            continue
        for ch_node in L.d(bn, 'chapter'):
            ch_num = int(F.chapter.v(ch_node))
            if not any(s <= ch_num <= e for s, e in ch_ranges):
                continue
            for vs_node in L.d(ch_node, 'verse'):
                vs_num = int(F.verse.v(vs_node))
                # build verse text
                tokens = [F.g_cons_utf8.v(w) for w in L.d(vs_node, 'word')
                          if F.g_cons_utf8.v(w)]
                if not tokens:
                    continue
                text   = WORD_SEP + WORD_SEP.join(tokens) + WORD_SEP
                n_chars = len(text)
                counts  = Counter()
                for n in CNG_SIZES:
                    for i in range(len(text) - n + 1):
                        counts[text[i:i+n]] += 1
                verse_counts[(ch_num, vs_num)] = dict(counts)
                verse_chars[(ch_num, vs_num)]  = n_chars
    return verse_counts, verse_chars


def extract_verse_level_wng(book_ch_pairs, F, L, T):
    """
    Extract word n-gram raw counts per verse (Type A = POS, Type B = function-word).

    Returns:
        verse_counts_A: {(ch, vs): {ngram: count}}
        verse_counts_B: {(ch, vs): {ngram: count}}
        verse_words:    {(ch, vs): int}
    """
    def pos_token(word):
        sp = F.sp.v(word) or 'unkn'
        if sp == 'verb':
            return f'verb_{F.vt.v(word) or "unkn"}'
        return sp

    verse_counts_A = {}
    verse_counts_B = {}
    verse_words    = {}
    for book, ch_ranges in book_ch_pairs:
        bn = T.nodeFromSection((book,))
        if bn is None:
            continue
        for ch_node in L.d(bn, 'chapter'):
            ch_num = int(F.chapter.v(ch_node))
            if not any(s <= ch_num <= e for s, e in ch_ranges):
                continue
            for vs_node in L.d(ch_node, 'verse'):
                vs_num  = int(F.verse.v(vs_node))
                pos_seq = []
                fw_seq  = []
                n_words = 0
                for word in L.d(vs_node, 'word'):
                    sp = F.sp.v(word)
                    if not sp:
                        continue
                    pos_seq.append(pos_token(word))
                    fw_seq.append(F.lex.v(word) or sp if sp in FUNCTION_POS else None)
                    n_words += 1
                if n_words == 0:
                    continue

                def count_ngrams(tokens, skip_null=False):
                    c = Counter()
                    for sz in WNG_SIZES:
                        for i in range(len(tokens) - sz + 1):
                            gram = tokens[i:i+sz]
                            if skip_null and any(t is None for t in gram):
                                continue
                            c['·'.join(str(t) for t in gram)] += 1
                    return dict(c)

                verse_counts_A[(ch_num, vs_num)] = count_ngrams(pos_seq, skip_null=False)
                verse_counts_B[(ch_num, vs_num)] = count_ngrams(fw_seq,  skip_null=True)
                verse_words[(ch_num, vs_num)]    = n_words
    return verse_counts_A, verse_counts_B, verse_words


def extract_unit_cng(book_ch_pairs, F, L, T):
    """Extract aggregated char n-gram rates for a full training/test unit."""
    total_counts = Counter()
    total_chars  = 0
    for book, ch_ranges in book_ch_pairs:
        bn = T.nodeFromSection((book,))
        if bn is None:
            continue
        for ch_node in L.d(bn, 'chapter'):
            ch_num = int(F.chapter.v(ch_node))
            if not any(s <= ch_num <= e for s, e in ch_ranges):
                continue
            for word in L.d(ch_node, 'word'):
                cons = F.g_cons_utf8.v(word)
                if not cons:
                    continue
                total_chars += len(cons) + 1  # approximate char count
        # More precisely, build the text
    # rebuild as text for accurate n-gram extraction
    tokens = []
    for book, ch_ranges in book_ch_pairs:
        bn = T.nodeFromSection((book,))
        if bn is None:
            continue
        for ch_node in L.d(bn, 'chapter'):
            ch_num = int(F.chapter.v(ch_node))
            if not any(s <= ch_num <= e for s, e in ch_ranges):
                continue
            for word in L.d(ch_node, 'word'):
                c = F.g_cons_utf8.v(word)
                if c:
                    tokens.append(c)
    if not tokens:
        return {}, 0
    text  = WORD_SEP + WORD_SEP.join(tokens) + WORD_SEP
    nch   = len(text)
    for n in CNG_SIZES:
        for i in range(len(text) - n + 1):
            total_counts[text[i:i+n]] += 1
    return {ng: cnt / nch * 1000 for ng, cnt in total_counts.items()}, len(tokens)


def extract_unit_wng(book_ch_pairs, F, L, T):
    """Extract aggregated word n-gram rates for a full training/test unit."""
    def pos_token(word):
        sp = F.sp.v(word) or 'unkn'
        if sp == 'verb':
            return f'verb_{F.vt.v(word) or "unkn"}'
        return sp

    pos_seq = []
    fw_seq  = []
    n_words = 0
    for book, ch_ranges in book_ch_pairs:
        bn = T.nodeFromSection((book,))
        if bn is None:
            continue
        for ch_node in L.d(bn, 'chapter'):
            ch_num = int(F.chapter.v(ch_node))
            if not any(s <= ch_num <= e for s, e in ch_ranges):
                continue
            for word in L.d(ch_node, 'word'):
                sp = F.sp.v(word)
                if not sp:
                    continue
                pos_seq.append(pos_token(word))
                fw_seq.append(F.lex.v(word) or sp if sp in FUNCTION_POS else None)
                n_words += 1

    def to_rates(tokens, skip_null=False):
        if n_words == 0:
            return {}
        c = Counter()
        for sz in WNG_SIZES:
            for i in range(len(tokens) - sz + 1):
                gram = tokens[i:i+sz]
                if skip_null and any(t is None for t in gram):
                    continue
                c['·'.join(str(t) for t in gram)] += 1
        return {ng: cnt / n_words * 1000 for ng, cnt in c.items()}

    return to_rates(pos_seq, False), to_rates(fw_seq, True), n_words


# ===========================================================================
# MVN MODEL
# ===========================================================================

def build_mvn_model(rates_matrix, dates_bce, feature_names):
    """
    rates_matrix: dict {feature: [rate_unit_0, ..., rate_unit_N-1]}
    Returns (ols_params, Sigma_inv)
    """
    ols_params    = {}
    residual_rows = []
    for fn in feature_names:
        y     = np.array(rates_matrix[fn], dtype=float)
        x     = np.array(dates_bce, dtype=float)
        valid = np.isfinite(y) & np.isfinite(x)
        if valid.sum() < 4:
            ols_params[fn] = (np.nanmean(y), 0.0)
            residual_rows.append(np.zeros(len(dates_bce)))
            continue
        slope, intercept, *_ = stats.linregress(x[valid], y[valid])
        ols_params[fn] = (intercept, slope)
        pred  = intercept + slope * x
        resid = y - pred
        resid[~np.isfinite(resid)] = 0.0
        residual_rows.append(resid)
    R         = np.array(residual_rows).T
    Sigma     = R.T @ R / max(len(dates_bce) - 2, 1)
    K         = len(feature_names)
    lam       = RIDGE * np.trace(Sigma) / K if K > 0 else 0.0
    Sigma_reg = Sigma + lam * np.eye(K)
    Sigma_inv = np.linalg.inv(Sigma_reg)
    return ols_params, Sigma_inv


def compute_map(obs_rates, feature_names, ols_params, Sigma_inv, date_grid):
    obs      = np.array([obs_rates.get(fn, 0.0) for fn in feature_names], dtype=float)
    log_post = np.zeros(len(date_grid))
    for i, d in enumerate(date_grid):
        pred  = np.array([ols_params[fn][0] + ols_params[fn][1] * d
                          for fn in feature_names], dtype=float)
        diff  = obs - pred
        valid = np.isfinite(diff)
        if valid.sum() < 2:
            log_post[i] = -1e9; continue
        dv = diff[valid]; Sv = Sigma_inv[np.ix_(valid, valid)]
        log_post[i] = -0.5 * float(dv @ Sv @ dv)
    log_prior  = -0.5 * ((date_grid - PRIOR_MU) / PRIOR_SIGMA) ** 2
    log_post  += log_prior
    log_post  -= log_post.max()
    post       = np.exp(log_post)
    post      /= post.sum()
    map_date   = float(date_grid[np.argmax(post)])
    cdf        = np.cumsum(post)
    ci_lo      = float(date_grid[np.searchsorted(cdf, 0.16)])
    ci_hi      = float(date_grid[np.searchsorted(cdf, 0.84)])
    return map_date, ci_lo, ci_hi


# ===========================================================================
# MONTE CARLO CORE
# ===========================================================================

def compute_flagged_unit_rates_cng(unit_name, book,
                                    verse_counts, verse_chars,
                                    removed_verses, feature_names):
    """
    Compute char n-gram rates for a flagged training unit with some verses removed.
    verse_counts: {(ch,vs): {ngram: count}}
    verse_chars:  {(ch,vs): int}
    removed_verses: set of (ch, vs) to exclude
    """
    total_count = Counter()
    total_chars = 0
    for (ch, vs), counts in verse_counts.items():
        if (ch, vs) in removed_verses:
            continue
        for ng, cnt in counts.items():
            total_count[ng] += cnt
        total_chars += verse_chars[(ch, vs)]
    if total_chars == 0:
        return {fn: 0.0 for fn in feature_names}
    return {fn: total_count.get(fn, 0) / total_chars * 1000
            for fn in feature_names}


def compute_flagged_unit_rates_wng(verse_counts_A, verse_counts_B,
                                    verse_words, removed_verses, feature_names_A, feature_names_B):
    total_A = Counter()
    total_B = Counter()
    total_w = 0
    for (ch, vs) in verse_words:
        if (ch, vs) in removed_verses:
            continue
        for ng, cnt in verse_counts_A.get((ch, vs), {}).items():
            total_A[ng] += cnt
        for ng, cnt in verse_counts_B.get((ch, vs), {}).items():
            total_B[ng] += cnt
        total_w += verse_words[(ch, vs)]
    if total_w == 0:
        rates_A = {fn: 0.0 for fn in feature_names_A}
        rates_B = {fn: 0.0 for fn in feature_names_B}
    else:
        rates_A = {fn: total_A.get(fn, 0) / total_w * 1000 for fn in feature_names_A}
        rates_B = {fn: total_B.get(fn, 0) / total_w * 1000 for fn in feature_names_B}
    return rates_A, rates_B


def draw_removed_verses(unit_name, book):
    """
    For one Monte Carlo iteration, decide which flagged verse positions to remove.
    Returns a set of (ch, vs) keys.
    """
    removed = set()
    for entry in INTERPOLATIONS:
        b, sch, svs, ech, evs, tier, _ = entry
        if BOOK_TO_UNIT.get(b) != unit_name:
            continue
        prob = TIER_PROB[tier]
        if np.random.random() < prob:
            # remove all verses in this passage
            for ch in range(sch, ech + 1):
                vs_start = svs if ch == sch else 1
                vs_end   = evs if ch == ech else 200  # generous upper bound
                for vs in range(vs_start, vs_end + 1):
                    removed.add((ch, vs))
    return removed


# ===========================================================================
# PLOTTING
# ===========================================================================

UNIT_GROUPS_ORDERED = [
    'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
    'D_source', 'P_source', 'JE_source', 'D_Code', 'D_Frame',
    'Lev_Holiness', 'Lev_Priestly',
    'Joshua', 'Judges', '1_Samuel', '2_Samuel', '1_Kings', '2_Kings',
    'Jer_DTR', 'Jer_oracle',
    'Song_Sea', 'Song_Deborah', 'D_Song',
]

def make_violin_plot(all_maps, s0_maps, model_label, out_path):
    """
    Violin plot: per-unit distribution of MAP dates across 200 iterations.
    S0 baseline shown as red dot.
    """
    units  = [u for u in UNIT_GROUPS_ORDERED if u in all_maps]
    n      = len(units)
    data   = [all_maps[u] for u in units]

    fig, ax = plt.subplots(figsize=(14, 8))

    vp = ax.violinplot(data, positions=range(n),
                       vert=False, showmedians=True, showextrema=False)
    for body in vp['bodies']:
        body.set_alpha(0.55)
        body.set_facecolor('#4575b4')
    vp['cmedians'].set_color('#1a1a2e')
    vp['cmedians'].set_linewidth(1.5)

    # S0 baseline dots
    for yi, unit in enumerate(units):
        if unit in s0_maps and np.isfinite(s0_maps[unit]):
            ax.scatter(s0_maps[unit], yi, color='#d73027', s=40, zorder=5,
                       label='S0 baseline' if yi == 0 else '')

    # Noisy markers
    for yi, unit in enumerate(units):
        if unit in NOISY_UNITS:
            ax.text(DATE_LO - 80, yi, '⚠', va='center', fontsize=9, color='#888')

    ax.set_yticks(range(n))
    ax.set_yticklabels([u.replace('_', ' ') for u in units], fontsize=8)
    ax.set_xlabel('MAP date (BCE)  —  higher = older', fontsize=10)
    ax.set_xlim(DATE_HI + 100, DATE_LO - 120)
    ax.invert_xaxis()
    ax.set_title(f'Monte Carlo interpolation sensitivity — {model_label}\n'
                 f'N={N_ITERATIONS} iterations  ·  Two-tier removal (LIKELY=75%, POSSIBLE=30%)\n'
                 f'Violin = date distribution under random interpolation removal; '
                 f'red dot = S0 baseline',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.25, lw=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Violin plot saved: {Path(out_path).name}')


def make_shift_distribution_plot(char_maps_all, word_maps_all,
                                  char_s0, word_s0, out_path):
    """
    For key units, show histogram of (MAP_iteration - MAP_S0),
    i.e. the date shift caused by interpolation removal.
    """
    KEY_UNITS = ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
                 'P_source', 'JE_source', 'D_source',
                 'Joshua', '2_Kings', 'Jer_oracle',
                 'Song_Sea', 'Song_Deborah']

    cols  = 4
    rows  = (len(KEY_UNITS) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3), sharey=False)
    axes = axes.flatten()

    for ax, unit in zip(axes, KEY_UNITS):
        c_shifts = (np.array(char_maps_all.get(unit, [])) - char_s0.get(unit, np.nan))
        w_shifts = (np.array(word_maps_all.get(unit, [])) - word_s0.get(unit, np.nan))
        c_shifts = c_shifts[np.isfinite(c_shifts)]
        w_shifts = w_shifts[np.isfinite(w_shifts)]

        bins = np.linspace(-250, 250, 26)
        if len(c_shifts) > 0:
            ax.hist(c_shifts, bins=bins, alpha=0.6, color='#4dac26',
                    label='Char n-gram', density=True)
        if len(w_shifts) > 0:
            ax.hist(w_shifts, bins=bins, alpha=0.6, color='#d01c8b',
                    label='Word n-gram', density=True)
        ax.axvline(0, color='#333', lw=1, ls='--', alpha=0.5)
        ax.set_title(unit.replace('_', ' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Δ MAP (years)', fontsize=7)
        ax.tick_params(labelsize=7)
        c_std = np.std(c_shifts) if len(c_shifts) > 0 else np.nan
        w_std = np.std(w_shifts) if len(w_shifts) > 0 else np.nan
        ax.text(0.03, 0.95,
                f'σ_char={c_std:.0f}\nσ_word={w_std:.0f}',
                transform=ax.transAxes, va='top', fontsize=7,
                color='#333333')
        if unit == KEY_UNITS[0]:
            ax.legend(fontsize=7, loc='upper right')

    for ax in axes[len(KEY_UNITS):]:
        ax.set_visible(False)

    fig.suptitle('Distribution of MAP shifts under Monte Carlo interpolation removal\n'
                 'Δ = MAP(iteration) − MAP(S0 baseline)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Shift distribution plot saved: {Path(out_path).name}')


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    np.random.seed(RNG_SEED)
    outdir    = WORKSPACE
    date_grid = np.linspace(DATE_HI, DATE_LO, N_GRID)

    # -----------------------------------------------------------------------
    # STEP 1 — Load BHSA
    # -----------------------------------------------------------------------
    print('=' * 70)
    print('STEP 1 — Load BHSA')
    print('=' * 70)
    sys.path.insert(0, '/sessions/relaxed-modest-dirac/text-fabric-data')
    from tf.app import use as tf_use
    A = tf_use('ETCBC/bhsa', hoist=globals(), checkout='local', silent='deep')
    print('  BHSA loaded.')

    # -----------------------------------------------------------------------
    # STEP 2 — Load S0 pre-selected feature sets
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 2 — Load S0 pre-selected features')
    print('=' * 70)

    cng_features = pd.read_csv(outdir / 'ngram_selected_features.csv')['ngram'].tolist()
    wng_featA    = pd.read_csv(outdir / 'word_ngram_typeA_features.csv')['ngram'].tolist()
    wng_featB    = pd.read_csv(outdir / 'word_ngram_typeB_features.csv')['ngram'].tolist()
    wng_featAB   = wng_featA + wng_featB

    print(f'  Char n-gram features: {len(cng_features)}')
    print(f'  Word n-gram Type A:   {len(wng_featA)}')
    print(f'  Word n-gram Type B:   {len(wng_featB)}')

    # -----------------------------------------------------------------------
    # STEP 3 — Extract verse-level features for flagged training units
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 3 — Extract verse-level features for flagged training units')
    print('=' * 70)

    flagged_verse_cng   = {}  # {unit: ({(ch,vs): {ng: cnt}}, {(ch,vs): n_chars})}
    flagged_verse_wngA  = {}  # {unit: {(ch,vs): {ng: cnt}}}
    flagged_verse_wngB  = {}
    flagged_verse_words = {}  # {unit: {(ch,vs): n_words}}

    for unit in FLAGGED_UNITS:
        pairs = ALL_TRAINING_UNITS[unit]
        print(f'  {unit} ...', end=' ', flush=True)
        v_cng, v_chars = extract_verse_level_cng(pairs, F, L, T)
        vA, vB, v_words = extract_verse_level_wng(pairs, F, L, T)
        flagged_verse_cng[unit]   = (v_cng, v_chars)
        flagged_verse_wngA[unit]  = vA
        flagged_verse_wngB[unit]  = vB
        flagged_verse_words[unit] = v_words
        total_v = len(v_cng)
        print(f'{total_v} verses')

    # -----------------------------------------------------------------------
    # STEP 4 — Extract unit-level features for non-flagged training units
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 4 — Extract unit-level features for non-flagged training units')
    print('=' * 70)

    fixed_cng   = {}   # {unit: {ng: rate}}
    fixed_wngA  = {}
    fixed_wngB  = {}

    for unit in UNIT_ORDER:
        if unit in FLAGGED_UNITS:
            continue
        pairs = ALL_TRAINING_UNITS[unit]
        print(f'  {unit} ...', end=' ', flush=True)
        cng_rates, nw = extract_unit_cng(pairs, F, L, T)
        wA, wB, nw2   = extract_unit_wng(pairs, F, L, T)
        fixed_cng[unit]  = cng_rates
        fixed_wngA[unit] = wA
        fixed_wngB[unit] = wB
        print(f'{nw} words')

    # Also get the S0 baseline rates for flagged units (no removal)
    for unit in FLAGGED_UNITS:
        v_cng, v_chars = flagged_verse_cng[unit]
        fixed_cng[unit]  = compute_flagged_unit_rates_cng(
            unit, None, v_cng, v_chars, set(), cng_features)
        rA, rB = compute_flagged_unit_rates_wng(
            flagged_verse_wngA[unit], flagged_verse_wngB[unit],
            flagged_verse_words[unit], set(), wng_featA, wng_featB)
        fixed_wngA[unit] = rA
        fixed_wngB[unit] = rB

    # -----------------------------------------------------------------------
    # STEP 5 — Extract features for test units
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 5 — Extract test unit features')
    print('=' * 70)

    test_cng  = {}
    test_wngA = {}
    test_wngB = {}
    for unit, pairs in TEST_UNITS.items():
        print(f'  {unit} ...', end=' ', flush=True)
        cng_rates, nw = extract_unit_cng(pairs, F, L, T)
        wA, wB, nw2   = extract_unit_wng(pairs, F, L, T)
        test_cng[unit]  = cng_rates
        test_wngA[unit] = wA
        test_wngB[unit] = wB
        print(f'{nw} words')

    # -----------------------------------------------------------------------
    # STEP 6 — Compute S0 baseline dates (no removal)
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 6 — S0 baseline dates (no removal)')
    print('=' * 70)

    def make_training_matrix_cng(rates_by_unit):
        return {fn: [rates_by_unit[u].get(fn, 0.0) for u in UNIT_ORDER]
                for fn in cng_features}

    def make_training_matrix_wng(ratesA_by_unit, ratesB_by_unit):
        mat = {fn: [ratesA_by_unit[u].get(fn, 0.0) for u in UNIT_ORDER]
               for fn in wng_featA}
        mat.update({fn: [ratesB_by_unit[u].get(fn, 0.0) for u in UNIT_ORDER]
                    for fn in wng_featB})
        return mat

    s0_ols_cng,  s0_sinv_cng  = build_mvn_model(
        make_training_matrix_cng(fixed_cng), TRAIN_DATES, cng_features)
    s0_ols_wngAB, s0_sinv_wngAB = build_mvn_model(
        make_training_matrix_wng(fixed_wngA, fixed_wngB), TRAIN_DATES, wng_featAB)

    s0_char_maps = {}
    s0_word_maps = {}
    for unit in TEST_UNITS:
        m, lo, hi = compute_map(test_cng[unit], cng_features,
                                s0_ols_cng, s0_sinv_cng, date_grid)
        s0_char_maps[unit] = m

        obs_wAB = {fn: test_wngA[unit].get(fn, 0.0) for fn in wng_featA}
        obs_wAB.update({fn: test_wngB[unit].get(fn, 0.0) for fn in wng_featB})
        m2, lo2, hi2 = compute_map(obs_wAB, wng_featAB,
                                   s0_ols_wngAB, s0_sinv_wngAB, date_grid)
        s0_word_maps[unit] = m2

    print('  S0 dates computed.')

    # -----------------------------------------------------------------------
    # STEP 7 — Monte Carlo iterations
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print(f'STEP 7 — Monte Carlo ({N_ITERATIONS} iterations)')
    print('=' * 70)

    mc_char_maps = {u: [] for u in TEST_UNITS}
    mc_word_maps = {u: [] for u in TEST_UNITS}

    for it in range(N_ITERATIONS):
        if (it + 1) % 25 == 0:
            print(f'  Iteration {it+1}/{N_ITERATIONS}', flush=True)

        # Draw removed verses for each flagged unit
        iter_cng  = dict(fixed_cng)   # shallow copy
        iter_wngA = dict(fixed_wngA)
        iter_wngB = dict(fixed_wngB)

        for unit in FLAGGED_UNITS:
            book = ALL_TRAINING_UNITS[unit][0][0]
            removed = draw_removed_verses(unit, book)
            if not removed:
                continue
            v_cng, v_chars = flagged_verse_cng[unit]
            iter_cng[unit]  = compute_flagged_unit_rates_cng(
                unit, book, v_cng, v_chars, removed, cng_features)
            rA, rB = compute_flagged_unit_rates_wng(
                flagged_verse_wngA[unit], flagged_verse_wngB[unit],
                flagged_verse_words[unit], removed, wng_featA, wng_featB)
            iter_wngA[unit] = rA
            iter_wngB[unit] = rB

        # Build training matrices and fit models
        ols_cng, sinv_cng = build_mvn_model(
            make_training_matrix_cng(iter_cng), TRAIN_DATES, cng_features)
        ols_wngAB, sinv_wngAB = build_mvn_model(
            make_training_matrix_wng(iter_wngA, iter_wngB), TRAIN_DATES, wng_featAB)

        # Predict test units
        for unit in TEST_UNITS:
            m, _, _ = compute_map(test_cng[unit], cng_features,
                                  ols_cng, sinv_cng, date_grid)
            mc_char_maps[unit].append(m)

            obs = {fn: test_wngA[unit].get(fn, 0.0) for fn in wng_featA}
            obs.update({fn: test_wngB[unit].get(fn, 0.0) for fn in wng_featB})
            m2, _, _ = compute_map(obs, wng_featAB, ols_wngAB, sinv_wngAB, date_grid)
            mc_word_maps[unit].append(m2)

    print(f'  Monte Carlo complete.')

    # -----------------------------------------------------------------------
    # STEP 8 — Assemble results
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 8 — Assemble results')
    print('=' * 70)

    def pct(arr, p):
        a = np.array(arr)
        a = a[np.isfinite(a)]
        return round(float(np.percentile(a, p)), 1) if len(a) > 0 else ''

    def std_(arr):
        a = np.array(arr)
        a = a[np.isfinite(a)]
        return round(float(np.std(a)), 1) if len(a) > 0 else ''

    rows = []
    for unit in TEST_UNITS:
        cm = mc_char_maps[unit]
        wm = mc_word_maps[unit]
        row = {
            'unit':            unit,
            'n_words':         len(list(test_cng[unit].keys())),
            'noisy':           unit in NOISY_UNITS,
            # S0 baselines
            'map_char_S0':     round(s0_char_maps[unit], 1),
            'map_word_S0':     round(s0_word_maps[unit], 1),
            # MC char n-gram
            'mc_char_mean':    pct(cm, 50),
            'mc_char_std':     std_(cm),
            'mc_char_p16':     pct(cm, 16),
            'mc_char_p84':     pct(cm, 84),
            'mc_char_p2p5':    pct(cm, 2.5),
            'mc_char_p97p5':   pct(cm, 97.5),
            # MC word n-gram
            'mc_word_mean':    pct(wm, 50),
            'mc_word_std':     std_(wm),
            'mc_word_p16':     pct(wm, 16),
            'mc_word_p84':     pct(wm, 84),
            'mc_word_p2p5':    pct(wm, 2.5),
            'mc_word_p97p5':   pct(wm, 97.5),
            # Shift = MC median - S0
            'shift_char':      round(float(np.median([x for x in cm if np.isfinite(x)])) - s0_char_maps[unit], 1),
            'shift_word':      round(float(np.median([x for x in wm if np.isfinite(x)])) - s0_word_maps[unit], 1),
        }
        rows.append(row)
        print(f"  {unit:<18} char: S0={row['map_char_S0']:.0f} "
              f"mc_med={row['mc_char_mean']} σ={row['mc_char_std']}  "
              f"word: S0={row['map_word_S0']:.0f} "
              f"mc_med={row['mc_word_mean']} σ={row['mc_word_std']}")

    df = pd.DataFrame(rows)
    df.to_csv(outdir / 'mc_sensitivity_summary.csv', index=False)
    print(f"\n  Saved: mc_sensitivity_summary.csv")

    # -----------------------------------------------------------------------
    # STEP 9 — Plots
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 9 — Plots')
    print('=' * 70)

    make_violin_plot(mc_char_maps, s0_char_maps,
                     'Character n-gram model',
                     outdir / 'mc_sensitivity_violin_char.png')

    make_violin_plot(mc_word_maps, s0_word_maps,
                     'Word n-gram model (A+B combined)',
                     outdir / 'mc_sensitivity_violin_word.png')

    make_shift_distribution_plot(mc_char_maps, mc_word_maps,
                                  s0_char_maps, s0_word_maps,
                                  outdir / 'mc_sensitivity_shift_dist.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
