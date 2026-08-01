"""
Script 22a — Extract and cache verse-level features for Monte Carlo sensitivity.

Saves to workspace:
  mc_cache_verse_cng.json     — verse-level char n-gram raw counts (7 flagged units)
  mc_cache_verse_wng.json     — verse-level word n-gram raw counts (7 flagged units)
  mc_cache_fixed_cng.json     — unit-level char n-gram rates (15 non-flagged units)
  mc_cache_fixed_wng.json     — unit-level word n-gram rates (15 non-flagged units)
  mc_cache_test_cng.json      — test unit char n-gram rates (23 units)
  mc_cache_test_wng.json      — test unit word n-gram rates (23 units)
  mc_cache_nwords.json        — word counts per test unit

Stores ONLY the S0 pre-selected features to keep files small.
"""
import sys
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
WORKSPACE = Path('/sessions/relaxed-modest-dirac/mnt/Diachronic Hebrew')

# ---------------------------------------------------------------------------
# Configuration — same as Script 22
# ---------------------------------------------------------------------------
FLAGGED_UNITS = {'Amos', 'Hosea', 'Micah', 'Isaiah_1', 'Nahum', 'Habakkuk', 'Zephaniah'}

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

CNG_SIZES = [3, 4]
WNG_SIZES = [2, 3]
WORD_SEP  = '_'
FUNCTION_POS = frozenset({'prep', 'conj', 'art', 'nega', 'prps', 'prde', 'inrg'})


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_verse_cng(book_ch_pairs, F, L, T, feat_set):
    """Returns {ch_vs_str: {ng: count}}, {ch_vs_str: n_chars}"""
    verse_counts = {}
    verse_chars  = {}
    for book, ch_ranges in book_ch_pairs:
        bn = T.nodeFromSection((book,))
        if bn is None: continue
        for ch_node in L.d(bn, 'chapter'):
            ch = int(F.chapter.v(ch_node))
            if not any(s <= ch <= e for s, e in ch_ranges): continue
            for vs_node in L.d(ch_node, 'verse'):
                vs = int(F.verse.v(vs_node))
                toks = [F.g_cons_utf8.v(w) for w in L.d(vs_node, 'word') if F.g_cons_utf8.v(w)]
                if not toks: continue
                text = WORD_SEP + WORD_SEP.join(toks) + WORD_SEP
                c = Counter()
                for n in CNG_SIZES:
                    for i in range(len(text) - n + 1):
                        ng = text[i:i+n]
                        if ng in feat_set:
                            c[ng] += 1
                key = f'{ch}:{vs}'
                verse_counts[key] = dict(c)
                verse_chars[key]  = len(text)
    return verse_counts, verse_chars


def pos_tok(word, F):
    sp = F.sp.v(word) or 'unkn'
    return f'verb_{F.vt.v(word) or "unkn"}' if sp == 'verb' else sp


def extract_verse_wng(book_ch_pairs, F, L, T, featA_set, featB_set):
    verse_A = {}; verse_B = {}; verse_w = {}
    for book, ch_ranges in book_ch_pairs:
        bn = T.nodeFromSection((book,))
        if bn is None: continue
        for ch_node in L.d(bn, 'chapter'):
            ch = int(F.chapter.v(ch_node))
            if not any(s <= ch <= e for s, e in ch_ranges): continue
            for vs_node in L.d(ch_node, 'verse'):
                vs = int(F.verse.v(vs_node))
                pos_seq = []; fw_seq = []; nw = 0
                for word in L.d(vs_node, 'word'):
                    sp = F.sp.v(word)
                    if not sp: continue
                    pos_seq.append(pos_tok(word, F))
                    fw_seq.append(F.lex.v(word) or sp if sp in FUNCTION_POS else None)
                    nw += 1
                if nw == 0: continue
                cA = Counter(); cB = Counter()
                for sz in WNG_SIZES:
                    for i in range(len(pos_seq) - sz + 1):
                        k = '·'.join(pos_seq[i:i+sz])
                        if k in featA_set: cA[k] += 1
                    for i in range(len(fw_seq) - sz + 1):
                        gram = fw_seq[i:i+sz]
                        if any(t is None for t in gram): continue
                        k = '·'.join(str(t) for t in gram)
                        if k in featB_set: cB[k] += 1
                key = f'{ch}:{vs}'
                verse_A[key] = dict(cA); verse_B[key] = dict(cB); verse_w[key] = nw
    return verse_A, verse_B, verse_w


def extract_unit_cng(book_ch_pairs, F, L, T, feat_set):
    toks = []
    for book, ch_ranges in book_ch_pairs:
        bn = T.nodeFromSection((book,))
        if bn is None: continue
        for ch_node in L.d(bn, 'chapter'):
            ch = int(F.chapter.v(ch_node))
            if not any(s <= ch <= e for s, e in ch_ranges): continue
            for w in L.d(ch_node, 'word'):
                c = F.g_cons_utf8.v(w)
                if c: toks.append(c)
    if not toks: return {}, 0
    text = WORD_SEP + WORD_SEP.join(toks) + WORD_SEP
    nch  = len(text)
    cnt  = Counter()
    for n in CNG_SIZES:
        for i in range(len(text) - n + 1):
            ng = text[i:i+n]
            if ng in feat_set: cnt[ng] += 1
    return {ng: v / nch * 1000 for ng, v in cnt.items()}, len(toks)


def extract_unit_wng(book_ch_pairs, F, L, T, featA_set, featB_set):
    pos_seq = []; fw_seq = []; nw = 0
    for book, ch_ranges in book_ch_pairs:
        bn = T.nodeFromSection((book,))
        if bn is None: continue
        for ch_node in L.d(bn, 'chapter'):
            ch = int(F.chapter.v(ch_node))
            if not any(s <= ch <= e for s, e in ch_ranges): continue
            for word in L.d(ch_node, 'word'):
                sp = F.sp.v(word)
                if not sp: continue
                pos_seq.append(pos_tok(word, F))
                fw_seq.append(F.lex.v(word) or sp if sp in FUNCTION_POS else None)
                nw += 1
    if nw == 0: return {}, {}, 0
    cA = Counter(); cB = Counter()
    for sz in WNG_SIZES:
        for i in range(len(pos_seq) - sz + 1):
            k = '·'.join(pos_seq[i:i+sz])
            if k in featA_set: cA[k] += 1
        for i in range(len(fw_seq) - sz + 1):
            gram = fw_seq[i:i+sz]
            if any(t is None for t in gram): continue
            k = '·'.join(str(t) for t in gram)
            if k in featB_set: cB[k] += 1
    rA = {ng: v / nw * 1000 for ng, v in cA.items()}
    rB = {ng: v / nw * 1000 for ng, v in cB.items()}
    return rA, rB, nw


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print('Loading BHSA...')
    sys.path.insert(0, '/sessions/relaxed-modest-dirac/text-fabric-data')
    from tf.app import use as tf_use
    A = tf_use('ETCBC/bhsa', hoist=globals(), checkout='local', silent='deep')
    print('BHSA loaded.')

    # Load S0 feature sets
    cng_feats = pd.read_csv(WORKSPACE / 'ngram_selected_features.csv')['ngram'].tolist()
    wngA_feats = pd.read_csv(WORKSPACE / 'word_ngram_typeA_features.csv')['ngram'].tolist()
    wngB_feats = pd.read_csv(WORKSPACE / 'word_ngram_typeB_features.csv')['ngram'].tolist()
    cng_set = set(cng_feats); wngA_set = set(wngA_feats); wngB_set = set(wngB_feats)
    print(f'Features: cng={len(cng_feats)} wngA={len(wngA_feats)} wngB={len(wngB_feats)}')

    # --- Verse-level for flagged units ---
    cache_vcng  = {}; cache_vchars = {}
    cache_vwngA = {}; cache_vwngB  = {}; cache_vwords = {}
    print('\nExtracting verse-level features for flagged units...')
    for unit in FLAGGED_UNITS:
        pairs = ALL_TRAINING_UNITS[unit]
        print(f'  {unit}...', end=' ', flush=True)
        vc, vch = extract_verse_cng(pairs, F, L, T, cng_set)
        vA, vB, vw_dict = extract_verse_wng(pairs, F, L, T, wngA_set, wngB_set)
        cache_vcng[unit]   = vc
        cache_vchars[unit] = vch
        cache_vwngA[unit]  = vA
        cache_vwngB[unit]  = vB
        cache_vwords[unit] = vw_dict
        print(f'{len(vc)} verses')

    # --- Unit-level for non-flagged training units ---
    cache_fixed_cng  = {}; cache_fixed_wngA = {}; cache_fixed_wngB = {}
    print('\nExtracting unit-level features for non-flagged training units...')
    for unit, pairs in ALL_TRAINING_UNITS.items():
        if unit in FLAGGED_UNITS: continue
        print(f'  {unit}...', end=' ', flush=True)
        rc, nw  = extract_unit_cng(pairs, F, L, T, cng_set)
        rA, rB, nw2 = extract_unit_wng(pairs, F, L, T, wngA_set, wngB_set)
        cache_fixed_cng[unit]  = rc
        cache_fixed_wngA[unit] = rA
        cache_fixed_wngB[unit] = rB
        print(f'{nw} words')

    # --- Test units ---
    cache_test_cng  = {}; cache_test_wngA = {}; cache_test_wngB = {}
    cache_nwords    = {}
    print('\nExtracting test unit features...')
    for unit, pairs in TEST_UNITS.items():
        print(f'  {unit}...', end=' ', flush=True)
        rc, nw  = extract_unit_cng(pairs, F, L, T, cng_set)
        rA, rB, nw2 = extract_unit_wng(pairs, F, L, T, wngA_set, wngB_set)
        cache_test_cng[unit]  = rc
        cache_test_wngA[unit] = rA
        cache_test_wngB[unit] = rB
        cache_nwords[unit]    = nw
        print(f'{nw} words')

    # --- Save ---
    print('\nSaving caches...')
    def save(obj, name):
        path = WORKSPACE / f'mc_cache_{name}.json'
        with open(path, 'w') as f:
            json.dump(obj, f)
        print(f'  Saved {path.name}')

    # Verse-level data: combine into one dict per unit
    verse_data = {}
    for unit in FLAGGED_UNITS:
        verse_data[unit] = {
            'cng':   cache_vcng[unit],
            'chars': cache_vchars[unit],
            'wngA':  cache_vwngA[unit],
            'wngB':  cache_vwngB[unit],
            'words': cache_vwords[unit],
        }
    save(verse_data,          'verse_data')
    save(cache_fixed_cng,     'fixed_cng')
    save(cache_fixed_wngA,    'fixed_wngA')
    save(cache_fixed_wngB,    'fixed_wngB')
    save(cache_test_cng,      'test_cng')
    save(cache_test_wngA,     'test_wngA')
    save(cache_test_wngB,     'test_wngB')
    save(cache_nwords,        'nwords')

    print('\nDone. All caches saved.')


if __name__ == '__main__':
    main()
