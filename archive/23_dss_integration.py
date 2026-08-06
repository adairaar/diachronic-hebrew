"""
Script 23 — DSS / Extrabiblical corpus integration
====================================================
Loads the ETCBC/extrabiblical Text-Fabric corpus and applies the existing
char and word n-gram diachronic dating models.

Part A — Validation
  Apply the existing BHSA-calibrated model to all extrabiblical units.
  Compare predicted dates with known/expected dates.

Part B — Augmented calibration
  Add Community Rule (1QS, ~150 BCE) and War Scroll (1QM, ~100 BCE) to
  the training set; refit both models; compare how Torah/historical test
  unit dates change.

Outputs
-------
  dss_partA_results.csv          — MAP + CI68 for all extrabiblical units
  dss_partB_shift.csv            — Date shift on standard test units (B vs A)
  dss_validation_dotplot.png     — Predicted vs expected dates (Part A)
  dss_calibration_shift.png      — Shift dot-plot (Part B effect)

DATA INTAKE SPECIFICATION
  See bottom of file — format required to add Ben Sira / Jubilees.
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tf.fabric import Fabric

warnings.filterwarnings('ignore')

WORKSPACE   = pathlib.Path('/sessions/relaxed-modest-dirac/mnt/Diachronic Hebrew')
TF_EXTRAB   = str(pathlib.Path.home() /
                  'text-fabric-data' / 'ETCBC' / 'extrabiblical' / 'tf' / '0.2')

# ---------------------------------------------------------------------------
# Extrabiblical unit catalogue
# ---------------------------------------------------------------------------
# Prose DSS: Part A validation + Part B training augmentation
DSS_PROSE = {
    'B_1QS': {'date': 150, 'label': 'Community Rule (1QS)'},
    'B_1QM': {'date': 100, 'label': 'War Scroll (1QM)'},
}
# Poetry DSS: Part A only (flagged noisy, analogous to Song of Sea)
DSS_POETRY = {
    'B_1QHa': {'date': 150, 'label': 'Hodayot (1QHa) [poetry]'},
}
# Rabbinic texts: Part A only (outside date grid ~200 CE)
RABBINIC = {
    'Pirqe':   {'date': None, 'label': 'Pirqe Avot (~200 CE)'},
    'Shirata': {'date': None, 'label': 'Shirata (Rabbinic, poetry)'},
}
# Pre-exilic inscriptions: Part A spot-check (too small for calibration)
INSCRIPTIONS = {
    'Mesa':             {'date': 840, 'label': 'Mesha Stela (~840 BCE)'},
    'Balaam':           {'date': 800, 'label': 'Balaam/Deir Alla (~800 BCE) ⚠non-BH'},
    'Ajrud':            {'date': 800, 'label': 'Kuntillet Ajrud (~800 BCE)'},
    'Arad':             {'date': 600, 'label': 'Arad ostraca (~600 BCE)'},
    'Lachish':          {'date': 587, 'label': 'Lachish letters (~587 BCE)'},
    'Mesad_Hashavyahu': {'date': 630, 'label': 'Mesad Hashavyahu (~630 BCE)'},
    'Ketef_Hinnom':     {'date': 625, 'label': 'Ketef Hinnom (~625 BCE)'},
    'Siloam':           {'date': 700, 'label': 'Siloam inscription (~700 BCE)'},
}
ALL_EXTRAB = {**DSS_PROSE, **DSS_POETRY, **RABBINIC, **INSCRIPTIONS}

# ---------------------------------------------------------------------------
# BHSA training corpus (Scripts 16/17/22)
# ---------------------------------------------------------------------------
TRAIN_UNITS = [
    'Amos','Hosea','Micah','Nahum','Habakkuk','Zephaniah','Isaiah_1',
    'Isaiah_2','Isaiah_3','Jeremiah','Ezekiel','Haggai','Zechariah_1',
    'Malachi','Jonah','Lamentations','Ezra','Nehemiah','Chronicles',
    'Daniel','Ecclesiastes','Esther',
]
TRAIN_DATES = [760,740,720,620,600,620,720,540,450,590,580,520,518,
               450,400,580,350,350,350,167,250,350]
FLAGGED = {'Amos','Hosea','Micah','Isaiah_1','Nahum','Habakkuk','Zephaniah'}

# MVN hyper-parameters (same throughout pipeline)
N_GRID    = 500
DATE_HI   = 1200
DATE_LO   = 50
PRIOR_MU  = 600.0
PRIOR_SIG = 350.0
RIDGE     = 0.20


# ---------------------------------------------------------------------------
# Vectorised MVN helpers (from Script 22b)
# ---------------------------------------------------------------------------
def build_mvn(rates_mat, dates, features):
    x  = np.array(dates, dtype=float); N = len(x); K = len(features)
    Y  = np.array([rates_mat[fn] for fn in features], dtype=float)
    np.nan_to_num(Y, copy=False)
    xm = x.mean(); xd = x - xm; xv = float((xd**2).sum()) or 1.0
    ym = Y.mean(axis=1)
    slopes     = (Y * xd).sum(axis=1) / xv
    intercepts = ym - slopes * xm
    pred = intercepts[:,None] + slopes[:,None] * x[None,:]
    R    = (Y - pred).T
    Sig  = R.T @ R / max(N-2, 1)
    lam  = RIDGE * np.trace(Sig) / K if K > 0 else 0.0
    Sinv = np.linalg.inv(Sig + lam * np.eye(K))
    return intercepts, slopes, Sinv


def get_map_ci(obs_rates, features, intercepts, slopes, Sinv, date_grid):
    obs  = np.array([obs_rates.get(fn, 0.0) for fn in features], dtype=float)
    pred = intercepts[None,:] + slopes[None,:] * date_grid[:,None]
    diff = obs[None,:] - pred
    A    = diff @ Sinv
    lp   = -0.5 * (A * diff).sum(axis=1)
    lp  += -0.5 * ((date_grid - PRIOR_MU) / PRIOR_SIG) ** 2
    lp  -= lp.max()
    post = np.exp(lp); post /= post.sum()
    cdf  = np.cumsum(post)
    map_ = float(date_grid[np.argmax(post)])
    p16  = float(date_grid[np.searchsorted(cdf, 0.16)])
    p84  = float(date_grid[np.searchsorted(cdf, 0.84)])
    return map_, p16, p84


# ---------------------------------------------------------------------------
# Feature extraction from extrabiblical TF corpus
# ---------------------------------------------------------------------------
SEP = '·'   # middle dot — same separator used throughout pipeline

def unit_cng_rates(book_node, cng_feats, F, L):
    """Char n-gram rates per 1000 consonants for an extrabiblical book node."""
    words = L.d(book_node, otype='word')
    flat  = ''.join(F.g_cons_utf8.v(w) or '' for w in words)
    n_ch  = len(flat)
    if n_ch == 0:
        return {fn: 0.0 for fn in cng_feats}
    total = {fn: 0 for fn in cng_feats}
    feat_set = set(cng_feats)
    for i in range(n_ch):
        for n in (3, 4):
            ng = flat[i:i+n]
            if len(ng) == n and ng in feat_set:
                total[ng] += 1
    return {fn: total[fn] / n_ch * 1000 for fn in cng_feats}


def unit_wng_rates(book_node, wngA_feats, wngB_feats, F, L):
    """Word n-gram rates per 1000 words for an extrabiblical book node."""
    words  = L.d(book_node, otype='word')
    n_words = len(words)
    if n_words == 0:
        return ({fn: 0.0 for fn in wngA_feats},
                {fn: 0.0 for fn in wngB_feats})

    totA  = {fn: 0 for fn in wngA_feats}
    totB  = {fn: 0 for fn in wngB_feats}
    setA  = set(wngA_feats); setB = set(wngB_feats)
    FUNC_SP = {'conj','prep','art','prps','prde','prin','nega','inrg'}

    tag_seq = []
    lex_seq = []
    for w in words:
        sp  = F.sp.v(w)  or 'unknown'
        vt  = F.vt.v(w)  or 'NA'
        lex = F.lex.v(w) or ''
        tag = f'{sp}_{vt}' if sp == 'verb' else sp
        tag_seq.append(tag)
        if sp in FUNC_SP:
            lex_seq.append(lex)

    # Type A: tag n-grams (bigrams and trigrams)
    for i in range(len(tag_seq)):
        for n in (2, 3):
            ng = SEP.join(tag_seq[i:i+n])
            if len(ng.split(SEP)) == n and ng in setA:
                totA[ng] += 1

    # Type B: function-word lex bigrams
    for i in range(len(lex_seq)-1):
        ng = f'{lex_seq[i]}{SEP}{lex_seq[i+1]}'
        if ng in setB:
            totB[ng] += 1

    rA = {fn: totA[fn] / n_words * 1000 for fn in wngA_feats}
    rB = {fn: totB[fn] / n_words * 1000 for fn in wngB_feats}
    return rA, rB


# ---------------------------------------------------------------------------
# S0 training rate reconstruction from mc_cache files
# ---------------------------------------------------------------------------
def build_s0_rates(cng_feats, wngA_feats, wngB_feats):
    """Reconstruct full S0 training rates from existing JSON caches."""
    def load(name):
        with open(WORKSPACE / f'mc_cache_{name}.json') as f:
            return json.load(f)

    fixed_cng  = load('fixed_cng')
    fixed_wngA = load('fixed_wngA')
    fixed_wngB = load('fixed_wngB')
    verse_data = load('verse_data')

    s0_cng   = dict(fixed_cng)
    s0_wngA  = dict(fixed_wngA)
    s0_wngB  = dict(fixed_wngB)

    # Add flagged units at full (no-removal) rates
    for unit in FLAGGED:
        vd = verse_data[unit]
        total_cng = {fn: 0 for fn in cng_feats}
        total_chars = 0
        for key, cnt in vd['cng'].items():
            for fn, c in cnt.items():
                if fn in total_cng: total_cng[fn] += c
            total_chars += vd['chars'].get(key, 0)
        s0_cng[unit] = ({fn: total_cng[fn] / total_chars * 1000
                         for fn in cng_feats} if total_chars else
                        {fn: 0.0 for fn in cng_feats})

        totA = {fn: 0 for fn in wngA_feats}
        totB = {fn: 0 for fn in wngB_feats}
        nw   = 0
        for key in vd['words']:
            for fn, c in vd['wngA'].get(key, {}).items():
                if fn in totA: totA[fn] += c
            for fn, c in vd['wngB'].get(key, {}).items():
                if fn in totB: totB[fn] += c
            nw += vd['words'][key]
        s0_wngA[unit] = ({fn: totA[fn] / nw * 1000 for fn in wngA_feats}
                         if nw else {fn: 0.0 for fn in wngA_feats})
        s0_wngB[unit] = ({fn: totB[fn] / nw * 1000 for fn in wngB_feats}
                         if nw else {fn: 0.0 for fn in wngB_feats})

    return s0_cng, s0_wngA, s0_wngB


# ---------------------------------------------------------------------------
# Matrix builders (same layout as Script 22b)
# ---------------------------------------------------------------------------
def make_cng_matrix(rates_by_unit, feats, units):
    return {fn: [rates_by_unit[u].get(fn, 0.0) for u in units] for fn in feats}

def make_wng_matrix(rA_by_unit, rB_by_unit, wngA_feats, wngB_feats, units):
    mat = {fn: [rA_by_unit[u].get(fn, 0.0) for u in units] for fn in wngA_feats}
    mat.update({fn: [rB_by_unit[u].get(fn, 0.0) for u in units] for fn in wngB_feats})
    return mat


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def validation_plot(rows, out_path):
    """Dot plot: predicted vs expected dates for extrabiblical texts."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    for ax, model_key, model_label, ci_lo, ci_hi in [
        (axes[0], 'map_char', 'Char n-gram', 'char_p16', 'char_p84'),
        (axes[1], 'map_word', 'Word n-gram (Type A)', 'word_p16', 'word_p84'),
    ]:
        labels = [r['label'] for r in rows]
        predicted = [r[model_key] for r in rows]
        expected  = [r['expected_date'] for r in rows]
        colors    = [r['color'] for r in rows]

        for i, (pred, exp, lab, col) in enumerate(
                zip(predicted, expected, labels, colors)):
            if pred is None: continue
            # p16 = older (larger BCE), p84 = younger (smaller BCE)
            p_old = rows[i][ci_lo]  # char_p16 / word_p16 — older bound
            p_yng = rows[i][ci_hi]  # char_p84 / word_p84 — younger bound
            # on inverted axis: left = older, right = younger
            err_left  = max(0.0, p_old - pred)
            err_right = max(0.0, pred  - p_yng)
            ax.errorbar(pred, i, xerr=[[err_left],[err_right]],
                        fmt='o', color=col, capsize=4, markersize=7)
            if exp is not None:
                ax.scatter(exp, i, marker='|', s=200, color='#888', zorder=5,
                           linewidths=2)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Date (BCE)', fontsize=10)
        ax.invert_xaxis()
        ax.set_xlim(1300, -250)
        ax.axvline(50, color='#ccc', lw=0.8, ls='--')
        ax.set_title(f'{model_label}\n'
                     'Filled dot = predicted MAP  |  grey bar = known date',
                     fontsize=10, fontweight='bold')
        ax.grid(axis='x', alpha=0.2)

    # Legend
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color='#2166ac', label='DSS prose (~100-150 BCE)'),
        mpatches.Patch(color='#762a83', label='DSS poetry (Hodayot)'),
        mpatches.Patch(color='#d73027', label='Rabbinic (~200 CE)'),
        mpatches.Patch(color='#4dac26', label='Pre-exilic inscriptions'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Part A — Validation: extrabiblical texts applied to BHSA-trained model\n'
                 'Error bars = 68% credible interval', fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {pathlib.Path(out_path).name}')


def shift_plot(partA_dates, partB_dates, out_path):
    """Dot plot showing how test unit dates shift when DSS texts enter training."""
    test_units = sorted(partA_dates.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 9))
    for ax, model in [(axes[0], 'char'), (axes[1], 'word')]:
        key = f'map_{model}'
        shifts = [partB_dates[u][key] - partA_dates[u][key] for u in test_units]
        colors = ['#2166ac' if s >= 0 else '#d73027' for s in shifts]
        ax.barh(range(len(test_units)), shifts, color=colors, alpha=0.7)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_yticks(range(len(test_units)))
        ax.set_yticklabels([u.replace('_',' ') for u in test_units], fontsize=8)
        ax.set_xlabel('Shift in MAP date (years; +ve = older)', fontsize=10)
        ax.set_title(f'{model.capitalize()} n-gram model\n'
                     'Shift = Part B − Part A (adding 1QS + 1QM to training)',
                     fontsize=10, fontweight='bold')
        ax.grid(axis='x', alpha=0.25)
    fig.suptitle('Part B — Calibration augmentation: effect of adding DSS prose texts to training',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {pathlib.Path(out_path).name}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    date_grid = np.linspace(DATE_HI, DATE_LO, N_GRID)

    # ---- Feature lists ----
    cng_feats   = pd.read_csv(WORKSPACE / 'ngram_selected_features.csv')['ngram'].tolist()
    wngA_feats  = pd.read_csv(WORKSPACE / 'word_ngram_typeA_features.csv')['ngram'].tolist()
    wngB_feats  = pd.read_csv(WORKSPACE / 'word_ngram_typeB_features.csv')['ngram'].tolist()
    wngAB_feats = wngA_feats + wngB_feats
    print(f'Features: {len(cng_feats)} char, {len(wngA_feats)} wngA, {len(wngB_feats)} wngB')

    # ---- Load BHSA S0 training rates ----
    print('\nReconstructing BHSA S0 training rates from cache...')
    s0_cng, s0_wngA, s0_wngB = build_s0_rates(cng_feats, wngA_feats, wngB_feats)
    print(f'  Training units: {len(s0_cng)}')

    # ---- Load test unit rates ----
    def load(name):
        with open(WORKSPACE / f'mc_cache_{name}.json') as f:
            return json.load(f)
    test_cng  = load('test_cng')
    test_wngA = load('test_wngA')
    test_wngB = load('test_wngB')
    test_units = list(test_cng.keys())
    print(f'  Test units: {len(test_units)}')

    # ---- Load extrabiblical TF corpus ----
    print('\nLoading ETCBC/extrabiblical corpus...')
    TF  = Fabric(locations=[TF_EXTRAB], silent=True)
    api = TF.load('g_cons_utf8 sp vt lex book', silent=True)
    F, L = api.F, api.L

    book_nodes = {F.book.v(b): b for b in F.otype.s('book')}
    print(f'  Books found: {sorted(book_nodes.keys())}')

    # ---- Extract extrabiblical rates ----
    print('\nExtracting extrabiblical features...')
    extrab_cng  = {}
    extrab_wngA = {}
    extrab_wngB = {}
    extrab_nwords = {}

    for unit in ALL_EXTRAB:
        if unit not in book_nodes:
            print(f'  WARNING: {unit} not found in corpus — skipping')
            continue
        bn = book_nodes[unit]
        words = L.d(bn, otype='word')
        extrab_nwords[unit] = len(words)

        extrab_cng[unit]  = unit_cng_rates(bn, cng_feats, F, L)
        rA, rB = unit_wng_rates(bn, wngA_feats, wngB_feats, F, L)
        extrab_wngA[unit] = rA
        extrab_wngB[unit] = rB
        print(f'  {unit:<20} {len(words):>6} words')

    # ---- Check Type B lex compatibility ----
    # Count how many Type B features have non-zero rates across DSS prose
    typeB_hits = sum(
        1 for fn in wngB_feats
        if any(extrab_wngB.get(u, {}).get(fn, 0) > 0
               for u in ('B_1QS', 'B_1QM'))
    )
    print(f'\n  Type B coverage in DSS prose: {typeB_hits}/{len(wngB_feats)} features non-zero')

    # ---- Fit Part A model (BHSA S0 training only) ----
    print('\nFitting Part A model (BHSA training only)...')
    ic_c, sl_c, sinv_c = build_mvn(
        make_cng_matrix(s0_cng, cng_feats, TRAIN_UNITS),
        TRAIN_DATES, cng_feats)
    ic_w, sl_w, sinv_w = build_mvn(
        make_wng_matrix(s0_wngA, s0_wngB, wngA_feats, wngB_feats, TRAIN_UNITS),
        TRAIN_DATES, wngAB_feats)

    # ---- Part A: apply to all extrabiblical units ----
    print('\nPart A — dates for extrabiblical units:')
    COLOR_MAP = {
        **{u: '#2166ac' for u in DSS_PROSE},
        **{u: '#762a83' for u in DSS_POETRY},
        **{u: '#d73027' for u in RABBINIC},
        **{u: '#4dac26' for u in INSCRIPTIONS},
    }
    partA_extrab = {}
    for unit in ALL_EXTRAB:
        if unit not in extrab_cng: continue
        obs_c = extrab_cng[unit]
        obs_w = {**{fn: extrab_wngA[unit].get(fn,0.) for fn in wngA_feats},
                 **{fn: extrab_wngB[unit].get(fn,0.) for fn in wngB_feats}}
        mc, c16, c84 = get_map_ci(obs_c, cng_feats, ic_c, sl_c, sinv_c, date_grid)
        mw, w16, w84 = get_map_ci(obs_w, wngAB_feats, ic_w, sl_w, sinv_w, date_grid)
        partA_extrab[unit] = dict(map_char=mc, char_p16=c16, char_p84=c84,
                                  map_word=mw, word_p16=w16, word_p84=w84)
        exp = ALL_EXTRAB[unit].get('date')
        lab = ALL_EXTRAB[unit]['label']
        # p16=older bound, p84=younger bound → half-width = (p16-p84)/2
        print(f'  {unit:<20} char={mc:>5.0f} [CI68: {c84:.0f}–{c16:.0f}]  '
              f'word={mw:>5.0f} [CI68: {w84:.0f}–{w16:.0f}]'
              + (f'  expected≈{exp}' if exp else '  [outside grid]'))

    # ---- Part A: also apply to BHSA test units ----
    print('\nPart A — BHSA test unit dates (baseline):')
    partA_test = {}
    for u in test_units:
        obs_c = test_cng[u]
        obs_w = {**{fn: test_wngA[u].get(fn,0.) for fn in wngA_feats},
                 **{fn: test_wngB[u].get(fn,0.) for fn in wngB_feats}}
        mc, c16, c84 = get_map_ci(obs_c, cng_feats, ic_c, sl_c, sinv_c, date_grid)
        mw, w16, w84 = get_map_ci(obs_w, wngAB_feats, ic_w, sl_w, sinv_w, date_grid)
        partA_test[u] = dict(map_char=mc, char_p16=c16, char_p84=c84,
                             map_word=mw, word_p16=w16, word_p84=w84)
        print(f'  {u:<18} char={mc:.0f}  word={mw:.0f}')

    # ---- Part B: augment training with 1QS + 1QM ----
    print('\nFitting Part B model (BHSA + 1QS + 1QM)...')
    b_train_units = TRAIN_UNITS + ['B_1QS', 'B_1QM']
    b_train_dates = TRAIN_DATES + [150, 100]

    b_cng  = {**s0_cng,  'B_1QS': extrab_cng['B_1QS'],  'B_1QM': extrab_cng['B_1QM']}
    b_wngA = {**s0_wngA, 'B_1QS': extrab_wngA['B_1QS'], 'B_1QM': extrab_wngA['B_1QM']}
    b_wngB = {**s0_wngB, 'B_1QS': extrab_wngB['B_1QS'], 'B_1QM': extrab_wngB['B_1QM']}

    ic_cb, sl_cb, sinv_cb = build_mvn(
        make_cng_matrix(b_cng, cng_feats, b_train_units),
        b_train_dates, cng_feats)
    ic_wb, sl_wb, sinv_wb = build_mvn(
        make_wng_matrix(b_wngA, b_wngB, wngA_feats, wngB_feats, b_train_units),
        b_train_dates, wngAB_feats)

    print('\nPart B — BHSA test unit dates (augmented training):')
    partB_test = {}
    for u in test_units:
        obs_c = test_cng[u]
        obs_w = {**{fn: test_wngA[u].get(fn,0.) for fn in wngA_feats},
                 **{fn: test_wngB[u].get(fn,0.) for fn in wngB_feats}}
        mc, c16, c84 = get_map_ci(obs_c, cng_feats, ic_cb, sl_cb, sinv_cb, date_grid)
        mw, w16, w84 = get_map_ci(obs_w, wngAB_feats, ic_wb, sl_wb, sinv_wb, date_grid)
        partB_test[u] = dict(map_char=mc, char_p16=c16, char_p84=c84,
                             map_word=mw, word_p16=w16, word_p84=w84)
        dc = mc - partA_test[u]['map_char']
        dw = mw - partA_test[u]['map_word']
        print(f'  {u:<18} char={mc:.0f} (Δ{dc:+.0f})  word={mw:.0f} (Δ{dw:+.0f})')

    # ---- Save CSVs ----
    print('\nSaving results...')
    rows_A = []
    for unit, d in partA_extrab.items():
        meta = ALL_EXTRAB[unit]
        rows_A.append({
            'unit':          unit,
            'label':         meta['label'],
            'n_words':       extrab_nwords.get(unit, 0),
            'expected_date': meta.get('date'),
            'map_char':      round(d['map_char'], 1),
            'char_p16':      round(d['char_p16'], 1),
            'char_p84':      round(d['char_p84'], 1),
            'map_word':      round(d['map_word'], 1),
            'word_p16':      round(d['word_p16'], 1),
            'word_p84':      round(d['word_p84'], 1),
        })
    pd.DataFrame(rows_A).to_csv(WORKSPACE / 'dss_partA_results.csv', index=False)
    print('  Saved: dss_partA_results.csv')

    rows_B = []
    for u in test_units:
        a, b_ = partA_test[u], partB_test[u]
        rows_B.append({
            'unit':           u,
            'partA_char':     round(a['map_char'],1),
            'partA_word':     round(a['map_word'],1),
            'partB_char':     round(b_['map_char'],1),
            'partB_word':     round(b_['map_word'],1),
            'shift_char':     round(b_['map_char'] - a['map_char'], 1),
            'shift_word':     round(b_['map_word'] - a['map_word'], 1),
        })
    pd.DataFrame(rows_B).to_csv(WORKSPACE / 'dss_partB_shift.csv', index=False)
    print('  Saved: dss_partB_shift.csv')

    # ---- Plots ----
    print('\nGenerating plots...')
    plot_rows = [
        {**partA_extrab[u],
         'label':         ALL_EXTRAB[u]['label'],
         'expected_date': ALL_EXTRAB[u].get('date'),
         'color':         COLOR_MAP[u]}
        for u in ALL_EXTRAB if u in partA_extrab
    ]
    validation_plot(plot_rows, WORKSPACE / 'dss_validation_dotplot.png')
    shift_plot(partA_test, partB_test, WORKSPACE / 'dss_calibration_shift.png')

    print('\nDone.')


# ---------------------------------------------------------------------------
# DATA INTAKE SPECIFICATION
# ---------------------------------------------------------------------------
DATA_FORMAT_SPEC = """
=======================================================================
DATA INTAKE FORMAT FOR NEW TEXTS (e.g. Ben Sira, Jubilees)
=======================================================================

To integrate a new Hebrew text into this pipeline, two levels of data
are needed.  Level 1 alone enables the char n-gram model; Level 2
additionally enables the word n-gram (Type A) model.

-----------------------------------------------------------------------
LEVEL 1 — Consonantal text only  (char n-gram model)
-----------------------------------------------------------------------
File:  <unit_name>_cons.tsv
Format: TSV, one verse (or equivalent section unit) per row.

Columns:
  unit      str   Unit identifier, e.g. 'Ben_Sira' or 'Jubilees_Heb'
  chapter   int   Chapter number (or equivalent major section)
  verse     int   Verse / line number within chapter
  consonants str  Space-separated consonantal Hebrew words (Unicode,
                  no vowels, no cantillation, no final forms needed—
                  but consistent casing within the corpus).
                  Example: "בראשית ברא אלהים את השמים ואת הארץ"
                  Lacunae / gaps: omit the word entirely (do not use
                  placeholder tokens such as [...]).

Example rows:
  Ben_Sira  1  1  כל חכמה מאת יהוה ועמו לעולם
  Ben_Sira  1  2  חול ימים ותרומות ארץ ומעמקי תהום מי ימדד

Minimum viable corpus: ~3,000 words (char n-gram rates become
  unreliable below this threshold; flag results accordingly).

-----------------------------------------------------------------------
LEVEL 2 — Morphological annotations  (word n-gram Type A model)
-----------------------------------------------------------------------
File:  <unit_name>_morph.tsv
Format: TSV, one WORD per row (not one verse per row).

Columns:
  unit     str   Same identifier as Level 1
  chapter  int
  verse    int
  wnum     int   Word position within verse (1-indexed)
  g_cons   str   Consonantal form of this individual word (Unicode)
  sp       str   Part of speech — MUST use ETCBC values exactly:
                   verb  subs  adjv  conj  prep  art
                   prps  prde  prin  advb  nega  nmpr  inrg  intj
  vt       str   Verb tense — MUST use ETCBC values exactly (verb only):
                   perf  impf  wayq  ptca  ptcp  infc  infa  impv
                   Use 'NA' for non-verb words.
  lex      str   Lexeme in ETCBC transliteration (for Type B features).
                 Required for function words (sp in: conj prep art prps
                 prde prin nega inrg).  May be left blank for content
                 words if Type B features are not needed.
                 ETCBC transliteration key (partial):
                   aleph=>  bet=B   gimel=G  dalet=D  he=H
                   vav=W    zayin=Z  xet=X   tet=V   yod=J
                   kaf=K    lamed=L  mem=M   nun=N   samek=S
                   ayin=<   pe=P    tsade=Y  qof=Q   resh=R
                   shin=C   tav=T
                 Common function-word lexemes:
                   W (conj ו)   H (art ה)   L (prep ל)
                   B (prep ב)   K (prep כ)  KJ (conj כי)
                   LO/ (neg לא) >CR (rel אשׁר)  ZH (dem זה)
                   >L (prep אל)

Example rows (Ben Sira 1:1):
  Ben_Sira  1  1  1  כל  subs  NA   KL
  Ben_Sira  1  1  2  חכמה subs  NA  XKM
  Ben_Sira  1  1  3  מאת  prep  NA  M>T
  Ben_Sira  1  1  4  יהוה nmpr  NA  JHW
  Ben_Sira  1  1  5  ו    conj  NA  W
  Ben_Sira  1  1  6  עמו  subs  NA  <M

-----------------------------------------------------------------------
MANUSCRIPT SOURCE METADATA  (required alongside either level)
-----------------------------------------------------------------------
Provide a JSON or CSV file with:
  unit           str   Same identifier
  date_bce       int   Best estimated date in BCE (positive integer)
                       For texts post-200 BCE use negative numbers
                       (e.g. 200 CE = -200).  Must be within the
                       model date grid (50–1200 BCE for calibration;
                       apply-only outside this range).
  date_uncertainty int Rough ±years on the date estimate
  noisy          bool  True if the text is poetry / hymnic (excluded
                       from training; applied as test only)
  sources        str   Comma-separated list of manuscript sources used,
                       e.g. "Masada,MS_A,MS_B" for Ben Sira
  notes          str   Free-text; lacunae fraction, dialect notes, etc.

-----------------------------------------------------------------------
PRACTICAL NOTES FOR BEN SIRA
-----------------------------------------------------------------------
Recommended source hierarchy:
  1. Masada scroll (MasSir): ~50 BCE copy; most reliable
  2. DSS fragment 2Q18: 1st century BCE
  3. Cairo Geniza MSS A–F: medieval copies; use with caution
     (potential scribal modernisation toward Masoretic Hebrew)
     Suggested: include Geniza MSS but flag as 'secondary_source=True'
     so their contribution can be toggled in sensitivity analysis.

Suggested date: 180 BCE  (composition date of Ben Sira himself)
Estimated words: ~15,000 if Masada + all Geniza (~2/3 of full text)
                 ~3,500 if Masada + DSS only

-----------------------------------------------------------------------
PRACTICAL NOTES FOR JUBILEES HEBREW FRAGMENTS
-----------------------------------------------------------------------
Sources: 4Q216–4Q224 and 11Q12 (15 manuscripts at Qumran).
  Hebrew text covers roughly 15–20% of the full book (scattered).
  DJD XIII (VanderKam & Milik, 1994) is the critical edition.
  Combined Hebrew word count: ~2,000–3,000 words.

Suggested date: 160 BCE  (generally accepted composition date)
Flag: noisy=False (prose narrative, re-writes of Genesis)
Note: Because of the fragmentary nature, the char n-gram model
  is preferred over word n-gram for Jubilees.  Flag min_words warning
  (below 3,000 = reliability threshold).

Benefit to pipeline: direct in-genre comparison with Genesis /
  JE_source at the same date grid position; tests whether the model
  can distinguish a ~160 BCE re-write of Genesis from the Genesis
  source itself.
=======================================================================
"""

if __name__ == '__main__':
    main()
    print(DATA_FORMAT_SPEC)
    # Also save the spec to a standalone file
    spec_path = WORKSPACE / 'data_intake_spec_new_texts.txt'
    spec_path.write_text(DATA_FORMAT_SPEC)
    print(f'  Data intake spec saved to: data_intake_spec_new_texts.txt')
