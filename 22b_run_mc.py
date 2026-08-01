"""
Script 22b — Monte Carlo interpolation sensitivity (uses cached verse features).

Loads caches written by 22a, runs 200 iterations of random passage removal,
saves results + plots.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
WORKSPACE = Path('/sessions/relaxed-modest-dirac/mnt/Diachronic Hebrew')

RNG_SEED      = 42
N_ITERATIONS  = 200
TIER_PROB     = {'LIKELY': 0.75, 'POSSIBLE': 0.30}

# ---------------------------------------------------------------------------
# Interpolation catalogue (same as Script 22)
# ---------------------------------------------------------------------------
INTERPOLATIONS = [
    ('Amos',     4, 13,  4, 13, 'LIKELY'),
    ('Amos',     5,  8,  5,  9, 'LIKELY'),
    ('Amos',     9,  5,  9,  6, 'LIKELY'),
    ('Amos',     9, 11,  9, 15, 'POSSIBLE'),
    ('Amos',     1,  9,  1, 12, 'POSSIBLE'),
    ('Amos',     2,  4,  2,  5, 'POSSIBLE'),
    ('Hosea',   14, 10, 14, 10, 'LIKELY'),
    ('Hosea',    1,  7,  1,  7, 'POSSIBLE'),
    ('Hosea',    2, 14,  2, 23, 'POSSIBLE'),
    ('Hosea',    3,  1,  3,  5, 'POSSIBLE'),
    ('Hosea',   11,  8, 11, 11, 'POSSIBLE'),
    ('Hosea',   14,  2, 14,  9, 'POSSIBLE'),
    ('Micah',    4,  1,  4,  5, 'LIKELY'),
    ('Micah',    2, 12,  2, 13, 'POSSIBLE'),
    ('Micah',    4,  9,  5, 15, 'POSSIBLE'),
    ('Micah',    7,  8,  7, 20, 'POSSIBLE'),
    ('Isaiah',  24,  1, 27, 13, 'LIKELY'),
    ('Isaiah',  34,  1, 35, 10, 'LIKELY'),
    ('Isaiah',  36,  1, 39,  8, 'LIKELY'),
    ('Isaiah',  12,  1, 12,  6, 'POSSIBLE'),
    ('Isaiah',  13,  1, 14, 27, 'POSSIBLE'),
    ('Isaiah',  16,  1, 16,  5, 'POSSIBLE'),
    ('Isaiah',  19, 16, 19, 25, 'POSSIBLE'),
    ('Isaiah',  23,  8, 23, 18, 'POSSIBLE'),
    ('Isaiah',  32,  9, 32, 20, 'POSSIBLE'),
    ('Nahum',    1,  2,  1, 10, 'LIKELY'),
    ('Habakkuk', 3,  1,  3, 19, 'LIKELY'),
    ('Habakkuk', 1,  5,  1, 11, 'POSSIBLE'),
    ('Zephaniah',3, 14,  3, 20, 'LIKELY'),
    ('Zephaniah',2,  3,  2,  3, 'POSSIBLE'),
    ('Zephaniah',2,  7,  2,  7, 'POSSIBLE'),
    ('Zephaniah',2,  8,  2, 11, 'POSSIBLE'),
    ('Zephaniah',3,  9,  3, 13, 'POSSIBLE'),
]

BOOK_TO_UNIT = {
    'Amos': 'Amos', 'Hosea': 'Hosea', 'Micah': 'Micah',
    'Isaiah': 'Isaiah_1', 'Nahum': 'Nahum',
    'Habakkuk': 'Habakkuk', 'Zephaniah': 'Zephaniah',
}
FLAGGED_UNITS = set(BOOK_TO_UNIT.values())

# Training unit order and dates (same as all other scripts)
UNIT_ORDER = [
    'Amos','Hosea','Micah','Nahum','Habakkuk','Zephaniah','Isaiah_1',
    'Isaiah_2','Isaiah_3','Jeremiah','Ezekiel','Haggai','Zechariah_1',
    'Malachi','Jonah','Lamentations','Ezra','Nehemiah','Chronicles',
    'Daniel','Ecclesiastes','Esther',
]
TRAIN_DATES = [760,740,720,620,600,620,720,540,450,590,580,520,518,
               450,400,580,350,350,350,167,250,350]
NOISY_UNITS = {'D_Song','Song_Sea','Song_Deborah'}

# MVN parameters
N_GRID      = 500
DATE_HI     = 1200
DATE_LO     = 50
PRIOR_MU    = 600.0
PRIOR_SIGMA = 350.0
RIDGE       = 0.20


# ---------------------------------------------------------------------------
# Pre-build passage lookup: for each unit, list of (key_set, tier)
# ---------------------------------------------------------------------------
def build_passage_sets():
    """Pre-compute the set of 'ch:vs' keys for each interpolation passage."""
    passages = {}   # {unit: [(frozen_set_of_keys, tier), ...]}
    for entry in INTERPOLATIONS:
        book, sch, svs, ech, evs, tier = entry
        unit = BOOK_TO_UNIT[book]
        keys = set()
        for ch in range(sch, ech + 1):
            v_s = svs if ch == sch else 1
            v_e = evs if ch == ech else 200
            for vs in range(v_s, v_e + 1):
                keys.add(f'{ch}:{vs}')
        passages.setdefault(unit, []).append((frozenset(keys), tier))
    return passages

PASSAGE_SETS = build_passage_sets()


# ---------------------------------------------------------------------------
# MVN model helpers  (fully vectorised — no Python loops over grid or features)
# ---------------------------------------------------------------------------
def build_mvn(rates_mat, dates, features):
    """Vectorised OLS over all features at once; returns (intercepts, slopes, Sigma_inv).

    intercepts, slopes: 1-D arrays aligned with `features`
    """
    x  = np.array(dates, dtype=float)          # (N,)
    N  = len(x)
    K  = len(features)
    # Build (K, N) rate matrix
    Y  = np.array([rates_mat[fn] for fn in features], dtype=float)
    np.nan_to_num(Y, copy=False)

    # Vectorised OLS: one pass for all K features
    xm  = x.mean()
    xd  = x - xm                              # (N,)
    xv  = float((xd ** 2).sum()) or 1.0
    ym  = Y.mean(axis=1)                      # (K,)
    slopes     = (Y * xd).sum(axis=1) / xv    # (K,)  (ym*xd sums to 0)
    intercepts = ym - slopes * xm             # (K,)

    # Residuals: (K, N) → transpose to (N, K)
    pred = intercepts[:, None] + slopes[:, None] * x[None, :]   # (K, N)
    R    = (Y - pred).T                        # (N, K)

    Sig  = R.T @ R / max(N - 2, 1)
    lam  = RIDGE * np.trace(Sig) / K if K > 0 else 0.0
    Sinv = np.linalg.inv(Sig + lam * np.eye(K))
    return intercepts, slopes, Sinv


def get_map(obs_rates, features, intercepts, slopes, Sinv, date_grid):
    """Vectorised MAP: no Python loop over grid points.

    pred: (N_GRID, K)  diff: (N_GRID, K)
    lp  = -0.5 * rowwise_quadratic(diff, Sinv)
    """
    obs  = np.array([obs_rates.get(fn, 0.0) for fn in features], dtype=float)  # (K,)
    # predictions for every grid point simultaneously
    pred = intercepts[None, :] + slopes[None, :] * date_grid[:, None]  # (N_GRID, K)
    diff = obs[None, :] - pred                                          # (N_GRID, K)
    A    = diff @ Sinv                                                  # (N_GRID, K)
    lp   = -0.5 * (A * diff).sum(axis=1)                               # (N_GRID,)
    lp  += -0.5 * ((date_grid - PRIOR_MU) / PRIOR_SIGMA) ** 2
    lp  -= lp.max()
    post = np.exp(lp); post /= post.sum()
    return float(date_grid[np.argmax(post)])


# ---------------------------------------------------------------------------
# Rate computation from verse cache
# ---------------------------------------------------------------------------
def verse_rates_cng(unit, removed_keys, cng_feats, verse_data):
    vd = verse_data[unit]
    total = {fn: 0 for fn in cng_feats}
    total_chars = 0
    for key, cnt in vd['cng'].items():
        if key in removed_keys: continue
        for fn, c in cnt.items():
            if fn in total: total[fn] += c
        total_chars += vd['chars'].get(key, 0)
    if total_chars == 0:
        return {fn: 0.0 for fn in cng_feats}
    return {fn: total[fn] / total_chars * 1000 for fn in cng_feats}


def verse_rates_wng(unit, removed_keys, wngA_feats, wngB_feats, verse_data):
    vd   = verse_data[unit]
    totA = {fn: 0 for fn in wngA_feats}
    totB = {fn: 0 for fn in wngB_feats}
    nw   = 0
    for key in vd['words']:
        if key in removed_keys: continue
        for fn, c in vd['wngA'].get(key, {}).items():
            if fn in totA: totA[fn] += c
        for fn, c in vd['wngB'].get(key, {}).items():
            if fn in totB: totB[fn] += c
        nw += vd['words'][key]
    if nw == 0:
        return ({fn: 0.0 for fn in wngA_feats}, {fn: 0.0 for fn in wngB_feats})
    rA = {fn: totA[fn] / nw * 1000 for fn in wngA_feats}
    rB = {fn: totB[fn] / nw * 1000 for fn in wngB_feats}
    return rA, rB


# ---------------------------------------------------------------------------
# Build training matrix helpers
# ---------------------------------------------------------------------------
def make_cng_matrix(rates_by_unit, cng_feats):
    return {fn: [rates_by_unit[u].get(fn, 0.0) for u in UNIT_ORDER] for fn in cng_feats}


def make_wng_matrix(ratesA_by_unit, ratesB_by_unit, wngA_feats, wngB_feats):
    mat = {fn: [ratesA_by_unit[u].get(fn, 0.0) for u in UNIT_ORDER] for fn in wngA_feats}
    mat.update({fn: [ratesB_by_unit[u].get(fn, 0.0) for u in UNIT_ORDER] for fn in wngB_feats})
    return mat


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
UNIT_DISPLAY_ORDER = [
    'Genesis','Exodus','Leviticus','Numbers','Deuteronomy',
    'D_source','P_source','JE_source','D_Code','D_Frame',
    'Lev_Holiness','Lev_Priestly',
    'Joshua','Judges','1_Samuel','2_Samuel','1_Kings','2_Kings',
    'Jer_DTR','Jer_oracle',
    'Song_Sea','Song_Deborah','D_Song',
]


def violin_plot(mc_maps, s0_maps, model_label, out_path):
    units = [u for u in UNIT_DISPLAY_ORDER if u in mc_maps and len(mc_maps[u]) > 1]
    data  = [mc_maps[u] for u in units]
    fig, ax = plt.subplots(figsize=(13, 9))
    vp = ax.violinplot(data, positions=range(len(units)),
                       vert=False, showmedians=True, showextrema=False)
    for body in vp['bodies']:
        body.set_alpha(0.55); body.set_facecolor('#4575b4')
    vp['cmedians'].set_color('#1a1a2e'); vp['cmedians'].set_linewidth(1.8)
    for yi, unit in enumerate(units):
        if unit in s0_maps and np.isfinite(s0_maps[unit]):
            ax.scatter(s0_maps[unit], yi, color='#d73027', s=45, zorder=5,
                       label='S0 baseline (no removal)' if yi == 0 else '')
        if unit in NOISY_UNITS:
            ax.text(DATE_LO - 100, yi, '⚠', va='center', fontsize=9, color='#888')
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels([u.replace('_',' ') for u in units], fontsize=8)
    ax.set_xlabel('MAP date (BCE)', fontsize=10)
    ax.set_xlim(DATE_HI + 150, DATE_LO - 150); ax.invert_xaxis()
    ax.set_title(
        f'Monte Carlo interpolation sensitivity — {model_label}\n'
        f'N={N_ITERATIONS} iterations | LIKELY passages: 75% removal prob | '
        f'POSSIBLE passages: 30% removal prob\n'
        f'Violin = date distribution; red dot = S0 baseline (full corpus)',
        fontsize=10, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.25, lw=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


def shift_dist_plot(mc_char, mc_word, s0_char, s0_word, out_path):
    KEY = ['Genesis','Exodus','Leviticus','Numbers','Deuteronomy',
           'P_source','JE_source','D_source',
           'Joshua','2_Kings','Jer_oracle','Song_Sea','Song_Deborah']
    cols = 4; rows = (len(KEY) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3))
    axes = axes.flatten()
    bins = np.linspace(-300, 300, 31)
    for ax, unit in zip(axes, KEY):
        cs = np.array(mc_char.get(unit, []))
        ws = np.array(mc_word.get(unit, []))
        cs = cs[np.isfinite(cs)] - s0_char.get(unit, np.nan)
        ws = ws[np.isfinite(ws)] - s0_word.get(unit, np.nan)
        cs = cs[np.isfinite(cs)]; ws = ws[np.isfinite(ws)]
        if len(cs): ax.hist(cs, bins=bins, alpha=0.6, color='#4dac26', label='Char', density=True)
        if len(ws): ax.hist(ws, bins=bins, alpha=0.6, color='#d01c8b', label='Word', density=True)
        ax.axvline(0, color='#555', lw=1, ls='--', alpha=0.6)
        ax.set_title(unit.replace('_',' '), fontsize=9, fontweight='bold')
        ax.set_xlabel('Δ MAP (years)', fontsize=7); ax.tick_params(labelsize=7)
        cstd = f'{np.std(cs):.0f}' if len(cs) else '—'
        wstd = f'{np.std(ws):.0f}' if len(ws) else '—'
        ax.text(0.03, 0.95, f'σc={cstd}\nσw={wstd}',
                transform=ax.transAxes, va='top', fontsize=7, color='#333')
        if unit == KEY[0]: ax.legend(fontsize=7)
    for ax in axes[len(KEY):]: ax.set_visible(False)
    fig.suptitle('Distribution of MAP shifts: MAP(iteration) − MAP(S0 baseline)\n'
                 'Centred on zero = robust; wide/shifted = sensitive to interpolation removal',
                 fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(RNG_SEED)
    date_grid = np.linspace(DATE_HI, DATE_LO, N_GRID)

    # ---- Load caches ----
    print('Loading caches...')
    def load(name):
        with open(WORKSPACE / f'mc_cache_{name}.json') as f:
            return json.load(f)

    verse_data   = load('verse_data')
    fixed_cng    = load('fixed_cng')
    fixed_wngA   = load('fixed_wngA')
    fixed_wngB   = load('fixed_wngB')
    test_cng     = load('test_cng')
    test_wngA    = load('test_wngA')
    test_wngB    = load('test_wngB')
    nwords       = load('nwords')

    # ---- Feature lists ----
    cng_feats  = pd.read_csv(WORKSPACE / 'ngram_selected_features.csv')['ngram'].tolist()
    wngA_feats = pd.read_csv(WORKSPACE / 'word_ngram_typeA_features.csv')['ngram'].tolist()
    wngB_feats = pd.read_csv(WORKSPACE / 'word_ngram_typeB_features.csv')['ngram'].tolist()
    wngAB_feats = wngA_feats + wngB_feats

    all_test_units = list(test_cng.keys())

    # ---- S0 baseline (no removal) ----
    print('Computing S0 baseline...')
    s0_cng_rates  = dict(fixed_cng)
    s0_wngA_rates = dict(fixed_wngA)
    s0_wngB_rates = dict(fixed_wngB)

    # add flagged units at full rates (no removal)
    for unit in FLAGGED_UNITS:
        s0_cng_rates[unit]  = verse_rates_cng(unit, set(), cng_feats, verse_data)
        rA, rB = verse_rates_wng(unit, set(), wngA_feats, wngB_feats, verse_data)
        s0_wngA_rates[unit] = rA
        s0_wngB_rates[unit] = rB

    ic_c0, sl_c0, sinv_c0 = build_mvn(make_cng_matrix(s0_cng_rates, cng_feats),
                                       TRAIN_DATES, cng_feats)
    ic_w0, sl_w0, sinv_w0 = build_mvn(make_wng_matrix(s0_wngA_rates, s0_wngB_rates,
                                                        wngA_feats, wngB_feats),
                                       TRAIN_DATES, wngAB_feats)

    s0_char = {u: get_map(test_cng[u], cng_feats, ic_c0, sl_c0, sinv_c0, date_grid)
               for u in all_test_units}
    s0_word = {u: get_map({**{fn: test_wngA[u].get(fn,0.) for fn in wngA_feats},
                           **{fn: test_wngB[u].get(fn,0.) for fn in wngB_feats}},
                           wngAB_feats, ic_w0, sl_w0, sinv_w0, date_grid)
               for u in all_test_units}

    print(f'  S0 sample: Genesis char={s0_char["Genesis"]:.0f} word={s0_word["Genesis"]:.0f}')

    # ---- Monte Carlo iterations ----
    print(f'\nRunning {N_ITERATIONS} Monte Carlo iterations...')
    mc_char = {u: [] for u in all_test_units}
    mc_word = {u: [] for u in all_test_units}

    for it in range(N_ITERATIONS):
        if (it+1) % 50 == 0:
            print(f'  Iteration {it+1}/{N_ITERATIONS}')

        # Draw removals
        iter_cng  = dict(fixed_cng)
        iter_wngA = dict(fixed_wngA)
        iter_wngB = dict(fixed_wngB)

        for unit in FLAGGED_UNITS:
            removed = set()
            for key_set, tier in PASSAGE_SETS.get(unit, []):
                if np.random.random() < TIER_PROB[tier]:
                    removed |= key_set
            iter_cng[unit]  = verse_rates_cng(unit, removed, cng_feats, verse_data)
            rA, rB = verse_rates_wng(unit, removed, wngA_feats, wngB_feats, verse_data)
            iter_wngA[unit] = rA; iter_wngB[unit] = rB

        ic_c, sl_c, sinv_c = build_mvn(make_cng_matrix(iter_cng, cng_feats),
                                        TRAIN_DATES, cng_feats)
        ic_w, sl_w, sinv_w = build_mvn(make_wng_matrix(iter_wngA, iter_wngB,
                                                         wngA_feats, wngB_feats),
                                        TRAIN_DATES, wngAB_feats)

        for u in all_test_units:
            mc_char[u].append(get_map(test_cng[u], cng_feats, ic_c, sl_c, sinv_c, date_grid))
            obs_w = {fn: test_wngA[u].get(fn,0.) for fn in wngA_feats}
            obs_w.update({fn: test_wngB[u].get(fn,0.) for fn in wngB_feats})
            mc_word[u].append(get_map(obs_w, wngAB_feats, ic_w, sl_w, sinv_w, date_grid))

    # ---- Results table ----
    print('\nAssembling results...')
    def q(arr, p): return round(float(np.percentile([x for x in arr if np.isfinite(x)], p)), 1)
    def sd(arr):   return round(float(np.std([x for x in arr if np.isfinite(x)])), 1)

    rows = []
    for u in all_test_units:
        cm = mc_char[u]; wm = mc_word[u]
        row = dict(unit=u, n_words=nwords.get(u,0), noisy=u in NOISY_UNITS,
                   map_char_S0=round(s0_char[u],1), map_word_S0=round(s0_word[u],1),
                   mc_char_median=q(cm,50), mc_char_std=sd(cm),
                   mc_char_p16=q(cm,16),   mc_char_p84=q(cm,84),
                   mc_char_p2p5=q(cm,2.5), mc_char_p97p5=q(cm,97.5),
                   mc_word_median=q(wm,50), mc_word_std=sd(wm),
                   mc_word_p16=q(wm,16),   mc_word_p84=q(wm,84),
                   mc_word_p2p5=q(wm,2.5), mc_word_p97p5=q(wm,97.5),
                   shift_char=round(q(cm,50)-s0_char[u],1),
                   shift_word=round(q(wm,50)-s0_word[u],1))
        rows.append(row)
        print(f"  {u:<18} char: S0={row['map_char_S0']:.0f} med={row['mc_char_median']} "
              f"σ={row['mc_char_std']}  "
              f"word: S0={row['map_word_S0']:.0f} med={row['mc_word_median']} "
              f"σ={row['mc_word_std']}")

    pd.DataFrame(rows).to_csv(WORKSPACE / 'mc_sensitivity_summary.csv', index=False)
    print('\n  Saved: mc_sensitivity_summary.csv')

    # ---- Plots ----
    print('\nGenerating plots...')
    violin_plot(mc_char, s0_char, 'Character n-gram model',
                WORKSPACE / 'mc_sensitivity_violin_char.png')
    violin_plot(mc_word, s0_word, 'Word n-gram model (A+B combined)',
                WORKSPACE / 'mc_sensitivity_violin_word.png')
    shift_dist_plot(mc_char, mc_word, s0_char, s0_word,
                    WORKSPACE / 'mc_sensitivity_shift_dist.png')

    print('\nDone.')


if __name__ == '__main__':
    main()
