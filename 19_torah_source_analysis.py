"""
Script 19 — Torah Source Layer and Compilation Date Analysis
============================================================
Applies chapter-level dating across all five Torah books using the character
n-gram model (script 16) and word n-gram model (script 17), then interprets
the resulting age profiles against known source-critical strata.

Three signals are combined to characterise each chapter:

  char_MAP   — character n-gram MAP date: sensitive to orthographic profile
               (plene spelling, consonantal distribution, morphological shape)
  word_MAP   — word n-gram A+B MAP date: sensitive to syntactic collocations
               and function-word sequences; more easily archaized
  arc_idx    — archaism index = word_MAP − char_MAP
               > 0  : word patterns older than orthographic profile
                      → either archaizing syntax OR embedded old source
                      whose orthography was updated during transmission
               ≈ 0  : methods agree → consistent dating signal
               < 0  : orthographic profile older than word patterns
                      → archaic orthography preserved with less archaic syntax

Classification heuristic (per chapter):
  OLD_SOURCE   : avg_date > 650 BCE and −200 ≤ arc_idx ≤ +300
  ARCHAIZED    : arc_idx > +300  (syntax far older than orthography)
  EDITORIAL    : avg_date < 425 BCE on BOTH models, |arc_idx| < 250
  UNCERTAIN    : everything else, or n_words < 200 (noisy)

Compilation date estimation:
  For each book, the three latest-dating datable chapters on BOTH models
  are identified as candidate editorial/compilation sections; their average
  (char_MAP + word_MAP)/2 date provides a terminus post quem estimate for
  when the book reached roughly its current compiled form.

Models are reconstructed from the saved training-rate CSV files produced by
scripts 16 and 17.  No BHSA re-extraction of the training corpus is needed.

Outputs
-------
  torah_chapter_dates.csv          — per-chapter dates + archaism index
  torah_compilation_report.csv     — per-book compilation summary
  torah_age_profiles.png           — 5-row age-profile plot per book
  torah_archaism_map.png           — heatmap of archaism index per book
  torah_source_comparison.png      — source-layer dates across models
  torah_compilation_summary.png    — compilation TPQ bar chart per book
"""

import argparse
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')

WORKSPACE = Path('/sessions/relaxed-modest-dirac/mnt/Diachronic Hebrew')

# ---------------------------------------------------------------------------
# Model parameters (must match scripts 16 / 17)
# ---------------------------------------------------------------------------
RIDGE_CHAR  = 0.20
RIDGE_WORD  = 0.20
N_GRID      = 500
DATE_HI     = 1200
DATE_LO     =   50
PRIOR_MU    = 600.0
PRIOR_SIGMA = 350.0

NGRAM_SIZES_CHAR = [3, 4]
NGRAM_SIZES_WORD = [2, 3]
WORD_SEP         = '_'
FUNCTION_POS     = frozenset({'prep', 'conj', 'art', 'nega', 'prps', 'prde', 'inrg'})

MIN_WORDS_FOR_DATING = 150   # chapters shorter than this are flagged noisy

# ---------------------------------------------------------------------------
# Torah books and their chapter counts
# ---------------------------------------------------------------------------
TORAH_BOOKS = {
    'Genesis':     50,
    'Exodus':      40,
    'Leviticus':   27,
    'Numbers':     36,
    'Deuteronomy': 34,
}

# ---------------------------------------------------------------------------
# Simplified source-critical labels per chapter (mainstream consensus)
# P  = Priestly source
# H  = Holiness Code (within P tradition, older stratum)
# J  = Jahwist source
# E  = Elohist source
# JE = mixed / intertwined JE material
# D  = Deuteronomic/ist
# ?  = uncertain / unique
# ---------------------------------------------------------------------------
SOURCE_LABELS = {
    'Genesis': {
        1: 'P', 2: 'J', 3: 'J', 4: 'J', 5: 'P',
        6: 'JE+P', 7: 'JE+P', 8: 'JE+P', 9: 'JE+P',
        10: 'JE+P', 11: 'JE+P', 12: 'J', 13: 'J', 14: '?',
        15: 'JE', 16: 'JE+P', 17: 'P', 18: 'J', 19: 'J',
        20: 'E', 21: 'JE+P', 22: 'E', 23: 'P', 24: 'J',
        25: 'JE+P', 26: 'J', 27: 'J', 28: 'JE+P', 29: 'JE',
        30: 'JE', 31: 'JE', 32: 'JE', 33: 'JE', 34: 'JE+P',
        35: 'JE+P', 36: 'P', 37: 'JE+P', 38: 'J', 39: 'J',
        40: 'JE', 41: 'JE', 42: 'JE', 43: 'J', 44: 'J',
        45: 'J', 46: 'JE+P', 47: 'JE', 48: 'JE', 49: 'Poem',
        50: 'JE+P',
    },
    'Exodus': {
        1: 'JE+P', 2: 'JE', 3: 'JE', 4: 'JE', 5: 'JE',
        6: 'P', 7: 'JE+P', 8: 'JE+P', 9: 'JE+P', 10: 'JE',
        11: 'JE', 12: 'JE+P', 13: 'JE', 14: 'JE+P', 15: 'Poem',
        16: 'P', 17: 'JE', 18: 'JE', 19: 'JE+P', 20: 'JE',
        21: 'JE', 22: 'JE', 23: 'JE', 24: 'JE+P',
        25: 'P', 26: 'P', 27: 'P', 28: 'P', 29: 'P',
        30: 'P', 31: 'P', 32: 'JE', 33: 'JE', 34: 'JE',
        35: 'P', 36: 'P', 37: 'P', 38: 'P', 39: 'P', 40: 'P',
    },
    'Leviticus': {
        **{ch: 'P'  for ch in range(1, 17)},   # 1–16: Priestly rituals
        **{ch: 'H'  for ch in range(17, 27)},   # 17–26: Holiness Code
        27: 'P',                                 # 27: appendix
    },
    'Numbers': {
        **{ch: 'P'  for ch in range(1, 11)},    # 1–10: Priestly census/camp
        11: 'JE', 12: 'JE', 13: 'JE+P', 14: 'JE+P',
        15: 'P', 16: 'JE+P', 17: 'P', 18: 'P', 19: 'P',
        20: 'JE+P', 21: 'JE', 22: 'JE', 23: 'JE', 24: 'JE',
        25: 'JE+P', 26: 'P',
        **{ch: 'P'  for ch in range(27, 37)},   # 27–36: Priestly laws/land
    },
    'Deuteronomy': {
        **{ch: 'D-frame' for ch in [1, 2, 3, 4]},
        **{ch: 'D-code'  for ch in range(5, 27)},
        **{ch: 'D-frame' for ch in [27, 28, 29, 30, 31]},
        32: 'Poem',
        33: 'Poem',
        34: 'D-frame',
    },
}

# Colors for source labels in plots
SOURCE_COLORS = {
    'P':      '#4575b4',   # blue
    'H':      '#74add1',   # light blue
    'J':      '#d73027',   # red
    'E':      '#fc8d59',   # salmon
    'JE':     '#f46d43',   # orange-red
    'JE+P':   '#a50026',   # dark red
    'D-code': '#4dac26',   # green
    'D-frame':'#b8e186',   # light green
    'Poem':   '#984ea3',   # purple
    '?':      '#999999',   # grey
}

# ---------------------------------------------------------------------------
# Classification thresholds
# ---------------------------------------------------------------------------
OLD_SOURCE_MIN_DATE  = 650   # avg_MAP older than this → candidate old source
ARCHAISM_THRESHOLD   = 300   # arc_idx > this → archaized
EDITORIAL_MAX_DATE   = 425   # avg_MAP younger than this on both models → editorial
ARCHAISM_RANGE_LO    = -200  # for OLD_SOURCE classification
ARCHAISM_RANGE_HI    = +300


def classify_chapter(map_char, map_word, arc_idx, n_words):
    if n_words < MIN_WORDS_FOR_DATING:
        return 'INSUFFICIENT'
    if np.isnan(map_char) or np.isnan(map_word):
        return 'INSUFFICIENT'
    avg = (map_char + map_word) / 2.0
    if arc_idx > ARCHAISM_THRESHOLD:
        return 'ARCHAIZED'
    if avg > OLD_SOURCE_MIN_DATE and ARCHAISM_RANGE_LO <= arc_idx <= ARCHAISM_RANGE_HI:
        return 'OLD_SOURCE'
    if map_char < EDITORIAL_MAX_DATE and map_word < EDITORIAL_MAX_DATE and abs(arc_idx) < 250:
        return 'EDITORIAL'
    return 'MIXED'


# ===========================================================================
# TEXT EXTRACTION
# ===========================================================================

def extract_cons_text(book, ch, F, L, T):
    bn = T.nodeFromSection((book,))
    if bn is None:
        return '', 0
    tokens = []
    n_words = 0
    for ch_node in L.d(bn, 'chapter'):
        if int(F.chapter.v(ch_node)) != ch:
            continue
        for word in L.d(ch_node, 'word'):
            cons = F.g_cons_utf8.v(word)
            if not cons:
                continue
            tokens.append(cons)
            n_words += 1
        break
    if not tokens:
        return '', 0
    text = WORD_SEP + WORD_SEP.join(tokens) + WORD_SEP
    return text, n_words


def extract_word_seqs(book, ch, F, L, T):
    bn = T.nodeFromSection((book,))
    if bn is None:
        return [], [], 0
    pos_seq = []
    fw_seq  = []
    n_words = 0
    for ch_node in L.d(bn, 'chapter'):
        if int(F.chapter.v(ch_node)) != ch:
            continue
        for word in L.d(ch_node, 'word'):
            sp = F.sp.v(word)
            if not sp:
                continue
            vt = F.vt.v(word) if sp == 'verb' else None
            pos_seq.append(f'verb_{vt or "unkn"}' if sp == 'verb' else sp)
            fw_seq.append(F.lex.v(word) or sp if sp in FUNCTION_POS else None)
            n_words += 1
        break
    return pos_seq, fw_seq, n_words


def compute_char_freqs(text, sizes=NGRAM_SIZES_CHAR):
    if not text:
        return {}
    total = len(text)
    counts = Counter()
    for n in sizes:
        for i in range(len(text) - n + 1):
            counts[text[i:i+n]] += 1
    return {ng: cnt / total * 1000 for ng, cnt in counts.items()}


def build_word_ngrams(tokens, sizes=NGRAM_SIZES_WORD, skip_null=False):
    counts = Counter()
    for n in sizes:
        for i in range(len(tokens) - n + 1):
            gram = tokens[i:i+n]
            if skip_null and any(t is None for t in gram):
                continue
            key = '·'.join(str(t) for t in gram)
            counts[key] += 1
    return dict(counts)


def counts_to_rates(counts, n_words, scale=1000.0):
    if n_words == 0:
        return {}
    return {ng: cnt / n_words * scale for ng, cnt in counts.items()}


# ===========================================================================
# MVN MODEL (vectorized posterior)
# ===========================================================================

def build_mvn_model(rates_df, dates_bce, feature_names, ridge_frac=0.20):
    ols_params    = {}
    residual_rows = []
    for fn in feature_names:
        y     = rates_df[fn].values.astype(float)
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
    lam       = ridge_frac * np.trace(Sigma) / K
    Sigma_r   = Sigma + lam * np.eye(K)
    Sigma_inv = np.linalg.inv(Sigma_r)
    return ols_params, Sigma_r, Sigma_inv


def compute_posterior_fast(obs_vec, ols_params, Sigma_inv, feature_names, date_grid):
    """
    Vectorized posterior computation.  All date_grid points are evaluated
    simultaneously using numpy matrix operations — ~100× faster than the
    loop-based version for 500-point grids.
    """
    intercepts = np.array([ols_params[fn][0] for fn in feature_names])
    slopes     = np.array([ols_params[fn][1] for fn in feature_names])

    # pred: (N_GRID, K)
    pred = intercepts[np.newaxis, :] + slopes[np.newaxis, :] * date_grid[:, np.newaxis]
    diff = obs_vec[np.newaxis, :] - pred    # (N_GRID, K)

    valid = np.isfinite(diff[0])
    if valid.sum() < 2:
        post = np.ones(len(date_grid)) / len(date_grid)
        return post

    diff_v  = diff[:, valid]                # (N_GRID, K_valid)
    Si_v    = Sigma_inv[np.ix_(valid, valid)]  # (K_valid, K_valid)
    log_ll  = -0.5 * np.sum((diff_v @ Si_v) * diff_v, axis=1)

    log_prior = -0.5 * ((date_grid - PRIOR_MU) / PRIOR_SIGMA) ** 2
    log_post  = log_ll + log_prior
    log_post -= log_post.max()
    post  = np.exp(log_post)
    post /= post.sum()
    return post


def map_and_ci(posterior, date_grid, ci_level=0.68):
    map_date = date_grid[np.argmax(posterior)]
    cdf      = np.cumsum(posterior)
    lo_frac  = (1 - ci_level) / 2
    ci_lo    = date_grid[np.searchsorted(cdf, lo_frac)]
    ci_hi    = date_grid[np.searchsorted(cdf, 1 - lo_frac)]
    return float(map_date), float(ci_lo), float(ci_hi)


# ===========================================================================
# LOAD MODELS FROM CSV
# ===========================================================================

def load_char_model(workspace):
    rates_df = pd.read_csv(workspace / 'ngram_training_rates.csv', index_col=0)
    sel_df   = pd.read_csv(workspace / 'ngram_selected_features.csv')
    sel_ng   = sel_df['ngram'].tolist()
    dates    = rates_df['date_bce'].tolist()
    sel_ng   = [f for f in sel_ng if f in rates_df.columns]
    ols, Sigma, Si = build_mvn_model(rates_df[sel_ng], dates, sel_ng, RIDGE_CHAR)
    print(f'  Char model: {len(sel_ng)} features, cond={np.linalg.cond(Sigma):.0f}')
    return ols, Si, sel_ng


def load_word_model(workspace):
    rates_df = pd.read_csv(workspace / 'word_ngram_training_rates.csv', index_col=0)
    selA_df  = pd.read_csv(workspace / 'word_ngram_typeA_features.csv')
    selB_df  = pd.read_csv(workspace / 'word_ngram_typeB_features.csv')
    sel_A    = [f for f in selA_df['ngram'].tolist() if f in rates_df.columns]
    sel_B    = [f for f in selB_df['ngram'].tolist() if f in rates_df.columns]
    sel_AB   = sel_A + sel_B
    dates    = rates_df['date_bce'].tolist()
    ols_AB, Sigma_AB, Si_AB = build_mvn_model(
        rates_df[sel_AB], dates, sel_AB, RIDGE_WORD)
    print(f'  Word A+B model: {len(sel_AB)} features, '
          f'cond={np.linalg.cond(Sigma_AB):.0f}')
    return ols_AB, Si_AB, sel_A, sel_B, sel_AB


# ===========================================================================
# DATE A SINGLE CHAPTER
# ===========================================================================

def date_chapter(book, ch, F, L, T, date_grid,
                 ols_ng, Si_ng, sel_ng,
                 ols_AB, Si_AB, sel_A, sel_B, sel_AB):
    """
    Returns (map_char, ci68_lo_char, ci68_hi_char,
             map_word, ci68_lo_word, ci68_hi_word,
             n_words, post_char, post_word)
    """
    text, nw  = extract_cons_text(book, ch, F, L, T)
    pos_seq, fw_seq, _ = extract_word_seqs(book, ch, F, L, T)

    if nw == 0:
        nan6 = (np.nan,) * 6
        return nan6[0], nan6[1], nan6[2], nan6[3], nan6[4], nan6[5], 0, None, None

    # Char n-gram
    ng_freqs = compute_char_freqs(text)
    obs_ng   = np.array([ng_freqs.get(f, 0.0) for f in sel_ng], dtype=float)
    post_ng  = compute_posterior_fast(obs_ng, ols_ng, Si_ng, sel_ng, date_grid)
    map_ng, lo_ng, hi_ng = map_and_ci(post_ng, date_grid)

    # Word n-gram A+B
    cnt_A  = build_word_ngrams(pos_seq, NGRAM_SIZES_WORD, skip_null=False)
    cnt_B  = build_word_ngrams(fw_seq,  NGRAM_SIZES_WORD, skip_null=True)
    ra     = counts_to_rates(cnt_A, nw)
    rb     = counts_to_rates(cnt_B, nw)
    obs_AB = np.concatenate([
        np.array([ra.get(f, 0.0) for f in sel_A], dtype=float),
        np.array([rb.get(f, 0.0) for f in sel_B], dtype=float),
    ])
    post_AB = compute_posterior_fast(obs_AB, ols_AB, Si_AB, sel_AB, date_grid)
    map_AB, lo_AB, hi_AB = map_and_ci(post_AB, date_grid)

    return (map_ng, lo_ng, hi_ng,
            map_AB, lo_AB, hi_AB,
            nw, post_ng, post_AB)


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_age_profiles(chapter_data, out_path):
    """
    5-row figure (one per Torah book) showing chapter-level age profiles.
    Blue = char n-gram MAP; Orange = word n-gram MAP.
    Background stripes colored by source assignment.
    """
    books = list(TORAH_BOOKS.keys())
    fig, axes = plt.subplots(5, 1, figsize=(18, 20))

    for ax, book in zip(axes, books):
        df = chapter_data[chapter_data['book'] == book].copy()
        df = df.sort_values('chapter')
        chapters = df['chapter'].values
        map_char = df['map_char'].values
        map_word = df['map_word'].values
        noisy    = df['n_words'].values < MIN_WORDS_FOR_DATING

        # background coloring by source
        for _, row in df.iterrows():
            src   = SOURCE_LABELS.get(book, {}).get(int(row['chapter']), '?')
            color = SOURCE_COLORS.get(src, '#eeeeee')
            ax.axvspan(row['chapter'] - 0.5, row['chapter'] + 0.5,
                       alpha=0.15, color=color, linewidth=0)

        # CI shading for char
        ci68_lo_c = df['ci68_lo_char'].values
        ci68_hi_c = df['ci68_hi_char'].values
        ci68_lo_w = df['ci68_lo_word'].values
        ci68_hi_w = df['ci68_hi_word'].values

        valid_c = ~np.isnan(map_char) & ~noisy
        valid_w = ~np.isnan(map_word) & ~noisy

        if valid_c.any():
            ax.fill_between(chapters[valid_c], ci68_lo_c[valid_c], ci68_hi_c[valid_c],
                            alpha=0.15, color='#2166ac')
            ax.plot(chapters[valid_c], map_char[valid_c],
                    'o-', color='#2166ac', lw=1.8, ms=5, label='char n-gram MAP')
        if valid_w.any():
            ax.fill_between(chapters[valid_w], ci68_lo_w[valid_w], ci68_hi_w[valid_w],
                            alpha=0.15, color='#d73027')
            ax.plot(chapters[valid_w], map_word[valid_w],
                    's--', color='#d73027', lw=1.8, ms=5, label='word n-gram MAP')

        # noisy chapters as open markers
        if noisy.any():
            ax.scatter(chapters[noisy & ~np.isnan(map_char)],
                       map_char[noisy & ~np.isnan(map_char)],
                       marker='o', facecolors='none', edgecolors='#2166ac',
                       s=40, linewidths=0.8, zorder=3)
            ax.scatter(chapters[noisy & ~np.isnan(map_word)],
                       map_word[noisy & ~np.isnan(map_word)],
                       marker='s', facecolors='none', edgecolors='#d73027',
                       s=40, linewidths=0.8, zorder=3)

        # prior mean reference line
        ax.axhline(PRIOR_MU, color='#888888', lw=0.6, ls=':', alpha=0.5)

        ax.set_ylabel('MAP date (BCE)', fontsize=9)
        ax.set_xlim(0.5, TORAH_BOOKS[book] + 0.5)
        ax.set_ylim(DATE_LO - 50, DATE_HI + 50)
        ax.invert_yaxis()   # older dates at top
        ax.set_title(book, fontsize=11, fontweight='bold', loc='left')
        ax.tick_params(labelsize=8)

        # x-tick every 5 chapters
        xticks = np.arange(5, TORAH_BOOKS[book] + 1, 5)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], fontsize=7)

        if book == books[-1]:
            ax.set_xlabel('Chapter', fontsize=9)
        if ax == axes[0]:
            ax.legend(fontsize=8, loc='upper right')

        # collect source labels present in this book for legend
        src_drawn = sorted({SOURCE_LABELS.get(book, {}).get(ch, '?')
                            for ch in range(1, TORAH_BOOKS[book] + 1)})
        src_handles = [mpatches.Patch(facecolor=SOURCE_COLORS.get(s, '#cccccc'),
                                       alpha=0.5, label=s, linewidth=0)
                       for s in src_drawn]
        ax.legend(handles=src_handles + [
            plt.Line2D([0],[0], color='#2166ac', lw=1.8, marker='o', ms=5, label='char MAP'),
            plt.Line2D([0],[0], color='#d73027', lw=1.8, marker='s', ms=5, ls='--', label='word MAP'),
        ], fontsize=7, loc='lower right', ncol=4, framealpha=0.9)

    fig.suptitle('Torah chapter-level age profiles: character n-gram (blue) vs. word n-gram (red)\n'
                 'Background shading = source label (scholarly consensus)\n'
                 'Older dates plotted higher; open markers = chapters with < 150 words',
                 fontsize=11, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


def plot_archaism_map(chapter_data, out_path):
    """
    Heatmap of archaism index per chapter for each Torah book.
    Diverging colormap: orange = archaized (word older), blue = genuine old orthography.
    """
    books = list(TORAH_BOOKS.keys())
    max_chs = max(TORAH_BOOKS.values())

    fig, axes = plt.subplots(5, 1, figsize=(18, 10))

    for ax, book in zip(axes, books):
        df  = chapter_data[(chapter_data['book'] == book) &
                           (chapter_data['n_words'] >= MIN_WORDS_FOR_DATING)].copy()
        arc = np.full(TORAH_BOOKS[book], np.nan)
        for _, row in df.iterrows():
            ch = int(row['chapter']) - 1
            arc[ch] = row['archaism_index']

        im = ax.imshow(arc.reshape(1, -1), cmap='RdBu_r',
                       vmin=-400, vmax=400,
                       aspect='auto',
                       extent=[0.5, TORAH_BOOKS[book] + 0.5, 0, 1])
        ax.set_xlim(0.5, max_chs + 0.5)
        ax.set_yticks([])
        ax.set_ylabel(book, fontsize=9, rotation=0, labelpad=55)
        xticks = np.arange(5, TORAH_BOOKS[book] + 1, 5)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], fontsize=7)

        # source boundary markers
        src_labels = SOURCE_LABELS.get(book, {})
        prev_src = None
        for ch in range(1, TORAH_BOOKS[book] + 2):
            cur_src = src_labels.get(ch, '?')
            if cur_src != prev_src and prev_src is not None:
                ax.axvline(ch - 0.5, color='white', lw=0.8, alpha=0.7)
            prev_src = cur_src

        if ax == axes[-1]:
            ax.set_xlabel('Chapter', fontsize=9)

    # Single colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                                norm=plt.Normalize(vmin=-400, vmax=400))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Archaism index\n(word MAP − char MAP)\nOrange = archaized',
                   fontsize=9)

    fig.suptitle('Archaism index per Torah chapter\n'
                 'Orange = word model dates older (archaized syntax or old embedded source)\n'
                 'Blue = char model dates older (archaic orthography preserved)\n'
                 'White = methods agree',
                 fontsize=10, fontweight='bold')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


def plot_source_comparison(out_path, workspace):
    """
    Grouped bar chart: MAP dates for all source layers under linguistic
    (full model), char n-gram, and word n-gram models.
    """
    # Load results
    master_df  = pd.read_csv(workspace / 'master_dating_results.csv')
    master_df['unit_key'] = master_df['unit'].replace(
        {'D': 'D_source', 'P': 'P_source', 'JE': 'JE_source'})
    mdict = {r['unit_key']: r for _, r in master_df.iterrows()}

    wng_df  = pd.read_csv(workspace / 'word_ngram_dating_results.csv')
    wng_df['unit_key'] = wng_df['unit'].replace(
        {'D': 'D_source', 'P': 'P_source', 'JE': 'JE_source'})
    wdict = {r['unit_key']: r for _, r in wng_df.iterrows()}

    arc_df = pd.read_csv(workspace / 'archaism_diagnostic_results.csv')
    adict  = {r['unit']: r for _, r in arc_df.iterrows()}

    layers = ['P_source', 'JE_source', 'D_source', 'D_Code', 'D_Frame', 'D_Song',
              'Lev_Priestly', 'Lev_Holiness']
    labels = ['P source', 'JE source', 'D source', 'D Code', 'D Frame', 'D Song',
              'Lev P', 'Lev H']

    full_maps = [float(mdict.get(u, {}).get('map_full', np.nan)) for u in layers]
    ng_maps   = [float(mdict.get(u, {}).get('map_ngram', np.nan)) for u in layers]
    wng_maps  = [float(wdict.get(u, {}).get('map_AB', np.nan))    for u in layers]

    x   = np.arange(len(layers))
    w   = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, full_maps, w, label='Full linguistic model', color='#4d9221', alpha=0.85)
    ax.bar(x,     ng_maps,   w, label='Char n-gram',           color='#2166ac', alpha=0.85)
    ax.bar(x + w, wng_maps,  w, label='Word n-gram A+B',       color='#d73027', alpha=0.85)

    # Reference lines
    for date, label_txt, ls in [(760, 'Amos (760)', ':'), (550, '550 BCE', '--'),
                                  (450, '450 BCE', '-.')]:
        ax.axhline(date, color='#888888', lw=0.8, ls=ls, alpha=0.5,
                   label=f'ref: {date} BCE')

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('MAP date (BCE)', fontsize=10)
    ax.set_title('Source layer dating: three models compared\n'
                 'Higher = older date',
                 fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


def plot_compilation_summary(compilation_report, out_path):
    """
    Per-book bar showing: oldest source cluster / main composition / compilation TPQ.
    """
    books = list(compilation_report.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(books))
    bar_h = 0.25

    oldest_dates = [compilation_report[b]['oldest_source_date']  for b in books]
    main_dates   = [compilation_report[b]['main_composition_date'] for b in books]
    tpq_dates    = [compilation_report[b]['compilation_tpq']      for b in books]

    ax.barh(y + bar_h, oldest_dates, bar_h, label='Oldest source layer (10th pct)', color='#1a9850', alpha=0.85)
    ax.barh(y,          main_dates,  bar_h, label='Main composition (median)', color='#4575b4', alpha=0.85)
    ax.barh(y - bar_h,  tpq_dates,  bar_h, label='Compilation TPQ (90th pct)', color='#d73027', alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(books, fontsize=10)
    ax.set_xlabel('Date (BCE)', fontsize=10)
    ax.set_title('Torah stratigraphy: source layers and estimated compilation\n'
                 'Char + word n-gram models combined; based on chapter-level dates',
                 fontsize=11, fontweight='bold')
    ax.invert_xaxis()   # older dates on left

    # Reference lines
    for date, lbl in [(600, '600 BCE'), (450, '450 BCE'), (350, '350 BCE')]:
        ax.axvline(date, color='#aaaaaa', lw=0.8, ls='--', alpha=0.6)
        ax.text(date, len(books) - 0.2, lbl, fontsize=7, color='#666666', ha='center')

    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=str(WORKSPACE))
    args   = parser.parse_args()
    outdir = Path(args.outdir)

    # -----------------------------------------------------------------------
    # STEP 1 — Load BHSA
    # -----------------------------------------------------------------------
    print('=' * 70)
    print('STEP 1 — Load BHSA')
    print('=' * 70)
    import sys
    sys.path.insert(0, '/sessions/relaxed-modest-dirac/text-fabric-data')
    from tf.app import use as tf_use
    A = tf_use('ETCBC/bhsa', hoist=globals(), checkout='local', silent='deep')
    print('  BHSA loaded.')

    # -----------------------------------------------------------------------
    # STEP 2 — Reconstruct models from saved CSV files
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 2 — Reconstruct models')
    print('=' * 70)
    ols_ng, Si_ng, sel_ng               = load_char_model(WORKSPACE)
    ols_AB, Si_AB, sel_A, sel_B, sel_AB = load_word_model(WORKSPACE)
    date_grid = np.linspace(DATE_LO, DATE_HI, N_GRID)

    # -----------------------------------------------------------------------
    # STEP 3 — Date every Torah chapter
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 3 — Date Torah chapters')
    print('=' * 70)

    rows = []
    total_chapters = sum(TORAH_BOOKS.values())
    done = 0

    for book, n_chs in TORAH_BOOKS.items():
        print(f'\n  {book} ({n_chs} chapters):')
        print(f'  {"ch":>3}  {"words":>6}  {"char":>7}  '
              f'{"word":>7}  {"arc":>6}  {"class":<14}  src')
        print('  ' + '-' * 55)

        for ch in range(1, n_chs + 1):
            (map_ng, lo_ng, hi_ng,
             map_AB, lo_AB, hi_AB,
             nw, _, _) = date_chapter(
                book, ch, F, L, T, date_grid,
                ols_ng, Si_ng, sel_ng,
                ols_AB, Si_AB, sel_A, sel_B, sel_AB)

            arc_idx = (map_AB - map_ng) if not (np.isnan(map_AB) or np.isnan(map_ng)) \
                      else np.nan
            cls     = classify_chapter(map_ng, map_AB, arc_idx if not np.isnan(arc_idx) else 0, nw)
            src     = SOURCE_LABELS.get(book, {}).get(ch, '?')
            avg_date = (map_ng + map_AB) / 2.0 if not (np.isnan(map_ng) or np.isnan(map_AB)) \
                       else np.nan

            rows.append(dict(
                book=book, chapter=ch, n_words=nw,
                source_label=src,
                map_char=round(map_ng, 1), ci68_lo_char=round(lo_ng, 1), ci68_hi_char=round(hi_ng, 1),
                map_word=round(map_AB, 1), ci68_lo_word=round(lo_AB, 1), ci68_hi_word=round(hi_AB, 1),
                avg_date=round(avg_date, 1),
                archaism_index=round(arc_idx, 1) if not np.isnan(arc_idx) else np.nan,
                classification=cls,
            ))
            done += 1

            # progress every 5 chapters
            if ch % 5 == 0 or ch == n_chs:
                noisy_str = '⚠' if nw < MIN_WORDS_FOR_DATING else ''
                print(f'  {ch:>3}  {nw:>6,}  '
                      f'{map_ng:>7.0f}  {map_AB:>7.0f}  '
                      f'{arc_idx:>+6.0f}  {cls:<14}  {src}  {noisy_str}')

        print(f'  {done}/{total_chapters} chapters complete')

    chapter_data = pd.DataFrame(rows)
    chapter_csv  = outdir / 'torah_chapter_dates.csv'
    chapter_data.to_csv(chapter_csv, index=False)
    print(f'\n  Saved: {chapter_csv.name}')

    # -----------------------------------------------------------------------
    # STEP 4 — Compilation date estimates
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 4 — Compilation date estimates')
    print('=' * 70)

    compilation_report = {}
    print(f'\n  {"Book":<14}  {"Oldest":>7}  {"Median":>7}  {"TPQ":>7}  '
          f'{"Latest 3 chapters"}')
    print('  ' + '-' * 75)

    for book in TORAH_BOOKS:
        df = chapter_data[
            (chapter_data['book'] == book) &
            (chapter_data['n_words'] >= MIN_WORDS_FOR_DATING) &
            chapter_data['avg_date'].notna()
        ].copy()

        if df.empty:
            compilation_report[book] = {
                'oldest_source_date': np.nan,
                'main_composition_date': np.nan,
                'compilation_tpq': np.nan,
                'latest_chapters': [],
            }
            continue

        avg_dates = df['avg_date'].values
        oldest    = float(np.percentile(avg_dates, 90))   # oldest = high BCE
        median    = float(np.percentile(avg_dates, 50))
        tpq       = float(np.percentile(avg_dates, 10))   # youngest = low BCE

        # Find the 3 latest-dating chapters (youngest = smallest BCE value)
        df_sorted = df.sort_values('avg_date')
        latest3   = df_sorted.head(3)[['chapter', 'source_label', 'avg_date',
                                       'classification']].to_dict('records')

        compilation_report[book] = {
            'oldest_source_date':    round(oldest, 0),
            'main_composition_date': round(median, 0),
            'compilation_tpq':       round(tpq,    0),
            'latest_chapters':       latest3,
        }

        lc_str = ', '.join(f"ch{int(r['chapter'])}({r['source_label']},avg={r['avg_date']:.0f})"
                           for r in latest3)
        print(f'  {book:<14}  {oldest:>7.0f}  {median:>7.0f}  {tpq:>7.0f}  {lc_str}')

    # Save compilation report
    comp_rows = []
    for book, r in compilation_report.items():
        comp_rows.append(dict(
            book=book,
            oldest_source_date=r['oldest_source_date'],
            main_composition_date=r['main_composition_date'],
            compilation_tpq=r['compilation_tpq'],
            latest_ch_1=r['latest_chapters'][0]['chapter'] if len(r['latest_chapters']) > 0 else '',
            latest_ch_2=r['latest_chapters'][1]['chapter'] if len(r['latest_chapters']) > 1 else '',
            latest_ch_3=r['latest_chapters'][2]['chapter'] if len(r['latest_chapters']) > 2 else '',
        ))
    comp_df = pd.DataFrame(comp_rows)
    comp_csv = outdir / 'torah_compilation_report.csv'
    comp_df.to_csv(comp_csv, index=False)
    print(f'\n  Saved: {comp_csv.name}')

    # -----------------------------------------------------------------------
    # STEP 5 — Classification summary
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 5 — Classification summary')
    print('=' * 70)

    for book in TORAH_BOOKS:
        df  = chapter_data[chapter_data['book'] == book]
        cts = df['classification'].value_counts()
        print(f'\n  {book}:')
        for cls_label, count in cts.items():
            print(f'    {cls_label:<14}: {count} chapters')
            if cls_label in ('OLD_SOURCE', 'ARCHAIZED', 'EDITORIAL'):
                ch_list = df[df['classification'] == cls_label]['chapter'].tolist()
                print(f'      chapters: {ch_list}')

    # -----------------------------------------------------------------------
    # STEP 6 — Plots
    # -----------------------------------------------------------------------
    print('\n' + '=' * 70)
    print('STEP 6 — Plots')
    print('=' * 70)

    plot_age_profiles(chapter_data, str(outdir / 'torah_age_profiles.png'))
    plot_archaism_map(chapter_data, str(outdir / 'torah_archaism_map.png'))
    plot_source_comparison(str(outdir / 'torah_source_comparison.png'), WORKSPACE)
    plot_compilation_summary(compilation_report,
                             str(outdir / 'torah_compilation_summary.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
