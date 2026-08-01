#!/usr/bin/env python3
"""
Script 12: Tier 3 — Clause and Phrase-Level Diachronic Features
================================================================
Tests syntactic features that require traversal of BHSA clause and phrase
nodes rather than simple word-level counting.  These represent the most
linguistically sophisticated tier of our analysis.

Features tested:
  1.  frac_nmcl       — Nominal clause fraction of all content clauses
  2.  frac_fronted     — Fronted-constituent (non-V-initial) clause fraction
  3.  frac_cpen        — Casus pendens fraction (hanging-topic construction)
  4.  frac_wqtl_wayq   — Waw-qatal / (wayyiqtol + waw-qatal) narrative ratio
  5.  frac_ptcp_cl     — Participial clause fraction
  6.  frac_infc        — Infinitive-construct clause fraction
  7.  frac_sv          — S-before-V word-order fraction (verbal clauses)
  8.  frac_null_subj   — Null-subject fraction (verbal clauses, no Subj phrase)
  9.  rate_cpen_1k     — Casus pendens rate per 1,000 words
  10. frac_objc_before_pred — Object-before-verb (OV) order fraction

All features are computed for the broad training corpus (oracle Jeremiah),
then tested with Spearman ρ and LOO resampling.  Only features passing
p < 0.10 with LOO fraction ≥ 0.50 are recommended for inclusion in the
main analysis pipeline.

Usage
-----
    python 12_tier3_clause_features.py [--data-path PATH]
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter, defaultdict
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Corpus specs (same as scripts 06–11, oracle Jeremiah)
# ---------------------------------------------------------------------------

JEREMIAH_ORACLE_CHAPTERS = set(
    list(range(1, 7)) + list(range(8, 11)) + list(range(12, 17)) +
    [19, 20] + list(range(22, 24)) + list(range(30, 32)) + list(range(46, 52))
)

PROPHETIC_SPECS = [
    ('Amos',         ['Amos'],         None,       760, 15),
    ('Hosea',        ['Hosea'],        None,       725, 20),
    ('Micah',        ['Micah'],        None,       720, 20),
    ('Isaiah_1',     ['Isaiah'],       (1, 39),    700, 15),
    ('Zephaniah',    ['Zephaniah'],    None,       630, 15),
    ('Nahum',        ['Nahum'],        None,       620, 20),
    ('Habakkuk',     ['Habakkuk'],     None,       605, 20),
    ('Jeremiah',     ['Jeremiah'],     'oracle',   590, 15),
    ('Lamentations', ['Lamentations'], None,       586, 20),
    ('Ezekiel',      ['Ezekiel'],      None,       570, 15),
    ('Isaiah_2',     ['Isaiah'],       (40, 55),   550, 20),
    ('Haggai',       ['Haggai'],       None,       520,  5),
    ('Zechariah_1',  ['Zechariah'],    (1,  8),    518,  5),
    ('Isaiah_3',     ['Isaiah'],       (56, 66),   450, 100),  # revised: 450 BCE
    ('Malachi',      ['Malachi'],      None,       460, 20),
]

BROAD_EXTENSION = [
    ('Jonah',        ['Jonah'],                        None,           400,  50),
    ('Ezra',         ['Ezra'],                         None,           350,  75),  # revised: 350 BCE
    ('Nehemiah',     ['Nehemiah'],                     None,           350,  75),  # revised: 350 BCE
    ('Chronicles',   ['1_Chronicles', '2_Chronicles'], None,           350,  50),
    ('Esther',       ['Esther'],                       None,           350,  50),
    ('Ecclesiastes', ['Ecclesiastes'],                 None,           330,  80),
    ('Daniel',       ['Daniel'],                       [(1,1),(8,12)], 167,  10),  # Hebrew chs only
]

# Clause-type groupings
# ---- wayyiqtol (narrative waw-consecutive imperfect)
WAYQ_TYPES   = {'Way0', 'WayX', 'WaYX'}
# ---- waw-qatal (converted perfect / waw-consecutive perfect)
WQTL_TYPES   = {'WQt0', 'WQtX', 'WxQ0', 'WxQX'}
# ---- all waw-narrative = union (denominator for frac_wqtl_wayq)
WNARR_TYPES  = WAYQ_TYPES | WQTL_TYPES
# ---- fronted non-verb-initial clauses (x = non-predicate precedes verb)
FRONT_TYPES  = {'xQt0', 'xYq0', 'xQtX', 'xYqX', 'xIm0',
                'WxY0', 'WxQ0', 'WxQX', 'WxYX', 'WxI0',
                'XQtl', 'XYqt'}
# ---- casus pendens
CPEN_TYPES   = {'CPen'}
# ---- nominal clauses
NMCL_TYPES   = {'NmCl', 'AjCl'}
# ---- participial clauses
PTCP_TYPES   = {'Ptcp'}
# ---- infinitive-construct clauses
INFC_TYPES   = {'InfC'}
# ---- clause types that contain a finite/non-inf predicate (verbal clauses)
VERBAL_TYPES = (WAYQ_TYPES | WQTL_TYPES | FRONT_TYPES |
                {'ZQt0', 'ZQtX', 'ZYq0', 'ZYqX', 'ZIm0',
                 'WYq0', 'WYqX', 'WIm0', 'WXYq', 'WXQt',
                 'xQt0', 'xYq0', 'xQtX', 'xYqX',
                 'Way0', 'WayX'})
# ---- skip these uninformative/non-content types in denominators
SKIP_TYPES   = {'Ellp', 'Voct', 'MSyn', 'InfA'}


# ---------------------------------------------------------------------------
# BHSA loading
# ---------------------------------------------------------------------------

def load_bhsa(data_path):
    try:
        from tf.fabric import Fabric
    except ImportError:
        sys.exit('text-fabric not installed.')
    print(f'Loading BHSA from {data_path}...')
    TF = Fabric(locations=str(data_path), modules=[''], silent=True)
    api = TF.load(
        'otype lex sp vs vt typ rela function domain book chapter',
        silent=True)
    return api


# ---------------------------------------------------------------------------
# Clause iteration for a text unit
# ---------------------------------------------------------------------------

def clauses_for_spec(spec, F, L, T):
    """
    Yield (clause_node, chap_num) for every clause in the unit.
    Handles None / (start,end) / list-of-(start,end) / 'oracle' chapter ranges.
    """
    name, books, chap_range, *_ = spec
    for book in books:
        bn = T.nodeFromSection((book,))
        if bn is None:
            continue
        for ch_node in L.d(bn, 'chapter'):
            ch_num = int(F.chapter.v(ch_node))
            if chap_range == 'oracle':
                if ch_num not in JEREMIAH_ORACLE_CHAPTERS:
                    continue
            elif isinstance(chap_range, list):
                # list of (lo, hi) inclusive ranges — e.g. [(1,1),(8,12)] for Daniel
                if not any(lo <= ch_num <= hi for lo, hi in chap_range):
                    continue
            elif chap_range and not (chap_range[0] <= ch_num <= chap_range[1]):
                continue
            for cl in L.d(ch_node, 'clause'):
                yield cl, ch_num


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_tier3(spec, F, L, T):
    """
    Compute all Tier-3 features for one corpus unit.
    Returns dict of feature_name → value, plus metadata.
    """
    # Counters
    n_words       = 0
    n_clauses     = 0            # all non-skip clauses
    n_verbal      = 0            # clauses with finite predicate
    n_nmcl        = 0
    n_ptcp        = 0
    n_infc        = 0
    n_fronted     = 0
    n_cpen        = 0
    n_wayq        = 0
    n_wqtl        = 0
    n_wnarr       = 0            # wayq + wqtl
    n_sv          = 0            # verbal clauses with Subj before Pred
    n_sv_total    = 0            # verbal clauses with both Subj and Pred
    n_null_subj   = 0            # verbal clauses with no Subj phrase
    n_ov          = 0            # verbal clauses with Objc before Pred
    n_ov_total    = 0            # verbal clauses with both Objc and Pred

    for cl, _ in clauses_for_spec(spec, F, L, T):
        typ = F.typ.v(cl)
        if typ in SKIP_TYPES:
            # Still count words
            n_words += sum(1 for _ in L.d(cl, 'word'))
            continue

        words_in_cl = list(L.d(cl, 'word'))
        n_words += len(words_in_cl)
        n_clauses += 1

        # Clause-type tallies
        if typ in NMCL_TYPES:
            n_nmcl += 1
        if typ in PTCP_TYPES:
            n_ptcp += 1
        if typ in INFC_TYPES:
            n_infc += 1
        if typ in FRONT_TYPES:
            n_fronted += 1
        if typ in CPEN_TYPES:
            n_cpen += 1
        if typ in WAYQ_TYPES:
            n_wayq += 1
        if typ in WQTL_TYPES:
            n_wqtl += 1
        if typ in WNARR_TYPES:
            n_wnarr += 1

        # Phrase-function analysis for verbal clauses
        if typ in VERBAL_TYPES:
            n_verbal += 1
            phrases = list(L.d(cl, 'phrase'))
            ph_funcs = {F.function.v(ph): ph for ph in phrases}

            has_subj = 'Subj' in ph_funcs
            has_pred = 'Pred' in ph_funcs
            has_objc = 'Objc' in ph_funcs

            if not has_subj:
                n_null_subj += 1

            if has_subj and has_pred:
                n_sv_total += 1
                if ph_funcs['Subj'] < ph_funcs['Pred']:   # node order = text order
                    n_sv += 1

            if has_objc and has_pred:
                n_ov_total += 1
                if ph_funcs['Objc'] < ph_funcs['Pred']:
                    n_ov += 1

    if n_clauses == 0 or n_words == 0:
        return None

    result = {
        '_n_words':          n_words,
        '_n_clauses':        n_clauses,
        # Fractions of all content clauses
        'frac_nmcl':         n_nmcl   / n_clauses,
        'frac_fronted':      n_fronted / n_clauses,
        'frac_cpen':         n_cpen   / n_clauses,
        'frac_ptcp_cl':      n_ptcp   / n_clauses,
        'frac_infc':         n_infc   / n_clauses,
        # Narrative waw ratio
        'frac_wqtl_wayq':    n_wqtl  / n_wnarr if n_wnarr > 5 else np.nan,
        # Per-1000-words rate
        'rate_cpen_1k':      n_cpen  / n_words * 1000,
        # Verbal clause word-order fractions
        'frac_sv':           n_sv    / n_sv_total   if n_sv_total   > 10 else np.nan,
        'frac_null_subj':    n_null_subj / n_verbal if n_verbal      > 10 else np.nan,
        'frac_ov':           n_ov    / n_ov_total   if n_ov_total   > 10 else np.nan,
    }
    return result


# ---------------------------------------------------------------------------
# Spearman ρ scan + LOO
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    'frac_nmcl', 'frac_fronted', 'frac_cpen', 'frac_ptcp_cl',
    'frac_infc', 'frac_wqtl_wayq', 'rate_cpen_1k',
    'frac_sv', 'frac_null_subj', 'frac_ov',
]

# Expected directions (positive = increases toward LBH / later periods)
EXPECTED_DIR = {
    'frac_nmcl':       +1,   # more nominal clauses in LBH
    'frac_fronted':    +1,   # more fronting in LBH
    'frac_cpen':       +1,   # casus pendens increases in LBH
    'frac_ptcp_cl':    +1,   # participial predication increases
    'frac_infc':        0,   # unclear
    'frac_wqtl_wayq':  +1,   # waw-qatal replaces wayyiqtol in LBH
    'rate_cpen_1k':    +1,   # same direction as frac_cpen
    'frac_sv':         +1,   # SV order increases in LBH (more analytical)
    'frac_null_subj':   0,   # unclear
    'frac_ov':          0,   # unclear direction
}


def spearman_scan(rates_df, dates_bce):
    """Run Spearman ρ for each feature against date (BCE, positive = older)."""
    rows = []
    for fn in FEATURE_NAMES:
        y = rates_df[fn].values.astype(float)
        x = np.array(dates_bce, dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 6:
            rows.append({'feature': fn, 'rho': np.nan, 'p': np.nan,
                         'n_valid': valid.sum()})
            continue
        # Negative x because dates_bce is positive = older; we want
        # positive ρ to mean "increases as time progresses forward"
        rho, p = spearmanr(-x[valid], y[valid])
        rows.append({'feature': fn, 'rho': rho, 'p': p,
                     'n_valid': int(valid.sum()),
                     'expected': EXPECTED_DIR.get(fn, 0)})
    return pd.DataFrame(rows)


def loo_fraction(rates_df, dates_bce, feature_name, observed_rho):
    """
    Leave-one-out: what fraction of LOO resamples give ρ with same sign as
    observed_rho?  Returns fraction [0, 1].
    """
    y = rates_df[feature_name].values.astype(float)
    x = np.array(dates_bce, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 6:
        return np.nan
    n = len(x)
    same_sign = 0
    total = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        xv = -x[mask & valid]
        yv =  y[mask & valid]
        if len(xv) < 5:
            continue
        rho_i, _ = spearmanr(xv, yv)
        total += 1
        if np.sign(rho_i) == np.sign(observed_rho):
            same_sign += 1
    return same_sign / total if total > 0 else np.nan


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tier3_scan(scan_df, output_path):
    """Bar chart of Spearman ρ for all Tier-3 features."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = []
    for _, row in scan_df.iterrows():
        exp = row.get('expected', 0)
        rho = row['rho']
        if not np.isfinite(rho):
            colors.append('#aaaaaa')
        elif exp == 0:
            colors.append('#888888')
        elif np.sign(rho) == np.sign(exp):
            colors.append('#2166ac')   # direction-consistent
        else:
            colors.append('#d6604d')   # wrong direction

    rhos = [r if np.isfinite(r) else 0 for r in scan_df['rho']]
    ax.barh(range(len(scan_df)), rhos, color=colors, alpha=0.8)
    ax.set_yticks(range(len(scan_df)))
    ax.set_yticklabels(scan_df['feature'].tolist(), fontsize=9)
    ax.axvline(0, color='black', lw=0.8)

    # Mark significance threshold
    sig_rows = scan_df[scan_df['p'] < 0.10]
    for i, row in sig_rows.iterrows():
        ax.text(rhos[i] + (0.01 if rhos[i] >= 0 else -0.01),
                list(scan_df.index).index(i),
                f'p={row["p"]:.3f}', va='center',
                ha='left' if rhos[i] >= 0 else 'right', fontsize=7)

    ax.set_xlabel('Spearman ρ  (positive = increases toward LBH)')
    ax.set_title('Tier 3 feature scan — clause & phrase-level features\n'
                 'Blue = direction-consistent; red = wrong direction; grey = undetermined')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Feature scan plot saved: {output_path}')


def plot_tier3_trajectories(rates_df, dates_bce, sig_features, output_path):
    """Scatter + regression line for each significant feature."""
    if not sig_features:
        return
    ncols = min(len(sig_features), 3)
    nrows = (len(sig_features) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    axes = np.array(axes).flatten()

    for i, fn in enumerate(sig_features):
        ax = axes[i]
        y = rates_df[fn].values.astype(float)
        x = np.array(dates_bce)
        valid = np.isfinite(y)
        ax.scatter(-x[valid], y[valid], s=40, alpha=0.8, color='#2166ac')
        for j, name in enumerate(rates_df.index):
            if valid[j]:
                ax.annotate(name[:6], (-x[j], y[j]),
                            fontsize=6, ha='left', va='bottom')
        if valid.sum() >= 4:
            b, a = np.polyfit(-x[valid], y[valid], 1)
            xr = np.array([(-x[valid]).min(), (-x[valid]).max()])
            ax.plot(xr, a + b * xr, 'r--', lw=1.5)
        ax.set_xlabel('Years BCE (→ later)')
        ax.set_ylabel(fn)
        ax.set_title(fn)
    for j in range(len(sig_features), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f'Trajectory plots saved: {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-path',
        default=str(Path.home()/'text-fabric-data'/'github'/'ETCBC'/'bhsa'/'tf'/'2021'))
    parser.add_argument('--corpus', choices=['prophetic', 'broad'],
        default='broad')
    parser.add_argument('--outdir', default='.')
    args = parser.parse_args()

    outdir = Path(args.outdir)

    specs = (PROPHETIC_SPECS + BROAD_EXTENSION
             if args.corpus == 'broad' else PROPHETIC_SPECS)
    print(f'\nUsing {"BROAD" if args.corpus == "broad" else "PROPHETIC"} corpus '
          f'({len(specs)} units, oracle Jeremiah)')

    # ── Load BHSA ──────────────────────────────────────────────────────────
    api = load_bhsa(args.data_path)
    F, L, T = api.F, api.L, api.T

    # ── Extract Tier-3 features for all training units ─────────────────────
    print('\nExtracting clause-level features...')
    rows      = []
    dates_bce = []
    names     = []

    for spec in specs:
        unit_name = spec[0]
        date_ctr  = spec[3]
        print(f'  {unit_name:<18}', end='', flush=True)

        feat = extract_tier3(spec, F, L, T)
        if feat is None:
            print('  (no data — skipped)')
            continue

        nw = feat['_n_words']
        nc = feat['_n_clauses']
        frac_wq = feat.get('frac_wqtl_wayq', np.nan)
        frac_nm = feat.get('frac_nmcl', np.nan)
        print(f'{nw:>8,} words  {nc:>6,} clauses  '
              f'wqtl/wayq={frac_wq:.3f}  nmcl={frac_nm:.3f}')

        row = {fn: feat.get(fn, np.nan) for fn in FEATURE_NAMES}
        rows.append(row)
        dates_bce.append(date_ctr)
        names.append(unit_name)

    rates_df = pd.DataFrame(rows, index=names)

    # ── Spearman ρ scan ─────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('TIER-3 FEATURE CORRELATION SCAN')
    print('='*70)

    scan_df = spearman_scan(rates_df, dates_bce)

    # Expected direction consistency
    scan_df['dir_consistent'] = scan_df.apply(
        lambda r: (r['expected'] == 0 or np.sign(r['rho']) == np.sign(r['expected']))
        if np.isfinite(r['rho']) else False, axis=1)

    # LOO for features with p < 0.20 (compute loo for borderline too)
    print('Computing LOO stability...')
    loo_fracs = []
    for _, row in scan_df.iterrows():
        if not np.isfinite(row['rho']) or abs(row['rho']) < 0.10:
            loo_fracs.append(np.nan)
            continue
        lf = loo_fraction(rates_df, dates_bce, row['feature'], row['rho'])
        loo_fracs.append(lf)
    scan_df['loo_frac'] = loo_fracs

    # Sort by |ρ|
    scan_df = scan_df.sort_values('rho', key=abs, ascending=False)

    header = f"  {'Feature':<22} {'ρ':>7}  {'p':>8}  {'Expected':>9}  {'Dir?':>5}  {'LOO':>6}"
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for _, row in scan_df.iterrows():
        if not np.isfinite(row['rho']):
            print(f"  {row['feature']:<22}   {'nan':>7}   {'nan':>8}  "
                  f"{'?':>9}  {'':>5}  {'':>6}  (insufficient data)")
            continue
        exp_str = ('+LBH' if row['expected'] > 0 else
                   '-LBH' if row['expected'] < 0 else '?')
        dir_str = '✓' if row['dir_consistent'] else '✗'
        loo_str = (f"{row['loo_frac']:.2f}" if np.isfinite(row.get('loo_frac', np.nan))
                   else '  — ')
        sig = ('***' if row['p'] < 0.01 else '**' if row['p'] < 0.05
               else '*' if row['p'] < 0.10 else '')
        print(f"  {row['feature']:<22} {row['rho']:>+7.3f}  {row['p']:>8.4f}  "
              f"{exp_str:>9}  {dir_str:>5}  {loo_str:>6}  {sig}")

    # ── Summary ──────────────────────────────────────────────────────────────
    sig_mask = (scan_df['p'] < 0.10) & scan_df['rho'].apply(np.isfinite)
    sig_features = scan_df[sig_mask]['feature'].tolist()
    loo_sig = [f for f in sig_features
               if np.isfinite(scan_df.set_index('feature').loc[f, 'loo_frac'] or np.nan)
               and (scan_df.set_index('feature').loc[f, 'loo_frac'] or 0) >= 0.50]

    print(f'\nSummary:')
    print(f'  Features tested:              {len(scan_df)}')
    print(f'  Features p < 0.10:            {len(sig_features)}')
    print(f'  Also LOO-stable (≥ 50 %):     {len(loo_sig)}')

    if loo_sig:
        print(f'\n  *** RECOMMENDED FOR PIPELINE INCLUSION ***')
        for fn in loo_sig:
            row = scan_df.set_index('feature').loc[fn]
            print(f'    {fn:<22}  ρ={row["rho"]:+.3f}  p={row["p"]:.4f}  '
                  f'LOO={row["loo_frac"]:.2f}')
    else:
        print(f'\n  No Tier-3 features meet p < 0.10 with LOO ≥ 0.50.')
        print(f'  Tier-3 clause features do not add reliable diachronic signal')
        print(f'  beyond the lexical/morphological features already in the pipeline.')

    # ── Torah book values for significant features ────────────────────────
    if sig_features:
        print(f'\n  Torah book values for significant Tier-3 features:')
        torah_books = ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy']
        hdr = f"  {'Feature':<22}  " + '  '.join(f'{b[:3]:>7}' for b in torah_books)
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        for fn in sig_features:
            row_vals = []
            for book in torah_books:
                spec = (book, [book], None, 0, 0)
                feat = extract_tier3(spec, F, L, T)
                v = feat.get(fn, np.nan) if feat else np.nan
                row_vals.append(v)
            vals_str = '  '.join(f'{v:>7.3f}' if np.isfinite(v) else '    nan'
                                  for v in row_vals)
            print(f'  {fn:<22}  {vals_str}')

    # ── Save and plot ─────────────────────────────────────────────────────
    scan_df.to_csv(outdir / 'tier3_feature_scan.csv', index=False)
    rates_df.to_csv(outdir / 'tier3_training_rates.csv')
    print(f'\nFeature scan saved: tier3_feature_scan.csv')
    print(f'Training rates saved: tier3_training_rates.csv')

    plot_tier3_scan(scan_df, str(outdir / 'tier3_scan.png'))
    if sig_features:
        plot_tier3_trajectories(rates_df, dates_bce, sig_features,
                                str(outdir / 'tier3_trajectories.png'))

    print('\nDone.')


if __name__ == '__main__':
    main()
