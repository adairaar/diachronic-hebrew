"""
Script 20 — Add CI68 Error Bars to Bar / Dot Charts
=====================================================
Regenerates three publication-quality plots from Scripts 18 and 19 with
68% credible-interval error bars added to every bar and dot:

  1. torah_source_comparison.png   — grouped vertical bars per source layer
  2. torah_compilation_summary.png — horizontal bars per Torah book
  3. archaism_kchr_comparison.png  — connected-dot Kings vs Chronicles plot

All data is read from existing CSV outputs; no BHSA re-extraction needed.

Error-bar convention
--------------------
  • For the source comparison and KChr dots, the CI68 comes directly from
    the individual unit posteriors (columns ci68_lo_* / ci68_hi_*).
  • For the compilation summary, the three "compilation date" bars are
    percentile estimates derived from the distribution of chapter-level MAP
    dates within each book.  The CI68 of each percentile estimate is
    represented by the inter-percentile range that covers roughly 68 % of
    the surrounding distribution:
        oldest source  (90th pct)  → 75th–97.5th pct spread
        main composition (50th pct) → 33rd–67th pct spread
        compilation TPQ  (10th pct) → 2.5th–20th  pct spread
    This shows how tightly the chapter dates cluster around each summary
    estimate, making it easy to judge whether two books' bars overlap.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ───────────────────────────────────────────────────────────────────
WORKSPACE = Path('/sessions/relaxed-modest-dirac/mnt/Diachronic Hebrew')

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
})

# ── Colour palette (consistent with Scripts 18–19) ───────────────────────────
C_FULL  = '#4d9221'   # full linguistic model
C_CHAR  = '#2166ac'   # char n-gram
C_WORD  = '#d73027'   # word n-gram A+B
C_KINGS = '#1f78b4'   # Kings / Isaiah (blue circle)
C_CHR   = '#e41a1c'   # Chronicles (red square)
C_OTHER = '#ff7f00'   # non-CHR comparison (orange square)

ERR_KW  = dict(fmt='none', capsize=4, elinewidth=1.4, capthick=1.4,
               ecolor='#333333', zorder=6)


# ════════════════════════════════════════════════════════════════════════════
# 1 — Source-layer comparison (vertical grouped bars, inverted y-axis)
# ════════════════════════════════════════════════════════════════════════════

def plot_source_comparison(out_path):
    """
    Grouped bar chart: MAP dates for all source layers under three models,
    with asymmetric CI68 error bars.
    """
    master  = pd.read_csv(WORKSPACE / 'master_dating_results.csv')
    master['unit_key'] = master['unit'].replace(
        {'D': 'D_source', 'P': 'P_source', 'JE': 'JE_source'})
    mdict = {r['unit_key']: r for _, r in master.iterrows()}

    wng    = pd.read_csv(WORKSPACE / 'word_ngram_dating_results.csv')
    wng['unit_key'] = wng['unit'].replace(
        {'D': 'D_source', 'P': 'P_source', 'JE': 'JE_source'})
    wdict = {r['unit_key']: r for _, r in wng.iterrows()}

    layers = ['P_source', 'JE_source', 'D_source', 'D_Code', 'D_Frame',
              'D_Song', 'Lev_Priestly', 'Lev_Holiness']
    labels = ['P source', 'JE source', 'D source', 'D Code', 'D Frame',
              'D Song', 'Lev P', 'Lev H']

    def ci_err_y(map_val, lo, hi):
        """Return ([below], [above]) in data-coord magnitude for errorbar."""
        return [abs(map_val - lo)], [abs(hi - map_val)]

    full_maps  = [float(mdict.get(u, {}).get('map_full',    np.nan)) for u in layers]
    full_lo    = [float(mdict.get(u, {}).get('ci68_lo_raw', np.nan)) for u in layers]
    full_hi    = [float(mdict.get(u, {}).get('ci68_hi_raw', np.nan)) for u in layers]

    ng_maps    = [float(mdict.get(u, {}).get('map_ngram',   np.nan)) for u in layers]
    ng_lo      = [float(mdict.get(u, {}).get('ci68_lo_ng',  np.nan)) for u in layers]
    ng_hi      = [float(mdict.get(u, {}).get('ci68_hi_ng',  np.nan)) for u in layers]

    wng_maps   = [float(wdict.get(u, {}).get('map_AB',      np.nan)) for u in layers]
    wng_lo     = [float(wdict.get(u, {}).get('ci68_lo_AB',  np.nan)) for u in layers]
    wng_hi     = [float(wdict.get(u, {}).get('ci68_hi_AB',  np.nan)) for u in layers]

    x  = np.arange(len(layers))
    w  = 0.25
    fig, ax = plt.subplots(figsize=(13, 5.5))

    # Bars
    ax.bar(x - w, full_maps, w, label='Full linguistic model',
           color=C_FULL, alpha=0.85, zorder=3)
    ax.bar(x,     ng_maps,   w, label='Char n-gram',
           color=C_CHAR, alpha=0.85, zorder=3)
    ax.bar(x + w, wng_maps,  w, label='Word n-gram A+B',
           color=C_WORD, alpha=0.85, zorder=3)

    # CI68 error bars (asymmetric; data coords work correctly on inverted axis)
    def yerr_arrays(maps, los, his):
        below = [abs(m - l) if np.isfinite(m) and np.isfinite(l) else 0
                 for m, l in zip(maps, los)]
        above = [abs(h - m) if np.isfinite(m) and np.isfinite(h) else 0
                 for m, h in zip(maps, his)]
        return [below, above]

    ax.errorbar(x - w, full_maps, yerr=yerr_arrays(full_maps, full_lo, full_hi),
                **ERR_KW)
    ax.errorbar(x,     ng_maps,   yerr=yerr_arrays(ng_maps, ng_lo, ng_hi),
                **ERR_KW)
    ax.errorbar(x + w, wng_maps,  yerr=yerr_arrays(wng_maps, wng_lo, wng_hi),
                **ERR_KW)

    # Reference lines
    for date, lbl_txt, ls in [(760, 'Amos (760)', ':'),
                               (550, '550 BCE', '--'),
                               (450, '450 BCE', '-.')]:
        ax.axhline(date, color='#888888', lw=0.8, ls=ls, alpha=0.5,
                   label=f'ref: {date} BCE')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('MAP date (BCE)  —  error bars = 68% CI', fontsize=10)
    ax.set_title('Source layer dating: three models compared\n'
                 'Higher = older date;  error bars show 68 % credible interval',
                 fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


# ════════════════════════════════════════════════════════════════════════════
# 2 — Compilation summary (horizontal bars, inverted x-axis)
# ════════════════════════════════════════════════════════════════════════════

def plot_compilation_summary(out_path):
    """
    Per-book horizontal bar chart with percentile-spread CI68 error bars.
    """
    BOOKS  = ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy']
    MIN_WORDS = 200

    chap_df = pd.read_csv(WORKSPACE / 'torah_chapter_dates.csv')
    comp_df = pd.read_csv(WORKSPACE / 'torah_compilation_report.csv')
    comp    = {r['book']: r for _, r in comp_df.iterrows()}

    def _pct(arr, p):
        return float(np.percentile(arr, p)) if len(arr) >= 3 else np.nan

    oldest_dates, oldest_lo, oldest_hi = [], [], []
    main_dates,   main_lo,   main_hi   = [], [], []
    tpq_dates,    tpq_lo,    tpq_hi    = [], [], []

    for book in BOOKS:
        df = chap_df[
            (chap_df['book'] == book) &
            (chap_df['n_words'] >= MIN_WORDS) &
            chap_df['avg_date'].notna()
        ]['avg_date'].values

        oldest_dates.append(comp[book]['oldest_source_date'])
        main_dates.append(comp[book]['main_composition_date'])
        tpq_dates.append(comp[book]['compilation_tpq'])

        # 68%-spread CI for each percentile estimate
        # oldest (90th pct): ±1 spread → use 75th–97.5th
        oldest_lo.append(_pct(df, 75))
        oldest_hi.append(_pct(df, 97.5))
        # main composition (50th pct): use 33rd–67th
        main_lo.append(_pct(df, 33))
        main_hi.append(_pct(df, 67))
        # TPQ (10th pct): use 2.5th–20th
        tpq_lo.append(_pct(df, 2.5))
        tpq_hi.append(_pct(df, 20))

    def xerr_arrays(maps, los, his):
        # for inverted x-axis: xerr[0] goes left (older, higher BCE = hi)
        #                       xerr[1] goes right (newer, lower BCE = lo)
        left  = [abs(h - m) if np.isfinite(m) and np.isfinite(h) else 0
                 for m, h in zip(maps, his)]
        right = [abs(m - l) if np.isfinite(m) and np.isfinite(l) else 0
                 for m, l in zip(maps, los)]
        return [left, right]

    y     = np.arange(len(BOOKS))
    bar_h = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.barh(y + bar_h, oldest_dates, bar_h,
            label='Oldest source layer (90th pct)',
            color='#1a9850', alpha=0.85, zorder=3)
    ax.barh(y,          main_dates,  bar_h,
            label='Main composition (median)',
            color='#4575b4', alpha=0.85, zorder=3)
    ax.barh(y - bar_h,  tpq_dates,  bar_h,
            label='Compilation TPQ (10th pct)',
            color='#d73027', alpha=0.85, zorder=3)

    kw_h = dict(ERR_KW)   # copy so we can adjust fmt/orient
    ax.errorbar(oldest_dates, y + bar_h,
                xerr=xerr_arrays(oldest_dates, oldest_lo, oldest_hi),
                **kw_h)
    ax.errorbar(main_dates,   y,
                xerr=xerr_arrays(main_dates, main_lo, main_hi),
                **kw_h)
    ax.errorbar(tpq_dates,    y - bar_h,
                xerr=xerr_arrays(tpq_dates, tpq_lo, tpq_hi),
                **kw_h)

    ax.set_yticks(y)
    ax.set_yticklabels(BOOKS, fontsize=10)
    ax.set_xlabel('Date (BCE)  —  error bars = 68 % spread of chapter dates',
                  fontsize=10)
    ax.set_title('Torah stratigraphy: source layers and estimated compilation\n'
                 'Char + word n-gram combined; error bars = chapter-date spread (68 %)',
                 fontsize=11, fontweight='bold')
    ax.invert_xaxis()

    for date, lbl in [(600, '600 BCE'), (450, '450 BCE'), (350, '350 BCE')]:
        ax.axvline(date, color='#aaaaaa', lw=0.8, ls='--', alpha=0.6)
        ax.text(date, len(BOOKS) - 0.2, lbl, fontsize=7,
                color='#666666', ha='center')

    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


# ════════════════════════════════════════════════════════════════════════════
# 3 — Kings vs Chronicles connected-dot plot with CI68 whiskers
# ════════════════════════════════════════════════════════════════════════════

def plot_kchr_comparison(out_path):
    """
    Connected-dot plot with CI68 horizontal error bars on every dot.
    Left panel = char n-gram; right panel = word n-gram A+B.
    """
    arc = pd.read_csv(WORKSPACE / 'archaism_diagnostic_results.csv')
    recs = {r['unit']: r for _, r in arc.iterrows()}

    pairs = [
        ('Kings_Solomon',  'Chr_Solomon',  'Solomon narrative'),
        ('Kings_Judah',    'Chr_Judah',    'Judean kings (broad)'),
        ('Kings_Hezekiah', 'Chr_Hezekiah', 'Hezekiah pericope'),
        ('Kings_Fall',     'Chr_Fall',     'Manasseh → Exile'),
        ('Kings_Hezekiah', 'Isa_Hezekiah', 'Hezekiah: Kings vs Isaiah'),
    ]
    valid = [(k, c, lbl) for k, c, lbl in pairs if k in recs and c in recs]
    if not valid:
        print('  No KChr pairs found — skipping.')
        return

    models = [
        ('map_ng', 'ci68_lo_ng', 'ci68_hi_ng',  'Char n-gram'),
        ('map_AB', 'ci68_lo_AB', 'ci68_hi_AB',  'Word n-gram A+B'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for ax_idx, (mkey, lo_key, hi_key, model_label) in enumerate(models):
        ax   = axes[ax_idx]
        y_pos = np.arange(len(valid))

        for i, (k_unit, c_unit, label) in enumerate(valid):
            rk  = recs[k_unit]
            rc  = recs[c_unit]
            mk  = float(rk.get(mkey,   np.nan))
            mc  = float(rc.get(mkey,   np.nan))
            mk_lo = float(rk.get(lo_key, np.nan))
            mk_hi = float(rk.get(hi_key, np.nan))
            mc_lo = float(rc.get(lo_key, np.nan))
            mc_hi = float(rc.get(hi_key, np.nan))

            if not (np.isfinite(mk) and np.isfinite(mc)):
                continue

            c_color = C_CHR if c_unit.startswith('Chr') else C_OTHER

            # Connecting line (zorder 2 so it sits behind dots)
            ax.plot([mk, mc], [i, i], '-', color='#aaaaaa', lw=1.2, zorder=2)

            # Kings / Isaiah dot with CI68 whiskers
            if np.isfinite(mk_lo) and np.isfinite(mk_hi):
                # xerr for inverted x: left = mk_hi - mk, right = mk - mk_lo
                ax.errorbar(mk, i,
                            xerr=[[abs(mk_hi - mk)], [abs(mk - mk_lo)]],
                            fmt='none', capsize=3, elinewidth=1.2,
                            capthick=1.2, ecolor=C_KINGS, zorder=5)
            ax.scatter(mk, i, color=C_KINGS, s=80, zorder=6,
                       label='Kings/Isaiah' if i == 0 else '')

            # Chronicles / counterpart dot with CI68 whiskers
            if np.isfinite(mc_lo) and np.isfinite(mc_hi):
                ax.errorbar(mc, i,
                            xerr=[[abs(mc_hi - mc)], [abs(mc - mc_lo)]],
                            fmt='none', capsize=3, elinewidth=1.2,
                            capthick=1.2, ecolor=c_color, zorder=5)
            ax.scatter(mc, i, color=c_color, s=80, zorder=6, marker='s',
                       label='Chronicles' if i == 0 else '')

            delta = mk - mc
            x_max = max(
                (mk_hi if np.isfinite(mk_hi) else mk),
                (mc_hi if np.isfinite(mc_hi) else mc),
            )
            ax.text(x_max + 18, i, f'Δ={delta:+.0f}',
                    va='center', fontsize=8, color='#444444')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([lbl for _, _, lbl in valid], fontsize=9)
        ax.set_xlabel('MAP date (BCE)  —  error bars = 68 % CI', fontsize=10)
        ax.set_title(f'{model_label}: Kings (●) vs. Chr (■)\n'
                     f'Error bars = 68 % CI;  Positive Δ = Kings dates older',
                     fontsize=9, fontweight='bold')
        for ref in (350, 550):
            ax.axvline(ref, color='#cccccc', lw=0.6, ls=':', alpha=0.7)
        ax.invert_xaxis()
        ax.legend(fontsize=8, loc='lower right')

    fig.suptitle('Kings vs. Chronicles: character n-gram vs. word n-gram dating\n'
                 'Larger Δ = model more sensitive to register/orthography change',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {Path(out_path).name}')


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Script 20 — CI68 error-bar plots')
    print('=' * 55)

    plot_source_comparison(
        WORKSPACE / 'torah_source_comparison.png')

    plot_compilation_summary(
        WORKSPACE / 'torah_compilation_summary.png')

    plot_kchr_comparison(
        WORKSPACE / 'archaism_kchr_comparison.png')

    print('\nDone — three plots regenerated with CI68 error bars.')
