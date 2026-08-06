#!/usr/bin/env python3
"""
kings_chronicles.py — matched-pairs diachronic test on synoptic parallels
=========================================================================
The Greek archaizing analysis collapsed because of an author-identity confound:
one text per author, so features tracked WHO wrote it, and an arbitrary split of
genuine Classical texts scored AUC 0.94 (placebo T3).

Kings/Chronicles removes that structurally. The Chronicler retells narratives
from Samuel-Kings centuries later. Content, genre, subject and language are held
constant within each pair; only the date of composition varies. This is a
matched-pairs design, so each pair is its own control and no cross-author
comparison is needed.

Three things are measured:

  1. PAIRED DIRECTION. For each synoptic pair, does the Chronicles member score
     more LBH than its Kings counterpart? Tested with a Wilcoxon signed-rank
     test across pairs, plus a plain sign count.

  2. NOISE FLOOR. 2 Kgs 18-20 and Isa 36-39 are near-verbatim duplicates of the
     same Hezekiah narrative. Whatever difference the method reports for THAT
     pair is measurement noise, and any real signal must exceed it.

  3. PLACEBO. Arbitrary splits WITHIN Kings and WITHIN Chronicles. If the
     method reports comparable differences for splits that carry no date
     contrast, the paired result means nothing. This is the control the Greek
     analysis failed.

Output: results_v2/kings_chronicles.csv
"""

from __future__ import annotations
import os, importlib.util
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
FM   = os.path.join(HERE, "data", "feature_matrix_v2.csv")
OUT  = os.path.join(HERE, "results_v2", "kings_chronicles.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

spec = importlib.util.spec_from_file_location(
    "ex", os.path.join(HERE, "hierarchical_bayes", "00_extract_features.py"))
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)

PRESPEC = ["frac_ani","frac_she","frac_ein","rate_wayyiqtol","rate_qatal",
           "frac_niphal","rate_inf_con","rate_gam","rate_ut_nouns",
           "frac_infc","frac_fronted","frac_null_subj"]

# Synoptic parallels (standard critical alignment)
PAIRS = [
    ("Solomon_accession", [("1_Kings",[(1,2)])],        [("1_Chronicles",[(28,29)])]),
    ("Solomon_reign",     [("1_Kings",[(3,5)])],        [("2_Chronicles",[(1,2)])]),
    ("Temple",            [("1_Kings",[(6,8)])],        [("2_Chronicles",[(3,7)])]),
    ("Solomon_later",     [("1_Kings",[(9,11)])],       [("2_Chronicles",[(8,9)])]),
    ("Rehoboam",          [("1_Kings",[(12,14)])],      [("2_Chronicles",[(10,12)])]),
    ("Asa",               [("1_Kings",[(15,15)])],      [("2_Chronicles",[(14,16)])]),
    ("Jehoshaphat",       [("1_Kings",[(22,22)])],      [("2_Chronicles",[(17,20)])]),
    ("Joash",             [("2_Kings",[(11,12)])],      [("2_Chronicles",[(22,24)])]),
    ("Amaziah",           [("2_Kings",[(14,14)])],      [("2_Chronicles",[(25,25)])]),
    ("Uzziah_Jotham",     [("2_Kings",[(15,15)])],      [("2_Chronicles",[(26,27)])]),
    ("Ahaz",              [("2_Kings",[(16,16)])],      [("2_Chronicles",[(28,28)])]),
    ("Hezekiah",          [("2_Kings",[(18,20)])],      [("2_Chronicles",[(29,32)])]),
    ("Manasseh",          [("2_Kings",[(21,21)])],      [("2_Chronicles",[(33,33)])]),
    ("Josiah",            [("2_Kings",[(22,23)])],      [("2_Chronicles",[(34,35)])]),
]

# Near-verbatim duplicate: measurement noise floor
NOISE_PAIR = ("Hezekiah_noise_floor",
              [("2_Kings", [(18,20)])], [("Isaiah", [(36,39)])])

# Placebo: arbitrary within-book splits carrying no date contrast
PLACEBO = [
    ("placebo_Kings_odd_even", [("1_Kings",[(1,11)])],  [("1_Kings",[(12,22)])]),
    ("placebo_Chron_early_late",[("2_Chronicles",[(1,17)])],
                                [("2_Chronicles",[(18,36)])]),
    ("placebo_2Kings_split",   [("2_Kings",[(1,12)])],  [("2_Kings",[(13,25)])]),
]


def main():
    df = pd.read_csv(FM, index_col="id")
    tr = df[(df["in_training"] == True) & (~df["holdout"].astype(bool))]  # noqa: E712
    feats = [f for f in PRESPEC if f in df.columns]

    # CBH/LBH anchors from the dated training corpus (external calibration)
    d = tr["date_bce"].values.astype(float)
    A, B = [], []
    for f in feats:
        y = pd.to_numeric(tr[f], errors="coerce").values.astype(float)
        m = np.isfinite(y)
        sl, ic, *_ = stats.linregress(d[m], y[m])
        A.append(ic); B.append(sl)
    A, B = np.array(A), np.array(B)
    mu_c, mu_l = A + B * 720.0, A + B * 250.0

    api = ex.load_bhsa(ex.TF_PATH)
    F, L, T = api.F, api.L, api.T

    def score(pairs):
        fe = ex.extract_unit("x", pairs, F, L, T)
        if fe is None:
            return None, None, 0
        n = fe.pop("n_words")
        x = np.array([fe.get(f, np.nan) for f in feats], float)
        den = mu_l - mu_c
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(np.abs(den) > 1e-12, (x - mu_c) / den, np.nan)
        return float(np.nanmean(s)), x, n

    rows = []
    print("SYNOPTIC PAIRS — does Chronicles read later than Kings?\n")
    print(f"  {'pair':22s}{'Kings s':>9s}{'Chron s':>9s}{'delta':>8s}"
          f"{'K words':>9s}{'C words':>9s}")
    print("  " + "-" * 68)
    deltas, names = [], []
    for name, kp, cp in PAIRS:
        sk, xk, nk = score(kp)
        sc, xc, nc = score(cp)
        if sk is None or sc is None:
            continue
        dl = sc - sk
        deltas.append(dl); names.append(name)
        rows.append(dict(pair=name, kind="synoptic", s_kings=sk, s_chron=sc,
                         delta=dl, n_kings=nk, n_chron=nc))
        print(f"  {name:22s}{sk:>9.3f}{sc:>9.3f}{dl:>+8.3f}{nk:>9d}{nc:>9d}")

    deltas = np.array(deltas)
    pos = int((deltas > 0).sum())
    print(f"\n  Chronicles scores later in {pos} of {len(deltas)} pairs")
    if len(deltas) >= 6:
        w = stats.wilcoxon(deltas)
        sg = stats.binomtest(pos, len(deltas), 0.5)
        print(f"  mean delta = {deltas.mean():+.3f}   median = {np.median(deltas):+.3f}")
        print(f"  Wilcoxon signed-rank : W={w.statistic:.1f}, p={w.pvalue:.4f}")
        print(f"  sign test            : p={sg.pvalue:.4f}")

    # Noise floor
    name, kp, cp = NOISE_PAIR
    sk, _, nk = score(kp); sc, _, nc = score(cp)
    if sk is not None and sc is not None:
        nf = sc - sk
        rows.append(dict(pair=name, kind="noise_floor", s_kings=sk, s_chron=sc,
                         delta=nf, n_kings=nk, n_chron=nc))
        print(f"\n  NOISE FLOOR (2 Kgs 18-20 vs Isa 36-39, near-verbatim)")
        print(f"    delta = {nf:+.3f}   vs synoptic mean {deltas.mean():+.3f}")
        print(f"    signal-to-noise ratio = {abs(deltas.mean()/nf):.1f}x"
              if abs(nf) > 1e-9 else "    (noise floor ~0)")

    # Placebo
    print(f"\n  PLACEBO — arbitrary within-book splits (no date contrast)")
    pl = []
    for name, ap, bp in PLACEBO:
        sa, _, na = score(ap); sb, _, nb = score(bp)
        if sa is None or sb is None:
            continue
        dl = sb - sa; pl.append(dl)
        rows.append(dict(pair=name, kind="placebo", s_kings=sa, s_chron=sb,
                         delta=dl, n_kings=na, n_chron=nb))
        print(f"    {name:26s} delta = {dl:+.3f}")
    if pl:
        print(f"\n    placebo |delta| mean = {np.mean(np.abs(pl)):.3f}")
        print(f"    synoptic |delta| mean = {np.mean(np.abs(deltas)):.3f}")
        verdict = ("SURVIVES placebo" if np.mean(np.abs(deltas)) >
                   2 * np.mean(np.abs(pl)) else
                   "FAILS placebo — within-book variation is comparable")
        print(f"    -> {verdict}")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
