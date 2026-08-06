#!/usr/bin/env python3
"""
resistant_v3.py — resistant model rebuilt on morphology, plus over-correction
=============================================================================
Motivated by the Greek ground-truth test (greek/archaism_resistance.py), where
the Second Sophistic Atticizers provide labelled archaizing data:

    morphology  R = +0.59   (optative 0.77, participle 0.70)  RESISTS imitation
    syntax      R = +0.01                                     FAKED completely

The previous Hebrew resistant model used clause-level syntax on the assumption
that syntactic templating lies below conscious control. The Greek evidence says
that assumption is wrong: educated imitators reproduce sentence shapes, but not
the frequency distribution of morphological forms across thousands of verbs.

This script therefore does two things.

  1. MORPHOLOGICAL RESISTANT MODEL. Rebuild the resistant instrument from verb
     stem and verb form distributions rather than clause fractions.

  2. OVER-CORRECTION TEST. In Greek the cleanest archaizing signature was not
     resistance but *overshoot*: Atticizers used oun LESS than Classical
     authors, avoided hina MORE, wrote LONGER sentences than Classical. A
     genuine ancient text sits AT the ancient extreme; an imitator goes past it.

     For each feature we compute how far a text lies beyond the most extreme
     value attained by any securely dated text, in SDs of the training spread.
     Positive overshoot on multiple features is the imitator's fingerprint, and
     it does NOT require dating the text.

Output: results_v2/resistant_v3.csv
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE   = os.path.dirname(os.path.abspath(__file__))
FM     = os.path.join(HERE, "data", "feature_matrix_v2.csv")
OUTDIR = os.path.join(HERE, "results_v2")
os.makedirs(OUTDIR, exist_ok=True)

GRID  = np.linspace(900, -100, 601)
RIDGE = 0.20

# ── Morphological resistant set (Greek-validated level) ──────────────────────
# Verb stem and verb form rates: the Hebrew analogue of optative/participle
# frequency distribution. These are hard to sustain consistently across a text.
MORPH_RESIST = [
    "rate_niphal", "rate_piel", "rate_hiphil", "rate_hithpael", "rate_hophal",
    "rate_qal", "rate_ptca", "rate_ptcp", "rate_inf_abs", "rate_impv",
    "frac_niphal", "rate_prs", "rate_const", "rate_pl_noun", "rate_f_noun",
]
# Old syntactic set, retained for direct comparison
SYNT_RESIST = ["frac_infc", "frac_fronted", "frac_null_subj", "frac_wqtl_wayq"]

# Features where "more archaic" means a LOWER value (LBH markers) vs HIGHER
# (CBH markers); direction is taken from the fitted slope, so no hand-coding.

META = {"date_bce","date_sigma","register","genre","holdout",
        "hbvi_holdout","in_training","n_words"}


def fit(df, feats, ridge=RIDGE):
    d = df["date_bce"].values.astype(float)
    A, B, res = [], [], []
    for f in feats:
        y = pd.to_numeric(df[f], errors="coerce").values.astype(float)
        m = np.isfinite(y)
        sl, ic, *_ = stats.linregress(d[m], y[m])
        A.append(ic); B.append(sl)
        r = np.full(len(d), np.nan); r[m] = y[m] - (ic + sl * d[m]); res.append(r)
    A, B = np.array(A), np.array(B)
    R = np.nan_to_num(np.vstack(res).T)
    S = (R.T @ R) / max(len(d) - 2, 1)
    S += ridge * (np.trace(S) / len(feats)) * np.eye(len(feats))
    return A, B, S


def post(x, A, B, S):
    m = np.isfinite(x)
    if m.sum() < 2:
        return np.nan
    Si = np.linalg.inv(S[np.ix_(m, m)])
    ll = np.array([-0.5 * (dv := (x - (A + B * g))[m]) @ Si @ dv for g in GRID])
    ll -= ll.max()
    p = np.exp(ll - 0.5 * ((GRID - 464.0) / 300.0) ** 2)
    return float(GRID[np.argmax(p)])


def main():
    df = pd.read_csv(FM, index_col="id")
    tr = df[(df["in_training"] == True) & (~df["holdout"].astype(bool))]  # noqa: E712
    mr = [f for f in MORPH_RESIST if f in df.columns]
    sr = [f for f in SYNT_RESIST if f in df.columns]
    print(f"Morphological resistant set: {len(mr)} features")
    print(f"Syntactic resistant set (old): {len(sr)} features\n")

    Am, Bm, Sm = fit(tr, mr)
    As, Bs, Ss = fit(tr, sr)

    # ── Over-correction: how far beyond the dated extreme does a text sit? ────
    # For each feature, find the most archaic value attained by any dated text,
    # then measure overshoot beyond it in training SDs. Direction from slope.
    stats_tbl = {}
    for j, f in enumerate(mr):
        v = pd.to_numeric(tr[f], errors="coerce").values.astype(float)
        v = v[np.isfinite(v)]
        # positive slope B means the feature RISES with BCE date (older = higher)
        older_is_high = Bm[j] > 0
        extreme = v.max() if older_is_high else v.min()
        stats_tbl[f] = (extreme, float(np.std(v)), older_is_high)

    rows = []
    for uid, r in df.iterrows():
        xm = pd.to_numeric(r[mr], errors="coerce").values.astype(float)
        xs = pd.to_numeric(r[sr], errors="coerce").values.astype(float)
        over, n_over = [], 0
        for f in mr:
            val = pd.to_numeric(pd.Series([r[f]]), errors="coerce").iloc[0]
            if not np.isfinite(val):
                continue
            ext, sd, hi = stats_tbl[f]
            if sd <= 0:
                continue
            o = (val - ext) / sd if hi else (ext - val) / sd
            over.append(o)
            if o > 1.0:
                n_over += 1
        rows.append(dict(
            unit=uid, n_words=int(r["n_words"]),
            role=("holdout" if r["holdout"] else "hbvi_holdout" if r["hbvi_holdout"]
                  else "train" if r["in_training"] else "target"),
            scholarly=r["date_bce"],
            resist_morph=round(post(xm, Am, Bm, Sm)),
            resist_synt=round(post(xs, As, Bs, Ss)),
            overshoot_mean=round(float(np.mean(over)), 2) if over else np.nan,
            overshoot_max=round(float(np.max(over)), 2) if over else np.nan,
            n_feat_overshoot=n_over))
    out = pd.DataFrame(rows).set_index("unit")
    out.to_csv(os.path.join(OUTDIR, "resistant_v3.csv"))

    # ── Calibrate: what does overshoot look like for SECURELY DATED texts? ───
    t = out[out["role"].isin(["train", "hbvi_holdout"])]
    print("Over-correction calibration on securely dated texts")
    print(f"  overshoot_mean : mean {t.overshoot_mean.mean():+.2f}  "
          f"sd {t.overshoot_mean.std():.2f}  max {t.overshoot_mean.max():+.2f}")
    print(f"  n_feat_overshoot > 1 SD : max {int(t.n_feat_overshoot.max())} "
          f"of {len(mr)} features")
    thresh = t.overshoot_mean.mean() + 2 * t.overshoot_mean.std()
    print(f"  -> flag threshold (mean + 2sd) = {thresh:+.2f}\n")

    print("Resistant-model dates and over-correction")
    print(f"  {'unit':15s}{'morph':>8s}{'syntax':>8s}{'diff':>7s}"
          f"{'oversh':>9s}{'nfeat':>7s}  flag")
    print("  " + "-" * 62)
    for u in ["Song_Sea","Song_Deborah","D_Song","D_Code","D_Frame",
              "Lev_Holiness","Lev_Priestly","P_source","JE_source","D_source",
              "Jer_oracle","Jer_DTR","Haggai","Daniel"]:
        if u not in out.index:
            continue
        r = out.loc[u]
        flag = "*** OVER-CORRECTED" if (pd.notna(r.overshoot_mean)
                                        and r.overshoot_mean > thresh) else ""
        print(f"  {u:15s}{r.resist_morph:>8.0f}{r.resist_synt:>8.0f}"
              f"{r.resist_morph-r.resist_synt:>+7.0f}"
              f"{r.overshoot_mean:>9.2f}{int(r.n_feat_overshoot):>7d}  {flag}")

    print(f"\nSaved → {OUTDIR}/resistant_v3.csv")


if __name__ == "__main__":
    main()
