#!/usr/bin/env python3
"""
archaism_genre_v2.py — archaism diagnostic + genre correction on corpus v2
==========================================================================
Three products, all on the pre-specified feature set:

  1. LBH archaism score.  Each feature normalised against the CBH and LBH
     anchor dates; the mean score classifies a text's surface register.

  2. Archaism diagnostic.  delta_arch = MAP_full - MAP_resistant.  Positive
     means the lexical/morphological surface looks older than the clause-level
     syntax, i.e. archaizing.  Negative means the reverse: old syntax carrying
     modernised vocabulary.

  3. Genre correction (Strategy B).  Features whose between-genre variance
     (legal vs narrative, within the Torah) is large relative to their temporal
     variance are down-weighted by w_j = 1/(1+gamma_j).  The training corpus
     contains no legal prose, so legal-register texts are the ones at risk.

Outputs: results_v2/archaism_genre_v2.csv
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

RIDGE, RIDGE_RES = 0.10, 0.20
GRID = np.linspace(900, -100, 601)
D_CBH, D_LBH = 720.0, 250.0

PRESPEC = ["frac_ani","frac_she","frac_ein","rate_wayyiqtol","rate_qatal",
           "frac_niphal","rate_inf_con","rate_gam","rate_ut_nouns",
           "frac_infc","frac_fronted","frac_null_subj","frac_wqtl_wayq"]
RESISTANT = ["frac_infc","frac_fronted","frac_null_subj","frac_wqtl_wayq"]

LEGAL     = ["D_Code", "Lev_Priestly", "Lev_Holiness"]
NARRATIVE = ["Gen_JE", "Exo_JE", "Num_JE"]


def ols(df, feats):
    d = df["date_bce"].values.astype(float)
    A, B, sd = [], [], []
    for f in feats:
        y = pd.to_numeric(df[f], errors="coerce").values.astype(float)
        m = np.isfinite(y)
        sl, ic, *_ = stats.linregress(d[m], y[m])
        A.append(ic); B.append(sl)
        sd.append(float(np.nanstd(y[m] - (ic + sl * d[m]))))
    return np.array(A), np.array(B), np.array(sd)


def cov(df, feats, ridge):
    d = df["date_bce"].values.astype(float)
    res = []
    for f in feats:
        y = pd.to_numeric(df[f], errors="coerce").values.astype(float)
        m = np.isfinite(y)
        sl, ic, *_ = stats.linregress(d[m], y[m])
        r = np.full(len(d), np.nan); r[m] = y[m] - (ic + sl * d[m]); res.append(r)
    R = np.nan_to_num(np.vstack(res).T)
    S = (R.T @ R) / max(len(d) - 2, 1)
    return S + ridge * (np.trace(S) / len(feats)) * np.eye(len(feats))


def post_map(x, A, B, S, W=None):
    m = np.isfinite(x)
    if m.sum() < 2:
        return np.nan, (np.nan, np.nan)
    Ssub = S[np.ix_(m, m)]
    Si = np.linalg.inv(Ssub)
    if W is not None:
        Wm = np.diag(W[m]); Si = Wm @ Si @ Wm
    ll = np.array([-0.5 * (dv := (x - (A + B * g))[m]) @ Si @ dv for g in GRID])
    ll -= ll.max()
    p = np.exp(ll - 0.5 * ((GRID - 464.0) / 300.0) ** 2)
    p /= p.sum()
    o = np.argsort(p)[::-1]; c = np.cumsum(p[o]); sel = GRID[o[c <= 0.68]]
    return float(GRID[p.argmax()]), ((float(sel.min()), float(sel.max()))
                                     if len(sel) else (np.nan, np.nan))


def main():
    df = pd.read_csv(FM, index_col="id")
    tr = df[(df["in_training"] == True) & (~df["holdout"].astype(bool))]   # noqa: E712
    feats = [f for f in PRESPEC if f in df.columns]
    res_f = [f for f in RESISTANT if f in df.columns]

    A, B, sd = ols(tr, feats)
    S  = cov(tr, feats, RIDGE)
    Ar, Br, _ = ols(tr, res_f)
    Sr = cov(tr, res_f, RIDGE_RES)

    # ── Genre ratios (legal vs narrative within Torah) ────────────────────────
    gam = {}
    for j, f in enumerate(feats):
        lg = pd.to_numeric(df.loc[[u for u in LEGAL if u in df.index], f],
                           errors="coerce").values.astype(float)
        nr = pd.to_numeric(df.loc[[u for u in NARRATIVE if u in df.index], f],
                           errors="coerce").values.astype(float)
        lg, nr = lg[np.isfinite(lg)], nr[np.isfinite(nr)]
        if len(lg) < 2 or len(nr) < 2 or sd[j] <= 0:
            gam[f] = 0.0; continue
        between = (np.mean(lg) - np.mean(nr)) ** 2
        temporal = (B[j] * (tr["date_bce"].max() - tr["date_bce"].min())) ** 2
        gam[f] = float(between / temporal) if temporal > 0 else 0.0
    W = np.array([1.0 / (1.0 + gam[f]) for f in feats])

    print("Genre ratios (legal vs narrative, within Torah)")
    print(f"  {'feature':16s}{'gamma':>9s}{'weight':>9s}")
    for f in sorted(gam, key=lambda k: -gam[k]):
        flag = "  <- genre-confounded" if gam[f] > 1.5 else ""
        print(f"  {f:16s}{gam[f]:>9.2f}{1/(1+gam[f]):>9.2f}{flag}")

    # ── Per-text results ──────────────────────────────────────────────────────
    rows = []
    for uid, r in df.iterrows():
        x  = pd.to_numeric(r[feats], errors="coerce").values.astype(float)
        xr = pd.to_numeric(r[res_f], errors="coerce").values.astype(float)
        mf, (lo, hi) = post_map(x, A, B, S)
        mr, _        = post_map(xr, Ar, Br, Sr)
        mg, (glo, ghi) = post_map(x, A, B, S, W=W)
        # LBH archaism score
        mu_c = A + B * D_CBH; mu_l = A + B * D_LBH
        den = mu_l - mu_c
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(np.abs(den) > 1e-12, (x - mu_c) / den, np.nan)
        sbar = float(np.nanmean(s))
        cls = ("Archaizing" if sbar < 0 else "Archaic/SBH" if sbar < 0.30
               else "Transitional" if sbar <= 0.65 else "Modern/LBH")
        rows.append(dict(unit=uid, n_words=int(r["n_words"]),
                         role=("holdout" if r["holdout"] else
                               "hbvi_holdout" if r["hbvi_holdout"] else
                               "train" if r["in_training"] else "target"),
                         scholarly=r["date_bce"],
                         map_full=round(mf), ci68_lo=round(lo), ci68_hi=round(hi),
                         map_resist=round(mr),
                         delta_arch=round(mf - mr),
                         map_genre=round(mg),
                         genre_shift=round(mg - mf),
                         lbh_score=round(sbar, 3), lbh_class=cls))
    out = pd.DataFrame(rows).set_index("unit")
    out.to_csv(os.path.join(OUTDIR, "archaism_genre_v2.csv"))

    print("\n\nArchaism diagnostic  (delta_arch = full - resistant)")
    print(f"  {'unit':14s}{'full':>7s}{'resist':>8s}{'delta':>8s}{'LBH s':>8s}  class")
    for u in ["Song_Sea","Song_Deborah","D_Song","D_Code","D_Frame","Lev_Holiness",
              "Lev_Priestly","P_source","JE_source","D_source","Jer_oracle","Jer_DTR"]:
        if u in out.index:
            r = out.loc[u]
            print(f"  {u:14s}{r['map_full']:>7.0f}{r['map_resist']:>8.0f}"
                  f"{r['delta_arch']:>+8.0f}{r['lbh_score']:>8.2f}  {r['lbh_class']}")

    print("\n\nGenre correction (Strategy B)")
    print(f"  {'unit':14s}{'uncorrected':>13s}{'corrected':>11s}{'shift':>8s}")
    for u in ["D_source","D_Code","Lev_Priestly","Lev_Holiness",
              "P_source","JE_source","Song_Sea"]:
        if u in out.index:
            r = out.loc[u]
            print(f"  {u:14s}{r['map_full']:>13.0f}{r['map_genre']:>11.0f}"
                  f"{r['genre_shift']:>+8.0f}")

    print(f"\nSaved → {OUTDIR}/archaism_genre_v2.csv")


if __name__ == "__main__":
    main()
