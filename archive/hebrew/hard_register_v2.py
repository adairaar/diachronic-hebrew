#!/usr/bin/env python3
"""
hard_register_v2.py — register-conditioned MLE-MVN on corpus v2
================================================================
Per-register OLS + Tikhonov MVN, using the pre-specified literature feature
set (no screening), matching the design agreed for dating_v3.py.

Registers are taken as hard assignments from corpus_manifest_v2.json, so this
does not depend on 02_register_classifier.py.

IMPORTANT CAVEAT, reported in the output: per-register training counts are
small (SBH 6, Transitional 9, LBH 6 after holdouts are removed).  With 13
features this is J > n within every group, so each group covariance is
rank-deficient and the ridge is carrying the model.  Group-level MAPs should
be read as heavily regularised, not as independent estimates.  This is almost
certainly why the v1 register-conditioned model produced extreme values
(Haggai 361 BCE, D_source 54 BCE).

Outputs: results_v2/hard_register_v2.csv
"""

from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from scipy import stats

HERE   = os.path.dirname(os.path.abspath(__file__))
FM     = os.path.join(HERE, "data", "feature_matrix_v2.csv")
OUTDIR = os.path.join(HERE, "results_v2")
os.makedirs(OUTDIR, exist_ok=True)

RIDGE = 0.10
GRID  = np.linspace(900, -100, 601)
GROUPS = ["SBH", "Transitional", "LBH"]

PRESPEC = ["frac_ani","frac_she","frac_ein","rate_wayyiqtol","rate_qatal",
           "frac_niphal","rate_inf_con","rate_gam","rate_ut_nouns",
           "frac_infc","frac_fronted","frac_null_subj","frac_wqtl_wayq"]


class GroupMVN:
    def __init__(self, name, df, feats, ridge=RIDGE):
        self.name, self.feats = name, list(feats)
        d = df["date_bce"].values.astype(float)
        self.n, self.dmin, self.dmax = len(d), d.min(), d.max()
        A, B, res = [], [], []
        for f in self.feats:
            y = pd.to_numeric(df[f], errors="coerce").values.astype(float)
            m = np.isfinite(y)
            if m.sum() < 3 or np.nanstd(y[m]) < 1e-12:
                A.append(np.nanmean(y[m]) if m.sum() else 0.0); B.append(0.0)
                r = np.zeros(self.n)
            else:
                sl, ic, *_ = stats.linregress(d[m], y[m])
                A.append(ic); B.append(sl)
                r = np.full(self.n, np.nan); r[m] = y[m] - (ic + sl * d[m])
            res.append(r)
        self.A, self.B = np.array(A), np.array(B)
        R = np.nan_to_num(np.vstack(res).T)
        S = (R.T @ R) / max(self.n - 2, 1)
        S += ridge * (np.trace(S) / len(self.feats) + 1e-9) * np.eye(len(self.feats))
        self.S = S
        ev = np.linalg.eigvalsh(S)
        self.rank = int((ev > 1e-10 * ev.max()).sum())

    def loglik(self, x):
        m = np.isfinite(x)
        if m.sum() < 2:
            return None
        Si = np.linalg.inv(self.S[np.ix_(m, m)])
        out = np.empty(len(GRID))
        for i, g in enumerate(GRID):
            diff = (x - (self.A + self.B * g))[m]
            out[i] = -0.5 * diff @ Si @ diff
        return out


def hdi(post, lvl=0.68):
    o = np.argsort(post)[::-1]; c = np.cumsum(post[o]); sel = GRID[o[c <= lvl]]
    return (float(sel.min()), float(sel.max())) if len(sel) else (np.nan, np.nan)


def main():
    df = pd.read_csv(FM, index_col="id")
    feats = [f for f in PRESPEC if f in df.columns]
    tr_all = df[(df["in_training"] == True) &                       # noqa: E712
                (~df["holdout"].astype(bool)) &
                (~df["hbvi_holdout"].astype(bool))]

    print(f"Pre-specified features: {len(feats)}\n")
    models = {}
    print("Per-register training sets")
    for g in GROUPS:
        sub = tr_all[tr_all["register"] == g]
        models[g] = GroupMVN(g, sub, feats)
        print(f"  {g:14s} n={len(sub):2d}  J={len(feats)}  "
              f"cov rank={models[g].rank}  "
              f"range {sub['date_bce'].max():.0f}–{sub['date_bce'].min():.0f} BCE")
        print(f"      {', '.join(sub.index.tolist())}")
    print("\n  NOTE: J > n in every group; covariances are rank-deficient and")
    print("  the ridge is doing the work. Read group MAPs as regularised.\n")

    rows = []
    for uid, r in df.iterrows():
        x = pd.to_numeric(r[feats], errors="coerce").values.astype(float)
        known = r["register"] if isinstance(r["register"], str) and r["register"] in GROUPS else None
        best, bestpost, bestll = None, None, -np.inf
        for g in GROUPS:
            ll = models[g].loglik(x)
            if ll is None:
                continue
            marg = float(np.logaddexp.reduce(ll))       # marginal over the grid
            lp = -0.5 * ((GRID - 464.0) / 300.0) ** 2
            p = np.exp(ll - ll.max() + lp); s = p.sum()
            if s <= 0:
                continue
            p = p / s
            if marg > bestll:
                best, bestpost, bestll = g, p, marg
            if known == g:
                kp = p
        if bestpost is None:
            continue
        use_g, use_p = (known, kp) if known else (best, bestpost)
        mp = GRID[use_p.argmax()]; lo, hi = hdi(use_p)
        mb = GRID[bestpost.argmax()]
        rows.append(dict(unit=uid, n_words=int(r["n_words"]),
                         role=("holdout" if r["holdout"] else
                               "hbvi_holdout" if r["hbvi_holdout"] else
                               "train" if r["in_training"] else "target"),
                         scholarly=r["date_bce"],
                         assigned_group=use_g, best_group=best,
                         map_bce=round(mp), ci68_lo=round(lo), ci68_hi=round(hi),
                         map_bestgroup=round(mb)))
    out = pd.DataFrame(rows).set_index("unit")
    out.to_csv(os.path.join(OUTDIR, "hard_register_v2.csv"))

    t = out[out["role"] == "train"].dropna(subset=["scholarly"])
    print(f"In-sample MAE (own register): "
          f"{(t['map_bce']-t['scholarly']).abs().mean():.1f} yr\n")

    print("Holdouts (best-group selection, no register given)")
    for u in ["Habakkuk", "Jer_oracle", "Haggai", "Daniel"]:
        if u in out.index:
            r = out.loc[u]
            print(f"  {u:12s} scholarly {r['scholarly']:>4.0f}  "
                  f"best-group MAP {r['map_bestgroup']:>4.0f}  "
                  f"err {r['map_bestgroup']-r['scholarly']:+5.0f}  "
                  f"(grp {r['best_group']})")

    print("\nTargets (best-group selection)")
    for u in ["P_source","JE_source","D_source","D_Code","D_Frame",
              "Lev_Priestly","Lev_Holiness","Song_Sea","Jer_DTR"]:
        if u in out.index:
            r = out.loc[u]
            print(f"  {u:14s} MAP {r['map_bestgroup']:>4.0f}  "
                  f"[{r['ci68_lo']:.0f}–{r['ci68_hi']:.0f}]  grp {r['best_group']}")

    print(f"\nSaved → {OUTDIR}/hard_register_v2.csv")


if __name__ == "__main__":
    main()
