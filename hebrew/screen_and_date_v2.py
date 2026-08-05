#!/usr/bin/env python3
"""
screen_and_date_v2.py — feature screening sweep + MLE-MVN refit on corpus v2
============================================================================
Two jobs:

 1. SWEEP.  Report how many features survive at each (p-threshold x LOO-gate)
    combination, using the *sign-consistency* LOO definition the manuscript
    claims (fraction of leave-one-out refits preserving the sign of rho),
    not the significance-retention definition some scripts actually used.

 2. REFIT.  Fit MLE-MVN at the chosen operating point and date every target.

Audit note (2026-08): the manuscript stated p<0.30 and a 68% sign-consistency
LOO gate.  In the code, 11_comprehensive_dating used p<0.10 with LOO=1.00,
hebrew/03_hard_register_dating used p<0.25 with a 0.30 gate on a *different*
statistic (significance retention), and 06_feature_mining used p<0.10 with a
0.75 advisory flag.  This script establishes a single documented criterion.
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

META = {"date_bce","date_sigma","register","genre","holdout",
        "hbvi_holdout","in_training","n_words"}

# Operating point (documented, single source of truth)
P_THRESH  = 0.10     # matches 11_comprehensive_dating.py and the original design
LOO_GATE  = 0.68     # sign-consistency floor (1 sd of a binomial sign test)
RIDGE     = 0.10     # full model
RIDGE_RES = 0.20     # resistant model
TIER3 = ["frac_infc","frac_fronted","frac_null_subj","frac_wqtl_wayq"]


def loo_sign_consistency(dates, vals):
    """Fraction of leave-one-out refits preserving the sign of full-sample rho."""
    ok = np.isfinite(dates) & np.isfinite(vals)
    d, v = dates[ok], vals[ok]
    if len(d) < 6 or np.nanstd(v) < 1e-12:
        return np.nan, np.nan, np.nan
    rho_full, p_full = stats.spearmanr(d, v)
    if not np.isfinite(rho_full):
        return np.nan, np.nan, np.nan
    s_full = np.sign(rho_full)
    keep = 0
    for i in range(len(d)):
        r, _ = stats.spearmanr(np.delete(d, i), np.delete(v, i))
        if np.isfinite(r) and np.sign(r) == s_full:
            keep += 1
    return float(rho_full), float(p_full), keep / len(d)


def screen(df_tr):
    dates = df_tr["date_bce"].values.astype(float)
    recs = []
    for c in [c for c in df_tr.columns if c not in META]:
        vals = pd.to_numeric(df_tr[c], errors="coerce").values.astype(float)
        if np.isfinite(vals).sum() < 6:
            continue
        rho, p, loo = loo_sign_consistency(dates, vals)
        if rho is None or not np.isfinite(rho):
            continue
        recs.append(dict(feature=c, rho=rho, p_raw=p, loo_sign=loo,
                         n_obs=int(np.isfinite(vals).sum())))
    return pd.DataFrame(recs).sort_values("p_raw")


def fit_mvn(df_tr, feats, ridge):
    dates = df_tr["date_bce"].values.astype(float)
    A, B, resid = [], [], []
    for f in feats:
        y = pd.to_numeric(df_tr[f], errors="coerce").values.astype(float)
        m = np.isfinite(y)
        sl, ic, *_ = stats.linregress(dates[m], y[m])
        A.append(ic); B.append(sl)
        full = np.full(len(dates), np.nan)
        full[m] = y[m] - (ic + sl * dates[m])
        resid.append(full)
    A = np.array(A); B = np.array(B)
    R = np.vstack(resid).T
    Rf = np.where(np.isfinite(R), R, 0.0)
    S = (Rf.T @ Rf) / max(len(dates) - 1, 1)
    S = S + ridge * (np.trace(S) / len(feats)) * np.eye(len(feats))
    return A, B, np.linalg.inv(S)


def posterior(x, A, B, Sinv, grid, prior_mu=464.0, prior_sd=300.0, wscale=1.0):
    m = np.isfinite(x)
    ll = np.empty(len(grid))
    for i, d in enumerate(grid):
        diff = (x - (A + B * d))[m]
        ll[i] = -0.5 * wscale * diff @ Sinv[np.ix_(m, m)] @ diff
    ll = ll - ll.max()
    lp = -0.5 * ((grid - prior_mu) / prior_sd) ** 2
    post = np.exp(ll + lp)
    return post / post.sum()


def hdci(grid, post, lvl=0.68):
    o = np.argsort(post)[::-1]
    c = np.cumsum(post[o])
    sel = grid[o[c <= lvl]]
    return (float(sel.min()), float(sel.max())) if len(sel) else (np.nan, np.nan)


def main():
    df = pd.read_csv(FM, index_col="id")
    tr = df[df["in_training"] == True]                      # noqa: E712
    print(f"Training units: {len(tr)}   ({tr['date_bce'].min():.0f}"
          f"–{tr['date_bce'].max():.0f} BCE)\n")

    sc = screen(tr)
    sc.to_csv(os.path.join(OUTDIR, "feature_screen_v2.csv"), index=False)

    # ── 1. SWEEP ──────────────────────────────────────────────────────────────
    print("Feature survivors — corpus v2 (sign-consistency LOO)")
    print(f"{'p<':>6s}" + "".join(f"{g:>13s}" for g in
          ["LOO>=0.50","LOO>=0.68","LOO>=0.75","LOO>=0.90","LOO=1.00"]))
    sweep = []
    for p in [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        sub = sc[sc["p_raw"] < p]
        row = [int((sub["loo_sign"] >= g).sum()) for g in (0.50,0.68,0.75,0.90,1.0)]
        sweep.append([p] + row)
        print(f"{p:>6.2f}" + "".join(f"{v:>13d}" for v in row))
    pd.DataFrame(sweep, columns=["p","loo50","loo68","loo75","loo90","loo100"]).to_csv(
        os.path.join(OUTDIR, "screen_sweep_v2.csv"), index=False)

    # ── 2. REFIT at the documented operating point ────────────────────────────
    sel = sc[(sc["p_raw"] < P_THRESH) & (sc["loo_sign"] >= LOO_GATE)]["feature"].tolist()
    res_feats = [f for f in TIER3 if f in sc["feature"].values]
    print(f"\nOperating point: p<{P_THRESH}, LOO_sign>={LOO_GATE}"
          f"  ->  {len(sel)} features")
    print(f"Resistant model: {len(res_feats)} Tier-3 features\n")

    A, B, Si = fit_mvn(tr, sel, RIDGE)
    Ar, Br, Sir = fit_mvn(tr, res_feats, RIDGE_RES)
    grid = np.linspace(900, -100, 500)     # 900 BCE .. 100 CE

    rows = []
    for uid, r in df.iterrows():
        xs = pd.to_numeric(r[sel],       errors="coerce").values.astype(float)
        xr = pd.to_numeric(r[res_feats], errors="coerce").values.astype(float)
        w  = float(np.clip(r["n_words"] / 5000.0, 1, 5))
        pf = posterior(xs, A, B, Si,  grid, wscale=w)
        pr = posterior(xr, Ar, Br, Sir, grid, wscale=w)
        mf, mr = grid[pf.argmax()], grid[pr.argmax()]
        lo, hi = hdci(grid, pf)
        rows.append(dict(unit=uid, n_words=int(r["n_words"]),
                         role=("holdout" if r["holdout"] else
                               "hbvi_holdout" if r["hbvi_holdout"] else
                               "train" if r["in_training"] else "target"),
                         scholarly=r["date_bce"],
                         map_full=round(mf), ci68_lo=round(lo), ci68_hi=round(hi),
                         map_resist=round(mr), delta_arch=round(mf - mr)))
    out = pd.DataFrame(rows).set_index("unit")
    out.to_csv(os.path.join(OUTDIR, "dating_v2.csv"))

    print("── Training fit ──")
    t = out[out["role"].isin(["train","hbvi_holdout"])].dropna(subset=["scholarly"])
    err = (t["map_full"] - t["scholarly"]).abs()
    print(f"  in-sample MAE = {err.mean():.1f} yr   RMSE = {np.sqrt((err**2).mean()):.1f} yr\n")

    print("── Holdouts ──")
    for u in ["Jer_oracle","Haggai","Habakkuk","Daniel"]:
        if u in out.index:
            r = out.loc[u]
            print(f"  {u:12s} scholarly={r['scholarly']:>5.0f}  "
                  f"MAP={r['map_full']:>5.0f}  err={r['map_full']-r['scholarly']:+6.0f} yr")

    print("\n── Torah sources & key targets ──")
    for u in ["P_source","D_source","JE_source","D_Code","D_Frame",
              "Lev_Holiness","Lev_Priestly","Song_Sea","Song_Deborah","D_Song"]:
        if u in out.index:
            r = out.loc[u]
            print(f"  {u:14s} MAP={r['map_full']:>5.0f} "
                  f"[{r['ci68_lo']:.0f}–{r['ci68_hi']:.0f}]  "
                  f"resist={r['map_resist']:>5.0f}  D_arch={r['delta_arch']:+6.0f}")

    print(f"\nSaved → {OUTDIR}/")


if __name__ == "__main__":
    main()
