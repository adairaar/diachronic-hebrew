"""
Honest dating of undated targets (P, D, JE, Torah books, poems) with
CONFORMAL prediction intervals calibrated on leave-one-out residuals.

Why conformal: the pipeline's parametric intervals achieve 52% coverage at a
nominal 68%. Conformal intervals are distribution-free and have guaranteed
finite-sample marginal coverage under exchangeability, so "does this interval
stay inside one period" becomes a statement that means something.

Predictive distribution for a target = point prediction + the empirical set of
signed LOO residuals. Period probabilities are read off that distribution.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import numpy as np, pandas as pd
from scipy import stats
import json

MATRIX = DH.f("feature_matrix_v2.csv")
PERIODS = ["Pre-exilic", "Exilic", "Persian", "Hellenistic"]
BOUND = [586, 539, 332]
def to_period(d):
    return 0 if d > 586 else 1 if d > 539 else 2 if d > 332 else 3
GRID = np.linspace(900.0, 100.0, 401)
ALPHA = 0.05


def load():
    df = pd.read_csv(MATRIX)
    meta = {"id","date_bce","date_sigma","register","genre","holdout",
            "hbvi_holdout","n_words","in_training"}
    feats = [c for c in df.columns if c not in meta]
    X = df[feats].astype(float).values
    ok = np.isfinite(X).all(0) & (X.std(0) > 0)
    return df, np.array(feats)[ok], X[:, ok]


def screen(Ztr, dtr, alpha):
    n = len(dtr)
    rx = stats.rankdata(Ztr, axis=0); ry = stats.rankdata(dtr)
    rxc = rx - rx.mean(0); ryc = ry - ry.mean()
    den = np.sqrt((rxc**2).sum(0)*(ryc**2).sum()); den = np.where(den>0,den,np.inf)
    rho = np.clip((rxc*ryc[:,None]).sum(0)/den, -0.9999999, 0.9999999)
    t = rho*np.sqrt((n-2)/(1-rho**2))
    return np.flatnonzero(2*stats.t.sf(np.abs(t), n-2) < alpha)


def fit_generative(Xtr, dtr, xte, alpha=ALPHA):
    n = len(dtr)
    mu, sd = Xtr.mean(0), np.where(Xtr.std(0)>0, Xtr.std(0), 1.0)
    Ztr, zte = (Xtr-mu)/sd, (xte-mu)/sd
    keep = screen(Ztr, dtr, alpha)
    if keep.size == 0: return np.nan
    d0, ds = dtr.mean(), (dtr.std() or 1.0)
    t = (dtr-d0)/ds; tc = t-t.mean(); Y = Ztr[:,keep]
    b = (tc[:,None]*(Y-Y.mean(0))).sum(0)/(tc**2).sum()
    a = Y.mean(0)-b*t.mean()
    s = np.maximum(np.sqrt(((Y-(a+np.outer(t,b)))**2).sum(0)/max(n-2,1)), 1e-3)
    tg = (GRID-d0)/ds
    pred = a[None,:]+np.outer(tg,b)
    ll = (-0.5*((zte[keep][None,:]-pred)/s[None,:])**2 - np.log(s)[None,:]).sum(1)
    ll -= ll.max(); p = np.exp(ll)
    return float(GRID[np.argmax(p)])


def fit_ridge(Xtr, dtr, xte, alpha=ALPHA):
    mu, sd = Xtr.mean(0), np.where(Xtr.std(0)>0, Xtr.std(0), 1.0)
    Ztr, zte = (Xtr-mu)/sd, (xte-mu)/sd
    keep = screen(Ztr, dtr, alpha)
    if keep.size == 0: return np.nan
    A, y = Ztr[:,keep], dtr-dtr.mean(); n,k = A.shape
    G = A.T@A; best, bl = np.inf, 1.0
    for lam in 10.0**np.arange(-2,4.1,0.5):
        H = A@np.linalg.solve(G+lam*np.eye(k), A.T)
        h = np.clip(np.diag(H),0,1-1e-9)
        cv = (((y-H@y)/(1-h))**2).mean()
        if cv < best: best, bl = cv, lam
    w = np.linalg.solve(G+bl*np.eye(k), A.T@y)
    return float(zte[keep]@w + dtr.mean())


FAM = {"generative": fit_generative, "ridge": fit_ridge}


def main():
    df, feats, X = load()
    ids = df["id"].values
    dated = df["date_bce"].notna().values
    dtr_all = df.loc[dated, "date_bce"].values.astype(float)
    Xd = X[dated]
    print(f"{dated.sum()} dated calibration texts | {(~dated).sum()} undated targets "
          f"| {X.shape[1]} features\n")

    out = {}
    for fam, fn in FAM.items():
        # ── 1. LOO residuals on the dated texts -> conformal calibration set ──
        n = dated.sum()
        loo = np.array([fn(Xd[np.arange(n)!=i], dtr_all[np.arange(n)!=i], Xd[i])
                        for i in range(n)])
        resid = dtr_all - loo                       # signed: true - predicted
        absr = np.sort(np.abs(resid))
        q68 = absr[min(int(np.ceil((n+1)*0.68))-1, n-1)]
        q90 = absr[min(int(np.ceil((n+1)*0.90))-1, n-1)]

        # empirical coverage check of the conformal band on the calibration set
        cov68 = (np.abs(resid) <= q68).mean(); cov90 = (np.abs(resid) <= q90).mean()
        print(f"── {fam} ─────────────────────────────────────────────────")
        print(f"  LOO MAE {np.abs(resid).mean():.1f} yr   bias {resid.mean():+.1f} yr")
        print(f"  conformal half-width: 68% = +/-{q68:.0f} yr, 90% = +/-{q90:.0f} yr")
        print(f"  achieved coverage on calibration set: 68%->{cov68:.0%}  90%->{cov90:.0%}")
        rs = stats.spearmanr(df.loc[dated,'n_words'].values, np.abs(resid))
        print(f"  |residual| vs log word count: rho={rs.statistic:+.2f}, p={rs.pvalue:.3f}"
              f"  {'(size-dependent - flag)' if rs.pvalue<0.05 else '(no size dependence)'}")

        # ── 2. fit on ALL dated texts, predict every target ──
        rows = []
        for i in range(len(ids)):
            if dated[i]: continue
            pt = fn(Xd, dtr_all, X[i])
            draws = pt + resid                       # empirical predictive dist.
            probs = np.array([np.mean([to_period(v)==k for v in draws]) for k in range(4)])
            lo68, hi68 = pt-q68, pt+q68
            rows.append(dict(id=ids[i], n_words=int(df["n_words"].values[i]),
                             point=pt, lo68=lo68, hi68=hi68,
                             lo90=pt-q90, hi90=pt+q90,
                             span68=";".join(PERIODS[k] for k in
                                 range(to_period(hi68), to_period(lo68)+1)),
                             n_periods68=to_period(lo68)-to_period(hi68)+1,
                             **{f"P_{PERIODS[k]}": probs[k] for k in range(4)},
                             top_period=PERIODS[int(np.argmax(probs))],
                             top_prob=float(probs.max()),
                             p_post_exilic=float(np.mean(draws <= 586))))
        out[fam] = pd.DataFrame(rows)
        print()

    # ── report ──
    for fam in FAM:
        t = out[fam].sort_values("point", ascending=False)
        print("="*104)
        print(f"TARGETS — {fam}   (conformal 68% intervals)")
        print("="*104)
        print(f"{'unit':<15}{'words':>7}{'point':>7}{'68% interval':>18}{'periods':>9}"
              f"{'most likely':>14}{'p':>6}{'P(post-exilic)':>16}")
        print("-"*104)
        for _, r in t.iterrows():
            print(f"{r['id']:<15}{r['n_words']:7d}{r['point']:7.0f}"
                  f"{f'[{r.hi68:.0f}, {r.lo68:.0f}]':>18}{int(r['n_periods68']):9d}"
                  f"{r['top_period']:>14}{r['top_prob']:6.2f}{r['p_post_exilic']:16.2f}")
        print()

    merged = out["generative"].merge(out["ridge"], on="id", suffixes=("_gen","_ridge"))
    merged["agree_period"] = merged["top_period_gen"] == merged["top_period_ridge"]
    merged["point_gap"] = (merged["point_gen"]-merged["point_ridge"]).abs()
    print("="*80); print("CROSS-FAMILY AGREEMENT"); print("="*80)
    print(f"{'unit':<15}{'generative':>14}{'ridge':>14}{'gap':>7}   agree?")
    for _, r in merged.sort_values("point_gap").iterrows():
        print(f"{r['id']:<15}{r['top_period_gen']:>14}{r['top_period_ridge']:>14}"
              f"{r['point_gap']:7.0f}   {'yes' if r['agree_period'] else 'NO'}")
    print(f"\n  period agreement: {merged['agree_period'].mean():.0%} "
          f"({merged['agree_period'].sum()}/{len(merged)})")
    print(f"  median |point difference| between families: {merged['point_gap'].median():.0f} yr")

    for fam in FAM: out[fam].to_csv(DH.f(f"targets_{fam}.csv"), index=False)
    print("\nwrote targets_generative.csv, targets_ridge.csv")


if __name__ == "__main__":
    main()
