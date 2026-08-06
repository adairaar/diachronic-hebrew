"""
Greek corpus under the identical leakage-free protocol used for Hebrew.

Same two model families, same in-fold screening/standardisation/fitting,
same agnostic prior, same permutation null, same conformal calibration.
Dates are CE (negative = BCE), so time increases with the label.
"""
import numpy as np, pandas as pd
from scipy import stats
import json, sys

RNG = np.random.default_rng(20260806)
G = "/mnt/user-data/uploads/Diachronic Hebrew/greek"
GRID = np.linspace(-600.0, 550.0, 461)


def load():
    d = pd.read_csv(f"{G}/data/features/feature_matrix.csv")
    meta = {"id","author","date_ce","date_sigma","genre","holdout","word_count",
            "register","work","quote_heavy","notes","tlg_ref","first1k_hint"}
    feats = [c for c in d.columns if c not in meta]
    X = d[feats].astype(float).values
    ok = np.isfinite(X).all(0) & (X.std(0) > 0)
    return d, np.array(feats)[ok], X[:, ok]


def screen(Z, y, alpha):
    n = len(y)
    rx, ry = stats.rankdata(Z, axis=0), stats.rankdata(y)
    rxc, ryc = rx - rx.mean(0), ry - ry.mean()
    den = np.sqrt((rxc**2).sum(0)*(ryc**2).sum()); den = np.where(den > 0, den, np.inf)
    rho = np.clip((rxc*ryc[:,None]).sum(0)/den, -0.9999999, 0.9999999)
    t = rho*np.sqrt((n-2)/(1-rho**2))
    return np.flatnonzero(2*stats.t.sf(np.abs(t), n-2) < alpha)


def fit_generative(Xtr, dtr, xte, alpha):
    n = len(dtr)
    mu, sd = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1.0)
    Z, z = (Xtr-mu)/sd, (xte-mu)/sd
    keep = screen(Z, dtr, alpha)
    if keep.size == 0: return np.nan
    d0, ds = dtr.mean(), (dtr.std() or 1.0)
    t = (dtr-d0)/ds; tc = t-t.mean(); Y = Z[:,keep]
    b = (tc[:,None]*(Y-Y.mean(0))).sum(0)/(tc**2).sum(); a = Y.mean(0)-b*t.mean()
    s = np.maximum(np.sqrt(((Y-(a+np.outer(t,b)))**2).sum(0)/max(n-2,1)), 1e-3)
    tg = (GRID-d0)/ds
    ll = (-0.5*((z[keep][None,:]-(a[None,:]+np.outer(tg,b)))/s[None,:])**2
          - np.log(s)[None,:]).sum(1)
    ll -= ll.max(); post = np.exp(ll)
    return float(GRID[np.argmax(post)])


def fit_ridge(Xtr, dtr, xte, alpha):
    mu, sd = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1.0)
    Z, z = (Xtr-mu)/sd, (xte-mu)/sd
    keep = screen(Z, dtr, alpha)
    if keep.size == 0: return np.nan
    A, y = Z[:,keep], dtr-dtr.mean(); n,k = A.shape
    Gm = A.T@A; best, bl = np.inf, 1.0
    for lam in 10.0**np.arange(-2,4.1,0.5):
        H = A@np.linalg.solve(Gm+lam*np.eye(k), A.T)
        h = np.clip(np.diag(H),0,1-1e-9)
        cv = (((y-H@y)/(1-h))**2).mean()
        if cv < best: best, bl = cv, lam
    w = np.linalg.solve(Gm+bl*np.eye(k), A.T@y)
    return float(z[keep]@w + dtr.mean())


FAM = {"generative": fit_generative, "ridge": fit_ridge}


def run_loo(dates, X, alpha, fam):
    f = FAM[fam]; n = len(dates); out = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        try: out[i] = f(X[tr], dates[tr], X[i], alpha)
        except Exception: out[i] = np.nan
    return np.clip(out, GRID.min(), GRID.max())


def pairwise(d, m):
    n=len(d); c=t=0
    for i in range(n):
        for j in range(i+1,n):
            if d[i]==d[j]: continue
            t+=1; c += (np.sign(d[i]-d[j])==np.sign(m[i]-m[j]))
    return c/t, t


def score(d, m):
    ok = np.isfinite(m)
    if ok.sum() < 5: return None
    d, m = d[ok], m[ok]
    rho = stats.spearmanr(d, m).statistic if len(np.unique(m)) > 1 else 0.0
    pw, _ = pairwise(d, m)
    return dict(rho=(0.0 if not np.isfinite(rho) else float(rho)),
                pw=float(pw), mae=float(np.abs(d-m).mean()), n=int(ok.sum()))


def main(alpha=0.05, nperm=500):
    df, feats, X = load()
    dates = df["date_ce"].values.astype(float)
    print(f"GREEK: {len(dates)} texts | {X.shape[1]} usable features | "
          f"{dates.min():.0f} to {dates.max():.0f} CE | alpha={alpha}\n")

    res = {}
    for fam in FAM:
        m = run_loo(dates, X, alpha, fam); S = score(dates, m); res[fam] = (m, S)
        pw, npair = pairwise(dates[np.isfinite(m)], m[np.isfinite(m)])
        pb = stats.binomtest(int(round(pw*npair)), npair, 0.5, alternative="greater").pvalue
        print(f"── {fam} ──")
        print(f"   Spearman rho (LOO)      {S['rho']:+.3f}")
        print(f"   pairwise ordering       {pw*100:.1f}%  of {npair} pairs   p={pb:.2e}")
        print(f"   MAE                     {S['mae']:.1f} yr")
        print(f"   constant-predictor MAE  {np.abs(dates-dates.mean()).mean():.1f} yr"
              f"   (mean {dates.mean():+.0f} CE)")
        resid = dates - m
        absr = np.sort(np.abs(resid[np.isfinite(resid)])); n = len(absr)
        q68 = absr[min(int(np.ceil((n+1)*0.68))-1, n-1)]
        q90 = absr[min(int(np.ceil((n+1)*0.90))-1, n-1)]
        print(f"   conformal half-width    68% +/-{q68:.0f} yr   90% +/-{q90:.0f} yr")
        h = df["holdout"].values.astype(bool)
        print(f"   holdouts under this design (agnostic prior, in-fold everything):")
        for i in np.flatnonzero(h):
            print(f"      {df['id'].values[i]:<28} true {dates[i]:+6.0f}  "
                  f"pred {m[i]:+7.0f}  err {abs(dates[i]-m[i]):5.0f}")
        print(f"      holdout MAE {np.abs(dates[h]-m[h]).mean():.1f} yr\n")

    print(f"permutation null (n={nperm})...", flush=True)
    null = {f: {k: [] for k in ("rho","pw","mae")} for f in FAM}
    for _ in range(nperm):
        dp = RNG.permutation(dates)
        for fam in FAM:
            s = score(dp, run_loo(dp, X, alpha, fam))
            if s:
                for k in null[fam]: null[fam][k].append(s[k])
    print()
    for fam in FAM:
        S = res[fam][1]
        print(f"  --- {fam} ---")
        for k, lab, hi in [("rho","Spearman rho",True), ("pw","pairwise ordering",True),
                           ("mae","MAE (yr)",False)]:
            v = np.array(null[fam][k]); obs = S[k]
            p = ((np.sum(v>=obs) if hi else np.sum(v<=obs))+1)/(len(v)+1)
            print(f"    {lab:<20} observed {obs:8.3f}   null {v.mean():7.3f} +/- {v.std():.3f}"
                  f"   p={p:.4f}{'  *' if p<0.05 else ''}")

    out = pd.DataFrame(dict(id=df["id"], date_ce=dates, holdout=df["holdout"],
                            genre=df["genre"], word_count=df["word_count"],
                            loo_generative=res["generative"][0], loo_ridge=res["ridge"][0]))
    out.to_csv("/home/claude/greek_loo_results.csv", index=False)
    print("\nwrote greek_loo_results.csv")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv)>1 else 0.05,
         int(sys.argv[2]) if len(sys.argv)>2 else 500)
