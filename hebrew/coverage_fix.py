"""
Two routes to range coverage, tested honestly.

WHY REGRESSION CALIBRATION CANNOT GET THERE
-------------------------------------------
Fitting truth ~ a + s*pred gives s = cov(t,p)/var(p) = rho*sd(t)/sd(p), so the
calibrated predictions have sd = rho*sd(t) and coverage = rho exactly.  This is
the MSE-optimal answer -- shrinkage toward the mean is correct under squared
loss -- but it means coverage is capped at the correlation, which is ~0.65 here.
Observed: mean |coverage - rho| = 0.054 across configurations.

ROUTE 1  VARIANCE MATCHING (classical / inverse calibration)
    Rescale so sd(pred) = sd(truth) exactly.  Coverage -> 1.0 by construction.
    Costs MSE, because it deliberately abandons the shrinkage that minimises it.
    Standard in errors-in-variables and calibration-curve settings.

ROUTE 2  INVERSE-DENSITY WEIGHTING of the training books
    Weight each book by 1/f(date) so the effective design is uniform in date
    rather than concentrated where books happen to cluster.  Weighted ridge is
    ordinary ridge on sqrt(w)-scaled rows.  This attacks a different problem --
    uneven design density -- and can only help coverage by raising rho.

All scale factors and weights are computed from TRAINING books only, inside
each outer fold.
"""
import numpy as np, pandas as pd, sys, json
from scipy import stats

META = {"chunk_id", "unit", "date_bce", "genre", "register", "n_words"}


def load(target, kind):
    parts = []
    if kind in ("base", "both"):
        parts.append(pd.read_csv(f"/home/claude/big_features_{target}.csv"))
    if kind in ("ngram", "both"):
        d = pd.read_csv(f"/home/claude/ngram_features_{target}.csv")
        if parts:
            d = d.drop(columns=[c for c in d.columns if c in META and c != "chunk_id"])
        parts.append(d)
    D = parts[0] if len(parts) == 1 else parts[0].merge(parts[1], on="chunk_id")
    X = D[[c for c in D.columns if c not in META]].astype(float)
    keep = (X.std() > 0) & (X.isna().mean() < 0.2)
    X = X.loc[:, keep]; X = X.fillna(X.median())
    return D, X.values


def book_weights(dates_by_book, alpha, bw=90.0):
    """w ∝ (1/KDE(date))^alpha, normalised to mean 1."""
    d = np.asarray(dates_by_book, float)
    if alpha == 0: return np.ones_like(d)
    dens = np.array([np.exp(-0.5*((d-x)/bw)**2).sum() for x in d])
    w = (1.0/np.maximum(dens, 1e-9))**alpha
    return w/w.mean()


def fit_predict(Xtr, ytr, wtr, Xte, lam):
    mu, sd = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1.0)
    A = (Xtr-mu)/sd; B = (Xte-mu)/sd
    yb = np.average(ytr, weights=wtr)
    s = np.sqrt(wtr)[:, None]
    Aw = A*s; yw = (ytr-yb)*np.sqrt(wtr)
    K = Aw@Aw.T
    al = np.linalg.solve(K + lam*np.eye(K.shape[0]), yw)
    return (B@Aw.T)@al + yb


def evaluate(target, kind, mode, alpha, lam_grid):
    D, X = load(target, kind)
    y = D.date_bce.values.astype(float); g = D.unit.values
    books = list(pd.unique(g))
    bdate = {b: y[g == b][0] for b in books}
    rows = []
    for b in books:
        te = g == b; tr = ~te
        Xtr, ytr, gtr = X[tr], y[tr], g[tr]
        inner = [x for x in books if x != b]
        wb = dict(zip(inner, book_weights([bdate[x] for x in inner], alpha)))
        wtr = np.array([wb[u] for u in gtr])

        best, blam, boof = np.inf, lam_grid[0], None
        for lam in lam_grid:
            ot, op = [], []
            for bb in inner:
                m = gtr != bb
                pr = fit_predict(Xtr[m], ytr[m], wtr[m], Xtr[~m], lam)
                ot.append(ytr[~m][0]); op.append(np.median(pr))
            e = np.mean(np.abs(np.array(ot)-np.array(op)))
            if e < best: best, blam, boof = e, lam, (np.array(ot), np.array(op))

        pr = fit_predict(Xtr, ytr, wtr, X[te], blam)
        pred = float(np.median(pr))
        ot, op = boof
        if mode == "reg":                       # coverage -> rho
            v = op.var()
            s_ = float(np.cov(ot, op, bias=True)[0,1]/v) if v > 1e-9 else 1.0
            pred = float(ot.mean() + np.clip(s_, .5, 6.) * (pred - op.mean()))
        elif mode == "var":                     # coverage -> 1.0
            s_ = float(ot.std()/op.std()) if op.std() > 1e-9 else 1.0
            pred = float(ot.mean() + np.clip(s_, .5, 8.) * (pred - op.mean()))
        rows.append(dict(unit=b, truth=bdate[b], pred=pred))
    R = pd.DataFrame(rows); R["err"] = (R.truth-R.pred).abs()
    rho = stats.spearmanr(R.truth, R.pred)
    c = t = 0
    for i in range(len(R)):
        for j in range(i+1, len(R)):
            if R.truth[i] == R.truth[j]: continue
            t += 1
            c += np.sign(R.truth[i]-R.truth[j]) == np.sign(R.pred[i]-R.pred[j])
    return dict(target=target, kind=kind, mode=mode, alpha=alpha,
                mae=R.err.mean(), rho=rho.statistic, pair=c/t,
                cov=R.pred.std()/R.truth.std(),
                pmax=R.pred.max(), pmin=R.pred.min()), R


if __name__ == "__main__":
    LAM = 10.0**np.arange(1, 6.1, 1.0)
    CFG = [(1000, "both"), (500, "base")]
    res = []
    for target, kind in CFG:
        for mode in ("none", "reg", "var"):
            for alpha in (0.0, 0.5, 1.0):
                if mode == "none" and alpha == 0.5: continue
                m, R = evaluate(target, kind, mode, alpha, LAM)
                res.append(m)
                print(f"~{target}w {kind:<5} {mode:<5} w-alpha={alpha:<4} | "
                      f"MAE {m['mae']:6.1f}  rho {m['rho']:+.3f}  pair {m['pair']*100:4.1f}%  "
                      f"cov {m['cov']:.2f}  range [{m['pmax']:.0f},{m['pmin']:.0f}]", flush=True)
                R.to_csv(f"/home/claude/cov_{target}_{kind}_{mode}_{alpha}.csv", index=False)
                with open("/home/claude/coverage_fix.jsonl", "a") as fh:
                    fh.write(json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                         for k, v in m.items()})+"\n")
    pd.DataFrame(res).to_csv("/home/claude/coverage_fix_summary.csv", index=False)
