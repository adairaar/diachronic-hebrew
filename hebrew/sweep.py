"""
Chunk-size x feature-set x model sweep, with explicit de-attenuation.

THE COMPRESSION FIX
-------------------
Ridge shrinks predictions toward the mean, so predicted dates span less range
than true dates.  The correction is a linear recalibration, truth ~ a + s*pred,
with s > 1 expanding the range back.  It must be fitted on TRAINING books only,
using out-of-fold predictions, or it leaks.  So each outer fold runs an inner
leave-one-book-out over its 24 training books, fits the calibration on those
out-of-fold predictions, and applies it to the held-out book.

Ridge is solved in DUAL form, w = A'(AA' + lam I)^-1 y, which is exact and
O(n^3) rather than O(p^3) -- essential at p ~ 1300, n ~ 150-470.

Reported alongside MAE:
  range coverage   sd(pred)/sd(truth); 1.0 means the model can reach the ends
  two-sided calls  whether the model ever makes a CONFIDENT PRE-EXILIC call,
                   which is the failure that sank the previous design
"""
import numpy as np, pandas as pd, itertools, sys, json
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
    feats = [c for c in D.columns if c not in META]
    X = D[feats].astype(float)
    keep = (X.std() > 0) & (X.isna().mean() < 0.2)
    X = X.loc[:, keep]
    X = X.fillna(X.median())
    return D, X.values, X.shape[1]


def dual_ridge_fit(A, y, lam):
    K = A @ A.T
    return np.linalg.solve(K + lam * np.eye(K.shape[0]), y)


def dual_ridge_pred(A, alpha, B):
    return (B @ A.T) @ alpha


def std_fit(A):
    mu = A.mean(0); sd = A.std(0); sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def run_fold(Xtr, ytr, gtr, Xte, lam):
    mu, sd = std_fit(Xtr)
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    yb = ytr.mean()
    al = dual_ridge_fit(A, ytr - yb, lam)
    return dual_ridge_pred(A, al, B) + yb


def evaluate(target, kind, lam_grid, calibrate):
    D, X, p = load(target, kind)
    y = D.date_bce.values.astype(float); g = D.unit.values
    books = list(pd.unique(g))
    rows = []
    for b in books:
        te = g == b; tr = ~te
        Xtr, ytr, gtr = X[tr], y[tr], g[tr]
        inner = [bb for bb in books if bb != b]

        # inner LOBO: pick lambda AND collect out-of-fold predictions
        best, blam, boof = np.inf, lam_grid[0], None
        for lam in lam_grid:
            oof_t, oof_p = [], []
            for bb in inner:
                m = gtr != bb
                pr = run_fold(Xtr[m], ytr[m], gtr[m], Xtr[~m], lam)
                oof_t.append(ytr[~m][0]); oof_p.append(np.median(pr))
            e = np.mean(np.abs(np.array(oof_t) - np.array(oof_p)))
            if e < best:
                best, blam, boof = e, lam, (np.array(oof_t), np.array(oof_p))

        pr = run_fold(Xtr, ytr, gtr, X[te], blam)
        pred = float(np.median(pr)); spread = float(pr.std())

        if calibrate:
            ot, op = boof
            v = op.var()
            s = float(np.cov(ot, op, bias=True)[0, 1] / v) if v > 1e-9 else 1.0
            s = float(np.clip(s, 0.5, 4.0))          # guard against wild slopes
            a = float(ot.mean() - s * op.mean())
            pred = a + s * pred
            spread = spread * abs(s)
        rows.append(dict(unit=b, truth=y[te][0], pred=pred, spread=spread,
                         n_chunks=int(te.sum()), lam=blam))
    R = pd.DataFrame(rows); R["err"] = (R.truth - R.pred).abs()

    rho = stats.spearmanr(R.truth, R.pred)
    c = t = 0
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            if R.truth[i] == R.truth[j]: continue
            t += 1
            c += np.sign(R.truth[i]-R.truth[j]) == np.sign(R.pred[i]-R.pred[j])
    cov = R.pred.std() / R.truth.std()
    # two-sidedness: does it ever confidently call PRE-exilic, and is it right?
    z = (R.pred - 586) / R.spread.replace(0, np.nan)
    p_post = stats.norm.cdf(z.fillna(0) * -1)   # pred < 586 -> post-exilic
    lowconf = (p_post <= 0.2)
    hiconf = (p_post >= 0.8)
    truth_post = R.truth < 586
    return dict(target=target, kind=kind, p=p, calib=calibrate,
                mae=R.err.mean(), rho=rho.statistic, rho_p=rho.pvalue,
                pair=c / t, cov=cov,
                lo_calls=int(lowconf.sum()),
                lo_ok=int((lowconf & ~truth_post).sum()),
                hi_calls=int(hiconf.sum()),
                hi_ok=int((hiconf & truth_post).sum()),
                pmin=R.pred.min(), pmax=R.pred.max()), R


if __name__ == "__main__":
    LAM = 10.0 ** np.arange(1, 6.1, 1.0)
    import gc
    out = []
    targets = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [300, 500, 1000]
    for target in targets:
        for kind in ("base", "ngram", "both"):
            for cal in (False, True):
                try:
                    m, R = evaluate(target, kind, LAM, cal)
                except FileNotFoundError:
                    continue
                out.append(m)
                with open("/home/claude/sweep_results.jsonl","a") as fh:
                    fh.write(json.dumps({k:(float(v) if isinstance(v,(int,float,np.floating)) else v) for k,v in m.items()})+"\n")
                print(f"~{target:>4}w {kind:<6} calib={str(cal):<5} p={m['p']:>4} | "
                       f"MAE {m['mae']:6.1f}  rho {m['rho']:+.3f}  pair {m['pair']*100:4.1f}%  "
                       f"cov {m['cov']:.2f}  range [{m['pmax']:.0f},{m['pmin']:.0f}]  "
                       f"pre-calls {m['lo_ok']}/{m['lo_calls']}  post {m['hi_ok']}/{m['hi_calls']}",
                       flush=True)
                R.to_csv(f"/home/claude/sweep_{target}_{kind}_{int(cal)}.csv", index=False)
                del R; gc.collect()
    pd.DataFrame(out).to_csv("/home/claude/sweep_summary.csv", index=False)
    print("\nwrote sweep_summary.csv")
