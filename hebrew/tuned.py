"""
Fully tuned pipeline: lambda, weighting exponent alpha, and KDE bandwidth all
selected by inner leave-one-book-out; variance matching applied afterwards.

Selection criterion is Spearman rho on the inner out-of-fold predictions, NOT
MAE.  Rationale: variance matching sets the scale at the end, so what is needed
from the model is the ordering.  rho is scale-invariant, so selecting on it
selects for exactly the property that survives the rescaling.  Selecting on MAE
instead would favour shrinkage, which variance matching then has to undo.

Efficiency: for a fixed inner fold and weight vector, the kernel is eigen-
decomposed once and every lambda is then O(n^2), so the lambda grid is nearly
free.  Standardisation and weights are recomputed inside every inner fold, so
no quantity derived from a held-out book touches its own prediction.
"""
import numpy as np, pandas as pd, sys, json, itertools
from scipy import stats

META = {"chunk_id", "unit", "date_bce", "genre", "register", "n_words"}
LAM = 10.0 ** np.array([2, 3, 4, 5])
ALPHA = [0.0, 0.5, 1.0, 1.5]
BW = [60.0, 120.0, 200.0]


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


def wts(dates, alpha, bw):
    d = np.asarray(dates, float)
    if alpha == 0: return np.ones_like(d)
    dens = np.array([np.exp(-0.5 * ((d - x) / bw) ** 2).sum() for x in d])
    w = (1.0 / np.maximum(dens, 1e-9)) ** alpha
    return w / w.mean()


def fold_preds(Xtr, ytr, wtr, Xte, lams):
    """Return {lam: median prediction for the held-out block}."""
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 0, sd, 1.0)
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    yb = np.average(ytr, weights=wtr)
    s = np.sqrt(wtr)
    Aw = A * s[:, None]; yw = (ytr - yb) * s
    K = Aw @ Aw.T
    d, V = np.linalg.eigh(K)
    Vy = V.T @ yw
    KB = B @ Aw.T
    VB = KB @ V
    return {lam: float(np.median(VB @ (Vy / (d + lam)) + yb)) for lam in lams}


def evaluate(target, kind, variance_match=True, verbose=True):
    D, X = load(target, kind)
    y = D.date_bce.values.astype(float); g = D.unit.values
    books = list(pd.unique(g))
    bdate = {b: y[g == b][0] for b in books}
    rows = []
    for b in books:
        te = g == b; tr = ~te
        Xtr, ytr, gtr = X[tr], y[tr], g[tr]
        inner = [x for x in books if x != b]
        idates = [bdate[x] for x in inner]

        # inner LOBO over (alpha, bw), all lambdas at once
        best = (-2.0, ALPHA[0], BW[0], LAM[0], None)
        for alpha, bw in itertools.product(ALPHA, BW):
            wb = dict(zip(inner, wts(idates, alpha, bw)))
            acc = {lam: [] for lam in LAM}; truths = []
            for bb in inner:
                m = gtr != bb
                wv = np.array([wb[u] for u in gtr[m]])
                pv = fold_preds(Xtr[m], ytr[m], wv, Xtr[~m], LAM)
                for lam in LAM: acc[lam].append(pv[lam])
                truths.append(ytr[~m][0])
            for lam in LAM:
                r = stats.spearmanr(truths, acc[lam]).statistic
                if np.isfinite(r) and r > best[0]:
                    best = (r, alpha, bw, lam, (np.array(truths), np.array(acc[lam])))

        _, alpha, bw, lam, oof = best
        wb = dict(zip(inner, wts(idates, alpha, bw)))
        wv = np.array([wb[u] for u in gtr])
        pred = fold_preds(Xtr, ytr, wv, X[te], [lam])[lam]
        ot, op = oof
        if variance_match and op.std() > 1e-9:
            s_ = float(np.clip(ot.std() / op.std(), 0.5, 8.0))
            pred = float(ot.mean() + s_ * (pred - op.mean()))
        rows.append(dict(unit=b, truth=bdate[b], pred=pred,
                         alpha=alpha, bw=bw, lam=lam, inner_rho=best[0]))
        if verbose:
            print(f"    {b:<15} true {bdate[b]:4.0f} pred {pred:6.0f}  "
                  f"(a={alpha}, bw={bw:.0f}, lam=1e{np.log10(lam):.0f}, "
                  f"inner rho {best[0]:+.2f})", flush=True)

    R = pd.DataFrame(rows); R["err"] = (R.truth - R.pred).abs()
    rho = stats.spearmanr(R.truth, R.pred)
    c = t = 0
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            if R.truth[i] == R.truth[j]: continue
            t += 1
            c += np.sign(R.truth[i]-R.truth[j]) == np.sign(R.pred[i]-R.pred[j])
    pre = R[R.truth > 586]; post = R[R.truth < 586]
    m = dict(target=target, kind=kind, vm=variance_match, p=X.shape[1],
             mae=R.err.mean(), rho=rho.statistic, rho_p=rho.pvalue, pair=c/t,
             cov=R.pred.std()/R.truth.std(),
             side=float(((R.truth > 586) == (R.pred > 586)).mean()),
             pre_ok=int((pre.pred > 586).sum()), pre_n=len(pre),
             post_ok=int((post.pred < 586).sum()), post_n=len(post),
             pmax=R.pred.max(), pmin=R.pred.min(),
             alpha_mode=float(R.alpha.mode()[0]), bw_mode=float(R.bw.mode()[0]))
    return m, R


if __name__ == "__main__":
    cfgs = [(500, "base"), (500, "both"), (1000, "both")]
    out = []
    for target, kind in cfgs:
        for vm in (True, False):
            print(f"\n=== ~{target}w {kind}  variance_match={vm} ===", flush=True)
            m, R = evaluate(target, kind, vm, verbose=False)
            out.append(m)
            print(f"  MAE {m['mae']:6.1f}  rho {m['rho']:+.3f} (p={m['rho_p']:.4f})  "
                  f"pair {m['pair']*100:.1f}%  cov {m['cov']:.2f}  "
                  f"range [{m['pmax']:.0f},{m['pmin']:.0f}]  side {m['side']*100:.0f}%  "
                  f"pre {m['pre_ok']}/{m['pre_n']}  post {m['post_ok']}/{m['post_n']}", flush=True)
            print(f"  modal hyperparams: alpha={m['alpha_mode']}, bw={m['bw_mode']:.0f}", flush=True)
            R.to_csv(f"/home/claude/tuned_{target}_{kind}_{int(vm)}.csv", index=False)
            with open("/home/claude/tuned.jsonl", "a") as fh:
                fh.write(json.dumps({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                     for k, v in m.items()}) + "\n")
    pd.DataFrame(out).to_csv("/home/claude/tuned_summary.csv", index=False)
