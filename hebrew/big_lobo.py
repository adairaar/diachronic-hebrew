"""
Leave-one-BOOK-out with the expanded feature space.

Screening is not used: with p ~ 580 and n_eff ~ 74-175 the false-discovery
proportion is >= 0.38 even at alpha = 0.001 (see power_analysis_chunks.py).
Regularisation and dimensionality reduction are used instead, since neither
incurs a selection penalty.  Lambda / n_components are chosen by an INNER
leave-one-book-out on the training books only.
"""
import numpy as np, pandas as pd, sys
from scipy import stats

META = {"chunk_id", "unit", "date_bce", "genre", "register", "n_words"}


def load(target):
    D = pd.read_csv(f"/home/claude/big_features_{target}.csv")
    feats = [c for c in D.columns if c not in META]
    X = D[feats].astype(float)
    keep = (X.std() > 0) & (X.isna().mean() < 0.2)
    X = X.loc[:, keep]
    X = X.fillna(X.median())
    return D, X.values, list(X.columns)


def ridge_w(A, y, lam):
    k = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(k), A.T @ y)


def pls_w(A, y, ncomp):
    """Simple NIPALS PLS1 regression -> weight vector in original space."""
    A = A.copy(); y = y.copy()
    W, P, Q = [], [], []
    for _ in range(ncomp):
        w = A.T @ y
        nw = np.linalg.norm(w)
        if nw < 1e-12: break
        w /= nw
        t = A @ w
        tt = t @ t
        if tt < 1e-12: break
        p = A.T @ t / tt
        q = float(y @ t / tt)
        A = A - np.outer(t, p); y = y - q * t
        W.append(w); P.append(p); Q.append(q)
    if not W: return np.zeros(A.shape[1])
    W = np.array(W).T; P = np.array(P).T; Q = np.array(Q)
    try:
        return W @ np.linalg.solve(P.T @ W, Q)
    except np.linalg.LinAlgError:
        return np.zeros(W.shape[0])


def inner_select(Xtr, ytr, gtr, kind, grid):
    """Choose the hyperparameter by leave-one-book-out inside the training set."""
    books = np.unique(gtr)
    best, bp = np.inf, grid[0]
    for g in grid:
        errs = []
        for b in books:
            m = gtr != b
            mu, sd = Xtr[m].mean(0), np.where(Xtr[m].std(0) > 0, Xtr[m].std(0), 1.0)
            A = (Xtr[m] - mu) / sd; B = (Xtr[~m] - mu) / sd
            yb = ytr[m].mean()
            w = ridge_w(A, ytr[m] - yb, g) if kind == "ridge" else pls_w(A, ytr[m] - yb, g)
            errs.append(abs(np.median(B @ w + yb) - ytr[~m][0]))
        e = np.mean(errs)
        if e < best: best, bp = e, g
    return bp


def run(target, kind, grid, label):
    D, X, feats = load(target)
    y = D.date_bce.values.astype(float); g = D.unit.values
    books = D.unit.unique()
    rows = []
    for b in books:
        te = g == b; tr = ~te
        hp = inner_select(X[tr], y[tr], g[tr], kind, grid)
        mu, sd = X[tr].mean(0), np.where(X[tr].std(0) > 0, X[tr].std(0), 1.0)
        A = (X[tr] - mu) / sd; B = (X[te] - mu) / sd
        yb = y[tr].mean()
        w = ridge_w(A, y[tr] - yb, hp) if kind == "ridge" else pls_w(A, y[tr] - yb, hp)
        pr = B @ w + yb
        rows.append(dict(unit=b, truth=y[te][0], pred=float(np.median(pr)),
                         sd=float(pr.std()), n_chunks=int(te.sum()), hp=hp))
    R = pd.DataFrame(rows); R["err"] = (R.truth - R.pred).abs()
    const = np.abs(R.truth - R.truth.mean()).mean()
    rho = stats.spearmanr(R.truth, R.pred)
    c = t = 0
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            if R.truth[i] == R.truth[j]: continue
            t += 1
            c += np.sign(R.truth[i]-R.truth[j]) == np.sign(R.pred[i]-R.pred[j])
    comp = R.pred.std() / R.truth.std()
    print(f"\n{label}: {len(D)} chunks x {len(feats)} feats")
    print(f"   MAE {R.err.mean():6.1f} yr (constant {const:.1f})   rho {rho.statistic:+.3f} "
          f"(p={rho.pvalue:.4f})   pairwise {c/t*100:.1f}%   range-coverage {comp:.2f}")
    print(f"   pred range [{R.pred.max():.0f}, {R.pred.min():.0f}]  "
          f"true range [{R.truth.max():.0f}, {R.truth.min():.0f}]")
    return R


if __name__ == "__main__":
    LAM = 10.0 ** np.arange(0, 5.1, 0.5)
    NC = [1, 2, 3, 4, 6, 8, 12]
    best = None
    for target in (300, 500, 1000):
        R1 = run(target, "ridge", LAM, f"ridge  ~{target}w")
        R2 = run(target, "pls", NC, f"PLS    ~{target}w")
        for R, k in ((R1, f"ridge{target}"), (R2, f"pls{target}")):
            if best is None or R.err.mean() < best[1]:
                best = (k, R.err.mean(), R)
    print(f"\n{'='*70}\nBEST: {best[0]}  MAE {best[1]:.1f} yr\n{'='*70}")
    B = best[2].sort_values("truth", ascending=False)
    print(f"{'unit':<15}{'true':>7}{'pred':>8}{'err':>7}")
    for _, r in B.iterrows():
        print(f"{r.unit:<15}{r.truth:7.0f}{r.pred:8.0f}{r.err:7.0f}")
    B.to_csv("/home/claude/big_lobo_best.csv", index=False)
