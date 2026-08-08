"""
Leave-one-BOOK-out evaluation of the chunk-level design.

Train on chunks from 24 units, predict every chunk of the held-out unit,
aggregate to a book-level estimate.  Chunks from a unit never straddle the
fold boundary, so nothing leaks.

Compares directly against the book-level result this replaces:
  book-level LOO MAE  156.2 yr (generative) / 120.9 yr (ridge)
  constant predictor  137.3 yr
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import numpy as np, pandas as pd, sys
from scipy import stats

META = {"chunk_id", "unit", "date_bce", "sigma", "genre", "register", "n_words",
        "in_training", "id", "label"}


def load(target, min_icc=0.0):
    D = pd.read_csv(DH.f(f"chunk_features_{target}.csv"))
    diag = pd.read_csv(DH.f(f"chunk_diag_{target}.csv"))
    keep = diag[(diag.usable) & (diag.icc.fillna(0) >= min_icc)].feature.tolist()
    keep = [f for f in keep if f in D.columns]
    X = D[keep].copy()
    X = X.fillna(X.median())
    ok = X.std() > 0
    return D, X.loc[:, ok].values, np.array(keep)[ok.values]


def ridge_fit(A, y, lams=10.0 ** np.arange(-2, 4.6, 0.5)):
    n, k = A.shape
    G = A.T @ A
    best, bl = np.inf, lams[0]
    for lam in lams:
        try:
            H = A @ np.linalg.solve(G + lam * np.eye(k), A.T)
        except np.linalg.LinAlgError:
            continue
        h = np.clip(np.diag(H), 0, 1 - 1e-9)
        cv = (((y - H @ y) / (1 - h)) ** 2).mean()
        if cv < best:
            best, bl = cv, lam
    return np.linalg.solve(G + bl * np.eye(k), A.T @ y), bl


def run(target, min_icc, label):
    D, X, feats = load(target, min_icc)
    units = D.unit.unique()
    dates = D.date_bce.values
    rows = []
    for u in units:
        te = D.unit.values == u
        tr = ~te
        mu, sd = X[tr].mean(0), np.where(X[tr].std(0) > 0, X[tr].std(0), 1.0)
        Ztr, Zte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ybar = dates[tr].mean()
        w, lam = ridge_fit(Ztr, dates[tr] - ybar)
        pred = Zte @ w + ybar
        rows.append(dict(unit=u, truth=dates[te][0], n_chunks=te.sum(),
                         pred_median=float(np.median(pred)),
                         pred_mean=float(pred.mean()),
                         pred_sd=float(pred.std()),
                         pred_lo=float(np.percentile(pred, 16)),
                         pred_hi=float(np.percentile(pred, 84))))
    R = pd.DataFrame(rows)
    R["err"] = (R.truth - R.pred_median).abs()

    const = np.abs(R.truth - R.truth.mean()).mean()
    rho = stats.spearmanr(R.truth, R.pred_median)
    # pairwise ordering
    c = t = 0
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            if R.truth[i] == R.truth[j]: continue
            t += 1
            c += np.sign(R.truth[i] - R.truth[j]) == np.sign(R.pred_median[i] - R.pred_median[j])

    print(f"\n{'='*76}\n{label}  |  {len(D)} chunks, {len(feats)} features, "
          f"{len(R)} units\n{'='*76}")
    print(f"{'unit':<15}{'true':>7}{'pred':>8}{'|err|':>7}{'chunks':>8}"
          f"{'within-unit 68%':>20}")
    for _, r in R.sort_values("truth", ascending=False).iterrows():
        print(f"{r.unit:<15}{r.truth:7.0f}{r.pred_median:8.0f}{r.err:7.0f}"
              f"{r.n_chunks:8d}   [{r.pred_hi:5.0f}, {r.pred_lo:5.0f}]")
    print(f"\n  book-level MAE            : {R.err.mean():6.1f} yr")
    print(f"  constant predictor        : {const:6.1f} yr")
    print(f"  Spearman rho              : {rho.statistic:+.3f}  (p={rho.pvalue:.4f})")
    print(f"  pairwise ordering         : {c/t*100:5.1f}%  of {t} pairs")
    print(f"  median within-unit sd     : {R.pred_sd.median():6.1f} yr")
    print(f"  between-unit sd of preds  : {R.pred_median.std():6.1f} yr")
    ratio = R.pred_median.std() / R.pred_sd.median()
    print(f"  between/within ratio      : {ratio:6.2f}  "
          f"({'signal exceeds noise' if ratio > 1 else 'NOISE DOMINATES'})")
    return R


if __name__ == "__main__":
    for target in (500, 1000):
        for icc in (0.0, 0.3):
            run(target, icc, f"chunk~{target}w, features with ICC>={icc}")
