"""
Selection-corrected performance: choose the configuration inside each fold.

The headline figures are produced by a configuration (passage size, weighting
exponent, calibration, ridge penalty) that was itself chosen by comparing
out-of-sample performance across a search on the same corpus.  A permutation
null that holds the winning configuration fixed does not account for that
choice, so the reported rho is the maximum over a search and is optimistically
biased.

This script removes the bias by nesting the search.  For each held-out book,
the entire configuration is selected by an inner leave-one-book-out loop over
the remaining 24 units, and only then is the held-out book predicted.  The
resulting statistics are what a fresh corpus should be expected to give.

Configuration grid (72 combinations):
    passage size   300, 500, 1000 words
    weight exponent alpha = 0 (unweighted), 1 (inverse-density)
    ridge penalty  lambda in 10^1 .. 10^6

Variance matching is NOT part of the search.  It is a positive linear map of the
predictions, so it leaves Spearman rho and pairwise ordering exactly unchanged
and cannot inflate either; it is applied always, as a pre-specified correction
for shrinkage rather than as a tuned setting.  Including it in the grid only
produces arbitrary ties on a rank criterion.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import json, itertools
import numpy as np, pandas as pd
from scipy import stats

SIZES = [300, 500, 1000]
ALPHAS = [0.0, 1.0]
CALIBS = ["var"]   # pre-specified, not tuned: see note below
LAM = 10.0 ** np.arange(1, 6.1, 1.0)
META = {"chunk_id", "unit", "date_bce", "genre", "register", "n_words"}
BW = 90.0


def load(size):
    D = pd.read_csv(DH.f(f"big_features_{size}.csv"))
    f = [c for c in D.columns if c not in META]
    Xa = D[f].astype(float)
    keep = (Xa.std() > 0) & (Xa.isna().mean() < 0.2)
    f = list(np.array(f)[keep.values])
    X = D[f].astype(float).fillna(D[f].astype(float).median()).values
    return X, D.date_bce.values.astype(float), D.unit.values


DATA = {s: load(s) for s in SIZES}
BOOKS = list(pd.unique(DATA[500][2]))
BDATE = {b: DATA[500][1][DATA[500][2] == b][0] for b in BOOKS}
print(f"{len(BOOKS)} books | grid = {len(SIZES)}x{len(ALPHAS)}x{len(LAM)}"
      f" = {len(SIZES)*len(ALPHAS)*len(LAM)} configurations\n")


def wts(dates, alpha):
    d = np.asarray(dates, float)
    if alpha == 0: return np.ones_like(d)
    dens = np.array([np.exp(-0.5 * ((d - x) / BW) ** 2).sum() for x in d])
    w = (1.0 / np.maximum(dens, 1e-9)) ** alpha
    return w / w.mean()


def fit_all_lams(Xtr, ytr, wtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 0, sd, 1.0)
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    yb = np.average(ytr, weights=wtr); s = np.sqrt(wtr)
    Aw = A * s[:, None]; yw = (ytr - yb) * s
    ev, V = np.linalg.eigh(Aw @ Aw.T)
    Vt_y = V.T @ yw; BAt = (B @ Aw.T) @ V
    return {lam: BAt @ (Vt_y / (ev + lam)) + yb for lam in LAM}


def lobo_raw(size, alpha, subset):
    """Raw (uncalibrated) LOBO predictions for `subset`, per lambda."""
    X, y, g = DATA[size]
    m = np.isin(g, subset)
    X, y, g = X[m], y[m], g[m]
    out = {lam: [] for lam in LAM}
    truth = []
    for b in subset:
        te = g == b; tr = ~te
        inner = [x for x in subset if x != b]
        wb = dict(zip(inner, wts([BDATE[x] for x in inner], alpha)))
        wv = np.array([wb[u] for u in g[tr]])
        pr = fit_all_lams(X[tr], y[tr], wv, X[te])
        for lam in LAM: out[lam].append(float(np.median(pr[lam])))
        truth.append(BDATE[b])
    return np.array(truth), {lam: np.array(v) for lam, v in out.items()}


def score(t, p, calib):
    if calib == "var":
        S = float(np.clip(t.std() / p.std(), 0.5, 8.0))
        p = t.mean() + S * (p - p.mean())
    rho = stats.spearmanr(t, p)[0]
    return (float(rho) if np.isfinite(rho) else -1.0), p


def select(subset):
    """Pick (size, alpha, calib, lam) by leave-one-book-out within `subset`."""
    best, cfg = -np.inf, None
    for size, alpha in itertools.product(SIZES, ALPHAS):
        t, preds = lobo_raw(size, alpha, subset)
        for calib in CALIBS:
            for lam in LAM:
                r, _ = score(t, preds[lam], calib)
                if r > best: best, cfg = r, (size, alpha, calib, lam)
    return cfg, best


# ── outer loop: configuration chosen without the held-out book ───────────
print("outer fold: configuration selected on the other 24 books")
rows = []
for b in BOOKS:
    sub = [x for x in BOOKS if x != b]
    (size, alpha, calib, lam), inner_rho = select(sub)
    X, y, g = DATA[size]
    tr = g != b; te = g == b
    wb = dict(zip(sub, wts([BDATE[x] for x in sub], alpha)))
    wv = np.array([wb[u] for u in g[tr]])
    raw_te = float(np.median(fit_all_lams(X[tr], y[tr], wv, X[te])[lam]))
    if calib == "var":                       # constants from training books only
        t_in, preds_in = lobo_raw(size, alpha, sub)
        p_in = preds_in[lam]
        S = float(np.clip(t_in.std() / p_in.std(), 0.5, 8.0))
        pred = t_in.mean() + S * (raw_te - p_in.mean())
    else:
        pred = raw_te
    rows.append(dict(book=b, truth=BDATE[b], pred=pred, size=size, alpha=alpha,
                     calib=calib, lam=lam, inner_rho=inner_rho))
    print(f"  {b:<14} cfg=({size:4d}, a={alpha}, {calib:4s}, lam={lam:.0e}) "
          f"inner rho {inner_rho:+.3f}  ->  {pred:6.0f} (true {BDATE[b]:.0f})",
          flush=True)

R = pd.DataFrame(rows)
R.to_csv(DH.f("nested_selection_var.csv"), index=False)
t = R.truth.values; p = R.pred.values
rho, rp = stats.spearmanr(t, p)
n = len(t); ok = tot = 0
for i in range(n):
    for j in range(i + 1, n):
        if t[i] == t[j]: continue
        tot += 1; ok += int((p[i] - p[j]) * (t[i] - t[j]) > 0)
mae = float(np.abs(t - p).mean())
base = float(np.abs(t - t.mean()).mean())
pre = t > 586; post = t < 586

print("\n" + "=" * 70)
print("SELECTION-CORRECTED PERFORMANCE (configuration chosen inside each fold)")
print("=" * 70)
print(f"  MAE            {mae:.1f} yr   (constant predictor {base:.1f})")
print(f"  Spearman rho   {rho:+.3f}  (p = {rp:.4f})")
print(f"  pairwise       {100*ok/tot:.1f}% of {tot}")
print(f"  pre-exilic     {int((p[pre]>586).sum())}/{int(pre.sum())} placed correctly")
print(f"  post-exilic    {int((p[post]<586).sum())}/{int(post.sum())} placed correctly")
print(f"\n  configurations chosen across the 25 folds:")
print(R.groupby(['size', 'alpha', 'calib']).size().to_string())

json.dump(dict(mae=mae, base=base, rho=float(rho), rho_p=float(rp),
               pair=ok / tot, n_pair=tot,
               pre_ok=int((p[pre] > 586).sum()), n_pre=int(pre.sum()),
               post_ok=int((p[post] < 586).sum()), n_post=int(post.sum()),
               modal_size=int(R["size"].mode()[0]),
               modal_calib=R["calib"].mode()[0],
               n_distinct=int(R.groupby(['size','alpha','calib','lam']).ngroups)),
          open(DH.f("nested_selection_var.json"), "w"), indent=2)
print("\nwrote nested_selection.csv, nested_selection.json")
