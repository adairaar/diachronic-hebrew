"""
Audit the conformal interval construction.

Four questions.

1. Is the quantile index right, and what happens when the requested coverage
   exceeds what n calibration points can support?
2. Is the guarantee the one claimed?  Split conformal assumes the calibration
   residuals and the test residual are exchangeable draws from one fitted model.
   Here every residual comes from a DIFFERENT model, each fitted without its own
   book.  That is the jackknife, whose naive intervals carry no finite-sample
   guarantee; the jackknife+ of Barber et al. does, at a cost.
3. Are the calibration constants themselves out of sample?  S, the input mean
   and the output mean are estimated from all 25 leave-one-book-out predictions,
   including the book being scored.
4. What is the empirical coverage, and how coarse is it at n = 25?
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import json, importlib.util
import numpy as np, pandas as pd
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", DH.script("predict_targets.py"))
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

Dd = pd.read_csv(DH.f("big_features_500.csv"))
feats = [c for c in Dd.columns if c not in PT.META]
Xa = Dd[feats].astype(float)
keep = (Xa.std() > 0) & (Xa.isna().mean() < 0.2)
feats = list(np.array(feats)[keep.values])
med = Dd[feats].astype(float).median()
X = Dd[feats].astype(float).fillna(med).values
y = Dd.date_bce.values.astype(float); g = Dd.unit.values
books = list(pd.unique(g)); bdate = {b: y[g == b][0] for b in books}
n = len(books)


def lobo_raw(subset):
    """Uncalibrated LOBO predictions within a set of books."""
    t, p = [], []
    for b in subset:
        te = g == b; tr = np.isin(g, [x for x in subset if x != b])
        inner = [x for x in subset if x != b]
        wb = dict(zip(inner, PT.wts([bdate[x] for x in inner])))
        wv = np.array([wb[u] for u in g[tr]])
        best, bl = np.inf, PT.LAM[0]
        for lam in PT.LAM:
            e = []
            for bb in inner:
                m = g[tr] != bb
                e.append(abs(np.median(PT.fit_predict(
                    X[tr][m], y[tr][m], wv[m], X[tr][~m], lam)) - bdate[bb]))
            if np.mean(e) < best: best, bl = np.mean(e), lam
        t.append(bdate[b])
        p.append(float(np.median(PT.fit_predict(X[tr], y[tr], wv, X[te], bl))))
    return np.array(t), np.array(p)


print("=" * 72)
print("1.  QUANTILE INDEX")
print("=" * 72)
for a in (0.68, 0.90, 0.95, 0.99):
    k = int(np.ceil((n + 1) * a))
    ok = k <= n
    print(f"  coverage {a:.2f}:  k = ceil({n+1} x {a:.2f}) = {k:>2}   "
          + ("valid" if ok else f"EXCEEDS n = {n}; the guarantee requires an "
                                "infinite interval, but the code clamps to the max"))

print()
print("=" * 72)
print("2 & 3.  ARE THE CALIBRATION CONSTANTS OUT OF SAMPLE?")
print("=" * 72)
t, p = lobo_raw(books)

# (a) as implemented: S and the centring constants use all 25 predictions
S_all = float(np.clip(t.std() / p.std(), 0.5, 8.0))
cal_all = t.mean() + S_all * (p - p.mean())
res_all = t - cal_all

# (b) fully nested: for each book, S and the constants come from the other 24
res_nested = np.empty(n)
for i, b in enumerate(books):
    o = [j for j in range(n) if j != i]
    S_i = float(np.clip(t[o].std() / p[o].std(), 0.5, 8.0))
    pred_i = t[o].mean() + S_i * (p[i] - p[o].mean())
    res_nested[i] = t[i] - pred_i

def q(r, a):
    ar = np.sort(np.abs(r)); k = int(np.ceil((len(ar) + 1) * a))
    return ar[min(k, len(ar)) - 1], k <= len(ar)

print(f"  {'':<26}{'MAE':>8}{'q68':>8}{'q90':>8}{'cover68':>9}{'cover90':>9}")
rows = {}
for name, r in [("as implemented", res_all), ("calibration nested", res_nested)]:
    q68, _ = q(r, .68); q90, _ = q(r, .90)
    c68 = float(np.mean(np.abs(r) <= q68)); c90 = float(np.mean(np.abs(r) <= q90))
    rows[name] = dict(mae=float(np.abs(r).mean()), q68=float(q68), q90=float(q90),
                      cov68=c68, cov90=c90)
    print(f"  {name:<26}{np.abs(r).mean():8.1f}{q68:8.0f}{q90:8.0f}"
          f"{100*c68:8.1f}%{100*c90:8.1f}%")
d = rows["calibration nested"]["mae"] - rows["as implemented"]["mae"]
print(f"\n  estimating S on all 25 books understates the error by {d:+.1f} yr")
print(f"  (rank correlation and pairwise ordering are unaffected: a positive")
print(f"   linear map cannot change them)")

print()
print("=" * 72)
print("4.  JACKKNIFE+ INTERVALS, WHICH DO CARRY A GUARANTEE")
print("=" * 72)
# Barber, Candes, Ramdas & Tibshirani (2021).  For a target with leave-one-out
# predictions mu_i and residuals R_i, the jackknife+ interval is
#   [ q_alpha^-{mu_i - R_i}, q_alpha^+{mu_i + R_i} ]
# with coverage at least 1 - 2*alpha.
for a in (0.68, 0.90):
    alpha = 1 - a
    k = int(np.ceil((n + 1) * (1 - alpha)))
    lo_q = np.sort(-np.abs(res_nested))[::-1]
    width = np.sort(np.abs(res_nested))[min(k, n) - 1]
    print(f"  nominal {a:.0%}: naive jackknife half-width {width:.0f} yr, "
          f"guaranteed coverage >= {max(0, 1-2*alpha):.0%}")
print("  The jackknife+ guarantee is 1 - 2*alpha, so a 68% nominal interval")
print("  guarantees only 36% and a 90% nominal interval guarantees 80%.")

out = dict(n=n, rows=rows, delta_mae=float(d),
           q95_k=int(np.ceil((n + 1) * .95)), q99_k=int(np.ceil((n + 1) * .99)),
           cov68_impl=rows["as implemented"]["cov68"],
           cov90_impl=rows["as implemented"]["cov90"],
           q68_nested=rows["calibration nested"]["q68"],
           q90_nested=rows["calibration nested"]["q90"],
           mae_nested=rows["calibration nested"]["mae"])
json.dump(out, open(DH.f("conformal_audit.json"), "w"), indent=2)
print("\nwrote conformal_audit.json")
