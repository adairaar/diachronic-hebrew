"""
Leave-one-book-out jackknife of the headline statistics.

Drop each anchored unit in turn, re-run the entire pipeline on the remaining 24,
and record rho and pairwise accuracy.  This is the influence diagnostic a referee
would run, and it says how much any single unit is carrying.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import os
import numpy as np, pandas as pd, importlib.util, json
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", DH.script("predict_targets.py"))
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

Dd = pd.read_csv(DH.f("big_features_500.csv"))
feats0 = [c for c in Dd.columns if c not in PT.META]
ALL = list(pd.unique(Dd.unit.values))


def run(drop):
    D = Dd[~Dd.unit.isin(drop)]
    Xa = D[feats0].astype(float)
    keep = (Xa.std() > 0) & (Xa.isna().mean() < 0.2)
    f = list(np.array(feats0)[keep.values])
    med = D[f].astype(float).median()
    X = D[f].astype(float).fillna(med).values
    y = D.date_bce.values.astype(float); g = D.unit.values
    books = list(pd.unique(g)); bd = {b: y[g == b][0] for b in books}
    t, p = [], []
    for b in books:
        te = g == b; tr = ~te
        inner = [x for x in books if x != b]
        wb = dict(zip(inner, PT.wts([bd[x] for x in inner])))
        wv = np.array([wb[u] for u in g[tr]])
        best, bl = np.inf, PT.LAM[0]
        for lam in PT.LAM:
            e = []
            for bb in inner:
                m = g[tr] != bb
                pr = PT.fit_predict(X[tr][m], y[tr][m], wv[m], X[tr][~m], lam)
                e.append(abs(np.median(pr) - bd[bb]))
            if np.mean(e) < best: best, bl = np.mean(e), lam
        t.append(bd[b]); p.append(float(np.median(
            PT.fit_predict(X[tr], y[tr], wv, X[te], bl))))
    t, p = np.array(t), np.array(p)
    S = float(np.clip(t.std() / p.std(), .5, 8.))
    cal = t.mean() + S * (p - p.mean())
    rho = float(stats.spearmanr(t, cal)[0])
    n = len(t); ok = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            if t[i] == t[j]: continue
            tot += 1; ok += int((cal[i] - cal[j]) * (t[i] - t[j]) > 0)
    return rho, ok / tot, float(np.abs(t - cal).mean())


full = run([])
print(f"full corpus:  rho {full[0]:+.3f}  pair {100*full[1]:.1f}%  MAE {full[2]:.1f}\n")
# Checkpoint after each dropped book.  A full pass is 25 leave-one-out fits of
# a model that is itself leave-one-book-out, and in this environment a process
# that long is not guaranteed to survive; resuming beats restarting.
CKPT = DH.f(".jackknife_partial.csv")
rows, done = [], set()
if os.path.exists(CKPT):
    prev = pd.read_csv(CKPT)
    rows = prev.to_dict("records")
    done = set(prev.dropped)
    print(f"resuming: {len(done)} of {len(ALL)} books already done", flush=True)
for b in ALL:
    if b in done:
        continue
    r, pr, m = run([b])
    rows.append(dict(dropped=b, rho=r, pair=pr, mae=m, d_rho=r - full[0]))
    # fixed precision: a resumed row round-trips through CSV and a fresh one
    # does not, so without this the file's bytes depend on where the run was
    # interrupted even though every value is identical
    pd.DataFrame(rows).to_csv(CKPT, index=False, float_format="%.12g")
    print(f"  without {b:<14} rho {r:+.3f} ({r-full[0]:+.3f})  "
          f"pair {100*pr:4.1f}%  MAE {m:5.1f}", flush=True)
J = pd.DataFrame(rows).sort_values("rho")
J.to_csv(DH.f("jackknife.csv"), index=False, float_format="%.12g")
os.path.exists(CKPT) and os.remove(CKPT)
print(f"\nrho across all 25 leave-one-out fits: "
      f"min {J.rho.min():+.3f} ({J.iloc[0].dropped}), "
      f"median {J.rho.median():+.3f}, max {J.rho.max():+.3f}")
print(f"pairwise: min {100*J.pair.min():.1f}%  median {100*J.pair.median():.1f}%  "
      f"max {100*J.pair.max():.1f}%")
json.dump(dict(full_rho=full[0], full_pair=full[1], full_mae=full[2],
               rho_min=float(J.rho.min()), rho_med=float(J.rho.median()),
               rho_max=float(J.rho.max()), rho_min_book=J.iloc[0].dropped,
               pair_min=float(J.pair.min()), pair_med=float(J.pair.median()),
               pair_max=float(J.pair.max())),
          open(DH.f("jackknife.json"), "w"), indent=2)
