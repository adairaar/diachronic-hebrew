"""Does recovering the clause-type features improve the model?

BHSA encodes the Hebrew verbal system at clause level: Way0 for narrative
wayyiqtol, WQt0 for weqatal, xQt0 for fronted qatal, NmCl for verbless.  An
earlier version of the extractor counted clause TYPE but indexed the counts with
clause RELA keys, whose value sets are disjoint, so all 38 of those columns were
structurally zero and the model silently never saw the verbal system at all.

The two matrices compared here are both produced by the current extractor:

  big_features_500.csv        the reported model.  Its clause-type columns are
                              the broken ones, hence constant, hence dropped by
                              the zero-variance filter, leaving 578 features.
  big_features_500_ctyp.csv   the same extraction with the fix applied, giving
                              50 working clause-type features and 628 in total.

Everything else in the two files is numerically identical, column for column, so
this isolates the clause-type family and nothing else.
"""
import sys, numpy as np, pandas as pd, importlib.util
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", "/home/claude/predict_targets.py")
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)


def run(path, label):
    Dd = pd.read_csv(path)
    feats = [c for c in Dd.columns if c not in PT.META]
    Xa = Dd[feats].astype(float)
    keep = (Xa.std() > 0) & (Xa.isna().mean() < .2)
    feats = list(np.array(feats)[keep.values])
    med = Dd[feats].astype(float).median()
    X = Dd[feats].astype(float).fillna(med).values
    y = Dd.date_bce.values.astype(float); g = Dd.unit.values
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
                e.append(abs(np.median(PT.fit_predict(
                    X[tr][m], y[tr][m], wv[m], X[tr][~m], lam)) - bd[bb]))
            if np.mean(e) < best: best, bl = np.mean(e), lam
        t.append(bd[b])
        p.append(float(np.median(PT.fit_predict(X[tr], y[tr], wv, X[te], bl))))
    t, p = np.array(t), np.array(p)
    S = float(np.clip(t.std() / p.std(), .5, 8.))
    cal = t.mean() + S * (p - p.mean()); r = t - cal
    rho = stats.spearmanr(t, cal)[0]
    n = len(t); ok = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            if t[i] == t[j]: continue
            tot += 1; ok += int((cal[i] - cal[j]) * (t[i] - t[j]) > 0)
    pre = t > 586; post = t < 586
    print(f"{label:<28} p={len(feats):>4}  MAE {np.abs(r).mean():5.1f}  rho {rho:+.3f}  "
          f"pair {100*ok/tot:4.1f}%  pre {int((cal[pre]>586).sum())}/{int(pre.sum())} "
          f"post {int((cal[post]<586).sum())}/{int(post.sum())}", flush=True)


run("/home/claude/big_features_500.csv", "without clause-type feats")
run("/home/claude/big_features_500_ctyp.csv", "WITH clause-type feats")
