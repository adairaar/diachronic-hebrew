"""
Does the result survive dropping the training books whose dates are NOT fixed by
an external synchronism?

Six of the 25 anchors rest on literary or linguistic judgment rather than on a
datable external event.  Two of those (Jonah, Ecclesiastes) are dated partly by
their linguistic profile, which is the very thing the model is meant to measure.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import numpy as np, pandas as pd, importlib.util, json
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", DH.script("predict_targets.py"))
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

# anchors that are NOT an external synchronism or datable event
SOFT = ["Jonah", "Ecclesiastes", "Malachi", "Joel", "Isaiah_3", "Zechariah_2"]
CIRCULAR = ["Jonah", "Ecclesiastes"]          # dated partly BY linguistic profile

Dd = pd.read_csv(DH.f("big_features_500.csv"))
feats0 = [c for c in Dd.columns if c not in PT.META]


def run(drop, label):
    D = Dd[~Dd.unit.isin(drop)].copy()
    Xa = D[feats0].astype(float)
    keep = (Xa.std() > 0) & (Xa.isna().mean() < 0.2)
    feats = list(np.array(feats0)[keep.values])
    med = D[feats].astype(float).median()
    X = D[feats].astype(float).fillna(med).values
    y = D.date_bce.values.astype(float); g = D.unit.values
    books = list(pd.unique(g)); bdate = {b: y[g == b][0] for b in books}
    t, p = [], []
    for b in books:
        te = g == b; tr = ~te
        inner = [x for x in books if x != b]
        wb = dict(zip(inner, PT.wts([bdate[x] for x in inner])))
        wv = np.array([wb[u] for u in g[tr]])
        best, bl = np.inf, PT.LAM[0]
        for lam in PT.LAM:
            e = []
            for bb in inner:
                m = g[tr] != bb
                pr = PT.fit_predict(X[tr][m], y[tr][m], wv[m], X[tr][~m], lam)
                e.append(abs(np.median(pr) - bdate[bb]))
            if np.mean(e) < best: best, bl = np.mean(e), lam
        t.append(bdate[b])
        p.append(float(np.median(PT.fit_predict(X[tr], y[tr], wv, X[te], bl))))
    t, p = np.array(t), np.array(p)
    S = float(np.clip(t.std() / p.std(), .5, 8.))
    cal = t.mean() + S * (p - p.mean())
    r = t - cal
    rho = float(stats.spearmanr(t, cal)[0])
    n = len(t); ok = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            if t[i] == t[j]: continue
            tot += 1; ok += int((cal[i] - cal[j]) * (t[i] - t[j]) > 0)
    ar = np.sort(np.abs(r))
    q68 = ar[min(int(np.ceil((n + 1) * .68)) - 1, n - 1)]
    pre = t > 586; post = t < 586
    print(f"{label:<34} n={n:2d}  MAE {np.abs(r).mean():5.1f} (base "
          f"{np.abs(t-t.mean()).mean():5.1f})  rho {rho:+.3f}  "
          f"pair {100*ok/tot:4.1f}%  q68 {q68:3.0f}  "
          f"pre {int((cal[pre]>586).sum())}/{int(pre.sum())}  "
          f"post {int((cal[post]<586).sum())}/{int(post.sum())}")
    return dict(label=label, n=n, mae=float(np.abs(r).mean()),
                base=float(np.abs(t - t.mean()).mean()), rho=rho,
                pair=ok / tot, q68=float(q68))


out = []
out.append(run([], "A  all 25 anchors (as published)"))
out.append(run(CIRCULAR, "B  drop the 2 linguistically dated"))
out.append(run(SOFT, "C  drop all 6 non-external anchors"))
json.dump(out, open(DH.f("anchor_sensitivity.json"), "w"), indent=2)
