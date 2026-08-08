"""
Is the alternant-share advantage the encoding, or just having fewer features?

Seven share features give a higher genre-controlled correlation than \\HBfeat
rate features.  Two rival explanations have to be excluded before that means
anything.

  Parsimony.  A ridge fit over 578 features on 25 books is enormously
  overparameterised; seven features is not.  The comparison could be measuring
  dimensionality rather than encoding.  Controlled by drawing seven rate
  features at random, many times.

  Lexeme choice.  The pairs were assembled from lexemes already known to carry
  diachronic weight.  Controlled by refitting on exactly the same fourteen
  lexemes as plain rates: identical material, different encoding, so any
  selection bias applies equally to both arms.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import json
import numpy as np, pandas as pd, importlib.util
from scipy import stats

exec(open(DH.script("variationist_test.py")).read().split('print(f"  {\'model\'')[0])

y = Dd.date_bce.values.astype(float); g = Dd.unit.values
med = Dd[feats0].astype(float).median()
RAW = Dd[feats0].astype(float).fillna(med).values
SHV = SHARE.values
RNG = np.random.default_rng(4)
NDRAW = 40


def run(X):
    books = list(pd.unique(g)); bd = {b: y[g == b][0] for b in books}
    t, p = [], []
    for b in books:
        te = g == b; tr = ~te
        inner = [x for x in books if x != b]
        wb = dict(zip(inner, PT.wts([bd[x] for x in inner])))
        wv = np.array([wb[u] for u in g[tr]])
        err = {l: [] for l in PT.LAM}
        for bb in inner:
            m = g[tr] != bb
            pr = fit_all(X[tr][m], y[tr][m], wv[m], X[tr][~m])
            for l in PT.LAM:
                err[l].append(abs(np.median(pr[l]) - bd[bb]))
        bl = min(PT.LAM, key=lambda l: np.mean(err[l]))
        t.append(bd[b])
        p.append(float(np.median(fit_all(X[tr], y[tr], wv, X[te])[bl])))
    t, p = np.array(t), np.array(p)
    S = float(np.clip(t.std() / p.std(), 0.5, 8.0))
    return genre_rho(books, t, p.mean() + S * (p - p.mean()))


shares = run(SHV)
mem = [f"lex_{a}" for _, a, b, _ in have] + [f"lex_{b}" for _, a, b, _ in have]
same = run(RAW[:, [feats0.index(c) for c in mem if c in feats0]])
rand = [run(RAW[:, RNG.choice(len(feats0), 7, replace=False)]) for _ in range(NDRAW)]
rand = np.array(rand)

print(f"  {'model':<52}{'p':>4}{'rho|genre':>11}")
print(f"  {'alternant shares':<52}{SHV.shape[1]:>4}{shares:>+11.3f}")
print(f"  {'the same lexemes as plain rates':<52}{len(mem):>4}{same:>+11.3f}")
print(f"  {'random rate features':<52}{7:>4}{np.median(rand):>+11.3f}")
print(f"      over {NDRAW} draws: {rand.min():+.3f} to {rand.max():+.3f}; "
      f"{int((rand >= shares).sum())} reach the share model")
json.dump(dict(shares=float(shares), same_lexemes=float(same),
               random7=rand.tolist(), n_draws=NDRAW),
          open(DH.f("share_control.json"), "w"), indent=2)
print("\nwrote share_control.json")
