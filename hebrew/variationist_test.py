"""
Can alternant-share features be added to the model, and do they help?

The rate encoding measures how often a form appears per 1000 words.  That
conflates how often the author reaches for the relevant construction at all --
largely a function of genre -- with which of the competing forms he chooses when
he does, which is the diachronic signal.  Variationist linguistics isolates the
second by dividing within the set of alternants.

The question is whether enough alternant pairs survive this corpus's frequency
filter to build a useful feature set from them, and whether adding such features
to the existing model improves it.

Pairs are counted only where BOTH members clear the filter.  Shares use add-half
smoothing, (a + 1/2) / (a + b + 1), so a passage containing neither member gives
1/2 rather than an undefined value; without that, most pairs are undefined in
most 500-word passages and the feature is unusable at passage level.

Not every pair below is a linguistic variable in the strict sense.  The
first-person pronoun is the textbook case.  Others -- 'say' against 'speak',
'man' against 'human' -- are near-synonyms rather than competing realisations of
one function, and a variationist would not accept them without specifying the
envelope of variation.  They are included to bound what an automatic procedure
can reach; the strict subset is reported separately.
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

pt = importlib.util.spec_from_file_location("pt", DH.script("predict_targets.py"))
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

# (label, classical-or-A member, late-or-B member, is it a strict variable?)
PAIRS = [
    ("1cs pronoun 'anoki/'ani", ">NKJ", ">NJ", True),
    ("preposition 'el/le",      ">L",   "L",   True),
    ("negation lo'/'eyn",       "L>",   ">JN/", True),
    ("complementiser ki/'asher", "KJ",  ">CR", False),
    ("'stand' 'amad/qum",       "<MD[", "QWM[", False),
    ("'say' 'amar/dibber",      ">MR[", "DBR[", False),
    ("'man' 'ish/'adam",        ">JC/", ">DM/", False),
]

Dd = pd.read_csv(DH.f("big_features_500.csv"))
feats0 = [c for c in Dd.columns if c not in PT.META]
k = Dd[feats0].astype(float).std() > 0
feats0 = list(np.array(feats0)[k.values])
bgenre = Dd.groupby("unit").genre.first().to_dict()

have = [(n, a, b, s) for n, a, b, s in PAIRS
        if f"lex_{a}" in feats0 and f"lex_{b}" in feats0]
print(f"{len(have)} of {len(PAIRS)} candidate pairs have both members in the "
      f"{len(feats0)}-feature set")
print(f"  {sum(s for _, _, _, s in have)} of those are strict linguistic variables\n")

SH = {}
for n, a, b, strict in have:
    A = Dd[f"lex_{a}"].astype(float); B = Dd[f"lex_{b}"].astype(float)
    SH[f"share_{a}_{b}"] = (A + 0.5) / (A + B + 1.0)
SHARE = pd.DataFrame(SH)
STRICT = [f"share_{a}_{b}" for n, a, b, s in have if s]


def fit_all(Xtr, ytr, wtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 0, sd, 1.0)
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    yb = np.average(ytr, weights=wtr)
    s = np.sqrt(wtr)
    Aw = A * s[:, None]; yw = (ytr - yb) * s
    ev, V = np.linalg.eigh(Aw @ Aw.T)
    return {l: (B @ Aw.T) @ V @ (V.T @ yw / (ev + l)) + yb for l in PT.LAM}


def genre_rho(books, t, p):
    D = pd.DataFrame(dict(t=t, p=p, gn=[bgenre[b] for b in books]))
    D = D[D.gn.isin(["prophecy", "narrative"])]
    D = D.assign(tc=D.t - D.groupby("gn").t.transform("mean"),
                 pc=D.p - D.groupby("gn").p.transform("mean"))
    return float(stats.spearmanr(D.tc, D.pc)[0])


def lobo(X, y, g, label):
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
    cal = t.mean() + S * (p - p.mean())
    r = dict(model=label, n_feats=X.shape[1],
             mae=float(np.abs(t - cal).mean()),
             rho=float(stats.spearmanr(t, cal)[0]),
             rho_genre=genre_rho(books, t, cal))
    print(f"  {label:<44}{r['n_feats']:>5}{r['mae']:>6.0f}"
          f"{r['rho']:>+8.3f}{r['rho_genre']:>+11.3f}")
    return r


y = Dd.date_bce.values.astype(float); g = Dd.unit.values
med = Dd[feats0].astype(float).median()
RAW = Dd[feats0].astype(float).fillna(med).values
SHV = SHARE.values
STRV = SHARE[STRICT].values
LOGRAW = np.log1p(np.clip(RAW, 0, None))

print(f"  {'model':<44}{'p':>5}{'MAE':>6}{'rho':>8}{'rho|genre':>11}")
rows = [
    lobo(RAW, y, g, "rates only (as published)"),
    lobo(SHV, y, g, f"shares only ({len(have)} pairs)"),
    lobo(STRV, y, g, f"strict-variable shares only ({len(STRICT)} pairs)"),
    lobo(np.hstack([RAW, SHV]), y, g, "rates + shares"),
    lobo(np.hstack([LOGRAW, SHV]), y, g, "log rates + shares"),
]
json.dump(dict(n_pairs=len(have), n_strict=len(STRICT),
               pairs=[dict(label=n, a=a, b=b, strict=s) for n, a, b, s in have],
               results=rows),
          open(DH.f("variationist_test.json"), "w"), indent=2)
print("\nwrote variationist_test.json")
