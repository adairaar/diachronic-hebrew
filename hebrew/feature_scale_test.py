"""
Would a nonlinear feature scale, or a variationist denominator, do better?

Two distinct questions, raised by the observation that language change follows an
S-curve rather than a straight line.

1. SCALE.  The features are rates per 1000 words, not proportions, so they are
   not bounded above and a linear model does not imply an impossible value.  But
   if a rate's trajectory over time is logistic, then its logit is linear in
   time, and a model linear in the raw rate is fitting a curve with a line.  A
   log or logit transform would straighten it.  Tested here by refitting on
   transformed features.

2. DENOMINATOR.  A deeper issue.  Encoding a lexeme as occurrences per 1000
   words conflates two things: how often the author reaches for that kind of
   expression at all, and which of the available forms he chooses when he does.
   The first is largely genre; the second is the diachronic signal.
   Variationist linguistics measures the second by dividing within the set of
   alternants -- the share of 'anoki among first-person singular pronouns rather
   than 'anoki per 1000 words.  If the rate encoding is carrying genre because
   of its denominator, that bears directly on this paper's central confound.
   Tested here on the one alternant pair the corpus can support.
"""
import json
import numpy as np, pandas as pd, importlib.util
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", "/home/claude/predict_targets.py")
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

Dd = pd.read_csv("/home/claude/big_features_500.csv")
feats0 = [c for c in Dd.columns if c not in PT.META]
k = Dd[feats0].astype(float).std() > 0
feats0 = list(np.array(feats0)[k.values])
bgenre = Dd.groupby("unit").genre.first().to_dict()


def fit_all(Xtr, ytr, wtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 0, sd, 1.0)
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    yb = np.average(ytr, weights=wtr)
    s = np.sqrt(wtr)
    Aw = A * s[:, None]; yw = (ytr - yb) * s
    ev, V = np.linalg.eigh(Aw @ Aw.T)
    Vt = V.T @ yw
    return {l: (B @ Aw.T) @ V @ (Vt / (ev + l)) + yb for l in PT.LAM}


def genre_rho(books, t, p):
    D = pd.DataFrame(dict(t=t, p=p, gn=[bgenre[b] for b in books]))
    D = D[D.gn.isin(["prophecy", "narrative"])]
    D = D.assign(tc=D.t - D.groupby("gn").t.transform("mean"),
                 pc=D.p - D.groupby("gn").p.transform("mean"))
    return float(stats.spearmanr(D.tc, D.pc)[0])


def lobo(X, y, g):
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
    return dict(mae=float(np.abs(t - cal).mean()),
                rho=float(stats.spearmanr(t, cal)[0]),
                rho_genre=genre_rho(books, t, cal))


y = Dd.date_bce.values.astype(float); g = Dd.unit.values
med = Dd[feats0].astype(float).median()
RAW = Dd[feats0].astype(float).fillna(med).values

print("=" * 74)
print("1.  DOES A NONLINEAR FEATURE SCALE HELP?")
print("=" * 74)
print(f"  {len(feats0)} features, rates per 1000 words; "
      f"{(RAW.max(0) > 1).sum()} of them exceed 1.0, so they are not proportions\n")
TRANS = [
    ("raw rate (as published)", lambda Z: Z),
    ("log1p(rate)", lambda Z: np.log1p(np.clip(Z, 0, None))),
    ("sqrt(rate)  (variance-stabilizing for counts)", lambda Z: np.sqrt(np.clip(Z, 0, None))),
    ("logit of rate/1000, clipped", lambda Z: np.log(np.clip(Z / 1000, 1e-6, 1 - 1e-6)
                                                     / (1 - np.clip(Z / 1000, 1e-6, 1 - 1e-6)))),
]
print(f"  {'feature scale':<46}{'MAE':>6}{'rho':>8}{'rho|genre':>11}")
rows = []
for name, fn in TRANS:
    r = lobo(fn(RAW), y, g)
    rows.append(dict(scale=name, **r))
    print(f"  {name:<46}{r['mae']:>6.0f}{r['rho']:>+8.3f}{r['rho_genre']:>+11.3f}")

print()
print("=" * 74)
print("2.  THE DENOMINATOR: RATE PER 1000 WORDS vs SHARE OF THE ALTERNANT SET")
print("=" * 74)
A, B = "lex_>NKJ", "lex_>NJ"
Bk = Dd.groupby("unit").agg({A: "mean", B: "mean", "date_bce": "first"})
Bk["share"] = Bk[A] / (Bk[A] + Bk[B]).replace(0, np.nan)
Bk = Bk.dropna(subset=["share"])
gen = np.array([bgenre[b] for b in Bk.index])


def one(v, t, gn, lab):
    r = float(stats.spearmanr(t, v)[0])
    D = pd.DataFrame(dict(t=t, v=v, gn=gn))
    D = D[D.gn.isin(["prophecy", "narrative"])]
    D = D.assign(tc=D.t - D.groupby("gn").t.transform("mean"),
                 vc=D.v - D.groupby("gn").v.transform("mean"))
    rg = float(stats.spearmanr(D.tc, D.vc)[0])
    print(f"  {lab:<44}{r:>+8.3f}{rg:>11.3f}")
    return r, rg


print(f"  {'encoding of the anoki/ani variable':<44}{'rho':>8}{'rho|genre':>11}")
r1 = one(Bk[A].values, Bk.date_bce.values, gen, "'anoki per 1000 words (as published)")
r2 = one(Bk[B].values, Bk.date_bce.values, gen, "'ani per 1000 words (as published)")
r3 = one(Bk["share"].values, Bk.date_bce.values, gen,
         "'anoki as a SHARE of the pair (variationist)")
json.dump(dict(scales=rows,
               anoki_rate=r1, ani_rate=r2, share=r3,
               n_books_with_pair=int(len(Bk))),
          open("/home/claude/feature_scale_test.json", "w"), indent=2)
print(f"\n  ({len(Bk)} of 25 books contain at least one member of the pair)")
print("\nwrote feature_scale_test.json")
