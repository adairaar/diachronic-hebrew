"""
How much of the ordering is chronology, and how much is genre?

The leave-one-genre-out refits collapsed.  Before reading anything into that,
it has to be separated from the trivial explanation: those refits train on
fewer books, and a weaker model is not the same finding as a confounded one.

The question that does not depend on retraining is whether the model, fitted on
all 25 anchors, can order books WITHIN a genre.  Prophecy supplies 17 anchors
spanning 760 to 350 BCE -- a 410-year range inside a single register, which is
the best-controlled comparison the corpus allows.  If the ordering is
chronological, it should survive there.  If it is the narrative/prophecy
contrast, it should vanish there, because the narrative books happen to be
uniformly post-exilic and so "narrative" and "late" are the same axis.
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

Dd = pd.read_csv(DH.f("big_features_500.csv"))
B = pd.read_csv(DH.f("final_lobo_books.csv"))
gen = Dd.groupby("unit").genre.first()
B["genre"] = B.book.map(gen)
t, p = B.truth.values.astype(float), B.pred.values.astype(float)

print("=" * 78)
print("1.  ORDERING WITHIN GENRE, UNDER THE FULL 25-ANCHOR MODEL")
print("=" * 78)
print("  No retraining anywhere below.  These are the same leave-one-book-out")
print("  predictions the manuscript reports, sliced by genre.\n")
print(f"  {'subset':<26}{'n':>4}{'range':>12}{'rho':>9}{'p':>9}{'pair':>8}")
out = {}


def pairwise(tv, pv):
    ok = tot = 0
    for i in range(len(tv)):
        for j in range(i + 1, len(tv)):
            if tv[i] == tv[j]: continue
            tot += 1; ok += int((pv[i] - pv[j]) * (tv[i] - tv[j]) > 0)
    return ok / tot if tot else np.nan, tot


for name, m in [("all anchors", np.ones(len(B), bool)),
                ("prophecy only", (B.genre == "prophecy").values),
                ("narrative only", (B.genre == "narrative").values),
                ("everything but prophecy", (B.genre != "prophecy").values),
                ("everything but narrative", (B.genre != "narrative").values)]:
    tv, pv = t[m], p[m]
    if len(tv) < 3: continue
    rho, pval = stats.spearmanr(tv, pv)
    pr, npair = pairwise(tv, pv)
    out[name] = dict(n=int(m.sum()), rho=float(rho), p=float(pval),
                     pair=float(pr), lo=int(tv.min()), hi=int(tv.max()))
    print(f"  {name:<26}{m.sum():>4}{f'{int(tv.max())}-{int(tv.min())}':>12}"
          f"{rho:>+9.3f}{pval:>9.4f}{100*pr:>7.1f}%")

print()
print("=" * 78)
print("2.  PARTIAL CORRELATION, CONTROLLING FOR GENRE")
print("=" * 78)
print("  Remove each genre's mean from both the truth and the prediction, then")
print("  correlate what is left.  This is the ordering the model achieves that")
print("  genre alone could not have produced.\n")
Bc = B.copy()
Bc["t_c"] = Bc.truth - Bc.groupby("genre").truth.transform("mean")
Bc["p_c"] = Bc.pred - Bc.groupby("genre").pred.transform("mean")
big = Bc[Bc.genre.isin(["prophecy", "narrative"])]        # genres with n > 1
rho_raw = stats.spearmanr(B.truth, B.pred)[0]
rho_par = stats.spearmanr(big.t_c, big.p_c)[0]
rho_par_all = stats.spearmanr(Bc.t_c, Bc.p_c)[0]
print(f"  raw Spearman, all 25 anchors                      {rho_raw:+.3f}")
print(f"  genre-centred, the 22 books in a genre with n>1    {rho_par:+.3f}")
print(f"  genre-centred, all 25 (singleton genres centre to 0){rho_par_all:+.3f}")

print()
print("=" * 78)
print("3.  HOW MUCH OF THE SPREAD IS BETWEEN GENRES?")
print("=" * 78)
gm = B.groupby("genre").agg(n=("truth", "size"), truth=("truth", "mean"),
                            pred=("pred", "mean"))
print(gm.round(0).to_string())
ss_tot = float(((B.pred - B.pred.mean()) ** 2).sum())
ss_bet = float(sum(r.n * (r.pred - B.pred.mean()) ** 2 for _, r in gm.iterrows()))
print(f"\n  between-genre share of the variance in the predictions: "
      f"{100*ss_bet/ss_tot:.0f}%")

print()
print("=" * 78)
print("4.  THE 17 PROPHETIC BOOKS, IN ORDER")
print("=" * 78)
P = B[B.genre == "prophecy"].sort_values("truth", ascending=False)
print(f"  {'book':<14}{'truth':>7}{'pred':>7}{'resid':>8}")
for _, r in P.iterrows():
    print(f"  {r.book:<14}{int(r.truth):>7}{int(r.pred):>7}{int(r.resid):>+8}")
rho_p, pv_p = stats.spearmanr(P.truth, P.pred)
print(f"\n  Spearman within prophecy: {rho_p:+.3f} (p = {pv_p:.3f}, n = {len(P)})")

json.dump(dict(subsets=out, rho_raw=float(rho_raw), rho_partial=float(rho_par),
               rho_partial_all=float(rho_par_all),
               between_genre_share=float(ss_bet / ss_tot),
               rho_prophecy=float(rho_p), p_prophecy=float(pv_p)),
          open(DH.f("genre_confound.json"), "w"), indent=2)
print("\nwrote genre_confound.json")
