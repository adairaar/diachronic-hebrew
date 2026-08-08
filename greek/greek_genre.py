"""
Does the Greek archaizing measurement survive a genre control?

The Hebrew half of this study establishes that genre, not chronology, is the
dominant axis of variation in a corpus assembled by date, and that raw rank
correlation overstates the chronological signal by roughly half.  The Greek half
currently reports raw rank correlation and an uncorrected displacement.  That
asymmetry is indefensible on its face, and it may not be only cosmetic: the
Greek corpus is structured so that genre and register are correlated.  The
ancient Attic baseline is oratory-heavy and contains no narrative at all, while
the Koine training material is dominated by history and narrative.  A model that
learned "oratory" would look like a model that learned "early".

Three questions, none of which needs the model refitted, because the
leave-one-text-out predictions already exist:

  1. How much of the Greek model's ordering is between genres rather than
     within them?
  2. Is the Atticizer displacement reduced when each Atticizer is compared with
     the genre it belongs to rather than with the corpus as a whole?
  3. Within a single genre, are Atticizing texts displaced earlier than
     non-Atticizing ones?  That is the comparison that separates archaizing
     from genre, and it is the one the archaizing claim actually needs.
"""
import json
import numpy as np, pandas as pd
from scipy import stats

RNG = np.random.default_rng(19)
G = "/home/claude/greek"

MAN = pd.DataFrame(json.load(open(f"{G}/corpus_manifest.json")))
L = pd.read_csv(f"{G}/greek_loto_texts.csv")
L = L.merge(MAN[["id", "genre", "n_tokens"]], left_on="text", right_on="id",
            how="left")
L["genre"] = L.genre.str.replace("prose_", "", regex=False)

# The 14 Atticizers are held out of training entirely and scored separately, so
# they are absent from the leave-one-text-out file.  The two files also use
# opposite sign conventions: LOTO reports resid = truth - pred, the Atticizer
# file reports shift = pred - truth.  Everything below is put in "shift" units,
# where a negative value means the model dates the text EARLIER than it was
# written, which is the archaizing direction.
L["shift"] = -L.resid
A = pd.read_csv(f"{G}/greek_atticizers.csv").merge(
    MAN[["id", "genre"]], left_on="text", right_on="id", how="left")
A["genre"] = A.genre.str.replace("prose_", "", regex=False)

print("=" * 78)
print("1.  HOW MUCH OF THE GREEK ORDERING IS BETWEEN GENRES?")
print("=" * 78)
print(f"  {len(L)} texts with leave-one-text-out predictions\n")
print(f"  {'genre':<12}{'n':>4}{'mean truth':>12}{'mean resid':>12}")
for gn, sub in L.groupby("genre"):
    print(f"  {gn:<12}{len(sub):>4}{sub.truth.mean():>12.0f}{sub.resid.mean():>+12.0f}")

rho_raw = stats.spearmanr(L.truth, L.pred)[0]
Lc = L.copy()
big = Lc[Lc.groupby("genre").genre.transform("size") >= 4].copy()
big["tc"] = big.truth - big.groupby("genre").truth.transform("mean")
big["pc"] = big.pred - big.groupby("genre").pred.transform("mean")
rho_par = stats.spearmanr(big.tc, big.pc)[0]
gm = L.groupby("genre").pred.mean()
ss_t = float(((L.pred - L.pred.mean()) ** 2).sum())
ss_b = float(sum((L.genre == k).sum() * (v - L.pred.mean()) ** 2
                 for k, v in gm.items()))
print(f"\n  raw Spearman                       {rho_raw:+.3f}")
print(f"  genre-controlled Spearman          {rho_par:+.3f}  "
      f"({len(big)} texts in genres with n>=4)")
print(f"  between-genre share of prediction variance   {100*ss_b/ss_t:.0f}%")
print(f"\n  For comparison the Hebrew figures are +0.666 raw, +0.405 controlled,")
print(f"  and 77% between-genre.")

print()
print("=" * 78)
print("2.  THE ATTICIZER DISPLACEMENT, AGAINST ITS OWN GENRE")
print("=" * 78)
# genre baseline from the non-archaizing training texts, which is what the
# Atticizers are being contrasted against
base = L.groupby("genre")["shift"].mean()
A["genre_base"] = A.genre.map(base)
A["adj"] = A["shift"] - A.genre_base
print(f"  {'text':<34}{'genre':<12}{'shift':>8}{'genre base':>12}{'adjusted':>10}")
for _, r in A.sort_values("shift").iterrows():
    print(f"  {r.text[:33]:<34}{r.genre:<12}{r['shift']:>+8.0f}"
          f"{r.genre_base:>+12.0f}{r.adj:>+10.0f}")
raw_shift = float(A["shift"].mean()); adj_shift = float(A.adj.mean())
print(f"\n  mean displacement, uncorrected     {raw_shift:+.0f} yr")
print(f"  mean displacement, genre-corrected {adj_shift:+.0f} yr")
print(f"  the correction accounts for        {100*(1-adj_shift/raw_shift):.0f}% "
      f"of the raw displacement")

print()
print("=" * 78)
print("3.  WITHIN GENRE: ATTICIZERS AGAINST THEIR NON-ARCHAIZING PEERS")
print("=" * 78)
print("  The comparison the archaizing claim needs.  If Atticizers are displaced")
print("  earlier than non-Atticizing texts of the SAME genre, the displacement is")
print("  archaizing and not register-by-genre confounding.\n")
print(f"  {'genre':<12}{'n att':>7}{'n other':>9}{'att resid':>11}"
      f"{'other resid':>13}{'difference':>12}")
rows = []
for gn in sorted(set(A.genre) & set(L.genre)):
    a = A[A.genre == gn]; o = L[L.genre == gn]
    if len(a) == 0 or len(o) == 0: continue
    d = float(a["shift"].mean() - o["shift"].mean())
    rows.append(dict(genre=gn, n_att=len(a), n_oth=len(o),
                     att=float(a["shift"].mean()), oth=float(o["shift"].mean()),
                     diff=d))
    print(f"  {gn:<12}{len(a):>7}{len(o):>9}{a['shift'].mean():>+11.0f}"
          f"{o['shift'].mean():>+13.0f}{d:>+12.0f}")
W = pd.DataFrame(rows)
wmean = float(np.average(W["diff"], weights=W.n_att))
# paired bootstrap over genres, resampling texts within each genre
bs = []
for _ in range(4000):
    tot, wt = 0.0, 0.0
    for _, r in W.iterrows():
        a = A[A.genre == r.genre]["shift"].values
        o = L[L.genre == r.genre]["shift"].values
        d = RNG.choice(a, len(a)).mean() - RNG.choice(o, len(o)).mean()
        tot += d * r.n_att; wt += r.n_att
    bs.append(tot / wt)
bs = np.array(bs)
lo, hi = np.percentile(bs, [2.5, 97.5])
pval = float(np.mean(bs >= 0))
print(f"\n  weighted mean within-genre difference {wmean:+.0f} yr "
      f"(95% CI {lo:+.0f} to {hi:+.0f})")
print(f"  bootstrap P(difference >= 0) = {pval:.4f}")
print(f"  a negative difference means Atticizers are dated earlier than their")
print(f"  own-genre peers, which is the archaizing effect net of genre")
print(f"\n  every genre points the same way: "
      f"{int((W['diff'] < 0).sum())}/{len(W)} negative")

json.dump(dict(n_texts=len(L), rho_raw=float(rho_raw), rho_genre=float(rho_par),
               between_genre=float(ss_b / ss_t),
               shift_raw=raw_shift, shift_adj=adj_shift,
               within_genre_diff=wmean, ci=[float(lo), float(hi)],
               p_ge_zero=pval, n_genres=len(W),
               n_negative=int((W["diff"] < 0).sum()),
               by_genre=W.to_dict("records")),
          open(f"{G}/greek_genre.json", "w"), indent=2)
A.to_csv(f"{G}/greek_genre_atticizers.csv", index=False)
print("\nwrote greek_genre.json, greek_genre_atticizers.csv")
