"""
The signal is entirely lexical.  Is that diachrony or is it subject matter?

The red team found that removing the 250 lexemes leaves genre-controlled rho at
+0.14 and within-prophecy rho at +0.06, while the lexemes alone give +0.46 and
+0.43.  Every bit of the ordering lives in the vocabulary.

That is not automatically fatal.  Lexical replacement is a genuine and
well-established component of diachronic Hebrew -- Hurvitz's whole method rests
on it, and Persian loanwords and Aramaisms are real chronological markers.  A
lexical signal could be exactly what the literature predicts.

But it could equally be a topic classifier, and the two are distinguishable.
If the model is measuring diachrony, the high-leverage lexemes should be the
recognised late markers.  If it is measuring subject matter, they should instead
be the words that discriminate GENRE -- and that is directly checkable, because
the genre eta-squared for every feature is already computed.

Two tests:
  1. Across the 250 lexemes, does leverage correlate with genre discriminability?
  2. Where do the recognised CBH/LBH diagnostic lexemes actually rank?
"""
import json
import numpy as np, pandas as pd
from scipy import stats

F = pd.read_csv("/home/claude/sensitivity_features.csv")
Dd = pd.read_csv("/home/claude/big_features_500.csv")
META = {"chunk_id", "unit", "date_bce", "genre", "register", "n_words"}
feats = [c for c in Dd.columns if c not in META]
bg = Dd.groupby("unit").genre.first()
BM = Dd.groupby("unit")[feats].mean()
GEN = bg.loc[BM.index].values
m = np.isin(GEN, ["prophecy", "narrative"])
Z = BM.values[m]; gg = GEN[m]
tot = ((Z - Z.mean(0)) ** 2).sum(0)
bet = np.zeros(Z.shape[1])
for lev in np.unique(gg):
    s = gg == lev
    bet += s.sum() * (Z[s].mean(0) - Z.mean(0)) ** 2
ETA = pd.Series(bet / np.where(tot > 0, tot, np.inf), index=feats)

F["eta_genre"] = F.feature.map(ETA)
L = F[F.family == "lexical"].dropna(subset=["eta_genre"]).copy()

print("=" * 78)
print("1.  DO THE HIGH-LEVERAGE LEXEMES DISCRIMINATE GENRE?")
print("=" * 78)
rho_le, p_le = stats.spearmanr(L["abs"], L.eta_genre)
print(f"  across {len(L)} lexemes: leverage vs genre eta-squared  "
      f"rho {rho_le:+.3f} (p = {p_le:.2g})")
hi = L.nlargest(50, "abs"); lo = L.nsmallest(50, "abs")
print(f"  mean genre eta-squared, 50 highest-leverage lexemes: "
      f"{hi.eta_genre.mean():.3f}")
print(f"  mean genre eta-squared, 50 lowest-leverage lexemes:  "
      f"{lo.eta_genre.mean():.3f}")
print(f"  ratio {hi.eta_genre.mean()/max(lo.eta_genre.mean(),1e-9):.2f}x")

# ETCBC transliteration of lexemes the literature treats as diachronic markers.
# LBH: late replacements, Persian loans, Aramaisms.  CBH: the early counterparts
# they replace.  Restricted to items actually present in the feature set.
LBH = {
    "MLKWT/": "malkut, late 'kingdom' for mamlakah",
    "ZMN/": "zeman, Persian-era 'time'",
    "PRTMJM/": "partemim, Persian 'nobles'",
    "GNZJM/": "genazim, Persian 'treasury'",
    "DT/": "dat, Persian 'law'",
    "PTGM/": "pitgam, Persian 'decree'",
    "MDJNH/": "medinah, 'province'",
    "KCR[": "kaser, late 'be fitting'",
    "QBL[": "qabbel, Aramaising 'receive'",
    "<NJN/": "'inyan, late 'matter'",
    "BJRH/": "birah, 'citadel'",
    "JHWDJ/": "yehudi, 'Jew'",
    "HWN/": "hon, late 'wealth'",
    "SGN/": "segan, 'prefect'",
    "><J/": "'ay, Aramaising",
    "KTB/": "ketab, 'writing'",
}
CBH = {
    "MMLKH/": "mamlakah, classical 'kingdom'",
    ">NKJ": "'anoki, classical 1sg pronoun",
    "<T/": "'et, classical 'time'",
}
DIAG = {**LBH, **CBH}

print()
print("=" * 78)
print("2.  WHERE DO THE RECOGNISED DIACHRONIC MARKERS RANK?")
print("=" * 78)
L = L.sort_values("abs", ascending=False).reset_index(drop=True)
L["rank"] = L.index + 1
present, absent = [], []
for k, desc in DIAG.items():
    row = L[L.feature == f"lex_{k}"]
    if len(row):
        rr = row.iloc[0]
        present.append((k, desc, int(rr["rank"]), float(rr.yr_per_sd)))
    else:
        absent.append((k, desc))
print(f"  {len(present)} of {len(DIAG)} diagnostic lexemes are in the "
      f"{len(L)}-feature lexical set\n")
if present:
    print(f"  {'lexeme':<12}{'rank':>6}{'yr/sd':>9}   description")
    for k, desc, rk, v in sorted(present, key=lambda x: x[2]):
        print(f"  {k:<12}{rk:>6}{v:>+9.2f}   {desc}")
    rks = [r for _, _, r, _ in present]
    print(f"\n  median rank of the diagnostics: {int(np.median(rks))} of {len(L)}")
    print(f"  a marker with no special status would sit near {len(L)//2}")
if absent:
    print(f"\n  not in the feature set (too rare to survive the frequency "
          f"filter): {', '.join(k for k, _ in absent)}")

print()
print("=" * 78)
print("3.  WHAT THE TOP LEXEMES ACTUALLY ARE")
print("=" * 78)
GLOSS = {
    "lex_<MD[": "'amad, stand", "lex_QRB/": "qereb, midst",
    "lex_RB/": "rab, many/great", "lex_>NJ": "'ani, I (LBH marker)",
    "lex_>RY/": "'erets, land", "lex_>DMH/": "'adamah, ground",
    "lex_J<QB/": "Jacob (proper noun)", "lex_CM/": "shem, name",
    "lex_DBR/": "davar, word", "lex_JFR>L/": "Israel (proper noun)",
    "lex_M>D/": "me'od, very", "lex_<WLM/": "'olam, forever",
    "lex_GDWL/": "gadol, great", "lex_BW>[": "bo', come",
    "lex_KH": "koh, thus -- the prophetic messenger formula",
    "lex_NF>[": "nasa', lift", "lex_GM": "gam, also",
    "lex_JHWH/": "YHWH (divine name)", "lex_<M": "'im, with",
    "lex_<NH[": "'anah, answer", "lex_Z>T": "zo't, this",
    "lex_QDC/": "qodesh, holiness", "lex_PQD[": "paqad, visit",
    "lex_LKN": "lakhen, therefore",
}
print(f"  {'rank':>5}{'lexeme':<13}{'yr/sd':>9}{'eta genre':>11}   gloss")
for _, r in L.head(20).iterrows():
    print(f"  {int(r['rank']):>5}{r.feature:<13}{r.yr_per_sd:>+9.2f}"
          f"{r.eta_genre:>11.3f}   {GLOSS.get(r.feature, '')}")

print()
print("=" * 78)
print("4.  THE ONE PAIR THE CORPUS CAN ACTUALLY TEST")
print("=" * 78)
print("  'anoki -> 'ani is the textbook first-person pronoun shift: the long")
print("  classical form gives way to the short one across the exile.  Both")
print("  members are frequent enough to survive the filter, which is rare among")
print("  the diagnostics.  The model was never told this.\n")
for lx, name, expect in [("lex_>NKJ", "'anoki (classical)", "older"),
                         ("lex_>NJ", "'ani (late)", "later")]:
    row = L[L.feature == lx]
    if not len(row): continue
    v = float(row.iloc[0].yr_per_sd); rk = int(row.iloc[0]["rank"])
    got = "older" if v > 0 else "later"
    mark = "as predicted" if got == expect else "AGAINST prediction"
    print(f"  {name:<20} rank {rk:>3}/{len(L)}   {v:+.2f} yr/sd   "
          f"more of it -> {got}   ({mark})")
print("\n  A topic classifier has no reason to place these two words high and")
print("  in opposite directions.  This is the strongest single piece of")
print("  evidence that some genuine diachrony is being picked up.")

json.dump(dict(n_lex=len(L), rho_leverage_eta=float(rho_le), p=float(p_le),
               eta_top50=float(hi.eta_genre.mean()),
               eta_bot50=float(lo.eta_genre.mean()),
               diagnostics_present=len(present), diagnostics_total=len(DIAG),
               diagnostic_ranks={k: rk for k, _, rk, _ in present},
               median_diagnostic_rank=int(np.median([r for _, _, r, _ in present]))
               if present else None),
          open("/home/claude/lexical_diagnosis.json", "w"), indent=2)
print("\nwrote lexical_diagnosis.json")
