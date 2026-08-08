"""
The case against the result.

Not a sensitivity analysis.  The aim here is to break the finding, using the
strongest objections a hostile referee could raise, and to report only what is
left standing afterwards.

Four attacks, each with a decisive test.

  1. IT IS A TOPIC CLASSIFIER.  Lexical features carry 47% of the total
     leverage, and the highest-leverage lexemes are YHWH, Israel, Jacob, holy,
     people, king and speak.  Those are subject matter, not language change: a
     text about the monarchy uses the word king.  Test -- refit on the 384
     morphosyntactic features alone, with every lexeme removed, and separately
     on the 250 lexemes alone.  If the lexical model works and the
     morphosyntactic one does not, the objection is correct and the paper is
     measuring what its texts are about rather than when they were written.

  2. THE ANCHORS ARE PARTLY CIRCULAR.  Six of the 25 anchor dates rest on
     literary judgment rather than an external synchronism, and two of those
     (Jonah, Ecclesiastes) are dated partly BY their linguistic profile, which
     is the quantity being modelled.  Test -- refit on externally anchored books
     only.  Note in advance that four of the six are the late prophetic books,
     so this also strips most of the late end of the within-genre range; both
     effects will be visible and must not be conflated.

  3. IT IS GENRE, NOT TIME.  Handled by within_genre_null.py, running separately.

  4. IT IS DRIVEN BY BOOK SIZE.  Short books give noisier medians, and the
     shortest books in the corpus are disproportionately late.  If error tracks
     book length, the ordering could be a length artifact.  Test -- correlate
     absolute residual against chunk count and word count.

The strictest combination, morphosyntax only on externally anchored books, is
run last.  If anything survives that, it is the real result.
"""
import json, time
import numpy as np, pandas as pd, importlib.util
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", "/home/claude/predict_targets.py")
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

EX = 586
SRC = ["JE_source", "D_source", "P_source"]
SOFT = ["Jonah", "Ecclesiastes", "Malachi", "Joel", "Isaiah_3", "Zechariah_2"]

Dd = pd.read_csv("/home/claude/big_features_500.csv")
Dt = pd.read_csv("/home/claude/target_chunks_500.csv")
ALL = [c for c in Dd.columns if c not in PT.META and c in Dt.columns]
LEXF = [c for c in ALL if c.startswith("lex_")]
MORF = [c for c in ALL if not c.startswith("lex_")]
bgenre = Dd.groupby("unit").genre.first().to_dict()


def fit_all(Xtr, ytr, wtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 0, sd, 1.0)
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    yb = np.average(ytr, weights=wtr)
    s = np.sqrt(wtr)
    Aw = A * s[:, None]; yw = (ytr - yb) * s
    ev, V = np.linalg.eigh(Aw @ Aw.T)
    Vt = V.T @ yw
    BAt = (B @ Aw.T) @ V
    return {l: BAt @ (Vt / (ev + l)) + yb for l in PT.LAM}


def genre_rho(gen, t, p):
    D = pd.DataFrame(dict(t=t, p=p, gn=gen))
    D = D[D.gn.isin(["prophecy", "narrative"])].copy()
    if D.gn.nunique() < 2 or len(D) < 5: return np.nan
    D["tc"] = D.t - D.groupby("gn").t.transform("mean")
    D["pc"] = D.p - D.groupby("gn").p.transform("mean")
    if D.pc.std() == 0: return np.nan
    r = stats.spearmanr(D.tc, D.pc)[0]
    return np.nan if np.isnan(r) else float(r)


def run(featnames, drop_books, label):
    D = Dd[~Dd.unit.isin(drop_books)].copy()
    Xa = D[featnames].astype(float)
    k = (Xa.std() > 0) & (Xa.isna().mean() < 0.2)
    fe = list(np.array(featnames)[k.values])
    med = D[fe].astype(float).median()
    X = D[fe].astype(float).fillna(med).values
    XT = Dt[fe].astype(float).fillna(med).values
    y = D.date_bce.values.astype(float); g = D.unit.values
    gt = Dt.unit.values
    bk = list(pd.unique(g)); bd = {b: y[g == b][0] for b in bk}
    t, p, tgt_raw, R_self = [], [], {}, {}
    for b in bk:
        te = g == b; tr = ~te
        inner = [x for x in bk if x != b]
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
        tgt_raw[b] = fit_all(X[tr], y[tr], wv, XT)[bl]
    t, p = np.array(t), np.array(p)
    S = float(np.clip(t.std() / p.std(), 0.5, 8.0))
    cal = t.mean() + S * (p - p.mean())
    resid = t - cal
    gen = np.array([bgenre[b] for b in bk])
    pro = gen == "prophecy"
    ok = tot = 0
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            if t[i] == t[j]: continue
            tot += 1; ok += int((cal[i] - cal[j]) * (t[i] - t[j]) > 0)
    # final fit on every retained book, with lambda chosen by its own LOBO
    wb = dict(zip(bk, PT.wts([bd[x] for x in bk])))
    wv = np.array([wb[u] for u in g])
    err = {l: [] for l in PT.LAM}
    for b in bk:
        m = g != b
        pr = fit_all(X[m], y[m], wv[m], X[~m])
        for l in PT.LAM:
            err[l].append(abs(np.median(pr[l]) - bd[b]))
    bl = min(PT.LAM, key=lambda l: np.mean(err[l]))
    tgt = pd.Series(t.mean() + S * (fit_all(X, y, wv, XT)[bl] - p.mean())
                    ).groupby(gt).median()
    out = dict(label=label, n_feats=len(fe), n_books=len(bk),
               mae=float(np.abs(resid).mean()),
               rho_raw=float(stats.spearmanr(t, cal)[0]),
               rho_genre=genre_rho(gen, t, cal),
               rho_proph=float(stats.spearmanr(t[pro], cal[pro])[0])
               if pro.sum() > 3 else np.nan,
               p_proph=float(stats.spearmanr(t[pro], cal[pro])[1])
               if pro.sum() > 3 else np.nan,
               pair=ok / tot)
    for u in SRC:
        v = float(tgt[u])
        out[f"{u}_pred"] = round(v)
        out[f"{u}_ppost"] = round(float(np.mean((v + resid) < EX)), 2)
    return out


VARIANTS = [
    (ALL,  [],   "A  baseline: everything"),
    (MORF, [],   "B  morphosyntax only, no lexemes"),
    (LEXF, [],   "C  lexemes only, no morphosyntax"),
    (ALL,  SOFT, "D  externally anchored books only"),
    (MORF, SOFT, "E  morphosyntax only AND external anchors"),
]

print("=" * 96)
print("RED TEAM: THE STRONGEST CASE AGAINST THE RESULT")
print("=" * 96)
print(f"  {len(LEXF)} lexical features, {len(MORF)} morphosyntactic, "
      f"{len(SOFT)} soft anchors dropped in D and E\n")
print(f"  {'variant':<36}{'p':>5}{'bks':>5}{'MAE':>6}{'raw':>8}{'|genre':>8}"
      f"{'|proph':>8}{'pair':>7}{'JE':>6}{'D':>6}{'P':>6}")
print("  " + "-" * 92)
rows, t0 = [], time.time()
for fe, dr, lab in VARIANTS:
    r = run(fe, dr, lab)
    rows.append(r)
    print(f"  {lab:<36}{r['n_feats']:>5}{r['n_books']:>5}{r['mae']:>6.0f}"
          f"{r['rho_raw']:>+8.3f}{r['rho_genre']:>+8.3f}"
          f"{r['rho_proph']:>+8.3f}{100*r['pair']:>6.1f}%"
          f"{r['JE_source_pred']:>6.0f}{r['D_source_pred']:>6.0f}"
          f"{r['P_source_pred']:>6.0f}", flush=True)
R = pd.DataFrame(rows)
R.to_csv("/home/claude/red_team.csv", index=False)

print()
print("=" * 96)
print("ATTACK 4: IS IT BOOK LENGTH?")
print("=" * 96)
B = pd.read_csv("/home/claude/final_lobo_books.csv")
for col in ("n_chunks", "n_words"):
    r1, p1 = stats.spearmanr(B[col], B.resid.abs())
    r2, p2 = stats.spearmanr(B[col], B.truth)
    print(f"  {col:<10} vs |residual|  rho {r1:+.3f} (p={p1:.3f})   "
          f"vs true date  rho {r2:+.3f} (p={p2:.3f})")
print("  If short books were both noisier and systematically later, the")
print("  ordering could be a length artifact rather than a linguistic one.")

print()
print("=" * 96)
print("WHAT SURVIVES")
print("=" * 96)
base = R.iloc[0]
for _, r in R.iterrows():
    verdict = ("holds" if (r.rho_genre > 0.2 and r.rho_raw > 0.4)
               else "FAILS")
    print(f"  {r.label:<36} {verdict}")
print(f"\n  P(post-exilic) for the three sources by variant:")
for _, r in R.iterrows():
    print(f"    {r.label:<36} " + "  ".join(
        f"{u.split('_')[0]} {r[f'{u}_ppost']:.2f}" for u in SRC))
json.dump(R.to_dict("records"), open("/home/claude/red_team.json", "w"), indent=2)
print(f"\n{(time.time()-t0)/60:.1f} min.  wrote red_team.csv, red_team.json")
