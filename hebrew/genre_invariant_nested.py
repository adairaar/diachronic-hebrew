"""
The genre screen, with the retention fraction chosen inside the folds.

The sweep in genre_invariant.py reported its best row at 50% retention.  That
number cannot be quoted.  Reading five rows and keeping the best one is
selection on the validation set, and the resulting figure is optimistic by an
unknown amount -- the same mistake the nested configuration search was built to
avoid earlier in this project, and it would be embarrassing to make it twice.

So the retention fraction is chosen here the way lambda already is: inside each
leave-one-book-out fold, on an inner leave-one-book-out loop over the remaining
24 books, with the held-out book taking no part in its own configuration.  The
reported number is then honest by construction.

The inner criterion is genre-controlled rho, not raw rho.  This matters as much
as the nesting.  Raw rho on this corpus is 77% between-genre variance, so a
search that optimises it selects for genre discrimination; that is how the
configuration got here in the first place.

Lambda and the retention fraction are chosen jointly, since the eigendecomposition
makes the whole lambda path cost what one lambda costs.
"""
import json, time
import numpy as np, pandas as pd, importlib.util
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", "/home/claude/predict_targets.py")
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

EX = 586
SRC = ["JE_source", "D_source", "P_source"]
FRACS = [1.00, 0.75, 0.50, 0.30]

Dd = pd.read_csv("/home/claude/big_features_500.csv")
Dt = pd.read_csv("/home/claude/target_chunks_500.csv")
feats = [c for c in Dd.columns if c not in PT.META and c in Dt.columns]
Xa = Dd[feats].astype(float)
keep = (Xa.std() > 0) & (Xa.isna().mean() < 0.2)
feats = list(np.array(feats)[keep.values])
med = Dd[feats].astype(float).median()
X = Dd[feats].astype(float).fillna(med).values
XT = Dt[feats].astype(float).fillna(med).values
y = Dd.date_bce.values.astype(float); g = Dd.unit.values
gt = Dt.unit.values
books = list(pd.unique(g)); bdate = {b: y[g == b][0] for b in books}
bgenre = Dd.groupby("unit").genre.first().to_dict()
P = len(feats)

BM = pd.DataFrame(X, columns=feats).assign(unit=g).groupby("unit").mean().loc[books]
BMv = BM.values
GEN = np.array([bgenre[b] for b in books])
BIG = np.isin(GEN, ["prophecy", "narrative"])


def genre_eta2(train_books):
    m = np.isin(books, train_books) & BIG
    Z = BMv[m]; gg = GEN[m]
    tot = ((Z - Z.mean(0)) ** 2).sum(0)
    bet = np.zeros(Z.shape[1])
    for lev in np.unique(gg):
        s = gg == lev
        bet += s.sum() * (Z[s].mean(0) - Z.mean(0)) ** 2
    return bet / np.where(tot > 0, tot, np.inf)


def fit_all_lams(Xtr, ytr, wtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 0, sd, 1.0)
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    yb = np.average(ytr, weights=wtr)
    s = np.sqrt(wtr)
    Aw = A * s[:, None]; yw = (ytr - yb) * s
    ev, V = np.linalg.eigh(Aw @ Aw.T)
    Vt_y = V.T @ yw
    BAt = (B @ Aw.T) @ V
    return {lam: BAt @ (Vt_y / (ev + lam)) + yb for lam in PT.LAM}


def genre_rho(bk, tv, pv):
    """Spearman after centring truth and prediction within genre.

    Only genres with more than one book contribute; a singleton centres to zero
    and carries no ordering information.
    """
    D = pd.DataFrame(dict(b=bk, t=tv, p=pv,
                          gn=[bgenre[x] for x in bk]))
    D = D[D.gn.isin(["prophecy", "narrative"])].copy()
    if D.gn.nunique() < 1 or len(D) < 5: return -np.inf
    D["tc"] = D.t - D.groupby("gn").t.transform("mean")
    D["pc"] = D.p - D.groupby("gn").p.transform("mean")
    if D.pc.std() == 0: return -np.inf
    r = stats.spearmanr(D.tc, D.pc)[0]
    return -np.inf if np.isnan(r) else float(r)


def choose(train_books):
    """Pick (frac, lambda) by inner LOBO over `train_books` on genre-controlled rho."""
    eta_cache = {}
    inner_pred = {(f, l): {} for f in FRACS for l in PT.LAM}
    for bb in train_books:
        rest = [x for x in train_books if x != bb]
        eta = genre_eta2(rest)
        wb = dict(zip(rest, PT.wts([bdate[x] for x in rest])))
        tr = np.isin(g, rest); te = g == bb
        wv = np.array([wb[u] for u in g[tr]])
        for f in FRACS:
            k = max(int(round(f * P)), 20)
            sel = np.argsort(eta)[:k]
            pr = fit_all_lams(X[np.ix_(tr, sel)], y[tr], wv, X[np.ix_(te, sel)])
            for l in PT.LAM:
                inner_pred[(f, l)][bb] = float(np.median(pr[l]))
    best, bf, blam = -np.inf, FRACS[0], PT.LAM[0]
    for f in FRACS:
        for l in PT.LAM:
            d = inner_pred[(f, l)]
            bk = list(d)
            r = genre_rho(bk, [bdate[x] for x in bk], [d[x] for x in bk])
            if r > best: best, bf, blam = r, f, l
    return bf, blam, best


t0 = time.time()
print("=" * 78)
print("NESTED GENRE-INVARIANT MODEL")
print("=" * 78)
print(f"  retention fraction and lambda both chosen inside each fold,")
print(f"  on genre-controlled rho over the 24 training books\n")

truth, pred, chosen = [], [], []
for i, b in enumerate(books):
    inner = [x for x in books if x != b]
    f, lam, r_in = choose(inner)
    eta = genre_eta2(inner)
    k = max(int(round(f * P)), 20)
    sel = np.argsort(eta)[:k]
    wb = dict(zip(inner, PT.wts([bdate[x] for x in inner])))
    tr = g != b; te = g == b
    wv = np.array([wb[u] for u in g[tr]])
    pr = fit_all_lams(X[np.ix_(tr, sel)], y[tr], wv, X[np.ix_(te, sel)])[lam]
    truth.append(bdate[b]); pred.append(float(np.median(pr)))
    chosen.append(dict(book=b, frac=f, lam=float(lam), inner_rho=r_in))
    print(f"  [{i+1:>2}/25] {b:<14} kept {100*f:>3.0f}%  lambda {lam:.0e}  "
          f"inner rho|genre {r_in:+.3f}   ({(time.time()-t0)/60:.1f} min)", flush=True)

t = np.array(truth); p = np.array(pred)
S = float(np.clip(t.std() / p.std(), 0.5, 8.0))
cal = t.mean() + S * (p - p.mean())
resid = t - cal

D = pd.DataFrame(dict(book=books, truth=t, pred=cal, resid=resid,
                      genre=[bgenre[b] for b in books]))
D.to_csv("/home/claude/genre_invariant_nested_books.csv", index=False)
CH = pd.DataFrame(chosen)
CH.to_csv("/home/claude/genre_invariant_nested_choices.csv", index=False)

rho_raw = stats.spearmanr(t, cal)[0]
rho_par = genre_rho(books, t, cal)
pro = D[D.genre == "prophecy"]
rho_pro, p_pro = stats.spearmanr(pro.truth, pro.pred)
ok = tot = 0
for i in range(len(t)):
    for j in range(i + 1, len(t)):
        if t[i] == t[j]: continue
        tot += 1; ok += int((cal[i] - cal[j]) * (t[i] - t[j]) > 0)

print()
print("=" * 78)
print("RESULT, WITH NOTHING CHOSEN ON THE HELD-OUT BOOK")
print("=" * 78)
print(f"  retention fractions chosen: "
      + ", ".join(f"{100*f:.0f}% x{(CH.frac == f).sum()}" for f in FRACS
                  if (CH.frac == f).sum()))
print(f"  MAE                       {np.abs(resid).mean():>8.0f} yr")
print(f"  rho raw                   {rho_raw:>+8.3f}")
print(f"  rho genre-controlled      {rho_par:>+8.3f}")
print(f"  rho within prophecy       {rho_pro:>+8.3f}  (p = {p_pro:.3f}, n = {len(pro)})")
print(f"  pairwise ordering         {100*ok/tot:>7.1f}%")

# targets, using the modal configuration refit on all 25
modal_f = CH.frac.mode()[0]
modal_l = CH.lam.mode()[0]
eta = genre_eta2(books)
sel = np.argsort(eta)[:max(int(round(modal_f * P)), 20)]
wb = dict(zip(books, PT.wts([bdate[x] for x in books])))
wv = np.array([wb[u] for u in g])
raw = fit_all_lams(X[:, sel], y, wv, XT[:, sel])[modal_l]
tgt = pd.Series(t.mean() + S * (raw - p.mean())).groupby(gt).median()
print(f"\n  targets, modal configuration ({100*modal_f:.0f}% kept, "
      f"lambda {modal_l:.0e}):")
print(f"  {'unit':<15}{'date':>7}{'P(post-exilic)':>17}")
rows = []
for u in SRC + ["D_Code", "D_Frame", "Lev_Holiness", "Lev_Priestly", "Jer_DTR",
                "Song_Sea", "Song_Deborah"]:
    if u not in tgt.index: continue
    v = float(tgt[u]); pp = float(np.mean((v + resid) < EX))
    rows.append(dict(unit=u, pred=round(v), p_post=round(pp, 2)))
    print(f"  {u:<15}{v:>7.0f}{pp:>17.2f}")
pd.DataFrame(rows).to_csv("/home/claude/genre_invariant_nested_targets.csv",
                          index=False)

json.dump(dict(mae=float(np.abs(resid).mean()), rho_raw=float(rho_raw),
               rho_partial=float(rho_par), rho_prophecy=float(rho_pro),
               p_prophecy=float(p_pro), pair=ok / tot, S=S,
               modal_frac=float(modal_f), modal_lam=float(modal_l),
               n_feats=int(len(sel)),
               frac_counts={f"{f:.2f}": int((CH.frac == f).sum()) for f in FRACS},
               preds={r["unit"]: r["pred"] for r in rows},
               p_post={r["unit"]: r["p_post"] for r in rows}),
          open("/home/claude/genre_invariant_nested.json", "w"), indent=2)
print(f"\ntotal {(time.time()-t0)/60:.1f} min")
print("wrote genre_invariant_nested{,_books,_choices,_targets}")
