"""
A model that does not lean on genre in the first place.

The post-hoc correction was the wrong instrument.  It presumes the genre effect
is a constant offset per genre, estimates that offset from as little as one
book, and needs a genre label on the undated targets in order to apply it --
which is exactly what we do not have for the Pentateuchal sources.

The instrument used here is feature screening.  Rank each feature by how well it
discriminates GENRE among the anchor books, and discard the most genre-diagnostic
before fitting anything.  Two properties make this the right tool:

  * The screen never touches the dates.  It cannot leak the outcome, which
    matters because the same corpus is doing double duty as training and
    validation set.
  * It needs no genre label on the targets.  Whatever survives the screen is
    applied to the sources exactly as it is applied to the anchors.

Discriminability is measured at BOOK level, not chunk level: chunks within a
book are heavily correlated and a chunk-level F would be inflated by a factor of
the design effect.  Only prophecy (17 books) and narrative (5) enter the screen;
the three singleton genres cannot contribute a within-genre variance.

Selection runs INSIDE each leave-one-book-out fold, on training books only.

The evaluation metric changes too, and that is half the point.  Raw Spearman on
the 25 anchors is 77% between-genre variance, so every configuration choice made
by optimising it has been rewarding the confound.  What is reported here is
genre-controlled ordering: rho after centring truth and prediction within genre,
and rho within prophecy alone -- 17 books over 410 years in a single register,
the best-controlled comparison this corpus permits.
"""
import json, sys
import numpy as np, pandas as pd, importlib.util
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", "/home/claude/predict_targets.py")
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

EX = 586
SRC = ["JE_source", "D_source", "P_source"]
KEEP_FRACS = [1.00, 0.75, 0.50, 0.30, 0.15]

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

# book-level feature means: the unit at which genre discriminability is honest
BM = pd.DataFrame(X, columns=feats).assign(unit=g).groupby("unit").mean()
BM = BM.loc[books]
BMv = BM.values
GEN = np.array([bgenre[b] for b in books])
BIG = np.isin(GEN, ["prophecy", "narrative"])


def genre_eta2(train_books):
    """Share of each feature's between-book variance attributable to genre.

    Uses only `train_books`, and only the two genres with more than one book.
    No date enters this function.
    """
    m = np.isin(books, train_books) & BIG
    Z = BMv[m]; gg = GEN[m]
    tot = ((Z - Z.mean(0)) ** 2).sum(0)
    bet = np.zeros(Z.shape[1])
    for lev in np.unique(gg):
        s = gg == lev
        bet += s.sum() * (Z[s].mean(0) - Z.mean(0)) ** 2
    return bet / np.where(tot > 0, tot, np.inf)


def fit_all_lams(Xtr, ytr, wtr, Xte):
    """Ridge predictions for every lambda from one eigendecomposition."""
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 0, sd, 1.0)
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    yb = np.average(ytr, weights=wtr)
    s = np.sqrt(wtr)
    Aw = A * s[:, None]; yw = (ytr - yb) * s
    ev, V = np.linalg.eigh(Aw @ Aw.T)
    Vt_y = V.T @ yw
    BAt = (B @ Aw.T) @ V
    return {lam: BAt @ (Vt_y / (ev + lam)) + yb for lam in PT.LAM}


def run(frac):
    """Full LOBO with the genre screen re-run inside every fold."""
    t, p, tgt_folds, nkept = [], [], [], []
    for b in books:
        te = g == b; tr = ~te
        inner = [x for x in books if x != b]
        eta = genre_eta2(inner)                    # training books only
        k = max(int(round(frac * P)), 20)
        sel = np.argsort(eta)[:k]                  # least genre-diagnostic
        nkept.append(k)
        wb = dict(zip(inner, PT.wts([bdate[x] for x in inner])))
        wv = np.array([wb[u] for u in g[tr]])
        Xs, XTs = X[:, sel], XT[:, sel]
        err = {lam: [] for lam in PT.LAM}
        for bb in inner:
            m = g[tr] != bb
            pr = fit_all_lams(Xs[tr][m], y[tr][m], wv[m], Xs[tr][~m])
            for lam in PT.LAM:
                err[lam].append(abs(np.median(pr[lam]) - bdate[bb]))
        bl = min(PT.LAM, key=lambda l: np.mean(err[l]))
        t.append(bdate[b])
        p.append(float(np.median(fit_all_lams(Xs[tr], y[tr], wv, Xs[te])[bl])))
    t, p = np.array(t), np.array(p)
    S = float(np.clip(t.std() / p.std(), 0.5, 8.0))
    cal = t.mean() + S * (p - p.mean())
    resid = t - cal

    # final fit on all anchors, same screen, for the target predictions
    eta = genre_eta2(books)
    k = max(int(round(frac * P)), 20)
    sel = np.argsort(eta)[:k]
    wb = dict(zip(books, PT.wts([bdate[x] for x in books])))
    wv = np.array([wb[u] for u in g])
    err = {lam: [] for lam in PT.LAM}
    for b in books:
        m = g != b
        pr = fit_all_lams(X[:, sel][m], y[m], wv[m], X[:, sel][~m])
        for lam in PT.LAM:
            err[lam].append(abs(np.median(pr[lam]) - bdate[b]))
    bl = min(PT.LAM, key=lambda l: np.mean(err[l]))
    raw = fit_all_lams(X[:, sel], y, wv, XT[:, sel])[bl]
    tgt = pd.Series(t.mean() + S * (raw - p.mean())).groupby(gt).median()
    return t, cal, resid, tgt, int(np.mean(nkept)), [feats[i] for i in sel]


def metrics(t, cal):
    D = pd.DataFrame(dict(t=t, p=cal, genre=[bgenre[b] for b in books]))
    raw = stats.spearmanr(D.t, D.p)[0]
    big = D[D.genre.isin(["prophecy", "narrative"])].copy()
    big["tc"] = big.t - big.groupby("genre").t.transform("mean")
    big["pc"] = big.p - big.groupby("genre").p.transform("mean")
    par = stats.spearmanr(big.tc, big.pc)[0]
    pro = D[D.genre == "prophecy"]
    rp, pp = stats.spearmanr(pro.t, pro.p)
    gm = D.groupby("genre").p.mean()
    ss_t = float(((D.p - D.p.mean()) ** 2).sum())
    ss_b = float(sum((D.genre == k).sum() * (v - D.p.mean()) ** 2
                     for k, v in gm.items()))
    return dict(rho_raw=float(raw), rho_partial=float(par),
                rho_prophecy=float(rp), p_prophecy=float(pp),
                between_genre=float(ss_b / ss_t))


print("=" * 78)
print("GENRE-INVARIANT FEATURE SCREEN")
print("=" * 78)
print(f"  {P} features, {len(books)} anchor books, screen refitted in every fold")
print(f"  eta-squared computed on {int(BIG.sum())} books in prophecy or narrative\n")
print(f"  {'kept':>6}{'n':>6}{'MAE':>7}{'rho raw':>10}{'rho|genre':>11}"
      f"{'rho proph':>11}{'btwn-genre':>12}" + "".join(f"{u.split('_')[0]:>8}" for u in SRC))
print("  " + "-" * 88)
OUT = {}
for frac in KEEP_FRACS:
    t, cal, resid, tgt, nk, selnames = run(frac)
    M = metrics(t, cal)
    ps = {u: float(np.mean((float(tgt[u]) + resid) < EX)) for u in SRC}
    OUT[f"{frac:.2f}"] = dict(frac=frac, n_feats=nk, mae=float(np.abs(resid).mean()),
                              preds={u: round(float(tgt[u])) for u in SRC},
                              p_post={u: round(v, 2) for u, v in ps.items()},
                              **M)
    print(f"  {100*frac:>5.0f}%{nk:>6}{np.abs(resid).mean():>7.0f}"
          f"{M['rho_raw']:>+10.3f}{M['rho_partial']:>+11.3f}"
          f"{M['rho_prophecy']:>+11.3f}{100*M['between_genre']:>11.0f}%"
          + "".join(f"{float(tgt[u]):>8.0f}" for u in SRC), flush=True)
    if frac == 0.50:
        json.dump(selnames, open("/home/claude/genre_invariant_features.json", "w"))

print()
print("  rho raw    ordering on all 25 anchors, the number the paper reports")
print("  rho|genre  after centring truth and prediction within genre")
print("  rho proph  within the 17 prophetic books alone, 760-350 BCE")
print("  btwn-genre share of the prediction variance lying between genres")
print()
print("  P(post-exilic) for the three sources:")
for k, v in OUT.items():
    print(f"    kept {float(k)*100:>3.0f}%  " +
          "  ".join(f"{u.split('_')[0]} {v['p_post'][u]:.2f}" for u in SRC))

json.dump(OUT, open("/home/claude/genre_invariant.json", "w"), indent=2)
print("\nwrote genre_invariant.json, genre_invariant_features.json")
