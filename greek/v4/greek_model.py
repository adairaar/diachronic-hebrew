"""
Greek replication + the archaizing measurement Hebrew cannot supply.

Design mirrors the Hebrew pipeline exactly:
  ~500-token chunks, wide feature set, inverse-density weighting by date,
  ridge with lambda chosen by inner leave-one-TEXT-out, variance matching,
  conformal intervals on leave-one-text-out residuals.

The point of the Greek corpus is that archaizing is LABELLED here.  The Second
Sophistic Atticizers -- Lucian, Aelian, Philostratus, Aelius Aristides and
company -- deliberately imitated Classical Attic four to seven centuries after
the fact, and their true composition dates are securely known.  They are held
out of training entirely.  The displacement between their true date and the
date the model assigns them is the cost of skilled, sustained, deliberate
archaizing, measured directly.

Dates are CE (negative = BCE).  A NEGATIVE displacement means the text looks
OLDER than it is, i.e. the archaizing worked.
"""
import os, json, argparse
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
META = {"chunk_id", "unit", "date_ce", "register", "genre", "author", "n_tok"}
LAM = 10.0 ** np.arange(1, 6.1, 1.0)
ALPHA, BW = 1.0, 90.0


def wts(dates, alpha=ALPHA, bw=BW):
    d = np.asarray(dates, float)
    if alpha == 0: return np.ones_like(d)
    dens = np.array([np.exp(-0.5 * ((d - x) / bw) ** 2).sum() for x in d])
    w = (1.0 / np.maximum(dens, 1e-9)) ** alpha
    return w / w.mean()


class Gram:
    """Rank-updatable primal ridge solver.

    The Greek corpus has many more passages (~4200) than features (747), so the
    primal normal equations are the cheap route -- and, crucially, the inner
    leave-one-text-out folds differ from the full training set only by the
    removal of one text's rows.  Accumulating

        G  = sum_i w_i a_i a_i'      (p x p)
        S1 = sum_i w_i a_i y_i       (p,)
        S0 = sum_i w_i a_i           (p,)
        T1 = sum_i w_i y_i,  T0 = sum_i w_i

    once over the outer training fold makes each inner fold a subtraction plus
    one p x p eigendecomposition, instead of a fresh n x p decomposition.  That
    is the difference between minutes and hours at this corpus size.

    Standardisation constants and weights are computed on the OUTER training
    fold and held fixed through the inner lambda search.  The held-out text
    contributes to neither, so this is leakage-free with respect to the
    quantity being scored; the inner folds serve only to choose lambda.
    """

    def __init__(self, Xtr, ytr, wtr):
        self.mu = Xtr.mean(0)
        sd = Xtr.std(0); self.sd = np.where(sd > 0, sd, 1.0)
        A = (Xtr - self.mu) / self.sd
        self.Aw = A * wtr[:, None]          # w_i a_i, row-wise
        self.A = A
        self.y = ytr; self.w = wtr
        self.G = A.T @ self.Aw
        self.S1 = self.Aw.T @ ytr
        self.S0 = self.Aw.sum(0)
        self.T1 = float(wtr @ ytr); self.T0 = float(wtr.sum())

    def solve(self, Xte, drop=None):
        """Predictions for every lambda, optionally dropping a boolean row mask."""
        if drop is None or not drop.any():
            G, S1, S0, T1, T0 = self.G, self.S1, self.S0, self.T1, self.T0
        else:
            Ad, Awd = self.A[drop], self.Aw[drop]
            G = self.G - Ad.T @ Awd
            S1 = self.S1 - Awd.T @ self.y[drop]
            S0 = self.S0 - Awd.sum(0)
            T1 = self.T1 - float(self.w[drop] @ self.y[drop])
            T0 = self.T0 - float(self.w[drop].sum())
        yb = T1 / T0
        b = S1 - yb * S0
        ev, V = np.linalg.eigh(G)
        ev = np.maximum(ev, 0.0)
        Vtb = V.T @ b
        B = (Xte - self.mu) / self.sd
        BV = B @ V
        return {lam: BV @ (Vtb / (ev + lam)) + yb for lam in LAM}


def fit_all_lams(Xtr, ytr, wtr, Xte):
    return Gram(Xtr, ytr, wtr).solve(Xte)


def main(target, drop_lxx, npermute):
    D = pd.read_csv(os.path.join(HERE, f"greek_chunks_{target}.csv"))
    D = D[D.n_tok >= 100].copy()
    if drop_lxx:
        D = D[D.register != "LXX"].copy()
    att = D.register == "Atticizing"
    Tr, Te = D[~att].copy(), D[att].copy()

    feats = [c for c in D.columns if c not in META]
    Xa = Tr[feats].astype(float)
    keep = (Xa.std() > 0) & (Xa.isna().mean() < 0.2)
    feats = list(np.array(feats)[keep.values])
    med = Tr[feats].astype(float).median()
    X = Tr[feats].astype(float).fillna(med).values
    Xt = Te[feats].astype(float).fillna(med).values
    y = Tr.date_ce.values.astype(float); g = Tr.unit.values
    gt = Te.unit.values
    texts = list(pd.unique(g)); tdate = {t: y[g == t][0] for t in texts}
    print(f"{len(feats)} features | train {len(Tr)} chunks / {len(texts)} texts "
          f"({int(min(tdate.values()))} to {int(max(tdate.values()))} CE)"
          f" | holdout {len(Te)} chunks / {Te.unit.nunique()} Atticizing texts")
    print(f"training registers: {dict(Tr.groupby('register').unit.nunique())}\n")

    # ---------- leave-one-text-out on the non-archaizing corpus ----------
    def lobo(yv):
        td = {t: yv[g == t][0] for t in texts}
        tt, pp = [], []
        for t in texts:
            te = g == t; tr = ~te
            inner = [x for x in texts if x != t]
            wb = dict(zip(inner, wts([td[x] for x in inner])))
            wv = np.array([wb[u] for u in g[tr]])
            gr = Gram(X[tr], yv[tr], wv)          # built once per outer fold
            gin = g[tr]
            err = {lam: [] for lam in LAM}
            for bb in inner:
                m = gin == bb
                pr = gr.solve(X[tr][m], drop=m)   # rank-update, not a refit
                for lam in LAM: err[lam].append(abs(np.median(pr[lam]) - td[bb]))
            blam = min(LAM, key=lambda l: np.mean(err[l]))
            tt.append(td[t])
            pp.append(float(np.median(gr.solve(X[te])[blam])))
        return np.array(tt), np.array(pp)

    def metrics(t, p):
        S = float(np.clip(t.std() / p.std(), 0.5, 8.0))
        cal = t.mean() + S * (p - p.mean())
        r = t - cal
        rho, rp = stats.spearmanr(t, cal)
        n = len(t); ok = tot = 0
        for i in range(n):
            for j in range(i + 1, n):
                if t[i] == t[j]: continue
                tot += 1; ok += int((cal[i] - cal[j]) * (t[i] - t[j]) > 0)
        return dict(S=S, cal=cal, resid=r, mae=float(np.abs(r).mean()),
                    rho=float(rho), rho_p=float(rp), pair=ok / tot, n_pair=tot)

    t, p = lobo(y)
    M = metrics(t, p); cal, resid = M["cal"], M["resid"]
    ar = np.sort(np.abs(resid)); n = len(ar)
    q68 = ar[min(int(np.ceil((n + 1) * .68)) - 1, n - 1)]
    q90 = ar[min(int(np.ceil((n + 1) * .90)) - 1, n - 1)]
    base = float(np.abs(t - t.mean()).mean())
    print(f"variance-match scale S = {M['S']:.2f}")
    print(f"LOTO MAE {M['mae']:.1f} yr   (mean-predictor baseline {base:.1f} yr)")
    print(f"Spearman rho {M['rho']:+.3f} (p={M['rho_p']:.2g})")
    print(f"pairwise ordering {M['pair']*100:.1f}% of {M['n_pair']} pairs")
    print(f"conformal 68% +/-{q68:.0f} yr, 90% +/-{q90:.0f} yr\n")

    L = pd.DataFrame(dict(text=texts, truth=t.astype(int),
                          pred=np.round(cal).astype(int),
                          resid=np.round(resid).astype(int),
                          register=[Tr.register[g == x].iloc[0] for x in texts]))
    L = L.sort_values("truth")
    L.to_csv(os.path.join(HERE, "greek_loto_texts.csv"), index=False)
    print(L.to_string(index=False))

    # ---------- apply to the Atticizers ----------
    wb = dict(zip(texts, wts([tdate[x] for x in texts])))
    wv = np.array([wb[u] for u in g])
    gr = Gram(X, y, wv)
    err = {lam: [] for lam in LAM}
    for t_ in texts:
        m = g == t_
        pr = gr.solve(X[m], drop=m)
        for lam in LAM: err[lam].append(abs(np.median(pr[lam]) - tdate[t_]))
    blam = min(LAM, key=lambda l: np.mean(err[l]))
    raw = gr.solve(Xt)[blam]
    predt = t.mean() + M["S"] * (raw - p.mean())

    rows = []
    for u in pd.unique(gt):
        s = predt[gt == u]
        tru = float(Te.date_ce[gt == u].iloc[0])
        pv = float(np.median(s))
        rows.append(dict(text=u, author=Te.author[gt == u].iloc[0],
                         truth=int(tru), pred=round(pv), shift=round(pv - tru),
                         n_chunks=int((gt == u).sum()),
                         lo68=round(pv - q68), hi68=round(pv + q68)))
    A = pd.DataFrame(rows).sort_values("truth")
    A.to_csv(os.path.join(HERE, "greek_atticizers.csv"), index=False)
    print("\n" + "=" * 78)
    print("SECOND SOPHISTIC ATTICIZERS: apparent date vs true date")
    print("(negative shift = the text is dated EARLIER than it was written)")
    print("=" * 78)
    print(f"{'text':<32}{'author':<18}{'true':>6}{'pred':>7}{'shift':>8}{'chunks':>8}")
    for _, r in A.iterrows():
        print(f"{r.text:<32}{r.author:<18}{r.truth:6d}{r.pred:7d}{r['shift']:+8d}{r.n_chunks:8d}")
    print(f"\n  mean displacement   {A['shift'].mean():+.0f} yr")
    print(f"  median displacement {A['shift'].median():+.0f} yr")
    print(f"  texts dated too early: {(A['shift'] < 0).sum()}/{len(A)}")
    print(f"  LOTO MAE on non-archaizing texts, for comparison: {M['mae']:.0f} yr")

    json.dump(dict(n_feats=len(feats), n_train_chunks=int(len(Tr)),
                   n_train_texts=len(texts), n_att_texts=int(Te.unit.nunique()),
                   n_att_chunks=int(len(Te)), S=M["S"], mae=M["mae"],
                   mae_baseline=base, rho=M["rho"], rho_p=M["rho_p"],
                   pair=M["pair"], n_pair=M["n_pair"], q68=float(q68), q90=float(q90),
                   att_mean_shift=float(A['shift'].mean()),
                   att_median_shift=float(A["shift"].median()),
                   att_n_early=int((A["shift"] < 0).sum())),
              open(os.path.join(HERE, "greek_metrics.json"), "w"), indent=2)

    # ---------- permutation null ----------
    if npermute:
        rng = np.random.default_rng(0); nulls = []
        for k in range(npermute):
            perm = rng.permutation(texts)
            remap = {b: tdate[perm[i]] for i, b in enumerate(texts)}
            yp = np.array([remap[u] for u in g], float)
            tt, pp = lobo(yp); mm = metrics(tt, pp)
            nulls.append((mm["rho"], mm["pair"], mm["mae"]))
            if (k + 1) % 10 == 0: print(f"  perm {k+1}/{npermute}", flush=True)
        N = np.array(nulls)
        print(f"\npermutation null ({npermute} draws, full pipeline):")
        print(f"  rho  obs {M['rho']:+.3f} null med {np.median(N[:,0]):+.3f} "
              f"p={(np.sum(N[:,0]>=M['rho'])+1)/(npermute+1):.4f}")
        print(f"  pair obs {M['pair']*100:.1f}% null med {np.median(N[:,1])*100:.1f}% "
              f"p={(np.sum(N[:,1]>=M['pair'])+1)/(npermute+1):.4f}")
        np.savetxt(os.path.join(HERE, "greek_null.csv"), N, delimiter=",",
                   header="rho,pair,mae", comments="")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--keep-lxx", action="store_true")
    ap.add_argument("--permute", type=int, default=0)
    a = ap.parse_args()
    main(a.target, not a.keep_lxx, a.permute)
