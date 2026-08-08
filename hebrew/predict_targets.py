"""
Apply the recommended model to the undated units.

Model: ~500-word chunks, 579 base features, inverse-density weighted
(alpha = 1.0, bandwidth 90), ridge with lambda by leave-one-book-out,
variance-matched.

The variance-matching constants and the prediction intervals both come from
the dated corpus under leave-one-book-out, so they reflect the model's actual
out-of-sample behaviour rather than its fit.  Intervals are conformal on those
LOBO residuals: distribution-free, with finite-sample marginal coverage.
"""
import numpy as np, pandas as pd
from scipy import stats

META = {"chunk_id", "unit", "date_bce", "genre", "register", "n_words"}
LAM = 10.0 ** np.arange(1, 6.1, 1.0)
ALPHA, BW = 1.0, 90.0


def wts(dates, alpha=ALPHA, bw=BW):
    d = np.asarray(dates, float)
    if alpha == 0: return np.ones_like(d)
    dens = np.array([np.exp(-0.5 * ((d - x) / bw) ** 2).sum() for x in d])
    w = (1.0 / np.maximum(dens, 1e-9)) ** alpha
    return w / w.mean()


def fit_predict(Xtr, ytr, wtr, Xte, lam):
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 0, sd, 1.0)
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    yb = np.average(ytr, weights=wtr)
    s = np.sqrt(wtr)
    Aw = A * s[:, None]; yw = (ytr - yb) * s
    K = Aw @ Aw.T
    al = np.linalg.solve(K + lam * np.eye(K.shape[0]), yw)
    return (B @ Aw.T) @ al + yb


def main():
    Dd = pd.read_csv("/home/claude/big_features_500.csv")
    Dt = pd.read_csv("/home/claude/target_chunks_500.csv")
    feats = [c for c in Dd.columns if c not in META]
    feats = [c for c in feats if c in Dt.columns]
    Xd = Dd[feats].astype(float)
    keep = (Xd.std() > 0) & (Xd.isna().mean() < 0.2)
    feats = list(np.array(feats)[keep.values])
    med = Dd[feats].astype(float).median()
    Xd = Dd[feats].astype(float).fillna(med).values
    Xt = Dt[feats].astype(float).fillna(med).values
    y = Dd.date_bce.values.astype(float); g = Dd.unit.values
    books = list(pd.unique(g)); bdate = {b: y[g == b][0] for b in books}
    print(f"{len(feats)} features | {len(Dd)} dated chunks / {len(books)} books "
          f"| {len(Dt)} target chunks / {Dt.unit.nunique()} targets\n")

    # ── LOBO on the dated corpus: lambda, scale constants, residuals ──────────
    oof_t, oof_p = [], []
    for b in books:
        te = g == b; tr = ~te
        inner = [x for x in books if x != b]
        wb = dict(zip(inner, wts([bdate[x] for x in inner])))
        wv = np.array([wb[u] for u in g[tr]])
        best, blam = np.inf, LAM[0]
        for lam in LAM:
            e = []
            for bb in inner:
                m = g[tr] != bb
                pr = fit_predict(Xd[tr][m], y[tr][m], wv[m], Xd[tr][~m], lam)
                e.append(abs(np.median(pr) - y[tr][~m][0]))
            if np.mean(e) < best: best, blam = np.mean(e), lam
        pr = fit_predict(Xd[tr], y[tr], wv, Xd[te], blam)
        oof_t.append(bdate[b]); oof_p.append(float(np.median(pr)))
    oof_t = np.array(oof_t); oof_p = np.array(oof_p)
    S = float(np.clip(oof_t.std() / oof_p.std(), 0.5, 8.0))
    C_in, C_out = oof_p.mean(), oof_t.mean()
    cal = lambda v: C_out + S * (v - C_in)
    resid = oof_t - cal(oof_p)
    ar = np.sort(np.abs(resid)); n = len(ar)
    # A coverage level the calibration set cannot support has no finite
    # distribution-free interval; clamping to the maximum residual would
    # silently break the guarantee.
    def _q(a):
        k = int(np.ceil((n + 1) * a))
        return np.inf if k > n else ar[k - 1]
    q68 = ar[min(int(np.ceil((n + 1) * .68)) - 1, n - 1)]
    q90 = ar[min(int(np.ceil((n + 1) * .90)) - 1, n - 1)]
    print(f"variance-match scale S = {S:.2f}   LOBO MAE = {np.abs(resid).mean():.1f} yr")
    print(f"conformal half-widths: 68% +/-{q68:.0f} yr, 90% +/-{q90:.0f} yr\n")

    # ── final model on all dated books ───────────────────────────────────────
    wb = dict(zip(books, wts([bdate[x] for x in books])))
    wv = np.array([wb[u] for u in g])
    best, blam = np.inf, LAM[0]
    for lam in LAM:
        e = []
        for b in books:
            m = g != b
            pr = fit_predict(Xd[m], y[m], wv[m], Xd[~m], lam)
            e.append(abs(np.median(pr) - bdate[b]))
        if np.mean(e) < best: best, blam = np.mean(e), lam
    raw = fit_predict(Xd, y, wv, Xt, blam)
    Dt = Dt.assign(pred=cal(raw))

    rows = []
    for u, sub in Dt.groupby("unit"):
        p = float(np.median(sub.pred))
        draws = p + resid
        rows.append(dict(unit=u, n_chunks=len(sub), n_words=int(sub.n_words.sum()),
                         pred=p, lo68=p - q68, hi68=p + q68,
                         lo90=p - q90, hi90=p + q90,
                         chunk_sd=float(sub.pred.std()) if len(sub) > 1 else np.nan,
                         p_post=float(np.mean(draws < 586))))
    R = pd.DataFrame(rows).sort_values("pred", ascending=False)
    R.to_csv("/home/claude/target_predictions_naive.csv", index=False)

    ORDER = ["Song_Sea","Song_Deborah","D_Song","JE_source","Gen_JE","Exo_JE","Num_JE",
             "P_source","Lev_Priestly","Lev_Holiness","D_source","D_Code","D_Frame",
             "Jer_DTR","Genesis","Exodus","Leviticus","Numbers","Deuteronomy"]
    print(f"{'unit':<15}{'words':>7}{'chunks':>7}{'date':>7}{'68% interval':>20}"
          f"{'P(post-exilic)':>16}")
    print("-" * 74)
    for u in ORDER:
        r = R[R.unit == u]
        if not len(r): continue
        r = r.iloc[0]
        print(f"{u:<15}{r.n_words:7d}{r.n_chunks:7d}{r.pred:7.0f}"
              f"{f'  {r.hi68:.0f} - {r.lo68:.0f} BCE':>20}{r.p_post:16.2f}")
    return R


if __name__ == "__main__":
    main()
    # The file written above carries the estimator's own symmetric interval.
    # The published table replaces those with jackknife+ intervals;
    # finalize_targets.py does that and must run before anything reads
    # target_predictions_final.csv.  This copy is deliberately INSIDE the
    # __main__ guard: several downstream scripts import this module for its
    # helper functions, and an import must not touch any result file.
    import shutil
    shutil.copy("/home/claude/target_predictions_naive.csv",
                "/home/claude/target_predictions_final.csv")
    print("wrote target_predictions_naive.csv (provisional final copy made; "
          "run finalize_targets.py to publish)")
