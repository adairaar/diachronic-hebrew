"""
Jackknife+ prediction intervals for the undated units.

The intervals reported previously were built from leave-one-book-out residuals
and described as conformal with finite-sample marginal coverage.  That
description was wrong in two ways, and this script fixes both.

First, split conformal assumes every calibration residual comes from one model,
fitted on data exchangeable with the test point.  Here each residual comes from
a different model, fitted without its own book.  That is the jackknife, and
naive jackknife intervals carry no finite-sample guarantee at all.  The
jackknife+ of Barber, Candes, Ramdas and Tibshirani (2021) does: for a target x
with leave-one-out predictions mu_{-i}(x) and leave-one-out residuals R_i, the
interval

    [ q^-_alpha { mu_{-i}(x) - R_i } ,  q^+_{1-alpha} { mu_{-i}(x) + R_i } ]

has coverage at least 1 - 2*alpha, distribution-free and in finite samples.

Second, the variance-matching constants were estimated from all 25 leave-one-out
predictions, including the book being scored.  They are now recomputed inside
each fold, which widens the residuals and is the honest version.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import json, os
import numpy as np, pandas as pd, importlib.util

pt = importlib.util.spec_from_file_location("pt", DH.script("predict_targets.py"))
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

Dd = pd.read_csv(DH.f("big_features_500.csv"))
# The poem variants -- whole chapter, poem proper, prose remainder -- are scored
# alongside the ordinary targets so that they receive genuine jackknife+
# intervals from the same folds.  Computing their intervals separately from a
# point estimate would silently substitute a different estimator: jackknife+
# needs the per-fold predictions aligned with the per-fold residuals, not a
# scalar.
Dt = pd.read_csv(DH.f("target_chunks_500.csv"))
_poems = DH.f("poem_chunks.csv")
if os.path.exists(_poems):
    Dp = pd.read_csv(_poems)
    Dt = pd.concat([Dt, Dp[[c for c in Dp.columns if c in Dt.columns]]],
                   ignore_index=True)
    print(f"scoring {Dt.unit.nunique()} units "
          f"({Dp.unit.nunique()} of them poem variants)")
Pm = pd.read_csv(DH.f("poem_chunks.csv"))
Pm = Pm[Pm.unit.isin(["SongSea_poem", "SongDeborah_poem", "SongMoses_poem",
                      "SongSea_prose"])]
Dt = pd.concat([Dt, Pm], ignore_index=True)
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
n = len(books)
EX = 586

# ── leave-one-book-out: raw predictions for the held-out book and the targets ──
raw_self, raw_tgt, lam_used = {}, {}, {}
for b in books:
    te = g == b; tr = ~te
    inner = [x for x in books if x != b]
    wb = dict(zip(inner, PT.wts([bdate[x] for x in inner])))
    wv = np.array([wb[u] for u in g[tr]])
    best, bl = np.inf, PT.LAM[0]
    for lam in PT.LAM:
        e = []
        for bb in inner:
            m = g[tr] != bb
            e.append(abs(np.median(PT.fit_predict(
                X[tr][m], y[tr][m], wv[m], X[tr][~m], lam)) - bdate[bb]))
        if np.mean(e) < best: best, bl = np.mean(e), lam
    lam_used[b] = bl
    raw_self[b] = float(np.median(PT.fit_predict(X[tr], y[tr], wv, X[te], bl)))
    raw_tgt[b] = PT.fit_predict(X[tr], y[tr], wv, XT, bl)     # per target chunk
    print(f"  fitted without {b}", flush=True)

t = np.array([bdate[b] for b in books])
p = np.array([raw_self[b] for b in books])

# ── nested calibration: constants for fold i exclude book i ───────────────────
R = np.empty(n)                 # leave-one-out residuals, honestly calibrated
tgt_cal = {}                    # calibrated target predictions per fold
for i, b in enumerate(books):
    o = [j for j in range(n) if j != i]
    S_i = float(np.clip(t[o].std() / p[o].std(), 0.5, 8.0))
    m_in, m_out = p[o].mean(), t[o].mean()
    R[i] = abs(t[i] - (m_out + S_i * (p[i] - m_in)))
    raw_u = pd.Series(raw_tgt[b]).groupby(gt).median()
    tgt_cal[b] = m_out + S_i * (raw_u - m_in)

TC = pd.DataFrame(tgt_cal)      # rows: target units, columns: folds
units = list(TC.index)


def jkplus(v, R, alpha):
    """Jackknife+ interval.  v and R are aligned across the n folds."""
    lo_sorted = np.sort(v - R)
    hi_sorted = np.sort(v + R)
    k_lo = max(int(np.floor(alpha * (n + 1))), 1)
    k_hi = min(int(np.ceil((1 - alpha) * (n + 1))), n)
    return lo_sorted[k_lo - 1], hi_sorted[k_hi - 1]


rows = []
for u in units:
    v = TC.loc[u].values
    lo68, hi68 = jkplus(v, R, 0.16)      # central 68%
    lo90, hi90 = jkplus(v, R, 0.05)      # central 90%
    point = float(np.median(v))
    # P(post-exilic): fraction of the jackknife+ ensemble falling after the exile
    ens = np.concatenate([v - R, v + R])
    rows.append(dict(unit=u, pred=round(point),
                     lo68=round(lo68), hi68=round(hi68),
                     lo90=round(lo90), hi90=round(hi90),
                     p_post=round(float(np.mean(ens < EX)), 2)))
J = pd.DataFrame(rows).set_index("unit")
J.to_csv(DH.f("jackknife_plus_targets.csv"))

# compare against the estimator's own symmetric intervals, not against a file
# this script's own output may already have been merged into
old = pd.read_csv(DH.f("target_predictions_naive.csv")).set_index("unit")
print("\n" + "=" * 78)
print("JACKKNIFE+ INTERVALS vs THE INTERVALS AS PREVIOUSLY REPORTED")
print("=" * 78)
print(f"{'unit':<15}{'point':>7}{'68% as reported':>20}{'68% jackknife+':>20}{'P>ex':>7}")
ORDER = ["Song_Deborah","Song_Sea","D_Song","JE_source","D_source","P_source",
         "D_Code","D_Frame","Lev_Holiness","Lev_Priestly","Jer_DTR"]
for u in ORDER:
    if u not in J.index: continue
    r, o = J.loc[u], old.loc[u]
    print(f"{u:<15}{int(r.pred):>7}{f'{int(o.hi68)}-{int(o.lo68)}':>20}"
          f"{f'{int(r.hi68)}-{int(r.lo68)}':>20}{r.p_post:>7.2f}")

w_old = float((old.hi68 - old.lo68).mean())
w_new = float((J.hi68 - J.lo68).mean())
print(f"\n  mean 68% interval width: {w_old:.0f} -> {w_new:.0f} yr "
      f"({100*(w_new/w_old-1):+.0f}%)")
srcs = ["P_source", "JE_source", "D_source"]
print(f"  P(post-exilic) for the three sources: "
      f"{', '.join(f'{u.split(chr(95))[0]} {J.loc[u].p_post:.2f}' for u in srcs)}")
print(f"  minimum across the three: {J.loc[srcs].p_post.min():.2f} "
      f"(was {old.loc[srcs].p_post.min():.2f})")
# the leave-one-out residual ensemble is corpus-level, not unit-specific, so
# any prediction can be given a jackknife+ interval from it; finalize_poems.py
# uses it for the three poems, whose own script emits symmetric intervals
np.savetxt(DH.f("jackknife_plus_residuals.csv"), R, delimiter=",",
           header="residual", comments="")

json.dump(dict(width_old=w_old, width_new=w_new,
               minpost=float(J.loc[srcs].p_post.min()),
               sea=int(J.loc["Song_Sea"].pred) if "Song_Sea" in J.index else None,
               R_mean=float(R.mean())),
          open(DH.f("jackknife_plus.json"), "w"), indent=2)
print("\nwrote jackknife_plus_targets.csv, jackknife_plus.json")
