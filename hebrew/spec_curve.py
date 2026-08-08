"""
Specification curve: every defensible analysis at once.

The estimates in this project move when the specification moves, and no single
configuration has a strong claim to being the right one.  Reporting one number
and burying the rest in a sensitivity appendix invites the reader to assume the
reported number is special.  It is not.  So report all of them.

Four choices are crossed, giving 48 specifications, each a complete
leave-one-book-out run of the pipeline:

  passage size    300, 500, 700, 1000 words
  feature screen  keep all, keep the 75% or 50% least genre-diagnostic
  weighting       inverse-density (alpha=1) or unweighted
  calibration     variance-matched or raw

Every specification is scored on the metric that matters -- genre-controlled
rho, not raw rho -- and carries that score with it.  A specification that cannot
order the anchors has produced a number, not an estimate, and the figure marks
it so.  This is the part that ordinary specification curves omit and that this
corpus badly needs: two of the six configurations tried earlier today reversed
the conclusion, and both of them scored rho below zero on their own training
data.

Intervals are jackknife+, computed per specification from that specification's
own leave-one-out residuals and target predictions.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import json, itertools, time, sys
import numpy as np, pandas as pd, importlib.util
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", DH.script("predict_targets.py"))
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

EX = 586
SRC = ["JE_source", "D_source", "P_source"]
EXTRA = ["Song_Sea", "Song_Deborah", "D_Code", "D_Frame"]
# 300-word passages are dropped for cost, not for taste: at 465 chunks the
# eigendecomposition is 4.5x the 500-word one and the twelve specifications
# would take longer than the other thirty-six combined.  Four passage sizes
# remain, spanning a factor of 2.5.
SIZES = [400, 500, 700, 1000]
FRACS = [1.00, 0.75, 0.50]
ALPHAS = [1.0, 0.0]
CALIBS = ["var", "none"]

CACHE = {}


def load(size):
    if size in CACHE: return CACHE[size]
    Dd = pd.read_csv(DH.f(f"big_features_{size}.csv"))
    Dt = pd.read_csv(DH.f("target_chunks_500.csv"))
    fe = [c for c in Dd.columns if c not in PT.META and c in Dt.columns]
    Xa = Dd[fe].astype(float)
    k = (Xa.std() > 0) & (Xa.isna().mean() < 0.2)
    fe = list(np.array(fe)[k.values])
    med = Dd[fe].astype(float).median()
    X = Dd[fe].astype(float).fillna(med).values
    XT = Dt[fe].astype(float).fillna(med).values
    y = Dd.date_bce.values.astype(float); g = Dd.unit.values
    bk = list(pd.unique(g))
    bd = {b: y[g == b][0] for b in bk}
    bg = Dd.groupby("unit").genre.first().to_dict()
    BM = pd.DataFrame(X, columns=fe).assign(u=g).groupby("u").mean().loc[bk].values
    GEN = np.array([bg[b] for b in bk])
    CACHE[size] = (X, XT, y, g, Dt.unit.values, bk, bd, bg, BM, GEN, len(fe))
    return CACHE[size]


def eta2(BM, GEN, bk, train):
    m = np.isin(bk, train) & np.isin(GEN, ["prophecy", "narrative"])
    Z = BM[m]; gg = GEN[m]
    tot = ((Z - Z.mean(0)) ** 2).sum(0)
    bet = np.zeros(Z.shape[1])
    for lev in np.unique(gg):
        s = gg == lev
        bet += s.sum() * (Z[s].mean(0) - Z.mean(0)) ** 2
    return bet / np.where(tot > 0, tot, np.inf)


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


def genre_rho(GEN, t, p):
    D = pd.DataFrame(dict(t=t, p=p, gn=GEN))
    D = D[D.gn.isin(["prophecy", "narrative"])].copy()
    D["tc"] = D.t - D.groupby("gn").t.transform("mean")
    D["pc"] = D.p - D.groupby("gn").p.transform("mean")
    if D.pc.std() == 0: return 0.0
    r = stats.spearmanr(D.tc, D.pc)[0]
    return 0.0 if np.isnan(r) else float(r)


def jkplus(v, R, alpha, n):
    lo = np.sort(v - R); hi = np.sort(v + R)
    k_lo = max(int(np.floor(alpha * (n + 1))), 1)
    k_hi = min(int(np.ceil((1 - alpha) * (n + 1))), n)
    return float(lo[k_lo - 1]), float(hi[k_hi - 1])


def run(size, frac, alpha, calib):
    X, XT, y, g, gt, bk, bd, bg, BM, GEN, P = load(size)
    n = len(bk)
    self_p, tgt_raw = {}, {}
    for b in bk:
        te = g == b; tr = ~te
        inner = [x for x in bk if x != b]
        sel = np.argsort(eta2(BM, GEN, bk, inner))[:max(int(round(frac * P)), 20)]
        wb = dict(zip(inner, PT.wts([bd[x] for x in inner], alpha=alpha)))
        wv = np.array([wb[u] for u in g[tr]])
        err = {l: [] for l in PT.LAM}
        for bb in inner:
            m = g[tr] != bb
            pr = fit_all(X[np.ix_(tr, sel)][m], y[tr][m], wv[m],
                         X[np.ix_(tr, sel)][~m])
            for l in PT.LAM:
                err[l].append(abs(np.median(pr[l]) - bd[bb]))
        bl = min(PT.LAM, key=lambda l: np.mean(err[l]))
        self_p[b] = float(np.median(fit_all(X[np.ix_(tr, sel)], y[tr], wv,
                                            X[np.ix_(te, sel)])[bl]))
        tgt_raw[b] = fit_all(X[np.ix_(tr, sel)], y[tr], wv, XT[:, sel])[bl]

    t = np.array([bd[b] for b in bk]); p = np.array([self_p[b] for b in bk])
    R = np.empty(n); TC = {}
    for i, b in enumerate(bk):
        o = [j for j in range(n) if j != i]
        if calib == "var":
            S_i = float(np.clip(t[o].std() / p[o].std(), 0.5, 8.0))
            mi, mo = p[o].mean(), t[o].mean()
        else:
            S_i, mi, mo = 1.0, 0.0, 0.0
        R[i] = abs(t[i] - (mo + S_i * (p[i] - mi)))
        TC[b] = mo + S_i * (pd.Series(tgt_raw[b]).groupby(gt).median() - mi)
    TCd = pd.DataFrame(TC)

    if calib == "var":
        S = float(np.clip(t.std() / p.std(), 0.5, 8.0))
        cal = t.mean() + S * (p - p.mean())
    else:
        cal = p
    ok = tot = 0
    for i in range(n):
        for j in range(i + 1, n):
            if t[i] == t[j]: continue
            tot += 1; ok += int((cal[i] - cal[j]) * (t[i] - t[j]) > 0)
    pro = GEN == "prophecy"
    out = dict(size=size, frac=frac, alpha=alpha, calib=calib,
               n_feats=int(max(int(round(frac * P)), 20)),
               mae=float(np.abs(t - cal).mean()),
               rho_raw=float(stats.spearmanr(t, cal)[0]),
               rho_genre=genre_rho(GEN, t, cal),
               rho_proph=float(stats.spearmanr(t[pro], cal[pro])[0]),
               pair=ok / tot)
    for u in SRC + EXTRA:
        if u not in TCd.index: continue
        v = TCd.loc[u].values
        lo, hi = jkplus(v, R, 0.16, n)
        ens = np.concatenate([v - R, v + R])
        out[f"{u}_pred"] = float(np.median(v))
        out[f"{u}_lo"] = lo; out[f"{u}_hi"] = hi
        out[f"{u}_ppost"] = float(np.mean(ens < EX))
    return out


GRID = list(itertools.product(SIZES, FRACS, ALPHAS, CALIBS))
print(f"{len(GRID)} specifications\n", flush=True)

# The sandbox restarts without warning and takes running jobs with it, so each
# completed specification is checkpointed and a relaunch picks up where it left.
rows, t0 = [], time.time()
import os
if os.path.exists(DH.f("spec_curve.csv")):
    prev = pd.read_csv(DH.f("spec_curve.csv"))
    rows = prev.to_dict("records")
    print(f"resuming with {len(rows)} specifications already done\n", flush=True)
DONE = {(r["size"], r["frac"], r["alpha"], r["calib"]) for r in rows}

for i, (s, f, a, c) in enumerate(GRID):
    if (s, f, a, c) in DONE:
        continue
    rows.append(run(s, f, a, c))
    r = rows[-1]
    print(f"  [{i+1:>2}/{len(GRID)}] {s:>4}w kept{100*f:>4.0f}% a={a:.0f} {c:<4}  "
          f"MAE {r['mae']:>4.0f}  rho {r['rho_raw']:+.2f}  rho|g {r['rho_genre']:+.2f}"
          f"   JE {r.get('JE_source_pred', float('nan')):>4.0f}"
          f"  D {r.get('D_source_pred', float('nan')):>4.0f}"
          f"  P {r.get('P_source_pred', float('nan')):>4.0f}"
          f"   ({(time.time()-t0)/60:.1f} min)", flush=True)
    pd.DataFrame(rows).to_csv(DH.f("spec_curve.csv"), index=False)

D = pd.DataFrame(rows)
D.to_csv(DH.f("spec_curve.csv"), index=False)
passing = D[D.rho_genre > 0.2]
print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
print(f"  {len(D)} specifications, {len(passing)} with genre-controlled rho > 0.20")
for u in SRC:
    v = D[f"{u}_pred"]; vp = passing[f"{u}_pred"]
    pv = passing[f"{u}_ppost"]
    print(f"  {u:<12} all: {v.min():>4.0f}-{v.max():>4.0f}   "
          f"passing: {vp.min():>4.0f}-{vp.max():>4.0f}   "
          f"P(post) {pv.min():.2f}-{pv.max():.2f}")
json.dump(dict(n_spec=len(D), n_pass=int(len(passing)),
               ranges={u: dict(all_lo=float(D[f"{u}_pred"].min()),
                               all_hi=float(D[f"{u}_pred"].max()),
                               pass_lo=float(passing[f"{u}_pred"].min()),
                               pass_hi=float(passing[f"{u}_pred"].max()),
                               pass_ppost_lo=float(passing[f"{u}_ppost"].min()),
                               pass_ppost_hi=float(passing[f"{u}_ppost"].max()))
                       for u in SRC}),
          open(DH.f("spec_curve.json"), "w"), indent=2)
print("\nwrote spec_curve.csv, spec_curve.json")
