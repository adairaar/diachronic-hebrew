"""
Is there any chronological signal left once genre is held fixed?

This is the test the whole question reduces to.  The permutation null in the
manuscript shuffles book dates freely and scores raw Spearman.  Raw Spearman on
this corpus is 77% between-genre variance, so that null asks whether the model
can tell genres apart -- which it plainly can -- and not whether it can order
texts in time.

The null that asks the real question is a RESTRICTED permutation: shuffle dates
only within genre.  Prophecy's 17 dates are permuted among the prophetic books
and narrative's 5 among the narrative books, so the genre/date association that
the corpus happens to contain is preserved exactly and only the within-genre
chronology is destroyed.  The statistic scored against it is genre-controlled
rho, computed identically to the observed value.

If the observed genre-controlled rho sits inside this null, the model orders
genres and nothing else, and no amount of feature engineering will rescue the
dating claim.  If it sits outside, there is a chronological signal that survives
the confound, and it is that signal -- not the raw rho -- that the paper is
entitled to report.

Singleton genres take no part: a genre with one book has no within-genre
ordering to destroy, and it contributes nothing to the statistic either.
"""
import json, os, sys, time
import numpy as np, pandas as pd, importlib.util
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", "/home/claude/predict_targets.py")
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 500
CKPT = "/home/claude/within_genre_null.csv"
# A checkpoint written against a different feature matrix must not be resumed;
# see ckpt_guard.py for what went wrong when that was not enforced.
import importlib.util as _ilu
_g = _ilu.spec_from_file_location("ckpt_guard", "/home/claude/ckpt_guard.py")
_G = _ilu.module_from_spec(_g); _g.loader.exec_module(_G)
_RESUMABLE = _G.check(CKPT, ["/home/claude/big_features_500.csv"],
                      extra="seed=11")

Dd = pd.read_csv("/home/claude/big_features_500.csv")
feats = [c for c in Dd.columns if c not in PT.META]
Xa = Dd[feats].astype(float)
keep = (Xa.std() > 0) & (Xa.isna().mean() < 0.2)
feats = list(np.array(feats)[keep.values])
med = Dd[feats].astype(float).median()
X = Dd[feats].astype(float).fillna(med).values
y = Dd.date_bce.values.astype(float); g = Dd.unit.values
books = list(pd.unique(g)); bdate = {b: y[g == b][0] for b in books}
bgenre = Dd.groupby("unit").genre.first().to_dict()
GEN = np.array([bgenre[b] for b in books])
SHUF = [np.where(GEN == gn)[0] for gn in ("prophecy", "narrative")]


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


def lobo(yv):
    bd = {b: yv[g == b][0] for b in books}
    t, p = [], []
    for b in books:
        te = g == b; tr = ~te
        inner = [x for x in books if x != b]
        wb = dict(zip(inner, PT.wts([bd[x] for x in inner])))
        wv = np.array([wb[u] for u in g[tr]])
        err = {lam: [] for lam in PT.LAM}
        for bb in inner:
            m = g[tr] != bb
            pr = fit_all_lams(X[tr][m], yv[tr][m], wv[m], X[tr][~m])
            for lam in PT.LAM:
                err[lam].append(abs(np.median(pr[lam]) - bd[bb]))
        bl = min(PT.LAM, key=lambda l: np.mean(err[l]))
        t.append(bd[b]); p.append(float(np.median(fit_all_lams(
            X[tr], yv[tr], wv, X[te])[bl])))
    return np.array(t), np.array(p)


def stat(t, p):
    """Genre-controlled rho, and raw rho for reference."""
    S = float(np.clip(t.std() / p.std(), 0.5, 8.0))
    cal = t.mean() + S * (p - p.mean())
    D = pd.DataFrame(dict(t=t, p=cal, gn=GEN))
    raw = stats.spearmanr(D.t, D.p)[0]
    B = D[D.gn.isin(["prophecy", "narrative"])].copy()
    B["tc"] = B.t - B.groupby("gn").t.transform("mean")
    B["pc"] = B.p - B.groupby("gn").p.transform("mean")
    par = stats.spearmanr(B.tc, B.pc)[0]
    pro = D[D.gn == "prophecy"]
    rp = stats.spearmanr(pro.t, pro.p)[0]
    return (float(par) if not np.isnan(par) else 0.0,
            float(raw) if not np.isnan(raw) else 0.0,
            float(rp) if not np.isnan(rp) else 0.0)


t_obs, p_obs = lobo(y)
o_par, o_raw, o_pro = stat(t_obs, p_obs)
print(f"observed: rho|genre {o_par:+.3f}   rho raw {o_raw:+.3f}   "
      f"rho prophecy {o_pro:+.3f}", flush=True)
print(f"shuffling {len(SHUF[0])} prophecy and {len(SHUF[1])} narrative dates "
      f"within genre; {NPERM} draws\n", flush=True)

rng = np.random.default_rng(11)
rows, done = [], 0
if _RESUMABLE and os.path.exists(CKPT):
    prev = np.loadtxt(CKPT, delimiter=",", skiprows=1, ndmin=2)
    if prev.size:
        rows = [tuple(r) for r in prev]; done = len(rows)
        print(f"resuming from {done} draws", flush=True)

t0 = time.time()
for k in range(NPERM):
    idx = np.arange(len(books))
    for blk in SHUF:
        idx[blk] = rng.permutation(blk)
    if k < done:
        continue
    remap = {b: bdate[books[idx[i]]] for i, b in enumerate(books)}
    yp = np.array([remap[u] for u in g], float)
    rows.append(stat(*lobo(yp)))
    if (k + 1) % 2 == 0 or k == NPERM - 1:
        np.savetxt(CKPT, np.array(rows), delimiter=",",
                   header="par,raw,pro", comments="")
        el = time.time() - t0
        rate = el / max(k + 1 - done, 1)
        N = np.array(rows)
        print(f"  {k+1}/{NPERM}  p(rho|genre) = "
              f"{(np.sum(N[:,0] >= o_par)+1)/(len(N)+1):.4f}   "
              f"({el/60:.1f} min, eta {rate*(NPERM-k-1)/60:.0f} min)", flush=True)

N = np.array(rows)
p_par = float((np.sum(N[:, 0] >= o_par) + 1) / (len(N) + 1))
p_raw = float((np.sum(N[:, 1] >= o_raw) + 1) / (len(N) + 1))
p_pro = float((np.sum(N[:, 2] >= o_pro) + 1) / (len(N) + 1))
print("\n" + "=" * 70)
print("WITHIN-GENRE PERMUTATION NULL")
print("=" * 70)
print(f"  {'statistic':<22}{'observed':>10}{'null median':>14}{'p':>10}")
for nm, o, col, pv in [("genre-controlled rho", o_par, 0, p_par),
                       ("raw rho", o_raw, 1, p_raw),
                       ("rho within prophecy", o_pro, 2, p_pro)]:
    print(f"  {nm:<22}{o:>+10.3f}{np.median(N[:, col]):>+14.3f}{pv:>10.4f}")
print(f"\n  {len(N)} draws.  Raw rho is shown only to make the point that it")
print(f"  barely moves under a within-genre shuffle: the genre structure it")
print(f"  scores is left intact by this null, which is why it is the wrong")
print(f"  statistic for the dating claim.")
json.dump(dict(n=len(N), obs_partial=o_par, obs_raw=o_raw, obs_prophecy=o_pro,
               p_partial=p_par, p_raw=p_raw, p_prophecy=p_pro,
               null_partial_med=float(np.median(N[:, 0])),
               null_raw_med=float(np.median(N[:, 1])),
               null_prophecy_med=float(np.median(N[:, 2]))),
          open("/home/claude/within_genre_null.json", "w"), indent=2)
print("\nwrote within_genre_null.json")
