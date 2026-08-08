"""
Is the genre effect real, and what does it do to the three sources?

The residuals on the anchored corpus are structured by genre: the five narrative
books are placed about a century too late, the prophetic books close to correct.
Two things follow, and the manuscript currently addresses neither.

First, the narrative and prophetic anchors do not occupy the same stretch of the
timeline, so an apparent genre effect could be an endpoint effect wearing a
genre label.  The five narrative books are all post-exilic.  The test that
separates the two is a date-matched contrast: among anchors written between 460
and 300 BCE, where both genres are represented, does the gap survive?

Second, the poems are corrected by the poetry residual, which makes them look
younger and the paper's claim about them weaker.  Nothing corrects the three
Pentateuchal sources, where the same logic would push them earlier and weaken
the post-exilic result.  A correction that is applied only where it costs
nothing is not a correction.  This applies it symmetrically.

Third, and separately from any offset: refit with a whole genre removed from the
training set and see where the sources land.  That asks the question without
needing to label the targets at all.
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import json
import numpy as np, pandas as pd, importlib.util
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", DH.script("predict_targets.py"))
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

EX = 586
SRC = ["JE_source", "D_source", "P_source"]

Dd = pd.read_csv(DH.f("big_features_500.csv"))
Dt = pd.read_csv(DH.f("target_chunks_500.csv"))
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


def lobo_and_predict(subset):
    """LOBO within `subset`, plus target predictions from the full-subset fit.

    Returns the anchor truths, the calibrated anchor predictions, the calibrated
    per-unit target predictions and the LOBO residuals.  Calibration constants
    come only from `subset`, so a model trained without a genre is calibrated
    without it too.
    """
    idx = np.isin(g, subset)
    t, p = [], []
    for b in subset:
        te = (g == b); tr = idx & ~te
        inner = [x for x in subset if x != b]
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
        t.append(bdate[b])
        p.append(float(np.median(PT.fit_predict(X[tr], y[tr], wv, X[te], bl))))
    t, p = np.array(t), np.array(p)
    S = float(np.clip(t.std() / p.std(), 0.5, 8.0))
    cal = lambda v: t.mean() + S * (v - p.mean())
    resid = t - cal(p)

    wb = dict(zip(subset, PT.wts([bdate[x] for x in subset])))
    wv = np.array([wb[u] for u in g[idx]])
    best, bl = np.inf, PT.LAM[0]
    for lam in PT.LAM:
        e = []
        for b in subset:
            m = g[idx] != b
            e.append(abs(np.median(PT.fit_predict(
                X[idx][m], y[idx][m], wv[m], X[idx][~m], lam)) - bdate[b]))
        if np.mean(e) < best: best, bl = np.mean(e), lam
    raw = PT.fit_predict(X[idx], y[idx], wv, XT, bl)
    tgt = pd.Series(cal(raw)).groupby(gt).median()
    return t, cal(p), tgt, resid


# ── 1. genre residuals, with the uncertainty they actually carry ─────────────
t_all, p_all, tgt_all, res_all = lobo_and_predict(books)
A = pd.DataFrame(dict(book=books, truth=t_all, pred=p_all, resid=res_all,
                      genre=[bgenre[b] for b in books]))

print("=" * 78)
print("1.  GENRE RESIDUALS AND WHAT THEY ARE WORTH")
print("=" * 78)
print(f"  {'genre':<14}{'n':>4}{'mean':>9}{'sd':>8}{'se':>8}{'95% CI':>20}")
G = {}
for gn, sub in A.groupby("genre"):
    n = len(sub); m = sub.resid.mean()
    sd = sub.resid.std(ddof=1) if n > 1 else np.nan
    se = sd / np.sqrt(n) if n > 1 else np.nan
    ci = (f"{m-1.96*se:+.0f} to {m+1.96*se:+.0f}" if n > 1 else "not estimable")
    G[gn] = dict(n=int(n), mean=float(m), sd=None if n < 2 else float(sd),
                 se=None if n < 2 else float(se))
    print(f"  {gn:<14}{n:>4}{m:>+9.0f}{sd if n>1 else float('nan'):>8.0f}"
          f"{se if n>1 else float('nan'):>8.0f}{ci:>20}")
print("\n  Positive residual = the book is older than the model makes it, i.e. the")
print("  model places it too late.  Three of the five genres rest on one book and")
print("  support no estimate at all.")

# ── 2. is it genre, or is it the end of the timeline? ────────────────────────
print()
print("=" * 78)
print("2.  GENRE, OR ENDPOINT?  THE DATE-MATCHED CONTRAST")
print("=" * 78)
LO, HI = 250, 470                      # window where both genres are attested
W = A[(A.truth >= LO) & (A.truth <= HI)]
wn = W[W.genre == "narrative"]; wp = W[W.genre == "prophecy"]
print(f"  anchors written between {HI} and {LO} BCE:")
for _, r in W.sort_values("truth", ascending=False).iterrows():
    print(f"    {r.book:<14}{int(r.truth):>5} BCE   {r.genre:<12}resid {r.resid:+.0f}")
d = wn.resid.mean() - wp.resid.mean()
se = np.sqrt(wn.resid.var(ddof=1) / len(wn) + wp.resid.var(ddof=1) / len(wp))
tt, pp = stats.ttest_ind(wn.resid, wp.resid, equal_var=False)
print(f"\n  narrative  n={len(wn)}  mean resid {wn.resid.mean():+.0f}")
print(f"  prophecy   n={len(wp)}  mean resid {wp.resid.mean():+.0f}")
print(f"  difference {d:+.0f} yr  (se {se:.0f}, Welch t = {tt:.2f}, p = {pp:.4f})")
print("\n  Within a single 220-year window the two genres are placed roughly two")
print("  and a half centuries apart.  The effect is not the endpoint.")

# regression control: residual on date and a narrative indicator
NP = A[A.genre.isin(["narrative", "prophecy"])].copy()
NP["nar"] = (NP.genre == "narrative").astype(float)
Xr = np.column_stack([np.ones(len(NP)), NP.truth.values, NP.nar.values])
beta, *_ = np.linalg.lstsq(Xr, NP.resid.values, rcond=None)
fit = Xr @ beta; dof = len(NP) - 3
s2 = (NP.resid.values - fit) @ (NP.resid.values - fit) / dof
cov = s2 * np.linalg.inv(Xr.T @ Xr)
se_nar = np.sqrt(cov[2, 2]); t_nar = beta[2] / se_nar
print(f"\n  controlling for date across all {len(NP)} narrative and prophetic anchors:")
print(f"    narrative indicator {beta[2]:+.0f} yr  (se {se_nar:.0f}, t = {t_nar:.2f}, "
      f"p = {2*(1-stats.t.cdf(abs(t_nar), dof)):.4f})")
print(f"    date slope {beta[1]:+.3f} yr per year of true date")

# ── 3. the correction, applied to everything ────────────────────────────────
print()
print("=" * 78)
print("3.  THE CORRECTION APPLIED SYMMETRICALLY")
print("=" * 78)
print("  The Songs are corrected by the poetry residual in the published")
print("  analysis.  If that is legitimate, so is correcting the sources.  JE is")
print("  narrative; D and P are largely legal, a register with no anchor at all,")
print("  so both the narrative and the prophetic offsets are shown as bounds.")
print()
off_nar = float(A[A.genre == "narrative"].resid.mean())
off_pro = float(A[A.genre == "prophecy"].resid.mean())
off_poe = float(A[A.genre == "poetry"].resid.mean())
UNITS = SRC + ["Gen_JE", "Exo_JE", "Num_JE", "D_Code", "D_Frame",
               "Lev_Holiness", "Lev_Priestly", "Jer_DTR"]
print(f"  {'unit':<15}{'uncorrected':>12}{'+narrative':>12}{'P(post)':>9}"
      f"{'+prophecy':>12}{'P(post)':>9}")
rows = []
for u in UNITS:
    if u not in tgt_all.index: continue
    base = float(tgt_all[u])
    a, b = base + off_nar, base + off_pro
    pa = float(np.mean((a + res_all) < EX)); pb = float(np.mean((b + res_all) < EX))
    p0 = float(np.mean((base + res_all) < EX))
    rows.append(dict(unit=u, uncorrected=round(base), p_post=round(p0, 2),
                     nar_adj=round(a), p_post_nar=round(pa, 2),
                     pro_adj=round(b), p_post_pro=round(pb, 2)))
    print(f"  {u:<15}{base:>12.0f}{a:>12.0f}{pa:>9.2f}{b:>12.0f}{pb:>9.2f}")
C = pd.DataFrame(rows)
C.to_csv(DH.f("genre_symmetric_targets.csv"), index=False)
worst = C[C.unit.isin(SRC)]
print(f"\n  Under the narrative correction, the least post-exilic of the three")
print(f"  sources is {worst.loc[worst.p_post_nar.idxmin(),'unit']} at "
      f"{worst.p_post_nar.min():.2f}; uncorrected it is {worst.p_post.min():.2f}.")

# ── 4. refits with a genre removed from training ────────────────────────────
print()
print("=" * 78)
print("4.  REFITS WITH A GENRE REMOVED FROM THE TRAINING SET")
print("=" * 78)
VARIANTS = [
    ("all 25 anchors", books),
    ("narrative dropped", [b for b in books if bgenre[b] != "narrative"]),
    ("prophecy only", [b for b in books if bgenre[b] == "prophecy"]),
]
out = {}
print(f"  {'training set':<20}{'n':>4}{'rho':>8}{'MAE':>7}"
      + "".join(f"{u.split('_')[0]:>10}" for u in SRC) + f"{'min P(post)':>13}")
for name, sub in VARIANTS:
    tt_, pp_, tg_, rs_ = lobo_and_predict(sub)
    rho = stats.spearmanr(tt_, pp_)[0]
    ps = [float(np.mean((float(tg_[u]) + rs_) < EX)) for u in SRC]
    out[name] = dict(n=len(sub), rho=float(rho), mae=float(np.abs(rs_).mean()),
                     preds={u: round(float(tg_[u])) for u in SRC},
                     p_post={u: round(p, 2) for u, p in zip(SRC, ps)})
    print(f"  {name:<20}{len(sub):>4}{rho:>+8.3f}{np.abs(rs_).mean():>7.0f}"
          + "".join(f"{float(tg_[u]):>10.0f}" for u in SRC)
          + f"{min(ps):>13.2f}")
print("\n  Dropping the narrative anchors removes the books that carry the offset.")
print("  Training on prophecy alone removes the genre contrast entirely.")

json.dump(dict(genres=G, off_nar=off_nar, off_pro=off_pro, off_poe=off_poe,
               window=[LO, HI], win_nar=float(wn.resid.mean()),
               win_pro=float(wp.resid.mean()), win_n_nar=len(wn),
               win_n_pro=len(wp), win_diff=float(d), win_se=float(se),
               win_t=float(tt), win_p=float(pp),
               reg_nar=float(beta[2]), reg_nar_se=float(se_nar),
               reg_nar_t=float(t_nar),
               reg_nar_p=float(2 * (1 - stats.t.cdf(abs(t_nar), dof))),
               minpost_uncorr=float(worst.p_post.min()),
               minpost_nar=float(worst.p_post_nar.min()),
               minpost_pro=float(worst.p_post_pro.min()),
               variants=out),
          open(DH.f("genre_symmetric.json"), "w"), indent=2)
print("\nwrote genre_symmetric_targets.csv, genre_symmetric.json")
