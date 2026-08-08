"""
Do the block separations survive the genre screen?

The specification curve says they may not.  Across the 44 validated
specifications the Deuteronomic law code comes out EARLIER than its frame in
only 15, and the direction tracks one choice almost perfectly: with all features
retained the code is earlier in 12 of 12, at 75% retention in 3 of 16, and at
50% retention in 0 of 16.  The manuscript reports the +97-year gap from the
unscreened configuration and calls it layering.

That pattern has an obvious and unflattering reading.  The law code is legal
prose and the frame is hortatory narrative.  If the gap between them is carried
by the same genre-diagnostic vocabulary the paper screens out elsewhere, then
the "layering" is register, not date -- the very confound the Validation section
spends two pages establishing for the main results, never applied to this one.

This script applies it.  The chunk-level pipeline of internal_consistency.py is
rerun at three levels of the genre screen, and each block separation is
recomputed the way the manuscript computes it: difference of median passage
estimates, bootstrap interval, Mann--Whitney test.

Screening ranks features by eta-squared for genre across the anchor books and
drops the most genre-diagnostic, which is the same screen the specification
curve uses.
"""
import json
import numpy as np, pandas as pd, importlib.util
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", "/home/claude/predict_targets.py")
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

RNG = np.random.default_rng(11)
EX = 586
PAIRS = [("D_Code", "D_Frame", "Deuteronomic law code vs frame"),
         ("Lev_Holiness", "Lev_Priestly", "Holiness Code vs Leviticus 1-16"),
         ("Gen_JE", "Exo_JE", "Genesis JE vs Exodus JE")]

Dd = pd.read_csv("/home/claude/big_features_500.csv")
Dt = pd.read_csv("/home/claude/target_chunks_500.csv")
feats0 = [c for c in Dd.columns if c not in PT.META and c in Dt.columns]
Xa = Dd[feats0].astype(float)
keep0 = (Xa.std() > 0) & (Xa.isna().mean() < 0.2)
feats0 = list(np.array(feats0)[keep0.values])

# genre discriminability of each feature, measured on the anchors
BM = Dd.groupby("unit")[feats0].mean()
GEN = Dd.groupby("unit").genre.first().loc[BM.index].values
m = np.isin(GEN, ["prophecy", "narrative"])
Z = BM.values[m]; gg = GEN[m]
tot = ((Z - Z.mean(0)) ** 2).sum(0)
bet = np.zeros(Z.shape[1])
for lev in np.unique(gg):
    s = gg == lev
    bet += s.sum() * (Z[s].mean(0) - Z.mean(0)) ** 2
ETA = pd.Series(bet / np.where(tot > 0, tot, np.inf), index=feats0)


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


def run(frac):
    fe = list(ETA.sort_values().index[:int(round(frac * len(feats0)))])
    med = Dd[fe].astype(float).median()
    X = Dd[fe].astype(float).fillna(med).values
    XT = Dt[fe].astype(float).fillna(med).values
    y = Dd.date_bce.values.astype(float); g = Dd.unit.values
    books = list(pd.unique(g)); bd = {b: y[g == b][0] for b in books}

    cp = np.full(len(Dd), np.nan); bmed = {}
    for b in books:
        te = g == b; tr = ~te
        inner = [x for x in books if x != b]
        wb = dict(zip(inner, PT.wts([bd[x] for x in inner])))
        wv = np.array([wb[u] for u in g[tr]])
        err = {l: [] for l in PT.LAM}
        for bb in inner:
            mm = g[tr] != bb
            pr = fit_all(X[tr][mm], y[tr][mm], wv[mm], X[tr][~mm])
            for l in PT.LAM:
                err[l].append(abs(np.median(pr[l]) - bd[bb]))
        bl = min(PT.LAM, key=lambda l: np.mean(err[l]))
        p = fit_all(X[tr], y[tr], wv, X[te])[bl]
        cp[te] = p; bmed[b] = float(np.median(p))

    t = np.array([bd[b] for b in books]); pm = np.array([bmed[b] for b in books])
    S = float(np.clip(t.std() / pm.std(), 0.5, 8.0))
    cal = lambda v: t.mean() + S * (v - pm.mean())
    rho_g = None
    gcal = cal(np.array([bmed[b] for b in books]))
    Dg = pd.DataFrame(dict(t=t, p=gcal, gn=[Dd[Dd.unit == b].genre.iloc[0]
                                            for b in books]))
    Dg = Dg[Dg.gn.isin(["prophecy", "narrative"])]
    Dg = Dg.assign(tc=Dg.t - Dg.groupby("gn").t.transform("mean"),
                   pc=Dg.p - Dg.groupby("gn").p.transform("mean"))
    rho_g = float(stats.spearmanr(Dg.tc, Dg.pc)[0])

    wb = dict(zip(books, PT.wts([bd[x] for x in books])))
    wv = np.array([wb[u] for u in g])
    err = {l: [] for l in PT.LAM}
    for b in books:
        mm = g != b
        pr = fit_all(X[mm], y[mm], wv[mm], X[~mm])
        for l in PT.LAM:
            err[l].append(abs(np.median(pr[l]) - bd[b]))
    bl = min(PT.LAM, key=lambda l: np.mean(err[l]))
    tp = cal(fit_all(X, y, wv, XT)[bl])
    return len(fe), rho_g, pd.Series(tp, index=Dt.unit.values)


print("=" * 86)
print("DO THE BLOCK SEPARATIONS SURVIVE THE GENRE SCREEN?")
print("=" * 86)
rows = []
for frac, lab in [(1.0, "all features"), (0.75, "drop top 25% genre-diagnostic"),
                  (0.5, "drop top 50% genre-diagnostic")]:
    nfe, rho_g, tp = run(frac)
    print(f"\n{lab}  ({nfe} features, genre-controlled rho {rho_g:+.3f})")
    print(f"  {'comparison':<34}{'gap':>7}{'95% CI':>18}{'p':>9}")
    for a, b, name in PAIRS:
        x = tp[tp.index == a].values; y2 = tp[tp.index == b].values
        gap = float(np.median(x) - np.median(y2))
        d = np.array([np.median(RNG.choice(x, len(x))) -
                      np.median(RNG.choice(y2, len(y2))) for _ in range(4000)])
        lo, hi = np.percentile(d, [2.5, 97.5])
        pu = float(stats.mannwhitneyu(x, y2)[1])
        rows.append(dict(frac=frac, n_feats=nfe, rho_genre=rho_g, pair=name,
                         gap=gap, lo=float(lo), hi=float(hi), p=pu))
        print(f"  {name:<34}{gap:>+7.0f}{f'{lo:+.0f} to {hi:+.0f}':>18}{pu:>9.3f}")

R = pd.DataFrame(rows)
R.to_csv("/home/claude/block_robustness.csv", index=False)
print()
print("=" * 86)
print("VERDICT")
print("=" * 86)
for a, b, name in PAIRS:
    sub = R[R.pair == name]
    signs = set(np.sign(sub.gap))
    stable = len(signs) == 1
    sig = int((sub.p < 0.05).sum())
    print(f"  {name:<34} direction {'holds' if stable else 'REVERSES'}"
          f"   significant in {sig}/{len(sub)}")
json.dump(R.to_dict("records"), open("/home/claude/block_robustness.json", "w"),
          indent=2)
print("\nwrote block_robustness.csv/.json")
