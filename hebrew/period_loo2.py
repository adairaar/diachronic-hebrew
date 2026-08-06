"""
Period classification for Biblical Hebrew — honest LOO, two model families.

Both families are pre-specified and BOTH are reported, whatever they show.
Every step (screening, standardisation, fitting) happens inside the fold.
The permutation null re-runs the entire pipeline on shuffled dates.
"""
import numpy as np, pandas as pd
from scipy import stats
import sys, json

RNG = np.random.default_rng(20260806)
MATRIX = "/mnt/user-data/uploads/Diachronic Hebrew/hebrew/data/feature_matrix_v2.csv"

PERIODS = ["Pre-exilic", "Exilic", "Persian", "Hellenistic"]
def to_period(d):
    return 0 if d > 586 else 1 if d > 539 else 2 if d > 332 else 3
def to_binary(d):
    return 0 if d > 586 else 1

GRID = np.linspace(900.0, 100.0, 401)


def load():
    df = pd.read_csv(MATRIX)
    meta = {"id","date_bce","date_sigma","register","genre","holdout",
            "hbvi_holdout","n_words","in_training"}
    feats = [c for c in df.columns if c not in meta]
    d = df[df["date_bce"].notna()].reset_index(drop=True)
    X = d[feats].astype(float).values
    ok = np.isfinite(X).all(0) & (X.std(0) > 0)
    return d, np.array(feats)[ok], X[:, ok]


def screen(Ztr, dtr, alpha):
    n = len(dtr)
    rx = stats.rankdata(Ztr, axis=0); ry = stats.rankdata(dtr)
    rxc = rx - rx.mean(0); ryc = ry - ry.mean()
    den = np.sqrt((rxc**2).sum(0) * (ryc**2).sum())
    den = np.where(den > 0, den, np.inf)
    rho = np.clip((rxc * ryc[:,None]).sum(0)/den, -0.9999999, 0.9999999)
    t = rho*np.sqrt((n-2)/(1-rho**2))
    p = 2*stats.t.sf(np.abs(t), n-2)
    return np.flatnonzero(np.isfinite(p) & (p < alpha))


def fit_generative(Xtr, dtr, xte, alpha):
    """Invert per-feature OLS; flat prior over the date grid. (paper's family)"""
    n = len(dtr)
    mu, sd = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1.0)
    Ztr, zte = (Xtr-mu)/sd, (xte-mu)/sd
    keep = screen(Ztr, dtr, alpha)
    if keep.size == 0: return np.nan
    d0, ds = dtr.mean(), (dtr.std() or 1.0)
    t = (dtr-d0)/ds; tc = t-t.mean()
    Y = Ztr[:, keep]
    b = (tc[:,None]*(Y-Y.mean(0))).sum(0)/(tc**2).sum()
    a = Y.mean(0) - b*t.mean()
    s = np.maximum(np.sqrt(((Y-(a+np.outer(t,b)))**2).sum(0)/max(n-2,1)), 1e-3)
    tg = (GRID-d0)/ds
    pred = a[None,:] + np.outer(tg, b)
    ll = (-0.5*((zte[keep][None,:]-pred)/s[None,:])**2 - np.log(s)[None,:]).sum(1)
    ll -= ll.max(); post = np.exp(ll)
    return GRID[np.argmax(post)]


def fit_ridge(Xtr, dtr, xte, alpha):
    """Ridge regression of date on screened features; lambda by inner LOO."""
    mu, sd = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1.0)
    Ztr, zte = (Xtr-mu)/sd, (xte-mu)/sd
    keep = screen(Ztr, dtr, alpha)
    if keep.size == 0: return np.nan
    A, b0 = Ztr[:, keep], dtr - dtr.mean()
    n, k = A.shape
    lams = 10.0**np.arange(-2, 4.1, 0.5)
    best, bl = np.inf, lams[0]
    G = A.T @ A
    for lam in lams:                       # inner LOO via hat-matrix shortcut
        Hn = A @ np.linalg.solve(G + lam*np.eye(k), A.T)
        yh = Hn @ b0
        h = np.clip(np.diag(Hn), 0, 1-1e-9)
        cv = (((b0-yh)/(1-h))**2).mean()
        if cv < best: best, bl = cv, lam
    w = np.linalg.solve(G + bl*np.eye(k), A.T @ b0)
    return float(zte[keep] @ w + dtr.mean())


FAMILIES = {"generative": fit_generative, "ridge": fit_ridge}


def run_loo(dates, X, alpha, fam):
    f = FAMILIES[fam]; n = len(dates)
    out = np.full(n, np.nan)
    for i in range(n):
        tr = np.arange(n) != i
        try: out[i] = f(X[tr], dates[tr], X[i], alpha)
        except Exception: out[i] = np.nan
    return np.clip(out, GRID.min(), GRID.max())


def score(dates, maps):
    ok = np.isfinite(maps)
    if ok.sum() < 3: return None
    d, m = dates[ok], maps[ok]
    yt4 = np.array([to_period(v) for v in d]); yp4 = np.array([to_period(v) for v in m])
    yt2 = np.array([to_binary(v) for v in d]); yp2 = np.array([to_binary(v) for v in m])
    if len(np.unique(m)) < 2: rho = 0.0
    else: rho = float(stats.spearmanr(d, m).statistic)
    return dict(acc4=float((yt4==yp4).mean()), acc2=float((yt2==yp2).mean()),
                adj4=float((np.abs(yt4-yp4)<=1).mean()), mae=float(np.abs(d-m).mean()),
                rho=(0.0 if not np.isfinite(rho) else rho), n=int(ok.sum()),
                yt4=yt4, yp4=yp4)


def main(alpha=0.05, nperm=2000):
    dated, feats, X = load()
    dates = dated["date_bce"].values.astype(float); ids = dated["id"].values
    print(f"{len(dates)} dated texts | {X.shape[1]} usable features | alpha={alpha} | {nperm} permutations\n")

    yt4 = np.array([to_period(v) for v in dates])
    _, c4 = np.unique(yt4, return_counts=True)
    yt2 = np.array([to_binary(v) for v in dates])
    _, c2 = np.unique(yt2, return_counts=True)
    base4, base2 = c4.max()/len(dates), c2.max()/len(dates)

    results = {}
    for fam in FAMILIES:
        maps = run_loo(dates, X, alpha, fam)
        S = score(dates, maps); results[fam] = (maps, S)

        print("="*78); print(f"MODEL FAMILY: {fam}"); print("="*78)
        print(f"{'text':<15}{'true':>7}{'period':>13}{'LOO pred':>10}{'pred period':>14}{'err':>7}")
        print("-"*68)
        for i in np.argsort(-dates):
            tp, pp = to_period(dates[i]), to_period(maps[i])
            mk = "" if tp==pp else ("  <" if abs(tp-pp)==1 else "  <<")
            print(f"{ids[i]:<15}{dates[i]:7.0f}{PERIODS[tp]:>13}{maps[i]:10.0f}"
                  f"{PERIODS[pp]:>14}{abs(dates[i]-maps[i]):7.0f}{mk}")
        print(f"\n  4-period exact   {S['acc4']:.3f}   (baseline {base4:.3f})")
        print(f"  4-period +/-1    {S['adj4']:.3f}")
        print(f"  pre/post-exilic  {S['acc2']:.3f}   (baseline {base2:.3f})")
        print(f"  Spearman rho     {S['rho']:+.3f}")
        print(f"  MAE              {S['mae']:.1f} yr")
        print("\n  confusion (rows=true, cols=pred)")
        print("               " + "".join(f"{p[:6]:>8}" for p in PERIODS))
        for a_ in range(4):
            print(f"  {PERIODS[a_]:<12}" + "".join(
                f"{int(((S['yt4']==a_)&(S['yp4']==b_)).sum()):8d}" for b_ in range(4)))
        print()

    # ── permutation null, shared across families ──
    print("="*78); print(f"PERMUTATION NULL  (n={nperm}, full pipeline re-run per draw)"); print("="*78)
    null = {f: {k: [] for k in ("acc4","adj4","acc2","rho","mae")} for f in FAMILIES}
    dropped = {f: 0 for f in FAMILIES}
    for _ in range(nperm):
        dp = RNG.permutation(dates)
        for fam in FAMILIES:
            s = score(dp, run_loo(dp, X, alpha, fam))
            if s is None: dropped[fam] += 1; continue
            for k in null[fam]: null[fam][k].append(s[k])

    summary = {}
    for fam in FAMILIES:
        S = results[fam][1]
        print(f"\n  --- {fam} ---   ({dropped[fam]} of {nperm} draws yielded no usable model, excluded)")
        summary[fam] = {}
        for k, lab, hi in [("acc4","4-period exact",True), ("adj4","4-period +/-1",True),
                           ("acc2","pre/post-exilic",True), ("rho","Spearman rho",True),
                           ("mae","MAE (yr)",False)]:
            v = np.array(null[fam][k], float); v = v[np.isfinite(v)]
            obs = S[k]
            p = ((np.sum(v >= obs) if hi else np.sum(v <= obs)) + 1)/(len(v)+1)
            print(f"    {lab:<18} observed {obs:8.3f}   null {v.mean():7.3f} +/- {v.std():.3f}"
                  f"   p = {p:.4f}{'  *' if p<0.05 else ''}")
            summary[fam][k] = dict(observed=obs, null_mean=float(v.mean()),
                                   null_sd=float(v.std()), p=float(p))

    out = pd.DataFrame(dict(id=ids, date_bce=dates,
                            true_period=[PERIODS[to_period(d)] for d in dates],
                            loo_generative=results["generative"][0],
                            loo_ridge=results["ridge"][0],
                            n_words=dated["n_words"].values))
    out["pred_generative"] = [PERIODS[to_period(v)] for v in out["loo_generative"]]
    out["pred_ridge"]      = [PERIODS[to_period(v)] for v in out["loo_ridge"]]
    out.to_csv("/home/claude/loo_period_results.csv", index=False)
    json.dump(summary, open("/home/claude/loo_period_significance.json","w"), indent=2)
    print("\nwrote loo_period_results.csv, loo_period_significance.json")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv)>1 else 0.05,
         int(sys.argv[2]) if len(sys.argv)>2 else 2000)
