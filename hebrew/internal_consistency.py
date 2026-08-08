"""
How internally consistent is each source?

This asks something the absolute dates cannot answer reliably.  A source that is
a single composition should look, chunk by chunk, like a single dated book looks.
A source that is a rolling document -- an old core with later accretions -- should
be measurably more dispersed than that.

The quantity is a ratio, and that is why it is worth trusting when the absolute
dates are not.  Variance matching multiplies every prediction by one positive
constant; it therefore multiplies every within-unit dispersion by that same
constant and leaves the RANKING of units by dispersion untouched.  The
calibration choices that moved the absolute dates by centuries today cannot
change whether JE is more varied than P.

The benchmark is the anchor corpus itself.  Each anchor chunk is predicted by a
model trained without its own book, so the spread of chunk predictions within an
anchor book measures model noise plus whatever internal variation that book
actually has.  That is a conservative yardstick, because several anchors are
themselves composite -- Isaiah 1-39 most obviously -- so the reference
distribution is inflated in the direction that makes the sources look tame.

Hypotheses under test, stated in advance:
    P    largely internally consistent
    D    some variation; the law code and the frame are usually assigned layers
    JE   the most diverse; an early Jacob core with later additions would show
"""
import json
import numpy as np, pandas as pd, importlib.util
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", "/home/claude/predict_targets.py")
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

EX = 586
RNG = np.random.default_rng(7)

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


# ── per-CHUNK leave-one-book-out predictions for the anchors ────────────────
chunk_pred = np.full(len(Dd), np.nan)
book_med = {}
for b in books:
    te = g == b; tr = ~te
    inner = [x for x in books if x != b]
    wb = dict(zip(inner, PT.wts([bdate[x] for x in inner])))
    wv = np.array([wb[u] for u in g[tr]])
    err = {l: [] for l in PT.LAM}
    for bb in inner:
        m = g[tr] != bb
        pr = fit_all(X[tr][m], y[tr][m], wv[m], X[tr][~m])
        for l in PT.LAM:
            err[l].append(abs(np.median(pr[l]) - bdate[bb]))
    bl = min(PT.LAM, key=lambda l: np.mean(err[l]))
    p = fit_all(X[tr], y[tr], wv, X[te])[bl]
    chunk_pred[te] = p
    book_med[b] = float(np.median(p))

t = np.array([bdate[b] for b in books]); pm = np.array([book_med[b] for b in books])
S = float(np.clip(t.std() / pm.std(), 0.5, 8.0))
cal = lambda v: t.mean() + S * (v - pm.mean())
Dd = Dd.assign(cpred=cal(chunk_pred))

# ── per-chunk predictions for the targets from the full model ───────────────
wb = dict(zip(books, PT.wts([bdate[x] for x in books])))
wv = np.array([wb[u] for u in g])
err = {l: [] for l in PT.LAM}
for b in books:
    m = g != b
    pr = fit_all(X[m], y[m], wv[m], X[~m])
    for l in PT.LAM:
        err[l].append(abs(np.median(pr[l]) - bdate[b]))
bl = min(PT.LAM, key=lambda l: np.mean(err[l]))
Dt = Dt.assign(cpred=cal(fit_all(X, y, wv, XT)[bl]))


def disp(v):
    v = np.asarray(v, float)
    return dict(n=len(v), median=float(np.median(v)), sd=float(np.std(v, ddof=1)),
                iqr=float(np.subtract(*np.percentile(v, [75, 25]))),
                pct_post=float(np.mean(v < EX)))


pd.concat([
    Dd[["unit", "cpred"]].assign(kind="anchor"),
    Dt[["unit", "cpred"]].assign(kind="target"),
]).to_csv("/home/claude/chunk_preds.csv", index=False)

# ── the reference distribution: dispersion within single dated books ────────
REF = {b: disp(Dd.cpred[g == b]) for b in books if (g == b).sum() >= 4}
ref_sd = np.array([r["sd"] for r in REF.values()])
print("=" * 82)
print("1.  THE YARDSTICK: DISPERSION WITHIN SINGLE DATED BOOKS")
print("=" * 82)
print(f"  {len(REF)} anchor books with at least 4 passages")
print(f"  within-book SD of passage dates: median {np.median(ref_sd):.0f} yr, "
      f"range {ref_sd.min():.0f}-{ref_sd.max():.0f}")
hi = sorted(REF.items(), key=lambda kv: -kv[1]["sd"])[:4]
print("  most dispersed anchors: "
      + ", ".join(f"{k} {v['sd']:.0f}" for k, v in hi))
print("  (Isaiah 1-39 is itself composite, so this yardstick is generous to the")
print("   sources rather than harsh on them)")
r_n, p_n = stats.spearmanr([r["n"] for r in REF.values()], ref_sd)
print(f"  dispersion vs passage count across anchors: rho {r_n:+.3f} (p={p_n:.2f})"
      f"  -- {'no' if p_n > .05 else 'a'} size artifact")

# ── the sources ────────────────────────────────────────────────────────────
UNITS = [("JE_source", "JE composite"), ("Gen_JE", "  Genesis JE"),
         ("Exo_JE", "  Exodus JE"), ("Num_JE", "  Numbers JE"),
         ("D_source", "D source"), ("D_Code", "  law code, Deut 12-26"),
         ("D_Frame", "  frame, Deut 1-11 + 27-34"),
         ("P_source", "P source"), ("Lev_Priestly", "  Leviticus 1-16"),
         ("Lev_Holiness", "  Holiness Code, Lev 17-26"),
         ("Jer_DTR", "Jeremiah Dtr prose")]

print()
print("=" * 82)
print("2.  DISPERSION WITHIN EACH SOURCE")
print("=" * 82)
print(f"  {'unit':<28}{'n':>4}{'median':>8}{'SD':>7}{'IQR':>7}"
      f"{'vs anchors':>12}{'% post-exilic':>15}")
rows = []
for u, lab in UNITS:
    v = Dt.cpred[gt == u]
    if len(v) < 4: continue
    d = disp(v)
    pct = float(np.mean(ref_sd <= d["sd"]))
    rows.append(dict(unit=u, label=lab.strip(), **d, pctile=pct))
    print(f"  {lab:<28}{d['n']:>4}{d['median']:>8.0f}{d['sd']:>7.0f}{d['iqr']:>7.0f}"
          f"{pct*100:>11.0f}%{d['pct_post']*100:>14.0f}%")
R = pd.DataFrame(rows)
R.to_csv("/home/claude/internal_consistency.csv", index=False)
print("\n  'vs anchors' is the percentile of that unit's dispersion within the")
print("  reference distribution of single dated books.  50% means it looks like")
print("  an ordinary book; 90% means it is more varied than nine in ten of them.")

# ── is JE really more dispersed than P?  bootstrap the difference ───────────
print()
print("=" * 82)
print("3.  IS THE DIFFERENCE REAL?  BOOTSTRAP OVER PASSAGES")
print("=" * 82)


def boot_sd(v, B=4000):
    v = np.asarray(v, float)
    return np.array([np.std(RNG.choice(v, len(v), replace=True), ddof=1)
                     for _ in range(B)])


BS = {u: boot_sd(Dt.cpred[gt == u]) for u, _ in UNITS
      if (gt == u).sum() >= 4}
for u in ("JE_source", "D_source", "P_source"):
    q = np.percentile(BS[u], [2.5, 97.5])
    print(f"  {u:<12} SD {np.std(Dt.cpred[gt==u], ddof=1):>5.0f} yr   "
          f"95% CI {q[0]:.0f}-{q[1]:.0f}")
PAIRS = [("JE_source", "P_source"), ("JE_source", "D_source"),
         ("D_source", "P_source")]
print()
for a, b in PAIRS:
    d = BS[a] - BS[b]
    pr = float(np.mean(d > 0))
    print(f"  P(SD[{a.split('_')[0]}] > SD[{b.split('_')[0]}]) = {pr:.3f}   "
          f"difference {np.mean(d):+.0f} yr "
          f"(95% CI {np.percentile(d,2.5):+.0f} to {np.percentile(d,97.5):+.0f})")

print()
print("=" * 82)
print("4.  THE PREDICTION, AND WHAT HAPPENED")
print("=" * 82)
order = R[R.unit.isin(["JE_source", "D_source", "P_source"])].sort_values(
    "sd", ascending=False)
got = " > ".join(o.split("_")[0] for o in order.unit)
print(f"  predicted most to least varied:  JE > D > P")
print(f"  observed:                        {got}")
json.dump(dict(ref_median_sd=float(np.median(ref_sd)),
               ref_range=[float(ref_sd.min()), float(ref_sd.max())],
               units=R.to_dict("records"),
               observed_order=got,
               p_JE_gt_P=float(np.mean(BS["JE_source"] - BS["P_source"] > 0)),
               p_JE_gt_D=float(np.mean(BS["JE_source"] - BS["D_source"] > 0)),
               p_D_gt_P=float(np.mean(BS["D_source"] - BS["P_source"] > 0))),
          open("/home/claude/internal_consistency.json", "w"), indent=2)
print("\nwrote internal_consistency.csv, internal_consistency.json")
