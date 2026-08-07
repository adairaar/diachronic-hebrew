"""
Per-book LOBO diagnostics for the RECOMMENDED model, for the manuscript tables.

Recommended model = ~500-word chunks, 579 base features, inverse-density
weighted (alpha=1.0, bw=90), ridge with lambda by inner LOBO, variance-matched.

Dumps: per-book truth / prediction / residual / chunk sd / n_chunks / P(post-exilic),
headline metrics, and a book-level permutation null that re-runs the ENTIRE
pipeline (weights, lambda selection, variance matching) on shuffled dates.

The lambda sweep uses one eigendecomposition of the dual Gram matrix per fit,
so all lambdas cost what one costs.
"""
import numpy as np, pandas as pd, importlib.util, json, sys, time, os
from scipy import stats

pt = importlib.util.spec_from_file_location("pt", "/home/claude/predict_targets.py")
PT = importlib.util.module_from_spec(pt); pt.loader.exec_module(PT)

NPERM = int(sys.argv[1]) if len(sys.argv) > 1 else 200

Dd = pd.read_csv("/home/claude/big_features_500.csv")
feats = [c for c in Dd.columns if c not in PT.META]
Xa = Dd[feats].astype(float)
keep = (Xa.std() > 0) & (Xa.isna().mean() < 0.2)
feats = list(np.array(feats)[keep.values])
med = Dd[feats].astype(float).median()
X = Dd[feats].astype(float).fillna(med).values
y = Dd.date_bce.values.astype(float); g = Dd.unit.values
books = list(pd.unique(g)); bdate = {b: y[g == b][0] for b in books}
LAM = PT.LAM


def fit_all_lams(Xtr, ytr, wtr, Xte):
    """Ridge predictions for every lambda in LAM from one eigendecomposition."""
    mu = Xtr.mean(0); sd = Xtr.std(0); sd = np.where(sd > 0, sd, 1.0)
    A = (Xtr - mu) / sd; B = (Xte - mu) / sd
    yb = np.average(ytr, weights=wtr)
    s = np.sqrt(wtr)
    Aw = A * s[:, None]; yw = (ytr - yb) * s
    K = Aw @ Aw.T
    ev, V = np.linalg.eigh(K)
    Vt_y = V.T @ yw
    BAt = (B @ Aw.T) @ V                      # (n_te, n_tr) in eigenbasis
    return {lam: BAt @ (Vt_y / (ev + lam)) + yb for lam in LAM}


def lobo(yv):
    """Full LOBO under the recommended model, re-selecting lambda inside each fold."""
    bd = {b: yv[g == b][0] for b in books}
    t, p, sd, lams = [], [], [], []
    for b in books:
        te = g == b; tr = ~te
        inner = [x for x in books if x != b]
        wb = dict(zip(inner, PT.wts([bd[x] for x in inner])))
        wv = np.array([wb[u] for u in g[tr]])
        err = {lam: [] for lam in LAM}
        for bb in inner:
            m = g[tr] != bb
            preds = fit_all_lams(X[tr][m], yv[tr][m], wv[m], X[tr][~m])
            for lam in LAM:
                err[lam].append(abs(np.median(preds[lam]) - bd[bb]))
        blam = min(LAM, key=lambda l: np.mean(err[l]))
        pr = fit_all_lams(X[tr], yv[tr], wv, X[te])[blam]
        t.append(bd[b]); p.append(float(np.median(pr))); sd.append(float(np.std(pr)))
        lams.append(blam)
    return np.array(t), np.array(p), np.array(sd), lams


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


print(f"{len(feats)} features | {len(Dd)} chunks | {len(books)} books")
nch = pd.Series(g).value_counts()
print(f"chunks per book: min {nch.min()} median {int(nch.median())} max {nch.max()}")
print(f"words: {int(Dd.n_words.sum())} total, median chunk {int(Dd.n_words.median())}\n")

t0 = time.time()
t, p, sd, lams = lobo(y)
M = metrics(t, p)
cal, resid = M["cal"], M["resid"]
ar = np.sort(np.abs(resid)); n = len(ar)
def _q(a):
    k = int(np.ceil((n + 1) * a))
    return np.inf if k > n else ar[k - 1]
q68 = _q(.68)
q90 = _q(.90)
mean_pred = float(np.abs(t - t.mean()).mean())
print(f"one LOBO in {time.time()-t0:.1f} s")
print(f"variance-match scale S = {M['S']:.2f}   modal lambda = {max(set(lams), key=lams.count):.0e}")
print(f"LOBO MAE {M['mae']:.1f} yr   (mean-predictor baseline {mean_pred:.1f} yr)")
print(f"Spearman rho {M['rho']:+.3f} (p={M['rho_p']:.4f})")
print(f"pairwise ordering {M['pair']*100:.1f}% of {M['n_pair']} pairs")
print(f"conformal 68% +/-{q68:.0f} yr, 90% +/-{q90:.0f} yr\n")

rows = []
for i, b in enumerate(books):
    draws = cal[i] + resid
    rows.append(dict(book=b, truth=int(t[i]), pred=round(float(cal[i])),
                     resid=round(float(resid[i])), n_chunks=int(nch[b]),
                     n_words=int(Dd.n_words[g == b].sum()),
                     chunk_sd=round(float(sd[i] * M["S"])),
                     p_post=round(float(np.mean(draws < 586)), 2)))
R = pd.DataFrame(rows).sort_values("truth", ascending=False)
R.to_csv("/home/claude/final_lobo_books.csv", index=False)
print(R.to_string(index=False))

pre = R[R.truth > 586]; post = R[R.truth < 586]
n_pre_ok = int((pre.pred > 586).sum()); n_post_ok = int((post.pred < 586).sum())
print(f"\npre-exilic books placed pre-exilic : {n_pre_ok}/{len(pre)}")
print(f"post-exilic books placed post-exilic: {n_post_ok}/{len(post)}")
print(f"side accuracy overall              : {((R.pred>586)==(R.truth>586)).sum()}/{len(R)}")

# ---- book-level permutation null through the ENTIRE pipeline ----
CKPT = "/home/claude/final_lobo_null.csv"
rng = np.random.default_rng(0)
nulls = []
done = 0
if os.path.exists(CKPT):                      # resume a killed run
    prev = np.loadtxt(CKPT, delimiter=",", skiprows=1, ndmin=2)
    if prev.size:
        nulls = [tuple(r) for r in prev]
        done = len(nulls)
        print(f"resuming from {done} completed permutations", flush=True)
t0 = time.time()
for k in range(NPERM):
    perm = rng.permutation(books)             # advance the stream either way
    if k < done:
        continue
    remap = {b: bdate[perm[i]] for i, b in enumerate(books)}
    yp = np.array([remap[u] for u in g], float)
    tt, pp, _, _ = lobo(yp)
    mm = metrics(tt, pp)
    nulls.append((mm["rho"], mm["pair"], mm["mae"]))
    if (k + 1) % 5 == 0 or k == NPERM - 1:    # checkpoint often; these die
        np.savetxt(CKPT, np.array(nulls), delimiter=",",
                   header="rho,pair,mae", comments="")
        el = time.time() - t0
        rate = el / max(k + 1 - done, 1)
        print(f"  perm {k+1}/{NPERM}  ({el/60:.1f} min, "
              f"eta {rate*(NPERM-k-1)/60:.1f} min)", flush=True)
N = np.array(nulls)
p_rho = float((np.sum(N[:, 0] >= M["rho"]) + 1) / (NPERM + 1))
p_pair = float((np.sum(N[:, 1] >= M["pair"]) + 1) / (NPERM + 1))
p_mae = float((np.sum(N[:, 2] <= M["mae"]) + 1) / (NPERM + 1))
print(f"\nbook-level permutation null ({NPERM} draws, full pipeline re-run):")
print(f"  rho   observed {M['rho']:+.3f}  null median {np.median(N[:,0]):+.3f}  p = {p_rho:.4f}")
print(f"  pair  observed {M['pair']*100:.1f}%  null median {np.median(N[:,1])*100:.1f}%  p = {p_pair:.4f}")
print(f"  MAE   observed {M['mae']:.1f}   null median {np.median(N[:,2]):.1f}   p = {p_mae:.4f}")
np.savetxt("/home/claude/final_lobo_null.csv", N, delimiter=",",
           header="rho,pair,mae", comments="")

json.dump(dict(n_feats=len(feats), n_chunks=int(len(Dd)), n_books=len(books),
               n_words=int(Dd.n_words.sum()), S=M["S"], mae=M["mae"],
               mae_baseline=mean_pred, rho=M["rho"], rho_p=M["rho_p"],
               pair=M["pair"], n_pair=M["n_pair"], q68=float(q68), q90=float(q90),
               n_pre=len(pre), n_pre_ok=n_pre_ok, n_post=len(post), n_post_ok=n_post_ok,
               p_rho=p_rho, p_pair=p_pair, p_mae=p_mae, n_perm=NPERM,
               null_rho_med=float(np.median(N[:, 0])),
               null_pair_med=float(np.median(N[:, 1])),
               null_mae_med=float(np.median(N[:, 2]))),
          open("/home/claude/final_lobo_metrics.json", "w"), indent=2)
print("\nwrote final_lobo_books.csv, final_lobo_metrics.json, final_lobo_null.csv")
