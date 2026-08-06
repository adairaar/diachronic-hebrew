"""
Leakage decomposition computed on the GENERATIVE family the paper reports,
with no dependence on the withdrawn HB-VI run.

For each dated unit:
  1. Fit the model leave-one-out (unit contributes nothing to its own fit).
  2. Take the resulting likelihood over the date grid -- this is what the
     linguistic evidence alone says.
  3. Multiply by N(d_u, sigma_u^2), the unit's own scholarly date used as a
     prior.  This is the leaky design.
  4. Report the likelihood-only MAP, the leaky posterior MAP, the implied
     likelihood width, and the share of posterior precision the data supplies.
"""
import numpy as np, pandas as pd, json
from scipy import stats
import importlib.util

spec = importlib.util.spec_from_file_location("pl", "/home/claude/period_loo2.py")
pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)
GRID = pl.GRID
B = "/mnt/user-data/uploads/Diachronic Hebrew"


def loglik_grid(Xtr, dtr, xte, alpha=0.05):
    """Per-feature OLS inverted to a log-likelihood over the date grid."""
    n = len(dtr)
    mu, sd = Xtr.mean(0), np.where(Xtr.std(0) > 0, Xtr.std(0), 1.0)
    Z, z = (Xtr - mu) / sd, (xte - mu) / sd
    keep = pl.screen(Z, dtr, alpha)
    if keep.size == 0:
        return None
    d0, ds = dtr.mean(), (dtr.std() or 1.0)
    t = (dtr - d0) / ds; tc = t - t.mean(); Y = Z[:, keep]
    b = (tc[:, None] * (Y - Y.mean(0))).sum(0) / (tc ** 2).sum()
    a = Y.mean(0) - b * t.mean()
    s = np.maximum(np.sqrt(((Y - (a + np.outer(t, b))) ** 2).sum(0) / max(n - 2, 1)), 1e-3)
    tg = (GRID - d0) / ds
    return (-0.5 * ((z[keep][None, :] - (a[None, :] + np.outer(tg, b))) / s[None, :]) ** 2
            - np.log(s)[None, :]).sum(1)


def summarise(logp):
    p = np.exp(logp - logp.max()); p /= p.sum()
    m = GRID[np.argmax(p)]
    mean = (GRID * p).sum()
    sd = np.sqrt(((GRID - mean) ** 2 * p).sum())
    return m, sd


def main():
    dated, feats, X = pl.load()
    d = dated["date_bce"].values.astype(float)
    ids = dated["id"].values
    man = json.load(open(f"{B}/hebrew/corpus_manifest_v2.json"))
    sig = {t["id"]: t["date_sigma"] for k in ("training", "holdouts") for t in man[k]}
    role = {t["id"]: ("holdout" if k == "holdouts" or t.get("hbvi_holdout") else "training")
            for k in ("training", "holdouts") for t in man[k]}

    rows = []
    n = len(d)
    for i in range(n):
        tr = np.arange(n) != i
        ll = loglik_grid(X[tr], d[tr], X[i])
        if ll is None:
            continue
        lik_map, lik_sd = summarise(ll)
        su = float(sig[ids[i]])
        prior = -0.5 * ((GRID - d[i]) / su) ** 2
        post_map, post_sd = summarise(ll + prior)
        w = (1 / lik_sd ** 2) / (1 / lik_sd ** 2 + 1 / su ** 2) * 100
        rows.append(dict(id=ids[i], role=role[ids[i]], scholarly=d[i], sigma_u=su,
                         lik_map=lik_map, lik_sd=lik_sd, post_map=post_map,
                         post_sd=post_sd, data_share=w,
                         err_leaky=abs(post_map - d[i]), err_honest=abs(lik_map - d[i])))
    R = pd.DataFrame(rows)

    print("Leakage decomposition on the generative family (no HB-VI dependency)\n")
    print(f"{'unit':<15}{'sigma_u':>8}{'sigma_L':>8}{'data%':>7}"
          f"{'lik MAP':>9}{'leaky MAP':>10}{'|err| leaky':>12}{'|err| honest':>13}")
    print("-" * 82)
    for _, r in R.sort_values("sigma_u").iterrows():
        print(f"{r['id']:<15}{r['sigma_u']:8.0f}{r['lik_sd']:8.0f}{r['data_share']:7.1f}"
              f"{r['lik_map']:9.0f}{r['post_map']:10.0f}"
              f"{r['err_leaky']:12.0f}{r['err_honest']:13.0f}")
    print()
    print(f"  median sigma_L (likelihood width)      : {R.lik_sd.median():.0f} yr")
    print(f"  median data share of posterior precision: {R.data_share.median():.1f}%")
    print(f"  MAE under the leaky design              : {R.err_leaky.mean():.1f} yr")
    print(f"  MAE from the likelihood alone           : {R.err_honest.mean():.1f} yr")
    print()
    tight = R[R.sigma_u <= 20]
    print(f"  units with sigma_u <= 20 yr (n={len(tight)}):")
    print(f"    leaky MAE {tight.err_leaky.mean():.1f} yr vs honest MAE "
          f"{tight.err_honest.mean():.1f} yr; median data share {tight.data_share.median():.1f}%")
    R.to_csv("/home/claude/leakage_generative.csv", index=False)

    # ── LaTeX table ──
    NM = {"Isaiah_1":"Isaiah 1--39","Isaiah_2":"Isaiah 40--55","Isaiah_3":"Isaiah 56--66",
          "Zechariah_1":"Zechariah 1--8","Zechariah_2":"Zechariah 9--14",
          "Daniel":"Daniel (Heb.)","Jer_oracle":"Jeremiah oracle"}
    nm = lambda i: NM.get(i, i.replace("_", " "))
    sel = R.sort_values("sigma_u").head(14)
    L = [r"\begin{table}[!ht]", r"\centering",
         r"\caption{\textbf{What a scholarly prior contributes to a reported date.}",
         r"For each dated unit the model is fitted leave-one-out, giving a likelihood",
         r"over the date grid from the linguistic evidence alone (width",
         r"$\sigma_L$, mode $\mathrm{MAP}_L$).  Multiplying that likelihood by",
         r"$\mathcal{N}(d_u, \sigma_u^2)$ --- the unit's own scholarly date used as a",
         r"prior --- gives the posterior a holdout design of this kind would report.",
         r"``Data share'' is the fraction of posterior precision supplied by the",
         r"likelihood, $\sigma_L^{-2}/(\sigma_L^{-2}+\sigma_u^{-2})$.  Where the",
         r"scholarly date is tightly specified the share falls below 10\%, and the",
         r"reported error collapses toward zero while the error from the evidence",
         r"alone does not.  Units are ordered by $\sigma_u$.}",
         r"\begin{tabular}{lrrrrrr}", r"\hline",
         r"\textbf{Unit} & $\bm{\sigma_u}$ & $\bm{\sigma_L}$ & \textbf{data} &"
         r" $\bm{\mathrm{MAP}_L}$ & \textbf{err.\ reported} & \textbf{err.\ actual} \\",
         r" & (yr) & (yr) & (\%) & (BCE) & (yr) & (yr) \\", r"\hline"]
    L = [x.replace(r"\bm", r"\mathbf") for x in L]
    for _, r in sel.iterrows():
        L.append(f"{nm(r['id']):<18}& {r['sigma_u']:.0f} & {r['lik_sd']:.0f} & "
                 f"{r['data_share']:.1f} & {r['lik_map']:.0f} & {r['err_leaky']:.0f} & "
                 f"{r['err_honest']:.0f} " + r"\\")
    L += [r"\hline", r"\end{tabular}", r"\label{tab:leakage}", r"\end{table}"]
    open("/home/claude/paper/tab_leakage.tex", "w").write("\n".join(L) + "\n")
    print("\nwrote tab_leakage.tex (regenerated from the generative family)")


if __name__ == "__main__":
    main()
