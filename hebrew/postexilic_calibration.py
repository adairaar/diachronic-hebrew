"""
Does P(post-exilic) mean what it says?

The paper claims P/D/JE are post-exilic at P >= 0.92.  That number comes from
a conformal predictive distribution.  If the distribution is honest, then
among dated texts the model assigns P_post = 0.9, about 90% should actually
be post-exilic.  If it is not honest, the headline claim is not supportable
regardless of how the per-text errors look.

Protocol, fully nested so nothing leaks:
  for each dated text i:
      fit on the other 24                       -> point estimate for i
      compute LOO residuals WITHIN those 24     -> conformal residual set
      predictive distribution for i = point_i + those residuals
      P_post(i) = fraction of that distribution later than 586 BCE
"""
import importlib.util as _ilu, os as _os
_d = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_d, "hebrew", "dh_paths.py")) \
        and _os.path.dirname(_d) != _d:
    _d = _os.path.dirname(_d)
_p = _ilu.spec_from_file_location(
    "dh_paths", _os.path.join(_d, "hebrew", "dh_paths.py"))
DH = _ilu.module_from_spec(_p); _p.loader.exec_module(DH)
import numpy as np, pandas as pd
from scipy import stats
import importlib.util

spec = importlib.util.spec_from_file_location("pl", DH.script("period_loo2.py"))
pl = importlib.util.module_from_spec(spec); spec.loader.exec_module(pl)

BOUND = 586.0   # BCE; post-exilic means date < 586


def run(fam):
    dated, feats, X = pl.load()
    d = dated["date_bce"].values.astype(float)
    ids = dated["id"].values
    n = len(d)
    rows = []
    for i in range(n):
        tr = np.arange(n) != i
        Xtr, dtr = X[tr], d[tr]
        pt = pl.FAMILIES[fam](Xtr, dtr, X[i], 0.05)
        if not np.isfinite(pt):
            continue
        # inner LOO among the 24 training texts -> residual set, text i excluded
        res = []
        m = len(dtr)
        for j in range(m):
            tr2 = np.arange(m) != j
            q = pl.FAMILIES[fam](Xtr[tr2], dtr[tr2], Xtr[j], 0.05)
            if np.isfinite(q):
                res.append(dtr[j] - q)
        res = np.array(res)
        draws = pt + res
        p_post = float(np.mean(draws < BOUND))
        rows.append(dict(id=ids[i], truth=d[i], point=pt,
                         actually_post=int(d[i] < BOUND), p_post=p_post,
                         err=abs(d[i] - pt), n_words=dated["n_words"].values[i]))
    return pd.DataFrame(rows)


def report(R, fam):
    print(f"\n{'='*74}\n{fam}\n{'='*74}")
    print(f"{'unit':<15}{'true':>7}{'actually':>10}{'point':>8}{'|err|':>7}{'P(post-exilic)':>16}")
    print("-" * 68)
    for _, r in R.sort_values("truth", ascending=False).iterrows():
        flag = "post" if r["actually_post"] else "PRE"
        bad = "   <-- confident and wrong" if (
            (r["p_post"] > 0.8 and not r["actually_post"]) or
            (r["p_post"] < 0.2 and r["actually_post"])) else ""
        print(f"{r['id']:<15}{r['truth']:7.0f}{flag:>10}{r['point']:8.0f}"
              f"{r['err']:7.0f}{r['p_post']:16.2f}{bad}")

    # Brier score against a base-rate-only forecast
    base = R.actually_post.mean()
    brier = np.mean((R.p_post - R.actually_post) ** 2)
    brier_base = np.mean((base - R.actually_post) ** 2)
    print(f"\n  base rate post-exilic      : {base:.2f}  ({R.actually_post.sum()}/{len(R)})")
    print(f"  Brier score (model)        : {brier:.4f}")
    print(f"  Brier score (base rate)    : {brier_base:.4f}")
    skill = 1 - brier / brier_base
    print(f"  Brier skill score          : {skill:+.3f}   "
          f"({'better' if skill > 0 else 'WORSE'} than just quoting the base rate)")

    # calibration in coarse bins
    print("\n  calibration:")
    for lo, hi in [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.01)]:
        s = R[(R.p_post >= lo) & (R.p_post < hi)]
        if len(s):
            print(f"    P_post in [{lo:.1f},{hi:.1f}): n={len(s):2d}  "
                  f"mean predicted {s.p_post.mean():.2f}  actual {s.actually_post.mean():.2f}")

    # the number that matters for the paper's claim
    hi = R[R.p_post >= 0.90]
    if len(hi):
        print(f"\n  texts the model calls post-exilic at P>=0.90: n={len(hi)}, "
              f"actually post-exilic {hi.actually_post.sum()}/{len(hi)} "
              f"({hi.actually_post.mean():.0%})")
    conf_wrong = R[((R.p_post > 0.8) & (R.actually_post == 0)) |
                   ((R.p_post < 0.2) & (R.actually_post == 1))]
    print(f"  confident-and-wrong calls (P>0.8 or <0.2 on the wrong side): "
          f"{len(conf_wrong)}/{len(R)}")
    return dict(brier=brier, skill=skill, base=base,
                n_hi=len(hi), hi_correct=int(hi.actually_post.sum()) if len(hi) else 0)


if __name__ == "__main__":
    out = {}
    for fam in ("generative", "ridge"):
        R = run(fam)
        out[fam] = report(R, fam)
        R.to_csv(DH.f(f"postexilic_calibration_{fam}.csv"), index=False)
    print("\n" + "=" * 74)
    print("VERDICT")
    print("=" * 74)
    for fam, v in out.items():
        verdict = ("supports the claim" if v["skill"] > 0.15 else
                   "does not support a probabilistic claim" if v["skill"] <= 0 else
                   "marginal")
        print(f"  {fam:<12} Brier skill {v['skill']:+.3f} -> {verdict}")
