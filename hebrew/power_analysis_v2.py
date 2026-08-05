#!/usr/bin/env python3
"""
power_analysis_v2.py — set the screening p-threshold from statistical power
===========================================================================
With n training units and K candidate features, the screening threshold alpha
trades two errors against each other:

  * too strict -> genuine diachronic features are missed (low power)
  * too loose  -> the retained set is mostly noise (high false-discovery rate)

Both are quantified by simulation at the actual corpus size.  The false-
discovery estimate uses a permutation null built from the real feature matrix,
which preserves the between-feature correlation structure that an analytic
null ignores (features here are far from independent: verb-form rates,
stem rates and clause fractions all co-vary).

Spearman is computed as Pearson-on-ranks and fully vectorised, so the whole
analysis runs in seconds.

Output: hebrew/results_v2/power_analysis.csv
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE   = os.path.dirname(os.path.abspath(__file__))
FM     = os.path.join(HERE, "data", "feature_matrix_v2.csv")
OUTDIR = os.path.join(HERE, "results_v2")
os.makedirs(OUTDIR, exist_ok=True)

META = {"date_bce","date_sigma","register","genre","holdout",
        "hbvi_holdout","in_training","n_words"}

RNG    = np.random.default_rng(20260805)
N_SIM  = 20000
N_PERM = 5000
ALPHAS = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
RHOS   = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def rank_rows(A):
    """Average-tie ranks along axis 1."""
    return stats.rankdata(A, axis=1)


def rho_rows(Ra, Rb):
    """Pearson correlation between matching rows of two rank matrices."""
    a = Ra - Ra.mean(axis=1, keepdims=True)
    b = Rb - Rb.mean(axis=1, keepdims=True)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def p_from_rho(rho, n):
    """Two-sided p via the t approximation (adequate for n >= 20)."""
    rho = np.clip(rho, -0.999999, 0.999999)
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    return 2 * stats.t.sf(np.abs(t), df=n - 2)


def power_curve(n, rhos, alphas, n_sim=N_SIM):
    """Monte-Carlo power for each (rho, alpha)."""
    out = {}
    for r in rhos:
        L = np.linalg.cholesky(np.array([[1.0, r], [r, 1.0]]))
        z = RNG.standard_normal((n_sim, n, 2)) @ L.T
        p = p_from_rho(rho_rows(rank_rows(z[:, :, 0]), rank_rows(z[:, :, 1])), n)
        out[r] = [float((p < a).mean()) for a in alphas]
    return out


def min_detectable_rho(n, alpha, target=0.80, n_sim=8000):
    lo, hi = 0.05, 0.99
    for _ in range(14):
        mid = (lo + hi) / 2
        L = np.linalg.cholesky(np.array([[1.0, mid], [mid, 1.0]]))
        z = RNG.standard_normal((n_sim, n, 2)) @ L.T
        p = p_from_rho(rho_rows(rank_rows(z[:, :, 0]), rank_rows(z[:, :, 1])), n)
        if (p < alpha).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    df = pd.read_csv(FM, index_col="id")
    tr = df[df["in_training"] == True]                      # noqa: E712
    dates = tr["date_bce"].values.astype(float)
    n = len(tr)

    feats = [c for c in tr.columns if c not in META]
    X = tr[feats].apply(pd.to_numeric, errors="coerce")
    usable = [c for c in feats
              if np.isfinite(X[c].values).sum() >= n - 2
              and np.nanstd(X[c].values) > 1e-12]
    X = X[usable]
    # median-fill the handful of residual gaps so the null can be vectorised
    Xv = np.where(np.isfinite(X.values), X.values,
                  np.nanmedian(X.values, axis=0, keepdims=True))
    K = len(usable)
    print(f"Corpus v2: n = {n} training units, K = {K} candidate features")
    print(f"(features requiring >= {n-2} finite observations)\n")

    Rx = rank_rows(Xv.T)                       # (K, n) feature ranks
    rd = stats.rankdata(dates)

    obs_rho = rho_rows(Rx, np.tile(rd, (K, 1)))
    obs_p   = p_from_rho(obs_rho, n)

    # ── Permutation null ──────────────────────────────────────────────────────
    print(f"Permutation null: {N_PERM} shuffles of the date vector …")
    null_counts = {a: np.empty(N_PERM, dtype=int) for a in ALPHAS}
    for s in range(N_PERM):
        rp = RNG.permutation(rd)
        p  = p_from_rho(rho_rows(Rx, np.tile(rp, (K, 1))), n)
        for a in ALPHAS:
            null_counts[a][s] = int((p < a).sum())

    # ── Power ─────────────────────────────────────────────────────────────────
    print(f"\nPower to detect a true monotone trend (n = {n})\n")
    hdr = "  rho  " + "".join(f"{('a='+format(a,'.2f')):>9s}" for a in ALPHAS)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    pw = power_curve(n, RHOS, ALPHAS)
    for r in RHOS:
        print(f"  {r:.1f}  " + "".join(f"{v:>9.2f}" for v in pw[r]))

    # ── FDR accounting ────────────────────────────────────────────────────────
    print("\n\nFalse-discovery accounting\n")
    print(f"{'alpha':>7s}{'retained':>10s}{'exp.null':>10s}{'null p95':>10s}"
          f"{'est.FDP':>10s}{'min|rho| @80%':>15s}")
    print("  " + "-" * 60)
    rows = []
    for a in ALPHAS:
        obs   = int((obs_p < a).sum())
        nulls = null_counts[a]
        exp_n = float(nulls.mean())
        p95   = float(np.percentile(nulls, 95))
        fdp   = exp_n / obs if obs > 0 else np.nan
        mdr   = min_detectable_rho(n, a)
        i     = ALPHAS.index(a)
        rows.append(dict(alpha=a, retained=obs, expected_null=round(exp_n, 1),
                         null_p95=p95,
                         est_FDP=round(fdp, 3) if obs else np.nan,
                         min_rho_80pct=round(mdr, 3),
                         power_rho50=round(pw[0.5][i], 3),
                         power_rho70=round(pw[0.7][i], 3),
                         power_rho90=round(pw[0.9][i], 3)))
        print(f"{a:>7.2f}{obs:>10d}{exp_n:>10.1f}{p95:>10.1f}"
              f"{fdp:>10.3f}{mdr:>15.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTDIR, "power_analysis.csv"), index=False)

    print("\n\nInterpretation")
    print("  est.FDP = expected share of the retained set that is noise,")
    print("  calibrated against the real inter-feature correlation structure.\n")
    ok = out[out["est_FDP"] <= 0.10]
    if len(ok):
        b = ok.iloc[-1]
        print(f"  Largest alpha with est.FDP <= 0.10:  alpha = {b['alpha']:.2f}")
        print(f"    retains {int(b['retained'])} features "
              f"(~{b['expected_null']:.1f} expected noise)")
        print(f"    power at rho=0.9 : {b['power_rho90']:.2f}")
        print(f"    power at rho=0.7 : {b['power_rho70']:.2f}")
        print(f"    power at rho=0.5 : {b['power_rho50']:.2f}")
        print(f"    smallest |rho| detectable at 80% power: {b['min_rho_80pct']:.2f}")
    else:
        print("  No threshold reaches est.FDP <= 0.10.")
    print(f"\nSaved → {OUTDIR}/power_analysis.csv")


if __name__ == "__main__":
    main()
