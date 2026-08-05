#!/usr/bin/env python3
"""
loo_diagnostic_v2.py — does the LOO sign-consistency filter reject anything?
============================================================================
The manuscript treats LOO sign-consistency as a second, independent screen
that protects against false positives surviving a permissive p-threshold.

This script tests that claim directly.  The date vector is permuted, which
destroys any genuine diachronic signal, so EVERY feature is null by
construction.  We then ask: of the null features that sneak past p < alpha,
what fraction also achieve high LOO sign-consistency?

If that fraction is near 1, the LOO filter rejects essentially nothing and
provides no protection at all -- the retained set's false-discovery rate is
whatever the p-threshold alone delivers.

The mechanism to watch for: with n = 23, each leave-one-out refit shares 22
of 23 observations with the full fit.  The folds are almost perfectly
correlated with the full-sample statistic and with each other, so once |rho|
is moderate the sign simply cannot flip.  Sign-consistency is then a
near-deterministic function of |rho|, not independent evidence about it.

Output: hebrew/results_v2/loo_diagnostic.csv
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

RNG      = np.random.default_rng(20260805)
N_PERM   = 2000
ALPHAS   = [0.01, 0.05, 0.10, 0.20, 0.30]
LOO_GATE = 0.68


def p_from_rho(rho, n):
    rho = np.clip(rho, -0.999999, 0.999999)
    t = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    return 2 * stats.t.sf(np.abs(t), df=n - 2)


def corr_rows(A, b):
    """Correlation of each row of A (K,m) against vector b (m,)."""
    a = A - A.mean(axis=1, keepdims=True)
    bb = b - b.mean()
    num = a @ bb
    den = np.sqrt((a * a).sum(axis=1) * (bb * bb).sum())
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def rho_and_loo(Rx, rd, n):
    """Full-sample rho, p, and LOO sign-consistency for every feature.

    LOO is computed on the already-ranked data (ranks are not recomputed
    after deletion).  Since only the SIGN of each fold matters, this is
    equivalent for our purposes and keeps the simulation tractable.
    """
    rho  = corr_rows(Rx, rd)
    p    = p_from_rho(rho, n)
    sgn  = np.sign(rho)
    keep = np.zeros(Rx.shape[0])
    for i in range(n):
        m = np.arange(n) != i
        r_i = corr_rows(Rx[:, m], rd[m])
        keep += (np.sign(r_i) == sgn)
    return rho, p, keep / n


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
    Xv = X[usable].values
    Xv = np.where(np.isfinite(Xv), Xv, np.nanmedian(Xv, axis=0, keepdims=True))
    K = len(usable)
    Rx = stats.rankdata(Xv.T, axis=1)
    rd = stats.rankdata(dates)

    print(f"n = {n} training units, K = {K} candidate features\n")

    # ── Observed (real dates) ─────────────────────────────────────────────────
    o_rho, o_p, o_loo = rho_and_loo(Rx, rd, n)
    print("REAL DATA")
    for a in ALPHAS:
        sel = o_p < a
        both = sel & (o_loo >= LOO_GATE)
        print(f"  p<{a:.2f}: {sel.sum():3d} pass p   "
              f"{both.sum():3d} pass p AND LOO   "
              f"-> LOO removes {sel.sum()-both.sum()}")

    # ── Permutation null ──────────────────────────────────────────────────────
    print(f"\nNULL (dates permuted, {N_PERM} draws) — every feature is noise\n")
    cnt_p    = {a: [] for a in ALPHAS}
    cnt_both = {a: [] for a in ALPHAS}
    loo_given_p = {a: [] for a in ALPHAS}
    for _ in range(N_PERM):
        rp = RNG.permutation(rd)
        _, p, loo = rho_and_loo(Rx, rp, n)
        for a in ALPHAS:
            sel = p < a
            both = sel & (loo >= LOO_GATE)
            cnt_p[a].append(int(sel.sum()))
            cnt_both[a].append(int(both.sum()))
            if sel.sum():
                loo_given_p[a].append(float((loo[sel] >= LOO_GATE).mean()))

    rows = []
    print(f"{'alpha':>7s}{'null pass p':>13s}{'null pass p+LOO':>18s}"
          f"{'LOO survival':>14s}{'FDP (p)':>10s}{'FDP (p+LOO)':>14s}")
    print("  " + "-" * 74)
    for a in ALPHAS:
        np_p  = float(np.mean(cnt_p[a]))
        np_b  = float(np.mean(cnt_both[a]))
        surv  = float(np.mean(loo_given_p[a])) if loo_given_p[a] else float("nan")
        obs_p_n  = int((o_p < a).sum())
        obs_b_n  = int(((o_p < a) & (o_loo >= LOO_GATE)).sum())
        fdp_p = np_p / obs_p_n if obs_p_n else float("nan")
        fdp_b = np_b / obs_b_n if obs_b_n else float("nan")
        rows.append(dict(alpha=a, null_pass_p=round(np_p, 2),
                         null_pass_p_loo=round(np_b, 2),
                         loo_survival_rate=round(surv, 4),
                         obs_pass_p=obs_p_n, obs_pass_p_loo=obs_b_n,
                         FDP_p_only=round(fdp_p, 3), FDP_p_and_loo=round(fdp_b, 3)))
        print(f"{a:>7.2f}{np_p:>13.1f}{np_b:>18.1f}"
              f"{surv:>13.1%}{fdp_p:>10.3f}{fdp_b:>14.3f}")

    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "loo_diagnostic.csv"), index=False)

    # ── Why: sign-consistency as a function of |rho| ──────────────────────────
    print("\n\nLOO sign-consistency vs |rho|  (null features only)")
    allr, alll = [], []
    for _ in range(300):
        rp = RNG.permutation(rd)
        r, _, l = rho_and_loo(Rx, rp, n)
        allr.append(np.abs(r)); alll.append(l)
    allr = np.concatenate(allr); alll = np.concatenate(alll)
    print(f"  {'|rho| band':>14s}{'n':>8s}{'mean LOO':>11s}{'% at LOO=1.00':>16s}")
    for lo, hi in [(0.0,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,1.0)]:
        m = (allr >= lo) & (allr < hi)
        if m.sum():
            print(f"  {f'{lo:.1f}-{hi:.1f}':>14s}{m.sum():>8d}"
                  f"{alll[m].mean():>11.3f}{(alll[m] >= 0.999).mean():>15.1%}")

    print(f"\nSaved → {OUTDIR}/loo_diagnostic.csv")


if __name__ == "__main__":
    main()
