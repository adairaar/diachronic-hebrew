#!/usr/bin/env python3
"""
dating_v3.py — MLE-MVN with a pre-specified feature set and honest uncertainty
==============================================================================
Design agreed August 2026 after the corpus-v2 audit.

  Dating        pre-specified literature feature set, NOT screened on p.
                No selection on the outcome means no winner's curse and no
                multiple-testing penalty; the power/FDP analysis in
                power_analysis_v2.py applies to *screened* sets and does not
                apply here.

  Uncertainty   prediction variance rather than residual variance.  The old
                model treated the fitted alpha_j, beta_j as known exactly and
                used only sigma_j^2.  The correct dispersion for a new
                observation at date d is

                    Var(x_j - mu_j(d)) = sigma_j^2 * h(d),
                    h(d) = 1 + 1/n + (d - dbar)^2 / Sxx

                Because every feature is regressed on the same date vector,
                h(d) is a single scalar multiplying the whole covariance:

                    Sigma(d) = h(d) * Sigma_resid

                h(d) grows with distance from the training centroid, so texts
                dated outside the training range -- P, D, JE -- receive the
                wider intervals they deserve, while in-range texts barely move.

                NOTE: since Sigma now depends on d, the log-determinant term is
                no longer constant and MUST be retained in the log-likelihood.
                Dropping it (as a fixed-Sigma model may) biases the MAP toward
                the training centroid.

  Tikhonov      retained at all J.  cond(Sigma) ~ 9.2e4 even at J=4, so the
                ill-conditioning is driven by feature collinearity, not by
                J > N.

  Other analyses (genre correction, archaism diagnostic, n-gram instrument)
  continue to use the fuller feature set; that is a separate choice and needs
  no covariance justification.

Outputs
  results_v2/dating_v3.csv
  results_v2/calibration_v3.csv
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

RIDGE     = 0.10
RIDGE_RES = 0.20
GRID      = np.linspace(900, -100, 601)

# ── Pre-specified dating features (literature-derived, fixed a priori) ────────
# Cited to the diachronic Hebrew tradition; NOT selected on their behaviour in
# this corpus.  Provenance is given so the pre-registration is auditable.
PRESPEC = {
    "frac_ani":       "anoki -> ani 1sg pronoun shift (Hurvitz 1972)",
    "frac_she":       "asher -> she- relative particle (Polzin 1976)",
    "frac_ein":       "lo' -> 'en negation (Rooker 1990)",
    "rate_wayyiqtol": "wayyiqtol narrative decline (Polzin 1976)",
    "rate_qatal":     "qatal rise (Rooker 1990)",
    "frac_niphal":    "niphal stem expansion (Hurvitz 1982)",
    "rate_inf_con":   "infinitive-construct rate (Hurvitz 1982)",
    "rate_gam":       "gam particle rate (Rooker 1990)",
    "rate_ut_nouns":  "-ut nominalisation rate (Hornkohl 2024)",
    # Tier 3 resistant set (clause-level, below conscious control)
    "frac_infc":      "infinitive-construct clause fraction [R]",
    "frac_fronted":   "fronted (non-V-initial) clause fraction [R]",
    "frac_null_subj": "null-subject verbal clause fraction [R]",
    "frac_wqtl_wayq": "waw-qatal / (wayq+wqtl) narrative ratio [R]",
}
RESISTANT = ["frac_infc", "frac_fronted", "frac_null_subj", "frac_wqtl_wayq"]


class MVN:
    """Per-feature OLS + Tikhonov covariance, with prediction-variance dispersion."""

    def __init__(self, df_tr, feats, ridge, predictive=True):
        self.feats = list(feats)
        self.predictive = predictive
        d = df_tr["date_bce"].values.astype(float)
        self.n    = len(d)
        self.dbar = d.mean()
        self.Sxx  = float(((d - self.dbar) ** 2).sum())

        A, B, res = [], [], []
        for f in self.feats:
            y = pd.to_numeric(df_tr[f], errors="coerce").values.astype(float)
            m = np.isfinite(y)
            sl, ic, *_ = stats.linregress(d[m], y[m])
            A.append(ic); B.append(sl)
            r = np.full(self.n, np.nan); r[m] = y[m] - (ic + sl * d[m])
            res.append(r)
        self.A = np.array(A); self.B = np.array(B)
        R = np.nan_to_num(np.vstack(res).T)
        S = (R.T @ R) / max(self.n - 2, 1)          # n-2: OLS consumes two dof
        S += ridge * (np.trace(S) / len(self.feats)) * np.eye(len(self.feats))
        self.S = S

    def h(self, d):
        """Prediction-variance inflation factor."""
        if not self.predictive:
            return np.ones_like(np.atleast_1d(d), dtype=float)
        return 1.0 + 1.0 / self.n + (np.atleast_1d(d) - self.dbar) ** 2 / self.Sxx

    def posterior(self, x, wscale=1.0, prior_mu=464.0, prior_sd=300.0):
        m = np.isfinite(x)
        if m.sum() < 2:
            return None
        Ssub = self.S[np.ix_(m, m)]
        Sinv = np.linalg.inv(Ssub)
        sign, logdet = np.linalg.slogdet(Ssub)
        J = int(m.sum())
        hv = self.h(GRID)
        ll = np.empty(len(GRID))
        for i, g in enumerate(GRID):
            diff = (x - (self.A + self.B * g))[m]
            quad = diff @ Sinv @ diff
            # Sigma(d) = h(d) * S  ->  quad/h  and  logdet + J*log h
            ll[i] = -0.5 * (wscale * quad / hv[i] + J * np.log(hv[i]) + logdet)
        ll -= ll.max()
        lp = -0.5 * ((GRID - prior_mu) / prior_sd) ** 2
        p = np.exp(ll + lp)
        s = p.sum()
        return p / s if s > 0 else None


def hdi(post, lvl=0.68):
    o = np.argsort(post)[::-1]
    c = np.cumsum(post[o])
    sel = GRID[o[c <= lvl]]
    return (float(sel.min()), float(sel.max())) if len(sel) else (np.nan, np.nan)


def main():
    df = pd.read_csv(FM, index_col="id")
    tr = df[df["in_training"] == True].copy()               # noqa: E712
    n  = len(tr)

    feats = [f for f in PRESPEC if f in df.columns]
    res_f = [f for f in RESISTANT if f in df.columns]
    print(f"Corpus v2: {n} training units")
    print(f"Pre-specified dating features: {len(feats)}  (resistant subset: {len(res_f)})\n")
    for f in feats:
        print(f"   {f:16s} {PRESPEC[f]}")

    m_pred = MVN(tr, feats, RIDGE, predictive=True)
    m_old  = MVN(tr, feats, RIDGE, predictive=False)
    m_res  = MVN(tr, res_f, RIDGE_RES, predictive=True)

    # ── Calibration: LOO coverage, old vs predictive ──────────────────────────
    print("\n\nLOO calibration on training texts")
    cov = {}
    for tag, predictive in (("residual (old)", False), ("prediction (new)", True)):
        hits68 = hits95 = 0; errs = []; widths = []
        for uid in tr.index:
            sub = tr.drop(uid)
            mdl = MVN(sub, feats, RIDGE, predictive=predictive)
            x = pd.to_numeric(tr.loc[uid, feats], errors="coerce").values.astype(float)
            p = mdl.posterior(x)
            if p is None:
                continue
            mp = GRID[p.argmax()]
            lo, hi = hdi(p, 0.68); lo95, hi95 = hdi(p, 0.95)
            true = float(tr.loc[uid, "date_bce"])
            hits68 += (lo <= true <= hi); hits95 += (lo95 <= true <= hi95)
            errs.append(abs(mp - true)); widths.append(hi - lo)
        k = len(errs)
        cov[tag] = dict(n=k, cover68=hits68 / k, cover95=hits95 / k,
                        mae=float(np.mean(errs)), width68=float(np.mean(widths)))
        print(f"  {tag:18s} 68% CI covers {hits68/k:5.1%} (nominal 68%)   "
              f"95% covers {hits95/k:5.1%}   MAE {np.mean(errs):5.1f} yr   "
              f"mean 68% width {np.mean(widths):5.0f} yr")
    pd.DataFrame(cov).T.to_csv(os.path.join(OUTDIR, "calibration_v3.csv"))

    # ── Date everything ───────────────────────────────────────────────────────
    rows = []
    for uid, r in df.iterrows():
        x  = pd.to_numeric(r[feats], errors="coerce").values.astype(float)
        xr = pd.to_numeric(r[res_f], errors="coerce").values.astype(float)
        p  = m_pred.posterior(x); po = m_old.posterior(x); pr = m_res.posterior(xr)
        if p is None:
            continue
        mp = GRID[p.argmax()]; lo, hi = hdi(p)
        mo = GRID[po.argmax()] if po is not None else np.nan
        lo_o, hi_o = hdi(po) if po is not None else (np.nan, np.nan)
        mr = GRID[pr.argmax()] if pr is not None else np.nan
        rows.append(dict(
            unit=uid, n_words=int(r["n_words"]),
            role=("holdout" if r["holdout"] else
                  "hbvi_holdout" if r["hbvi_holdout"] else
                  "train" if r["in_training"] else "target"),
            scholarly=r["date_bce"],
            map_bce=round(mp), ci68_lo=round(lo), ci68_hi=round(hi),
            ci68_width=round(hi - lo),
            map_old=round(mo) if np.isfinite(mo) else np.nan,
            width_old=round(hi_o - lo_o) if np.isfinite(hi_o) else np.nan,
            map_resist=round(mr) if np.isfinite(mr) else np.nan,
            delta_arch=round(mp - mr) if np.isfinite(mr) else np.nan))
    out = pd.DataFrame(rows).set_index("unit")
    out.to_csv(os.path.join(OUTDIR, "dating_v3.csv"))

    print("\n\nHoldouts (never trained, either model)")
    for u in ["Jer_oracle", "Haggai"]:
        if u in out.index:
            r = out.loc[u]
            print(f"  {u:12s} scholarly {r['scholarly']:>4.0f}   "
                  f"MAP {r['map_bce']:>4.0f}  [{r['ci68_lo']:.0f}–{r['ci68_hi']:.0f}]  "
                  f"err {r['map_bce']-r['scholarly']:+5.0f} yr")

    print("\nTorah sources")
    for u in ["P_source", "JE_source", "D_source", "D_Code", "Lev_Priestly",
              "Song_Sea", "Jer_DTR"]:
        if u in out.index:
            r = out.loc[u]
            print(f"  {u:14s} MAP {r['map_bce']:>4.0f}  "
                  f"[{r['ci68_lo']:.0f}–{r['ci68_hi']:.0f}] "
                  f"width {r['ci68_width']:>3.0f}  "
                  f"(old width {r['width_old']:>3.0f})  "
                  f"resist {r['map_resist']:>4.0f}  D_arch {r['delta_arch']:+5.0f}")

    print(f"\nSaved → {OUTDIR}/dating_v3.csv, calibration_v3.csv")


if __name__ == "__main__":
    main()
