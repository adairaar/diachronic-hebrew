"""
03_hard_register_dating.py  —  Hebrew register-conditioned MVN dating
======================================================================
Hard-assignment register-conditioned Bayesian dating for Biblical Hebrew.
Mirrors the design of greek/10_hard_register_dating.py but adapted for the
Hebrew corpus.

Pipeline
--------
1.  Assign each text to exactly one register group via argmax of
    register probability from 02_register_classifier.py.

2.  For each group, screen features for within-group Spearman correlation
    with date (permissive thresholds: p < 0.25, LOO fraction ≥ 0.30).

3.  Train a ridge-regularised MVN model within each group using non-holdout
    texts (excluding LBH texts with large sigma that would dominate).

4.  Date every text: MAP = argmax[prior × likelihood] where
      prior  ~ N(date_bce, date_sigma²)
      likelihood = MVN model evaluated at feature vector

5.  Report per-holdout accuracy and date all test targets (D, P, JE).

Register groups
---------------
SBH          ~760–600 BCE  |  sub-model trained on SBH training texts
Transitional ~600–450 BCE  |  sub-model trained on Transitional training texts
LBH          ~400–167 BCE  |  sub-model trained on LBH training texts

Test targets are routed to the group with highest P(register) from the
classifier, and then dated using that group's sub-model.

Archaizing interpretation
--------------------------
If a test target (D, P, or JE) routes to the SBH/Transitional group but
has a positive archaizing_index AND its likelihood-only MAP is substantially
later than the prior mean, this is evidence of archaizing: the prior is
pulling the posterior to an early date, but the features alone prefer a
later date — exactly the pattern a 5th–4th century BCE author imitating
ancient Hebrew would produce.

Note on genre confound
-----------------------
The SBH training texts are all prophetic; the LBH training texts are
narrative/wisdom.  The Torah sources (D, P, JE) are narrative/legal.
Within-genre temporal signal is more reliable than cross-genre comparisons.
Features that are robustly genre-independent (frac_ani, frac_she, frac_ein,
rate_pen, rate_terem) are more diagnostic for D/P/JE than verb-form rates.
The screening step naturally down-weights genre-confounded features if they
show weak within-register temporal correlation.
"""

import json
import os
import sys
import warnings

import numpy as np

try:
    import pandas as pd
    from scipy import stats
    from scipy.optimize import minimize_scalar
except ImportError as e:
    sys.exit(f"Missing dependency: {e}")

HERE     = os.path.dirname(os.path.abspath(__file__))
FEAT_CSV = os.path.join(HERE, "data", "feature_matrix.csv")
REG_JSON = os.path.join(HERE, "results", "register_probs.json")
MANIF    = os.path.join(HERE, "corpus_manifest.json")
RESDIR   = os.path.join(HERE, "results")
os.makedirs(RESDIR, exist_ok=True)

P_THRESH  = 0.25
LOO_FRAC  = 0.30
MAX_FEATS = 15
MIN_TRAIN = 4
RIDGE     = 0.10

META_COLS = {"date_bce", "date_sigma", "register", "genre", "holdout",
             "group", "archaizing_flag"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bce(year: float) -> str:
    y = int(round(abs(year)))
    return f"{y} BCE"


def spearman_loo(dates, values, p_thresh, loo_frac):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = stats.spearmanr(dates, values, nan_policy="omit")
    rho, p = float(rho), float(p)
    if p >= p_thresh:
        return rho, p, 0.0
    n = len(dates)
    passing = sum(
        1 for i in range(n)
        if stats.spearmanr(
            np.delete(dates, i), np.delete(values, i),
            nan_policy="omit")[1] < p_thresh
    )
    return rho, p, passing / n


def screen_features(df_grp: pd.DataFrame) -> list:
    """Return features with significant within-group temporal correlation."""
    feat_cols = [c for c in df_grp.columns if c not in META_COLS]
    # date_bce is BCE positive, so higher = older; invert for temporal direction
    dates = df_grp["date_bce"].values.astype(float)
    records = []
    for col in feat_cols:
        vals = df_grp[col].values.astype(float)
        if np.nanstd(vals) < 1e-10 or np.isnan(vals).sum() > len(vals) * 0.3:
            continue
        rho, p, loo = spearman_loo(dates, vals, P_THRESH, LOO_FRAC)
        if p < P_THRESH and loo >= LOO_FRAC:
            records.append((col, abs(rho), rho, p, loo))
    records.sort(key=lambda x: -x[1])
    return [r[0] for r in records[:MAX_FEATS]]


class GroupMVN:
    """Ridge-regularised multivariate normal dating model for one register."""

    def __init__(self, name, ridge=RIDGE):
        self.name     = name
        self.ridge    = ridge
        self.alpha    = None
        self.beta     = None
        self.sigma_inv= None
        self.feats    = []
        self.d_min    = None
        self.d_max    = None

    def fit(self, df_train, feats):
        self.feats = feats
        dates = df_train["date_bce"].values.astype(float)
        self.d_min, self.d_max = dates.min(), dates.max()
        n = len(dates)
        X = df_train[feats].values.astype(float)

        self.alpha = np.zeros(len(feats))
        self.beta  = np.zeros(len(feats))
        resids = np.zeros_like(X)
        for j in range(len(feats)):
            x_j = X[:, j]
            mask = ~np.isnan(x_j)
            if mask.sum() < 3:
                continue
            A = np.column_stack([np.ones(mask.sum()), dates[mask]])
            coef, *_ = np.linalg.lstsq(A, x_j[mask], rcond=None)
            self.alpha[j], self.beta[j] = coef
            resids[mask, j] = x_j[mask] - (coef[0] + coef[1] * dates[mask])

        cov = (np.cov(resids.T) if len(feats) > 1
               else np.array([[np.var(resids)]]))
        cov += self.ridge * np.eye(len(feats))
        self.sigma_inv = np.linalg.inv(cov)

    def log_likelihood(self, x, d):
        mu = self.alpha + self.beta * d
        r  = x - mu
        return float(-0.5 * r @ self.sigma_inv @ r)

    def posterior(self, x, prior_mean, prior_sigma, n_grid=2000):
        d_lo = min(self.d_min, prior_mean - 4 * prior_sigma) - 50
        d_hi = max(self.d_max, prior_mean + 4 * prior_sigma) + 50
        d_lo = max(d_lo, 0)    # no negative BCE
        grid = np.linspace(d_lo, d_hi, n_grid)
        ll   = np.array([self.log_likelihood(x, d) for d in grid])
        lp   = -0.5 * ((grid - prior_mean) / prior_sigma) ** 2
        lpost= ll + lp
        lpost -= lpost.max()
        return grid, lpost

    def map_and_ci(self, x, prior_mean, prior_sigma):
        grid, lpost = self.posterior(x, prior_mean, prior_sigma)
        post = np.exp(lpost); post /= post.sum()
        cdf  = np.cumsum(post)
        map_d  = grid[np.argmax(post)]
        lo68   = grid[np.searchsorted(cdf, 0.16)]
        hi68   = grid[np.searchsorted(cdf, 0.84)]
        pmean  = np.dot(post, grid)
        pstd   = np.sqrt(np.dot(post, (grid - pmean)**2))
        z      = (map_d - prior_mean) / prior_sigma

        # Likelihood-only MAP
        ll = np.array([self.log_likelihood(x, d) for d in grid])
        lik = np.exp(ll - ll.max()); lik /= lik.sum()
        lik_map  = grid[np.argmax(lik)]
        lik_cdf  = np.cumsum(lik)
        lik_lo68 = grid[np.searchsorted(lik_cdf, 0.16)]
        lik_hi68 = grid[np.searchsorted(lik_cdf, 0.84)]
        lik_std  = np.sqrt(np.dot(lik, (grid - np.dot(lik, grid))**2))
        prior_prec = pstd**2 / prior_sigma**2 if prior_sigma > 0 else 1.0

        return dict(map_date=map_d, ci68_lo=lo68, ci68_hi=hi68, z_score=z,
                    post_mean=pmean, post_std=pstd,
                    lik_map=lik_map, lik_lo68=lik_lo68, lik_hi68=lik_hi68,
                    lik_std=lik_std, prior_prec=prior_prec)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv(FEAT_CSV, index_col="id")
    with open(REG_JSON, encoding="utf-8") as f:
        reg_probs = json.load(f)
    with open(MANIF, encoding="utf-8") as f:
        manifest = json.load(f)

    # ── Assign groups via argmax register probability ─────────────────────
    def assign_group(eid):
        r = reg_probs.get(eid, {})
        ps  = r.get("p_SBH", 0)
        pt  = r.get("p_Transitional", 0)
        pl  = r.get("p_LBH", 0)
        best = max(ps, pt, pl)
        if best == ps: return "SBH"
        if best == pt: return "Transitional"
        return "LBH"

    df["group"] = df.index.map(assign_group)
    df["archaizing_flag"] = df.index.map(
        lambda eid: reg_probs.get(eid, {}).get("archaizing_flag", False))

    feat_cols = [c for c in df.columns if c not in META_COLS | {"group", "archaizing_flag"}]

    # Separate training from test targets
    train_mask = df["register"] != "unknown"
    df_train   = df[train_mask].copy()
    df_test    = df[~train_mask].copy()

    # ── Group assignment report ───────────────────────────────────────────
    print("=" * 70)
    print("GROUP ASSIGNMENTS")
    print("=" * 70)
    for grp in ["SBH", "Transitional", "LBH"]:
        ids = df_train[df_train["group"] == grp].index.tolist()
        n_tr = sum(1 for i in ids if not df_train.loc[i, "holdout"])
        n_ho = sum(1 for i in ids if df_train.loc[i, "holdout"])
        print(f"\n  {grp:12s}  ({n_tr} training, {n_ho} holdout)")
        for eid in ids:
            flag = " [HOLDOUT]" if df_train.loc[eid, "holdout"] else ""
            true_reg = df_train.loc[eid, "register"]
            mismatch = " [MISMATCH]" if true_reg != grp else ""
            print(f"    {eid:22s}  {int(df_train.loc[eid,'date_bce'])} BCE{flag}{mismatch}")

    # ── Within-group feature screening ───────────────────────────────────
    print("\n" + "=" * 70)
    print("WITHIN-GROUP FEATURE SCREENING")
    print("=" * 70)

    group_feats: dict = {}
    for grp in ["SBH", "Transitional", "LBH"]:
        df_g = df_train[(df_train["group"] == grp) & (~df_train["holdout"])].copy()
        if len(df_g) < MIN_TRAIN:
            print(f"\n  {grp}: only {len(df_g)} training texts — skipping")
            group_feats[grp] = []
            continue
        feats = screen_features(df_g)
        group_feats[grp] = feats
        print(f"\n  {grp:12s}  (n={len(df_g)}, "
              f"range: {int(df_g['date_bce'].min())}–{int(df_g['date_bce'].max())} BCE)")
        if not feats:
            print("    No features passed screening — model will be prior-only.")
        for feat in feats:
            vals  = df_g[feat].values.astype(float)
            dates = df_g["date_bce"].values.astype(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, p = stats.spearmanr(dates, vals, nan_policy="omit")
            # Positive rho means feature correlates with higher date_bce = OLDER
            # Negative rho means feature increases as date_bce falls = MORE RECENT
            direction = "↑ (older)" if rho > 0 else "↓ (newer)"
            print(f"    {feat:28s}  ρ={rho:+.3f}  p={p:.3f}  {direction}")

    # ── Train sub-models ──────────────────────────────────────────────────
    models: dict = {}
    for grp in ["SBH", "Transitional", "LBH"]:
        feats = group_feats[grp]
        df_g  = df_train[(df_train["group"] == grp) & (~df_train["holdout"])].copy()
        if not feats or len(df_g) < MIN_TRAIN:
            models[grp] = None
            continue
        mdl = GroupMVN(name=grp)
        mdl.fit(df_g, feats)
        models[grp] = mdl

    # ── Date all training texts ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DATING RESULTS — TRAINING CORPUS")
    print("=" * 70)

    rows = []
    for grp in ["SBH", "Transitional", "LBH"]:
        mdl   = models[grp]
        feats = group_feats[grp]
        ids   = df_train[df_train["group"] == grp].index.tolist()
        print(f"\n── {grp} ─────────────────────────────────────────────────")
        if mdl is None:
            print("  (no model)")
            continue
        for eid in ids:
            row   = df_train.loc[eid]
            prior = float(row["date_bce"])
            sigma = float(row["date_sigma"])
            x = row[feats].values.astype(float)
            # Impute NaN with training-set feature mean
            x_clean = np.where(
                np.isnan(x),
                df_train[(df_train["group"] == grp) & (~df_train["holdout"])][feats]
                    .mean().values,
                x
            )
            res  = mdl.map_and_ci(x_clean, prior, sigma)
            hold = " [HOLDOUT]" if bool(row["holdout"]) else ""
            in68 = "✓" if res["ci68_lo"] <= prior <= res["ci68_hi"] else "✗"
            print(
                f"  {eid:22s}  "
                f"prior={bce(prior):8s}  "
                f"MAP={bce(res['map_date']):8s}  "
                f"68%=[{bce(res['ci68_lo'])},{bce(res['ci68_hi'])}]  "
                f"{in68}  "
                f"lik-only={bce(res['lik_map']):8s}±{res['lik_std']:.0f}yr  "
                f"prior={res['prior_prec']*100:.0f}%{hold}"
            )
            rows.append(dict(
                id=eid, group=grp, holdout=bool(row["holdout"]),
                true_register=row["register"],
                prior=prior, map_date=res["map_date"],
                ci68_lo=res["ci68_lo"], ci68_hi=res["ci68_hi"],
                z=res["z_score"], in_68=(res["ci68_lo"] <= prior <= res["ci68_hi"]),
                lik_map=res["lik_map"], lik_std=res["lik_std"],
                prior_prec=res["prior_prec"],
            ))

    # ── Date test targets (D, P, JE) ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("DATING RESULTS — TEST TARGETS (D / P / JE)")
    print("=" * 70)
    print()
    print("  These texts have NO confirmed dates — the dating is the result.")
    print("  'Prior' = scholarly consensus prior; likelihood is what the")
    print("  morphosyntactic features alone imply.")
    print()

    test_rows = []
    for eid in df_test.index:
        grp   = df_test.loc[eid, "group"]
        mdl   = models.get(grp)
        feats = group_feats.get(grp, [])
        prior = float(df_test.loc[eid, "date_bce"])
        sigma = float(df_test.loc[eid, "date_sigma"])
        arch_flag = bool(df_test.loc[eid, "archaizing_flag"])
        arch_idx  = float(reg_probs.get(eid, {}).get("archaizing_index", 0))
        p_sbh = float(reg_probs.get(eid, {}).get("p_SBH", 0))
        p_tr  = float(reg_probs.get(eid, {}).get("p_Transitional", 0))
        p_lbh = float(reg_probs.get(eid, {}).get("p_LBH", 0))

        if mdl is None or not feats:
            print(f"  {eid}: no model available for group {grp}")
            continue

        # Use only features available for this test target
        avail = [f for f in feats if not pd.isna(df_test.loc[eid, f])
                 if f in df_test.columns]
        if len(avail) < 2:
            print(f"  {eid}: insufficient features ({len(avail)} available) — prior-only")
            test_rows.append(dict(id=eid, group=grp, prior=prior,
                                  map_date=prior, ci68_lo=prior-sigma,
                                  ci68_hi=prior+sigma, lik_map=np.nan,
                                  lik_std=np.nan, prior_prec=1.0,
                                  archaizing_flag=arch_flag,
                                  archaizing_index=arch_idx))
            continue

        # Rebuild a smaller sub-model on available features only
        df_g_train = df_train[
            (df_train["group"] == grp) & (~df_train["holdout"])
        ].copy()
        # Re-screen on available subset
        avail_screened = [f for f in avail if f in group_feats[grp]]
        if len(avail_screened) < 2:
            avail_screened = avail[:min(3, len(avail))]

        sub_mdl = GroupMVN(name=f"{grp}_sub")
        sub_mdl.fit(df_g_train, avail_screened)
        x = df_test.loc[eid, avail_screened].values.astype(float)
        x_clean = np.nan_to_num(x, nan=df_g_train[avail_screened].mean().values.mean())
        res = sub_mdl.map_and_ci(x_clean, prior, sigma)

        arch_note = " ← ARCHAIZING CANDIDATE" if arch_flag else ""
        print(f"  {eid:22s}  "
              f"group={grp}  "
              f"P(SBH)={p_sbh:.2f} P(T)={p_tr:.2f} P(LBH)={p_lbh:.2f}")
        print(f"  {'':22s}  prior={bce(prior)}±{int(sigma)}yr")
        print(f"  {'':22s}  MAP={bce(res['map_date'])}  "
              f"68%=[{bce(res['ci68_lo'])},{bce(res['ci68_hi'])}]")
        print(f"  {'':22s}  lik-only={bce(res['lik_map'])}±{res['lik_std']:.0f}yr  "
              f"prior={res['prior_prec']*100:.0f}%  "
              f"arch_idx={arch_idx:+.2f}{arch_note}")
        print(f"  {'':22s}  features used: {avail_screened}")
        print()

        test_rows.append(dict(
            id=eid, group=grp, prior=prior,
            map_date=res["map_date"],
            ci68_lo=res["ci68_lo"], ci68_hi=res["ci68_hi"],
            lik_map=res["lik_map"], lik_std=res["lik_std"],
            prior_prec=res["prior_prec"],
            archaizing_flag=arch_flag, archaizing_index=arch_idx,
        ))

    # ── Holdout summary ───────────────────────────────────────────────────
    print("=" * 70)
    print("HOLDOUT SUMMARY")
    print("=" * 70)
    holdout_rows = [r for r in rows if r["holdout"]]
    fmt = "{:<22s}  {:>8s}  {:>8s}  {:>8s}  {:>10s}  {:>6s}  {:>5s}"
    print(fmt.format("Text", "Prior", "MAP", "In 68%?", "Lik-MAP", "Lik-σ", "Prior%"))
    print("─" * 80)
    for r in holdout_rows:
        print(fmt.format(
            r["id"], bce(r["prior"]), bce(r["map_date"]),
            "✓" if r["in_68"] else "✗",
            bce(r["lik_map"]), f"±{r['lik_std']:.0f}yr",
            f"{r['prior_prec']*100:.0f}%",
        ))

    # ── Interpretation ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()
    print("  KEY FINDING:")
    print("  The likelihood-only MAP date reflects what the morphosyntactic")
    print("  features alone imply, independent of any scholarly prior.")
    print()
    print("  For training holdouts: lik-only MAP should be within ~1 σ of")
    print("  the true date if the model has genuine temporal signal.")
    print()
    print("  For D/P/JE sources:")
    print("    lik_map >> prior → features look LATER than the scholarly prior")
    print("                       (consistent with late composition/editing)")
    print("    lik_map ≈ prior  → features are compatible with the prior")
    print("    lik_map << prior → features look OLDER than the scholarly prior")
    print()
    print("  Archaizing interpretation:")
    print("    If a source routes to a late register group (LBH/Transitional)")
    print("    AND lik_map > prior_bce (features prefer a later date)")
    print("    AND arch_idx > 0 (some archaic stylistic loading present),")
    print("    this is evidence that the author was IMITATING archaic Hebrew")
    print("    while writing in a later period.")
    print()

    for r in test_rows:
        eid = r["id"]
        if np.isnan(r.get("lik_map", np.nan)):
            continue
        lik_shift = r["lik_map"] - r["prior"]
        direction = "later" if lik_shift < 0 else "earlier"
        print(f"  {eid:22s}  lik-only {abs(lik_shift):.0f} yr {direction} than prior  "
              f"(prior={bce(r['prior'])} → lik-MAP={bce(r['lik_map'])})")

    # ── Save ──────────────────────────────────────────────────────────────
    all_rows = rows + test_rows
    out_csv  = os.path.join(RESDIR, "hard_register_dating_hebrew.csv")
    pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"\nResults → {out_csv}")


if __name__ == "__main__":
    main()
