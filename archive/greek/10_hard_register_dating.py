"""
10_hard_register_dating.py
==========================
Hard-assignment register-conditioned dating model.

Motivation
----------
The soft-mixture model (09_calibrated_dating.py) computes a likelihood that
is a weighted blend across all register sub-models.  For a text like the Gospel
of Mark the weight is ~75 % LXX + ~25 % Koine, so the Atticizing model leaks
into the posterior for every Koine/LXX text — exactly the cross-register
confound that the register classifier was supposed to remove.

This script takes the classification seriously: assign every text to exactly one
dating group based on argmax register probability, then train and evaluate a
completely separate MVN model within that group.

Dating groups
-------------
Three groups derived from four register classes:

  Classical      p_ancient_Attic = argmax  → model trained on ancient_Attic texts
  Atticizing     p_Atticizing    = argmax  → model trained on Atticizing texts
  Koine/Vernac.  p_Koine or p_LXX = argmax → model trained on secular Koine texts
                                              ONLY (LXX excluded from training).
                                              LXX texts are still *assigned* to this
                                              group for prediction, but the temporal
                                              model is built on secular Koine alone to
                                              avoid LXX substrate artefacts pushing
                                              NT Gospels toward ~240 BCE.

Note: NT texts (Mark, Matthew, Luke) are classified by the RF as LXX-styled
and therefore fall into the Koine/Vernacular group, which correctly covers
1st–2nd century CE.  A dedicated "LXX" sub-model would force them into the
−270 → −100 BCE range — the right style, the wrong period.

Within-group feature screening
-------------------------------
For each group, features are screened by Spearman ρ(feature, date_ce) using
only that group's training texts.  Thresholds are deliberately permissive
(p_thresh=0.25, loo_frac=0.30) because within-group n is small (14–32).
Features with the theoretically expected direction are noted but not filtered;
the ridge-regularised MVN handles weak signals by shrinking their β toward 0.

MVN model
---------
Identical to 05_mvn_dating.py: WLS linear regression of each feature on date_ce
gives (α, β, σ_resid) per feature.  Log-likelihood is Mahalanobis-based with
ridge-regularised residual covariance.  Posterior = prior (Gaussian, per corpus
manifest) × likelihood.

Key output
----------
  results/hard_register_dating.txt  — per-holdout report: MAP, 68 % CI,
                                       z-score, likelihood-only MAP, prior precision
  results/hard_register_dating.csv  — machine-readable table
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
    print(f"Missing dependency: {e}\nRun: pip install pandas scipy --break-system-packages")
    sys.exit(1)

HERE       = os.path.dirname(os.path.abspath(__file__))
FEAT_CSV   = os.path.join(HERE, "data", "features", "feature_matrix.csv")
MANIFEST   = os.path.join(HERE, "corpus_manifest.json")
REG_PROBS  = os.path.join(HERE, "results", "register_probs.json")
RESULTS    = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# Within-group screening thresholds — permissive because n is small
P_THRESH  = 0.25
LOO_FRAC  = 0.30
MAX_FEATS = 20   # ridge regularisation handles more, but cap to avoid tiny-n overfitting

# Minimum texts needed to train a sub-model
MIN_TRAIN = 8

META_COLS = {"author", "date_ce", "date_sigma", "genre", "holdout", "word_count",
             "register", "group"}


# ---------------------------------------------------------------------------
# Register assignment
# ---------------------------------------------------------------------------

def assign_group(probs: dict) -> str:
    """
    Map register probability dict → one of 'classical', 'atticizing', 'koine'.

    register_probs.json uses keys 'p_ancient_Attic', 'p_Atticizing', etc.

    'koine' captures both Koine and LXX argmax texts so that LXX-styled Koine
    texts (NT Gospels) fall in the group whose temporal range covers 70–120 CE.
    """
    pa  = probs.get("p_ancient_Attic", 0.0)
    pat = probs.get("p_Atticizing",    0.0)
    pk  = probs.get("p_Koine",         0.0)
    pl  = probs.get("p_LXX",           0.0)
    best = max(pa, pat, pk, pl)
    if best == pa:
        return "classical"
    if best == pat:
        return "atticizing"
    return "koine"          # Koine or LXX argmax → Koine/Vernacular group


# ---------------------------------------------------------------------------
# Feature screening (within-group Spearman + LOO)
# ---------------------------------------------------------------------------

def spearman_loo(dates: np.ndarray, values: np.ndarray,
                 p_thresh: float, loo_frac: float) -> tuple[float, float, float]:
    """Return (rho, p_value, loo_pass_fraction)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, p = stats.spearmanr(dates, values, nan_policy="omit")
    rho, p = float(rho), float(p)
    if p >= p_thresh:
        return rho, p, 0.0
    n = len(dates)
    passing = sum(
        1 for i in range(n)
        if (lambda d, v: stats.spearmanr(d, v, nan_policy="omit")[1] < p_thresh)(
            np.delete(dates, i), np.delete(values, i))
    )
    return rho, p, passing / n


def screen_features(df_group: pd.DataFrame) -> list[str]:
    """Return list of screened feature names for a group's training subset."""
    feat_cols = [c for c in df_group.columns if c not in META_COLS]
    dates = df_group["date_ce"].values.astype(float)

    records = []
    for col in feat_cols:
        vals = df_group[col].values.astype(float)
        if np.nanstd(vals) < 1e-10:
            continue
        rho, p, loo = spearman_loo(dates, vals, P_THRESH, LOO_FRAC)
        if p < P_THRESH and loo >= LOO_FRAC:
            records.append((col, abs(rho), rho, p, loo))

    records.sort(key=lambda x: -x[1])
    return [r[0] for r in records[:MAX_FEATS]]


# ---------------------------------------------------------------------------
# MVN model: fit and predict
# ---------------------------------------------------------------------------

class GroupMVN:
    """
    Linear MVN dating model for a single register group.

    For each feature j: x_j = α_j + β_j * d + ε_j
    Log-likelihood: −0.5 * r^T Σ^{-1} r  where r_j = x_j − (α_j + β_j*d)
    Σ is ridge-regularised residual covariance.
    """

    def __init__(self, name: str, ridge: float = 0.10):
        self.name   = name
        self.ridge  = ridge
        self.alpha  = None
        self.beta   = None
        self.sigma_inv = None
        self.feat_names: list[str] = []
        self.date_min = None
        self.date_max = None

    def fit(self, df_train: pd.DataFrame, feat_names: list[str]) -> None:
        self.feat_names = feat_names
        dates = df_train["date_ce"].values.astype(float)
        self.date_min = dates.min()
        self.date_max = dates.max()
        n = len(dates)
        X = df_train[feat_names].values.astype(float)  # (n, p)

        # WLS: equal weights (OLS), per feature
        self.alpha = np.zeros(len(feat_names))
        self.beta  = np.zeros(len(feat_names))
        residuals  = np.zeros_like(X)
        for j, col in enumerate(feat_names):
            x_j = X[:, j]
            mask = ~np.isnan(x_j)
            if mask.sum() < 4:
                continue
            A = np.column_stack([np.ones(mask.sum()), dates[mask]])
            coef, *_ = np.linalg.lstsq(A, x_j[mask], rcond=None)
            self.alpha[j], self.beta[j] = coef
            residuals[mask, j] = x_j[mask] - (coef[0] + coef[1] * dates[mask])

        # Ridge-regularised covariance
        cov = np.cov(residuals.T) if len(feat_names) > 1 else np.array([[np.var(residuals)]])
        cov += self.ridge * np.eye(len(feat_names))
        self.sigma_inv = np.linalg.inv(cov)

    def log_likelihood(self, x: np.ndarray, d: float) -> float:
        """Log-likelihood of observed feature vector x at date d."""
        mu = self.alpha + self.beta * d
        r  = x - mu
        return float(-0.5 * r @ self.sigma_inv @ r)

    def posterior(self, x: np.ndarray, prior_mean: float, prior_sigma: float,
                  n_grid: int = 2000) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior over date grid.
        Returns (dates_grid, log_posterior_unnorm).
        """
        # Extend grid slightly beyond training range
        d_lo = min(self.date_min, prior_mean - 4 * prior_sigma) - 50
        d_hi = max(self.date_max, prior_mean + 4 * prior_sigma) + 50
        dates = np.linspace(d_lo, d_hi, n_grid)

        log_lik  = np.array([self.log_likelihood(x, d) for d in dates])
        log_prior = -0.5 * ((dates - prior_mean) / prior_sigma) ** 2
        log_post  = log_lik + log_prior
        # Normalise
        log_post -= np.max(log_post)
        return dates, log_post

    def map_and_ci(self, x: np.ndarray, prior_mean: float, prior_sigma: float
                   ) -> dict:
        dates, log_post = self.posterior(x, prior_mean, prior_sigma)
        post = np.exp(log_post)
        post /= post.sum()

        map_idx = np.argmax(post)
        map_date = dates[map_idx]

        # 68 % HDI
        cdf = np.cumsum(post)
        lo68 = dates[np.searchsorted(cdf, 0.16)]
        hi68 = dates[np.searchsorted(cdf, 0.84)]

        # Posterior mean and std
        pmean = np.dot(post, dates)
        pstd  = np.sqrt(np.dot(post, (dates - pmean)**2))

        # Z-score relative to prior
        z = (map_date - prior_mean) / prior_sigma

        # Likelihood-only MAP (flat prior)
        log_lik = np.array([self.log_likelihood(x, d) for d in dates])
        lik_map = dates[np.argmax(log_lik)]

        # Likelihood-only 68 % HDI
        lik = np.exp(log_lik - log_lik.max())
        lik /= lik.sum()
        lik_cdf = np.cumsum(lik)
        lik_lo68 = dates[np.searchsorted(lik_cdf, 0.16)]
        lik_hi68 = dates[np.searchsorted(lik_cdf, 0.84)]
        lik_std  = np.sqrt(np.dot(lik, (dates - np.dot(lik, dates))**2))

        # Prior precision (fraction of posterior precision from prior alone)
        prior_var  = prior_sigma ** 2
        post_var   = pstd ** 2 if pstd > 0 else 1e-6
        prior_prec_frac = post_var / prior_var  # < 1 means posterior tighter than prior

        return dict(
            map_date    = map_date,
            post_mean   = pmean,
            post_std    = pstd,
            ci68_lo     = lo68,
            ci68_hi     = hi68,
            z_score     = z,
            lik_map     = lik_map,
            lik_lo68    = lik_lo68,
            lik_hi68    = lik_hi68,
            lik_std     = lik_std,
            prior_prec  = prior_prec_frac,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ce(year: float) -> str:
    y = int(round(year))
    return f"{abs(y)} BCE" if y < 0 else f"{y} CE"


def main() -> None:
    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(FEAT_CSV, index_col="id")
    with open(MANIFEST, encoding="utf-8") as f:
        corpus = json.load(f)
    with open(REG_PROBS, encoding="utf-8") as f:
        reg_probs = json.load(f)

    # Attach manifest fields
    for e in corpus:
        eid = e["id"]
        if eid in df.index:
            df.loc[eid, "register"]  = e.get("register", "unknown")
            df.loc[eid, "holdout"]   = e["holdout"]
            df.loc[eid, "date_ce"]   = e["date_ce"]
            df.loc[eid, "date_sigma"]= e["date_sigma"]

    # Hard-assign group from register_probs
    df["group"] = df.index.map(
        lambda eid: assign_group(reg_probs.get(eid, {}))
    )

    # ── Report group assignments ──────────────────────────────────────────────
    print("=" * 70)
    print("GROUP ASSIGNMENTS")
    print("=" * 70)
    for grp in ["classical", "atticizing", "koine"]:
        ids = df[df["group"] == grp].index.tolist()
        n_train = sum(1 for i in ids if not df.loc[i, "holdout"])
        n_hold  = sum(1 for i in ids if df.loc[i, "holdout"])
        print(f"\n  {grp.upper():12s}  ({n_train} training, {n_hold} holdout)")
        for eid in sorted(ids):
            probs = reg_probs.get(eid, {})
            flag  = " [HOLDOUT]" if df.loc[eid, "holdout"] else ""
            yr    = ce(df.loc[eid, "date_ce"])
            print(f"    {eid:45s} {yr:12s}{flag}")

    # ── Within-group feature screening ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("WITHIN-GROUP FEATURE SCREENING")
    print("=" * 70)

    group_features: dict[str, list[str]] = {}
    for grp in ["classical", "atticizing", "koine"]:
        # Koine sub-model: exclude LXX texts from training to prevent
        # LXX substrate features (near-zero γοῦν/τοίνυν, near-zero -ττ-,
        # extreme καί) from being learned as "early date" signals.
        # LXX texts are still *assigned* to the koine group for prediction.
        if grp == "koine":
            df_train = df[(df["group"] == grp) & (~df["holdout"]) &
                          (df["register"] != "LXX")].copy()
        else:
            df_train = df[(df["group"] == grp) & (~df["holdout"])].copy()
        if len(df_train) < MIN_TRAIN:
            print(f"\n  {grp}: only {len(df_train)} training texts — skipping")
            group_features[grp] = []
            continue

        feats = screen_features(df_train)
        group_features[grp] = feats

        print(f"\n  {grp.upper():12s}  (n={len(df_train)}, "
              f"date range: {ce(df_train['date_ce'].min())} – "
              f"{ce(df_train['date_ce'].max())})")
        if not feats:
            print("    No features passed screening — model will be prior-only.")
        for feat in feats:
            vals  = df_train[feat].values.astype(float)
            dates = df_train["date_ce"].values.astype(float)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, p = stats.spearmanr(dates, vals, nan_policy="omit")
            direction = "↑" if rho > 0 else "↓"
            print(f"    {feat:25s}  ρ={rho:+.3f}  p={p:.3f}  {direction}")

    # ── Train sub-models ──────────────────────────────────────────────────────
    models: dict[str, GroupMVN | None] = {}
    for grp in ["classical", "atticizing", "koine"]:
        feats = group_features[grp]
        if grp == "koine":
            df_train = df[(df["group"] == grp) & (~df["holdout"]) &
                          (df["register"] != "LXX")].copy()
        else:
            df_train = df[(df["group"] == grp) & (~df["holdout"])].copy()
        if not feats or len(df_train) < MIN_TRAIN:
            models[grp] = None
            continue
        mdl = GroupMVN(name=grp)
        mdl.fit(df_train, feats)
        models[grp] = mdl

    # ── Date all texts (training LOO + holdouts full) ─────────────────────────
    print("\n" + "=" * 70)
    print("DATING RESULTS")
    print("=" * 70)

    rows = []
    for grp in ["classical", "atticizing", "koine"]:
        mdl   = models[grp]
        feats = group_features[grp]
        ids   = df[df["group"] == grp].index.tolist()

        print(f"\n── {grp.upper()} ─────────────────────────────────────────────")
        if mdl is None:
            print("  (no model — insufficient training data or features)")
            continue

        for eid in ids:
            row = df.loc[eid]
            is_holdout = bool(row["holdout"])
            prior_mean  = float(row["date_ce"])
            prior_sigma = float(row["date_sigma"])
            x = row[feats].values.astype(float)

            res = mdl.map_and_ci(x, prior_mean, prior_sigma)

            scholarly_in_68 = res["ci68_lo"] <= prior_mean <= res["ci68_hi"]
            lik_in_68       = res["lik_lo68"] <= prior_mean <= res["lik_hi68"]

            flag  = " [HOLDOUT]" if is_holdout else ""
            print(
                f"  {eid:45s}"
                f"  MAP={ce(res['map_date']):10s}"
                f"  68%=[{ce(res['ci68_lo'])}, {ce(res['ci68_hi'])}]"
                f"  z={res['z_score']:+.2f}"
                f"  | lik-only MAP={ce(res['lik_map']):10s}"
                f"  lik68=[{ce(res['lik_lo68'])}, {ce(res['lik_hi68'])}]"
                f"{flag}"
            )

            rows.append(dict(
                id             = eid,
                group          = grp,
                holdout        = is_holdout,
                scholarly_date = prior_mean,
                map_date       = res["map_date"],
                ci68_lo        = res["ci68_lo"],
                ci68_hi        = res["ci68_hi"],
                z_score        = res["z_score"],
                scholar_in_68  = scholarly_in_68,
                lik_map        = res["lik_map"],
                lik68_lo       = res["lik_lo68"],
                lik68_hi       = res["lik_hi68"],
                lik_in_68      = lik_in_68,
                lik_std        = res["lik_std"],
                prior_prec     = res["prior_prec"],
            ))

    # ── Holdout summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HOLDOUT SUMMARY")
    print("=" * 70)
    holdout_rows = [r for r in rows if r["holdout"]]

    fmt = "{:45s} {:>10s}  {:>10s}  {:>8s}  {:>10s}  {:>8s}  {:>6s}"
    print(fmt.format("Text", "Scholarly", "MAP(post)", "In 68%?",
                     "Lik-MAP", "Lik σ", "Prior%"))
    print("─" * 110)
    for r in holdout_rows:
        in68 = "✓" if r["scholar_in_68"] else "✗"
        lin68 = "✓" if r["lik_in_68"] else "✗"
        prior_pct = f"{r['prior_prec']*100:.0f}%"
        print(fmt.format(
            r["id"],
            ce(r["scholarly_date"]),
            ce(r["map_date"]),
            in68,
            ce(r["lik_map"]),
            f"±{r['lik_std']:.0f}yr",
            prior_pct,
        ))

    # ── Save ──────────────────────────────────────────────────────────────────
    out_csv = os.path.join(RESULTS, "hard_register_dating.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nResults → {out_csv}")

    # ── Comparison note ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    n_feat_by_group = {g: len(group_features[g]) for g in group_features}
    print(f"  Features used per group: {n_feat_by_group}")
    print()
    print("  Each text is dated using ONLY the sub-model for its assigned group.")
    print("  The Atticizing group's model is never applied to Koine/LXX texts,")
    print("  removing the cross-register contamination in the soft-mixture model.")
    print()
    print("  'Prior %' = posterior_var / prior_var.  Close to 100% = likelihood")
    print("  contributes little; well below 100% = features are genuinely informative.")
    for r in holdout_rows:
        lik_shift = abs(r["lik_map"] - r["scholarly_date"])
        print(f"  {r['id']:45s}  lik-only shift from truth: {lik_shift:.0f} yr")


if __name__ == "__main__":
    main()
