"""
09_calibrated_dating.py
=======================
Stage 2 of the two-stage register-calibrated dating pipeline.

Motivation
----------
A single linear WLS trend (scripts 05–06) conflates two distinct stylistic
traditions: Atticizing authors (who deliberately archaise) and Koine authors
(whose language tracks the natural evolution of Greek). Features that increase
over time in Koine texts may show no trend — or the opposite trend — in
Atticizing texts, causing the single-model likelihood vertex to fall far from
the true date.

Three-model approach
--------------------
1. Fit a separate MVN model for each register group:
     - "Classical" group  : ancient_Attic + Atticizing texts
     - "Koine" group      : Koine texts only
     - "LXX" group        : LXX (Septuagint / Semitic-substrate) texts only

2. For any unknown text with feature vector x, compute:
     L_classical(x | d)   — log-likelihood under the Classical model
     L_koine(x | d)       — log-likelihood under the Koine model
     L_lxx(x | d)         — log-likelihood under the LXX model

3. Combine using P(register) from script 08:
     p_c = P(ancient_Attic) + P(Atticizing)
     p_k = P(Koine)
     p_l = P(LXX)

   Mixture log-likelihood (3-component):
     log L_mix(x | d) ≈ log[ p_c·exp(L_c(d)) + p_k·exp(L_k(d)) + p_l·exp(L_l(d)) ]

   Computed via log-sum-exp for numerical stability.

4. Multiply by Gaussian prior and report the posterior as before.

Comparison
----------
For each holdout text we report:
  - Baseline z-score (single-model, script 06)
  - Calibrated z-score (mixture model, this script)
  - Width of 68% HDI for both

Output
------
  results/calibrated_posteriors/<id>.json   — per-text posterior dicts
  results/calibrated_validation.csv         — holdout comparison table
  results/plots/calibrated_validation.png   — side-by-side holdout plots
  results/plots/calibrated_dating_summary.png — full corpus summary

Usage
-----
    python 09_calibrated_dating.py
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

HERE         = os.path.dirname(os.path.abspath(__file__))
RESULTS      = os.path.join(HERE, "results")
PLOTS_DIR    = os.path.join(HERE, "results", "plots")
FEAT_DIR     = os.path.join(HERE, "data", "features")
MANIFEST     = os.path.join(HERE, "corpus_manifest.json")
REG_PROBS    = os.path.join(RESULTS, "register_probs.json")
BASELINE_DIR = os.path.join(RESULTS, "posteriors")   # from 05_mvn_dating.py
CAL_POST_DIR = os.path.join(RESULTS, "calibrated_posteriors")

os.makedirs(CAL_POST_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DATE_GRID_MIN = -490
DATE_GRID_MAX =  410
DATE_GRID_STEP=   5
RIDGE_LAMBDA  = 1e-3


# ---------------------------------------------------------------------------
# MVN model (same as script 05 but self-contained here for clarity)
# ---------------------------------------------------------------------------

class MVNModel:
    """
    Weighted least-squares diachronic trend + multivariate Normal likelihood.

    For each feature j: x_j = α_j + β_j · d + ε_j
    Residual covariance Σ estimated from training residuals + ridge.

    Log-likelihood at date d:
        ℓ(d) = -½ (x - μ(d))ᵀ Σ⁻¹ (x - μ(d))  + const
    where μ(d) = α + β·d  (vector form)

    This is a downward quadratic parabola in d, so the likelihood-only
    MAP may lie off the evaluation grid.
    """

    def __init__(self, ridge: float = RIDGE_LAMBDA):
        self.ridge = ridge
        self.alpha = None  # (p,) intercepts
        self.beta  = None  # (p,) slopes
        self.Sigma_inv = None  # (p,p) precision matrix
        self.feature_names = None
        self.n_train = 0

    def fit(self, dates: np.ndarray, X: np.ndarray,
            weights: np.ndarray | None = None,
            feature_names: list | None = None):
        """
        Fit WLS trend per feature, then estimate residual covariance.

        dates   : (n,)   training dates in CE
        X       : (n, p) feature matrix
        weights : (n,)   optional WLS weights (default 1/sigma²)
        """
        n, p = X.shape
        self.n_train = n
        self.feature_names = feature_names or [f"f{i}" for i in range(p)]

        if weights is None:
            weights = np.ones(n)
        W = np.diag(weights)

        # WLS: for each feature j, fit x_j = alpha_j + beta_j * d
        alpha = np.zeros(p)
        beta  = np.zeros(p)
        A = np.column_stack([np.ones(n), dates])  # (n, 2)
        AW = A.T @ W  # (2, n)
        AWA = AW @ A  # (2, 2)
        for j in range(p):
            try:
                coef = np.linalg.solve(AWA, AW @ X[:, j])
                alpha[j], beta[j] = coef
            except np.linalg.LinAlgError:
                alpha[j], beta[j] = X[:, j].mean(), 0.0

        self.alpha = alpha
        self.beta  = beta

        # Residuals
        mu_hat = alpha[None, :] + beta[None, :] * dates[:, None]  # (n, p)
        R = X - mu_hat  # (n, p)

        # Weighted sample covariance
        w_sum = weights.sum()
        R_w   = R * np.sqrt(weights[:, None])
        Sigma = (R_w.T @ R_w) / w_sum

        # Ridge regularisation
        Sigma += self.ridge * np.eye(p)
        self.Sigma = Sigma   # store for diagnostics

        try:
            self.Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            self.Sigma_inv = np.linalg.pinv(Sigma)

    def log_likelihood(self, x: np.ndarray, date_grid: np.ndarray) -> np.ndarray:
        """
        Compute log L(x | d) for each d in date_grid.

        Returns array of shape (len(date_grid),) — unnormalised, up to constant.
        """
        p = len(x)
        # μ(d) for each date in grid: shape (G, p)
        G  = len(date_grid)
        mu = self.alpha[None, :] + self.beta[None, :] * date_grid[:, None]  # (G, p)
        diff = x[None, :] - mu  # (G, p)

        # Mahalanobis: -½ diff @ Sigma_inv @ diff.T  (diagonal of result)
        mah = np.einsum("gi,ij,gj->g", diff, self.Sigma_inv, diff)
        return -0.5 * mah   # shape (G,)

    def posterior(self, x: np.ndarray, date_grid: np.ndarray,
                  date_prior_mu: float | None = None,
                  date_prior_sigma: float | None = None) -> np.ndarray:
        """
        Compute normalised posterior P(d | x) on date_grid.

        If date_prior_mu is None, returns likelihood-only (uniform prior).
        """
        log_lik = self.log_likelihood(x, date_grid)

        if date_prior_mu is not None and date_prior_sigma is not None:
            log_prior = -0.5 * ((date_grid - date_prior_mu) / date_prior_sigma) ** 2
        else:
            log_prior = np.zeros_like(log_lik)

        log_post = log_lik + log_prior
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= post.sum()
        return post


# ---------------------------------------------------------------------------
# HDI helper (same as script 06)
# ---------------------------------------------------------------------------

def hdi(posterior: np.ndarray, date_grid: np.ndarray, frac: float = 0.68):
    """Shortest contiguous interval containing frac of posterior mass."""
    n   = len(posterior)
    cum = np.cumsum(posterior)
    best_span = n
    best_lo, best_hi = 0, n - 1
    for i in range(n):
        lo_mass = cum[i - 1] if i > 0 else 0.0
        target  = lo_mass + frac
        hi      = int(np.searchsorted(cum, target, side="left"))
        if hi >= n:
            break
        mass = cum[hi] - lo_mass
        if mass >= frac - 1e-9:
            span = hi - i
            if span < best_span:
                best_span = span
                best_lo   = i
                best_hi   = hi
    return float(date_grid[best_lo]), float(date_grid[best_hi])


# ---------------------------------------------------------------------------
# Mixture log-likelihood
# ---------------------------------------------------------------------------

def mixture_log_likelihood(
    x: np.ndarray,
    date_grid: np.ndarray,
    model_classical: MVNModel,
    model_koine: MVNModel,
    p_classical: float,
    p_koine: float,
    model_lxx: MVNModel | None = None,
    p_lxx: float = 0.0,
) -> np.ndarray:
    """
    Log-likelihood of x at each date under a 2- or 3-component MVN mixture.

    log L_mix(d) = log[ p_c·exp(ℓ_c(d)) + p_k·exp(ℓ_k(d)) [+ p_l·exp(ℓ_l(d))] ]

    Computed via log-sum-exp for numerical stability.
    Returns shape (G,).
    """
    log_p_c = np.log(p_classical + 1e-12)
    log_p_k = np.log(p_koine + 1e-12)

    a = model_classical.log_likelihood(x, date_grid) + log_p_c  # (G,)
    b = model_koine.log_likelihood(x, date_grid)     + log_p_k  # (G,)

    if model_lxx is not None and p_lxx > 1e-9:
        log_p_l = np.log(p_lxx + 1e-12)
        c = model_lxx.log_likelihood(x, date_grid) + log_p_l    # (G,)
        maxabc = np.maximum(np.maximum(a, b), c)
        log_mix = maxabc + np.log(
            np.exp(a - maxabc) + np.exp(b - maxabc) + np.exp(c - maxabc)
        )
    else:
        maxab = np.maximum(a, b)
        log_mix = maxab + np.log(np.exp(a - maxab) + np.exp(b - maxab))

    return log_mix


def mixture_posterior(
    x: np.ndarray,
    date_grid: np.ndarray,
    model_classical: MVNModel,
    model_koine: MVNModel,
    p_classical: float,
    p_koine: float,
    model_lxx: MVNModel | None = None,
    p_lxx: float = 0.0,
    date_prior_mu: float | None = None,
    date_prior_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (posterior_combined, posterior_lik_only) normalised on date_grid.
    """
    log_lik = mixture_log_likelihood(
        x, date_grid, model_classical, model_koine, p_classical, p_koine,
        model_lxx=model_lxx, p_lxx=p_lxx,
    )

    if date_prior_mu is not None and date_prior_sigma is not None:
        log_prior = -0.5 * ((date_grid - date_prior_mu) / date_prior_sigma) ** 2
    else:
        log_prior = np.zeros_like(log_lik)

    # Combined
    log_post = log_lik + log_prior
    log_post -= log_post.max()
    post = np.exp(log_post)
    post /= post.sum()

    # Likelihood only
    log_lik_n = log_lik - log_lik.max()
    post_lik  = np.exp(log_lik_n)
    post_lik /= post_lik.sum()

    return post, post_lik


# ---------------------------------------------------------------------------
# 3-component mixture posterior (each sub-model has its own feature subset)
# ---------------------------------------------------------------------------

def _mixture_posterior_3(
    x_c: np.ndarray, x_k: np.ndarray, x_l: np.ndarray,
    date_grid: np.ndarray,
    model_c: MVNModel, model_k: MVNModel, model_l: MVNModel,
    p_c: float, p_k: float, p_l: float,
    date_prior_mu: float | None = None,
    date_prior_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mixture posterior using separate feature vectors per sub-model.

    Each sub-model may use a different subset of features (feats_c, feats_k,
    feats_l), so x_c, x_k, x_l are pre-sliced to the appropriate columns.

    Returns (posterior_combined, posterior_lik_only) normalised on date_grid.
    """
    log_p_c = np.log(p_c + 1e-12)
    log_p_k = np.log(p_k + 1e-12)
    log_p_l = np.log(p_l + 1e-12)

    a = model_c.log_likelihood(x_c, date_grid) + log_p_c  # (G,)
    b = model_k.log_likelihood(x_k, date_grid) + log_p_k  # (G,)
    c = model_l.log_likelihood(x_l, date_grid) + log_p_l  # (G,)

    maxabc  = np.maximum(np.maximum(a, b), c)
    log_lik = maxabc + np.log(
        np.exp(a - maxabc) + np.exp(b - maxabc) + np.exp(c - maxabc)
    )

    if date_prior_mu is not None and date_prior_sigma is not None:
        log_prior = -0.5 * ((date_grid - date_prior_mu) / date_prior_sigma) ** 2
    else:
        log_prior = np.zeros_like(log_lik)

    # Combined posterior
    log_post = log_lik + log_prior
    log_post -= log_post.max()
    post = np.exp(log_post)
    post /= post.sum()

    # Likelihood-only
    log_lik_n = log_lik - log_lik.max()
    post_lik  = np.exp(log_lik_n)
    post_lik /= post_lik.sum()

    return post, post_lik


# ---------------------------------------------------------------------------
# Model parameter diagnostics
# ---------------------------------------------------------------------------

def _print_model_params(model: MVNModel, label: str, out_path: str,
                        feature_names: list, top_n: int = 25) -> None:
    """Print and save model parameters for one register sub-model."""
    p        = len(feature_names)
    eigvals  = np.linalg.eigvalsh(np.linalg.inv(model.Sigma_inv))
    cond     = eigvals[-1] / max(eigvals[0], 1e-15)

    lines = []
    lines.append("=" * 72)
    lines.append(f"CALIBRATED MODEL PARAMETERS — {label}")
    lines.append("=" * 72)
    lines.append(f"  Training texts              : {model.n_train}")
    lines.append(f"  Features (p)                : {p}")
    lines.append(f"  Ridge λ                     : {model.ridge}")
    lines.append(f"  Σ condition number           : {cond:.1f}")
    lines.append(f"  β̂ range                      : [{model.beta.min():.5f}, {model.beta.max():.5f}]")
    lines.append(f"  β̂ median |β|                 : {np.median(np.abs(model.beta)):.5f}")
    lines.append("")

    order    = np.argsort(np.abs(model.beta))[::-1]
    top_idxs = order[:top_n]
    lines.append(f"Top {top_n} features by |β̂|:")
    lines.append(f"  {'Feature':45s}  {'α̂':>10s}  {'β̂':>10s}  Direction")
    lines.append("  " + "-" * 75)
    for idx in top_idxs:
        fname = feature_names[idx]
        a     = model.alpha[idx]
        b     = model.beta[idx]
        direc = "↑ increasing" if b > 0 else "↓ decreasing"
        lines.append(f"  {fname:45s}  {a:>10.4f}  {b:>10.5f}  {direc}")

    report = "\n".join(lines)
    print(report)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"  Parameters saved → {out_path}\n")


def _print_beta_comparison(model_c: MVNModel, model_k: MVNModel,
                           feature_names: list, out_path: str,
                           top_n: int = 30) -> None:
    """
    Compare β vectors between Classical and Koine models.
    Highlights features where the two models DISAGREE on direction
    (opposite sign of β) — these are the features most confounded by
    register, and the ones that most hurt the single-model baseline.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("BETA COMPARISON: Classical vs Koine models")
    lines.append("  Features where sign(β_classical) ≠ sign(β_koine) are REGISTER-CONFOUNDED")
    lines.append("  — the single model's trend for these is a spurious compromise.")
    lines.append("=" * 80)

    # Extract β values for the common feature subset
    # (models may have been trained on different feature subsets)
    def _get_beta(model, feat):
        if feat in model.feature_names:
            idx = model.feature_names.index(feat)
            return model.beta[idx]
        return 0.0

    beta_c = np.array([_get_beta(model_c, f) for f in feature_names])
    beta_k = np.array([_get_beta(model_k, f) for f in feature_names])

    # Rank by |β_c - β_k| (largest disagreement first)
    diff   = np.abs(beta_c - beta_k)
    order  = np.argsort(diff)[::-1]

    lines.append(f"\n{'Feature':45s}  {'β_classical':>12s}  {'β_koine':>10s}  "
                 f"{'|Δβ|':>8s}  Conflict?")
    lines.append("-" * 90)
    for idx in order[:top_n]:
        fname = feature_names[idx]
        bc    = beta_c[idx]
        bk    = beta_k[idx]
        d     = diff[idx]
        conflict = "⚠ OPPOSITE TRENDS" if (bc * bk < 0 and abs(bc) > 1e-4 and abs(bk) > 1e-4) else ""
        lines.append(f"  {fname:43s}  {bc:>12.5f}  {bk:>10.5f}  {d:>8.5f}  {conflict}")

    # Summary count
    n_conflict = sum(
        1 for i in range(len(feature_names))
        if beta_c[i] * beta_k[i] < 0
        and abs(beta_c[i]) > 1e-4
        and abs(beta_k[i]) > 1e-4
    )
    lines.append(f"\nFeatures with conflicting trends: {n_conflict} / {len(feature_names)}")
    lines.append("These are the primary source of register-date confounding.\n")

    report = "\n".join(lines)
    print(report)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"Beta comparison saved → {out_path}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    date_grid = np.arange(DATE_GRID_MIN, DATE_GRID_MAX + DATE_GRID_STEP,
                          DATE_GRID_STEP, dtype=float)

    # ── Load manifest ─────────────────────────────────────────────────────────
    with open(MANIFEST, encoding="utf-8") as f:
        corpus = json.load(f)

    # ── Load feature matrix ───────────────────────────────────────────────────
    feat_path = os.path.join(FEAT_DIR, "feature_matrix.csv")
    feat_df   = pd.read_csv(feat_path, index_col=0)

    # ── Load robust feature names ─────────────────────────────────────────────
    robust_path = os.path.join(RESULTS, "robust_feature_names.json")
    with open(robust_path, encoding="utf-8") as f:
        robust_features = json.load(f)

    # Filter to only features present in the feature matrix
    robust_features = [f for f in robust_features if f in feat_df.columns]
    print(f"Robust features used: {len(robust_features)}")

    # ── Load register probabilities ───────────────────────────────────────────
    with open(REG_PROBS, encoding="utf-8") as f:
        reg_probs = json.load(f)

    # ── Build per-text info ────────────────────────────────────────────────────
    info = {e["id"]: e for e in corpus}

    # ── Split training into Classical, Koine, and LXX groups ─────────────────
    classical_ids = []
    koine_ids     = []
    lxx_ids       = []
    for entry in corpus:
        if entry.get("holdout", False):
            continue
        reg = entry.get("register", "")
        eid = entry["id"]
        if eid not in feat_df.index:
            continue
        if reg in ("ancient_Attic", "Atticizing"):
            classical_ids.append(eid)
        elif reg == "Koine":
            koine_ids.append(eid)
        elif reg == "LXX":
            lxx_ids.append(eid)

    print(f"\nClassical group training: {len(classical_ids)} texts")
    print(f"Koine    group training: {len(koine_ids)} texts")
    print(f"LXX      group training: {len(lxx_ids)} texts")

    # Helper: build X, dates, weights for a set of ids
    def build_Xd(ids):
        dates, X_rows, wts = [], [], []
        for eid in ids:
            entry = info[eid]
            dates.append(float(entry["date_ce"]))
            X_rows.append(feat_df.loc[eid, robust_features].values.astype(float))
            sig = float(entry.get("date_sigma", 20))
            wts.append(1.0 / (sig ** 2))
        return (np.array(dates), np.array(X_rows), np.array(wts))

    dates_c, X_c, wts_c = build_Xd(classical_ids)
    dates_k, X_k, wts_k = build_Xd(koine_ids)
    dates_l, X_l, wts_l = build_Xd(lxx_ids)

    # Limit features per sub-model to avoid overfitting when n is small.
    # Rule: at most n_train // 2 features for any sub-model (ridge handles the rest).
    # Load scan to get the |rho|-ranked feature order.
    scan_path  = os.path.join(RESULTS, "feature_scan_robust.csv")
    df_scan    = pd.read_csv(scan_path) if os.path.exists(scan_path) else None
    if df_scan is not None:
        ordered = df_scan.sort_values("abs_rho", ascending=False)["feature"].tolist()
        ordered = [f for f in ordered if f in robust_features]
    else:
        ordered = robust_features

    def cap_feats(n_train, max_global=80):
        """Return the feature list capped sensibly for the sub-model size.

        Ridge regularisation allows p > n/2, but heavily underdetermined
        systems drift toward prior-dominated posteriors.  Heuristic caps:
          n >= 25  →  min(80, n)   (classical: 31 → 31 feats)
          n >= 12  →  min(40, n)   (koine: 20 → 20 feats)
          n <  12  →  min(20, n*2) (lxx: 7 → 14 feats, heavier ridge)
        """
        if n_train >= 25:
            cap = min(max_global, n_train)
        elif n_train >= 12:
            cap = min(40, n_train)
        else:
            cap = min(20, n_train * 2)
        return ordered[:cap]

    feats_c = cap_feats(len(classical_ids))
    feats_k = cap_feats(len(koine_ids))
    feats_l = cap_feats(len(lxx_ids))

    # Use the intersection of all three as the common feature set for beta comparison
    common_feats = [f for f in ordered if f in feats_c and f in feats_k and f in feats_l]
    print(f"\nFeatures: classical={len(feats_c)}, koine={len(feats_k)}, "
          f"lxx={len(feats_l)}, common={len(common_feats)}")

    def sub_X(X_full, ids, feats):
        """Return X columns restricted to feats."""
        full_df = pd.DataFrame(X_full, index=ids, columns=robust_features)
        return full_df[feats].values

    # ── Train register-specific models ────────────────────────────────────────
    print("\nFitting Classical model …")
    model_c = MVNModel()
    model_c.fit(dates_c, sub_X(X_c, classical_ids, feats_c),
                weights=wts_c, feature_names=feats_c)

    print("Fitting Koine model …")
    model_k = MVNModel()
    model_k.fit(dates_k, sub_X(X_k, koine_ids, feats_k),
                weights=wts_k, feature_names=feats_k)

    print("Fitting LXX model …")
    model_l = MVNModel(ridge=0.05)   # heavier ridge for small n
    model_l.fit(dates_l, sub_X(X_l, lxx_ids, feats_l),
                weights=wts_l, feature_names=feats_l)

    # ── Print model parameters ────────────────────────────────────────────────
    _print_model_params(model_c, "CLASSICAL (ancient_Attic + Atticizing)",
                        os.path.join(RESULTS, "calibrated_classical_params.txt"),
                        feats_c, top_n=25)
    _print_model_params(model_k, "KOINE",
                        os.path.join(RESULTS, "calibrated_koine_params.txt"),
                        feats_k, top_n=25)
    _print_model_params(model_l, "LXX",
                        os.path.join(RESULTS, "calibrated_lxx_params.txt"),
                        feats_l, top_n=20)
    _print_beta_comparison(model_c, model_k, common_feats,
                           os.path.join(RESULTS, "calibrated_beta_comparison.txt"))

    # ── Score ALL texts (training + holdout) ──────────────────────────────────
    print("\nScoring all texts …")
    all_records = []
    for entry in corpus:
        eid  = entry["id"]
        if eid not in feat_df.index:
            print(f"  [SKIP] {eid}: not in feature matrix")
            continue

        x          = feat_df.loc[eid, robust_features].values.astype(float)
        prior_mu   = float(entry["date_ce"])
        prior_sig  = float(entry.get("date_sigma", 25))
        is_holdout = entry.get("holdout", False)

        # Register weights from classifier
        rp = reg_probs.get(eid, {})
        p_c = rp.get("p_ancient_Attic", 0.0) + rp.get("p_Atticizing", 0.0)
        p_k = rp.get("p_Koine", 0.0)
        p_l = rp.get("p_LXX", 0.0)
        total = p_c + p_k + p_l
        if total < 1e-9:
            p_c, p_k, p_l = 0.33, 0.34, 0.33
        else:
            p_c /= total; p_k /= total; p_l /= total

        # Build x restricted to each sub-model's feature set
        x_c = feat_df.loc[eid, feats_c].values.astype(float)
        x_k = feat_df.loc[eid, feats_k].values.astype(float)
        x_l = feat_df.loc[eid, feats_l].values.astype(float)

        # Create temporary single-feature-set wrapper using full feature x for
        # each sub-model (they each have their own internal feature selection)
        post, post_lik = _mixture_posterior_3(
            x_c, x_k, x_l,
            date_grid, model_c, model_k, model_l, p_c, p_k, p_l,
            date_prior_mu=prior_mu, date_prior_sigma=prior_sig,
        )

        # Save posterior JSON
        rec = {
            "id"                 : eid,
            "date_grid"          : date_grid.tolist(),
            "posterior"          : post.tolist(),
            "posterior_lik_only" : post_lik.tolist(),
            "p_classical"        : float(p_c),
            "p_koine"            : float(p_k),
            "p_lxx"              : float(p_l),
        }
        out_path = os.path.join(CAL_POST_DIR, f"{eid}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)

        map_date   = float(date_grid[np.argmax(post)])
        map_lik    = float(date_grid[np.argmax(post_lik)])
        lo68, hi68 = hdi(post, date_grid, 0.68)
        lo95, hi95 = hdi(post, date_grid, 0.95)
        lo68l, hi68l = hdi(post_lik, date_grid, 0.68)

        htype = " [HOLDOUT]" if is_holdout else ""
        print(f"  {eid:45s}  MAP={map_date:4.0f}  68%=[{lo68:4.0f},{hi68:4.0f}]"
              f"  p_c={p_c:.2f}  p_k={p_k:.2f}  p_l={p_l:.2f}{htype}")

        all_records.append({
            "id": eid, "holdout": is_holdout,
            "scholarly_date": prior_mu, "scholarly_sigma": prior_sig,
            "register_true": entry.get("register",""),
            "p_classical": float(p_c), "p_koine": float(p_k), "p_lxx": float(p_l),
            "cal_map": map_date, "cal_lik_map": map_lik,
            "cal_68lo": lo68, "cal_68hi": hi68,
            "cal_95lo": lo95, "cal_95hi": hi95,
            "cal_lik_68lo": lo68l, "cal_lik_68hi": hi68l,
        })

    results_df = pd.DataFrame(all_records)

    # ── Holdout comparison table ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("CALIBRATED vs BASELINE — HOLDOUT COMPARISON")
    print("=" * 80)

    holdout_df = results_df[results_df["holdout"]].copy()

    # Load baseline z-scores from script 06 CSV
    baseline_csv = os.path.join(RESULTS, "holdout_validation.csv")
    if os.path.exists(baseline_csv):
        base_df = pd.read_csv(baseline_csv)
        # Merge on id
        holdout_df = holdout_df.merge(
            base_df[["id", "lik_only_z_score", "lik_only_ci68_lo", "lik_only_ci68_hi"]].rename(
                columns={"lik_only_z_score": "base_z",
                         "lik_only_ci68_lo": "base_68lo",
                         "lik_only_ci68_hi": "base_68hi"}),
            on="id", how="left"
        )
    else:
        holdout_df["base_z"] = np.nan

    # Compute calibrated z-score (likelihood-only MAP vs scholarly date)
    holdout_df["cal_lik_z"] = (
        (holdout_df["cal_lik_map"] - holdout_df["scholarly_date"])
        / holdout_df["scholarly_sigma"]
    )

    # Width of 68% HDI
    holdout_df["cal_lik_width68"] = holdout_df["cal_lik_68hi"] - holdout_df["cal_lik_68lo"]
    if "base_68lo" in holdout_df.columns:
        holdout_df["base_width68"] = holdout_df["base_68hi"] - holdout_df["base_68lo"]
    else:
        holdout_df["base_width68"] = np.nan

    # Scholarly date in 68% HDI?
    holdout_df["cal_in_68"] = (
        (holdout_df["scholarly_date"] >= holdout_df["cal_lik_68lo"]) &
        (holdout_df["scholarly_date"] <= holdout_df["cal_lik_68hi"])
    )

    print(f"\n{'Text':35s}  {'Schl':>6s}  "
          f"{'Base|z|':>7s}  {'Cal|z|':>7s}  "
          f"{'BaseW68':>7s}  {'CalW68':>7s}  "
          f"{'In68?':>5s}  "
          f"{'P_c':>5s}  {'P_k':>5s}  {'P_l':>5s}")
    print("-" * 110)
    for _, row in holdout_df.sort_values("scholarly_date").iterrows():
        yr  = int(abs(row["scholarly_date"]))
        era = "BCE" if row["scholarly_date"] < 0 else "CE"
        bz  = f"{abs(row['base_z']):.2f}" if pd.notna(row.get("base_z")) else "  n/a"
        cz  = f"{abs(row['cal_lik_z']):.2f}"
        bw  = f"{row['base_width68']:.0f}" if pd.notna(row.get("base_width68")) else "  n/a"
        cw  = f"{row['cal_lik_width68']:.0f}"
        in68 = "YES ✓" if row["cal_in_68"] else "NO  ✗"
        pl   = row.get("p_lxx", 0.0)
        print(f"  {row['id']:33s}  {yr:3d}{era}  "
              f"{bz:>7s}  {cz:>7s}  "
              f"{bw:>7s}  {cw:>7s}  "
              f"{in68:>5s}  "
              f"{row['p_classical']:.2f}   {row['p_koine']:.2f}   {pl:.2f}")

    # Save CSV
    out_csv = os.path.join(RESULTS, "calibrated_validation.csv")
    holdout_df.to_csv(out_csv, index=False)
    print(f"\nCalibrated validation table → {out_csv}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if HAS_MPL:
        _plot_holdout_comparison(holdout_df, date_grid)
        _plot_calibrated_summary(results_df, date_grid)

    print("\nDone.")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _load_cal_posterior(eid: str, date_grid: np.ndarray):
    path = os.path.join(CAL_POST_DIR, f"{eid}.json")
    if not os.path.exists(path):
        return None, None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    post     = np.array(d["posterior"])
    post_lik = np.array(d["posterior_lik_only"])
    return post, post_lik


def _load_base_posterior(eid: str):
    path = os.path.join(BASELINE_DIR, f"{eid}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return np.array(d.get("posterior_lik_only", d["posterior"]))


def _plot_holdout_comparison(holdout_df: pd.DataFrame, date_grid: np.ndarray):
    """Side-by-side: baseline likelihood vs calibrated likelihood for each holdout."""
    n_h   = len(holdout_df)
    fig, axes = plt.subplots(n_h, 2, figsize=(14, 4 * n_h))
    if n_h == 1:
        axes = axes[np.newaxis, :]

    for i, (_, row) in enumerate(holdout_df.sort_values("scholarly_date").iterrows()):
        eid  = row["id"]
        sd   = row["scholarly_date"]
        ss   = row["scholarly_sigma"]

        # Baseline
        base_lik = _load_base_posterior(eid)
        # Calibrated
        cal_post, cal_lik = _load_cal_posterior(eid, date_grid)

        for j, (lik, title_sfx, col) in enumerate([
            (base_lik, "Baseline (single model)", "#e66101"),
            (cal_lik,  "Calibrated (mixture model)", "#1a9641"),
        ]):
            ax = axes[i, j]
            if lik is None:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            ax.plot(date_grid, lik / lik.max(), lw=1.8, color=col, label="Likelihood (norm.)")
            ax.fill_between(date_grid, lik / lik.max(), alpha=0.2, color=col)

            # Scholarly date
            ax.axvline(sd, color="steelblue", lw=1.5, ls="--", label=f"Scholarly ({int(abs(sd))}{'BCE' if sd<0 else 'CE'})")
            ax.axvspan(sd - ss, sd + ss, alpha=0.10, color="steelblue")

            # 68% HDI
            lo68, hi68 = hdi(lik, date_grid, 0.68)
            ax.axvspan(lo68, hi68, alpha=0.15, color=col, label=f"68% HDI [{int(lo68)}, {int(hi68)}]")

            # Truncation note
            if abs(hi68 - date_grid[-1]) < 10:
                ax.text(0.98, 0.92, "⚠ HDI truncated at boundary →",
                        transform=ax.transAxes, ha="right", fontsize=8, color="gray")

            map_lik = float(date_grid[np.argmax(lik)])
            z = (map_lik - sd) / ss
            in68 = (sd >= lo68) and (sd <= hi68)
            in68_str = "✓ in 68% HDI" if in68 else "✗ outside 68% HDI"
            ax.text(0.02, 0.96,
                    f"z = {z:+.2f}σ\nScholarly {in68_str}",
                    transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

            ax.set_title(f"{row['id']}  |  {title_sfx}", fontsize=9)
            ax.set_xlabel("Date (CE)")
            ax.set_ylabel("Normalised likelihood")
            ax.legend(fontsize=8, loc="upper left")
            ax.set_xlim(date_grid[0], date_grid[-1])
            ax.set_ylim(0, 1.05)

    plt.suptitle("Holdout Validation: Baseline vs Calibrated Likelihood",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "calibrated_validation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Holdout comparison plot → {path}")


def _plot_calibrated_summary(results_df: pd.DataFrame, date_grid: np.ndarray):
    """Summary plot: calibrated MAP vs scholarly date for all training texts."""
    train_df = results_df[~results_df["holdout"]].sort_values("scholarly_date")

    fig, ax = plt.subplots(figsize=(12, 7))

    colours = {"ancient_Attic": "#2166ac", "Atticizing": "#d6604d",
               "Koine": "#4dac26", "LXX": "#9b59b6"}
    for _, row in train_df.iterrows():
        col = colours.get(row["register_true"], "gray")
        ax.scatter(row["scholarly_date"], row["cal_map"], color=col, s=60,
                   alpha=0.8, zorder=3)
        ax.plot([row["scholarly_date"], row["scholarly_date"]],
                [row["cal_68lo"], row["cal_68hi"]],
                color=col, lw=1.5, alpha=0.5, zorder=2)

    # Diagonal
    lim = [date_grid[0], date_grid[-1]]
    ax.plot(lim, lim, "k--", lw=1, alpha=0.4, label="Perfect agreement")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=c,
               markersize=9, label=r)
        for r, c in colours.items()
    ]
    ax.legend(handles=legend_elements, fontsize=10)

    ax.set_xlabel("Scholarly date (CE)", fontsize=12)
    ax.set_ylabel("Calibrated model MAP date (CE)", fontsize=12)
    ax.set_title("Calibrated Dating Summary — Training Texts\n"
                 "(error bars = 68% HDI of combined posterior)", fontsize=12)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.axvline(0, color="gray", lw=0.5, ls=":")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "calibrated_dating_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Calibrated summary plot → {path}")


if __name__ == "__main__":
    main()
