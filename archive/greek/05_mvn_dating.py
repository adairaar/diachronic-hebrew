"""
05_mvn_dating.py
================
Multivariate Normal (MVN) Bayesian dating model for the Greek corpus.

The model mirrors the Hebrew pipeline's approach (scripts 10–11):

Model
-----
For each training text i with feature vector xᵢ and uncertain date dᵢ:

  dᵢ ~ N(μᵢ, σᵢ²)          [date prior: μᵢ = scholarly consensus, σᵢ = uncertainty]

  For each feature j:
    xᵢⱼ | dᵢ ~ N(αⱼ + βⱼ dᵢ, τⱼ²)   [linear trend with Gaussian noise]

  Parameters (αⱼ, βⱼ, τⱼ) estimated by WLS (weights = 1/σᵢ²) per feature.

  Residual covariance Σ captures correlation between features.

  For a new text with feature vector x*:
    log P(d* | x*) ∝ -½ (x* - μ(d*))ᵀ Σ⁻¹ (x* - μ(d*)) + log P(d*)

  where μ(d*) = α + β·d* is the predicted feature vector at date d*.
  P(d*) is a Gaussian prior centred on the scholarly consensus date for
  each text (both training texts and holdouts).  A uniform prior is NOT
  used for training texts: because the log-likelihood is exactly quadratic
  in d* (guaranteed by the linear model), a uniform prior means the MAP is
  the unconstrained parabola vertex, which frequently lies outside the
  training date range and is pinned to a grid boundary — producing a
  misleading MAP that does not represent the bulk of posterior probability.
  Using the scholarly Gaussian prior fixes this without changing the
  posterior for texts whose likelihood is informative.

  The likelihood-only posterior (no prior) is also saved alongside the
  combined posterior so the holdout validator can report raw model power.

  Tikhonov (ridge) regularization is applied to Σ⁻¹ for numerical stability.

Date uncertainty propagation
-----------------------------
When fitting feature regressions, each training observation is weighted by
1/σᵢ² where σᵢ is the date uncertainty (date_sigma). This down-weights texts
with poorly known dates, matching what the Hebrew pipeline does.

Outputs
-------
  results/dating_results.csv        — MAP date + 68%/95% credible intervals per entry
  results/posteriors/<id>.json      — full posterior distribution per entry
  results/plots/dating_posteriors.png

Usage
-----
    python 05_mvn_dating.py [--ridge 0.1] [--grid-step 5]
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available — skipping inline plots from this script.")

HERE       = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR   = os.path.join(HERE, "data", "features")
RESULTS    = os.path.join(HERE, "results")
PLOTS_DIR  = os.path.join(HERE, "results", "plots")
MANIFEST   = os.path.join(HERE, "corpus_manifest.json")

# ---------------------------------------------------------------------------
# Helper: weighted least-squares (WLS) linear regression
# ---------------------------------------------------------------------------

def wls_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """
    Fit y = α + β·x with weights w (= 1/σ²).
    Returns (alpha, beta, tau) where tau = weighted RMSE.
    """
    W  = np.diag(w)
    X  = np.column_stack([np.ones_like(x), x])
    try:
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        coef = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
    alpha, beta = coef
    residuals   = y - (alpha + beta * x)
    tau = np.sqrt(np.average(residuals**2, weights=w))
    tau = max(tau, 1e-6)   # prevent zero variance
    return float(alpha), float(beta), float(tau)


# ---------------------------------------------------------------------------
# MVN dating model class
# ---------------------------------------------------------------------------

class MVNDatingModel:
    """
    Multivariate Normal Bayesian dating model.

    Parameters
    ----------
    ridge : float
        Ridge regularization added to diagonal of Σ before inversion.
    """

    def __init__(self, ridge: float = 0.05):
        self.ridge = ridge
        self.features: list[str] = []
        self.alphas: np.ndarray  = None
        self.betas:  np.ndarray  = None
        self.taus:   np.ndarray  = None
        self.Sigma_inv: np.ndarray = None
        self.is_fitted  = False

    def fit(self, df_train: pd.DataFrame, feature_cols: list[str]) -> None:
        """
        Fit the model on training data.

        df_train must have columns: date_ce, date_sigma, + all feature_cols.
        """
        self.features = feature_cols
        p = len(feature_cols)

        dates  = df_train["date_ce"].values.astype(float)
        sigmas = df_train["date_sigma"].values.astype(float)
        weights = 1.0 / (sigmas**2 + 1.0)   # +1 for numerical safety

        X = df_train[feature_cols].values.astype(float)

        # WLS per feature
        alphas, betas, taus = [], [], []
        for j in range(p):
            y = X[:, j]
            # Impute NaN with column mean (should be rare after preprocessing)
            y = np.where(np.isnan(y), np.nanmean(y), y)
            a, b, t = wls_fit(dates, y, weights)
            alphas.append(a)
            betas.append(b)
            taus.append(t)

        self.alphas = np.array(alphas)
        self.betas  = np.array(betas)
        self.taus   = np.array(taus)

        # Compute residual covariance matrix
        predicted = self.alphas[None, :] + np.outer(dates, self.betas)
        residuals = X - predicted

        # Replace NaN residuals
        col_means = np.nanmean(residuals, axis=0)
        inds = np.where(np.isnan(residuals))
        residuals[inds] = np.take(col_means, inds[1])

        Sigma = np.cov(residuals.T)
        if Sigma.ndim == 0:
            Sigma = np.array([[float(Sigma)]])

        # Ridge regularization
        Sigma += self.ridge * np.eye(p)

        try:
            self.Sigma_inv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            self.Sigma_inv = np.linalg.pinv(Sigma)

        self.Sigma = Sigma   # store for diagnostics
        self.is_fitted = True
        print(f"  Model fitted: {p} features, "
              f"β range [{self.betas.min():.4f}, {self.betas.max():.4f}]")

    def log_likelihood(self, x: np.ndarray, date_grid: np.ndarray) -> np.ndarray:
        """
        Compute log-likelihood log P(x | d) for each d in date_grid.

        Parameters
        ----------
        x : np.ndarray shape (p,)
        date_grid : np.ndarray shape (G,)

        Returns
        -------
        np.ndarray shape (G,) of log-likelihoods
        """
        # Predicted mean at each grid point: (G, p)
        mu = self.alphas[None, :] + np.outer(date_grid, self.betas)
        # Residuals: (G, p)
        delta = x[None, :] - mu
        # Mahalanobis distance: sum_G (delta @ Sigma_inv @ delta.T) diagonal
        maha = np.einsum("gi,ij,gj->g", delta, self.Sigma_inv, delta)
        return -0.5 * maha

    def posterior(
        self,
        x: np.ndarray,
        date_grid: np.ndarray,
        date_prior_mu: float | None = None,
        date_prior_sigma: float | None = None,
    ) -> np.ndarray:
        """
        Compute normalized posterior P(d | x) on date_grid.

        If date_prior_mu/sigma are provided, uses a Gaussian prior on date;
        otherwise uses a uniform prior.
        """
        log_lk = self.log_likelihood(x, date_grid)

        if date_prior_mu is not None and date_prior_sigma is not None:
            log_prior = -0.5 * ((date_grid - date_prior_mu) / date_prior_sigma)**2
        else:
            log_prior = np.zeros_like(date_grid)

        log_post = log_lk + log_prior
        # Numerically stable softmax
        log_post -= log_post.max()
        post = np.exp(log_post)
        post /= post.sum()
        return post

    def likelihood_only_posterior(
        self,
        x: np.ndarray,
        date_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Compute normalized posterior using a uniform prior (likelihood only).
        Useful for diagnosing raw model discriminating power.
        """
        return self.posterior(x, date_grid,
                              date_prior_mu=None,
                              date_prior_sigma=None)

    def print_parameters(self, top_n: int = 25, out_path: str | None = None) -> None:
        """
        Print and optionally save the fitted model parameters for diagnostics.

        Outputs
        -------
        - Top features by |β| (most diachronically informative)
        - Covariance matrix condition number (numerical stability indicator)
        - β distribution summary (histogram-style)
        - Saved to out_path as a plain-text report if provided
        """
        if not self.is_fitted:
            print("  Model not fitted yet.")
            return

        p = len(self.features)
        eigvals = np.linalg.eigvalsh(self.Sigma)
        cond    = eigvals[-1] / max(eigvals[0], 1e-15)

        lines = []
        lines.append("=" * 72)
        lines.append("MVN MODEL PARAMETERS")
        lines.append("=" * 72)
        lines.append(f"  Features (p)            : {p}")
        lines.append(f"  Ridge λ                 : {self.ridge}")
        lines.append(f"  Σ condition number       : {cond:.1f}  "
                     f"({'well-conditioned' if cond < 1e4 else 'ill-conditioned — raise ridge'})")
        lines.append(f"  Σ eigenvalue range       : [{eigvals[0]:.4g}, {eigvals[-1]:.4g}]")
        lines.append(f"  β̂ range                  : [{self.betas.min():.5f}, {self.betas.max():.5f}]")
        lines.append(f"  β̂ median |β|             : {np.median(np.abs(self.betas)):.5f}")
        lines.append(f"  α̂ range                  : [{self.alphas.min():.5f}, {self.alphas.max():.5f}]")
        lines.append("")

        # Top features by |β|
        order    = np.argsort(np.abs(self.betas))[::-1]
        top_idxs = order[:top_n]

        lines.append(f"Top {top_n} features by |β̂|  (β > 0 means feature INCREASES over time):")
        lines.append(f"  {'Feature':45s}  {'α̂':>10s}  {'β̂':>10s}  {'τ̂ (residSD)':>12s}  Direction")
        lines.append("  " + "-" * 88)
        for idx in top_idxs:
            fname = self.features[idx]
            a     = self.alphas[idx]
            b     = self.betas[idx]
            t     = self.taus[idx]
            direc = "↑ increasing" if b > 0 else "↓ decreasing"
            lines.append(f"  {fname:45s}  {a:>10.4f}  {b:>10.5f}  {t:>12.4f}  {direc}")

        lines.append("")
        lines.append("β̂ distribution:")
        bins = [-1e6, -0.01, -0.001, 0.001, 0.01, 1e6]
        labels_b = ["β < -0.01", "-0.01≤β<-0.001", "|β|<0.001", "0.001≤β<0.01", "β ≥ 0.01"]
        counts = np.histogram(self.betas, bins=bins)[0]
        for lbl, cnt in zip(labels_b, counts):
            lines.append(f"  {lbl:22s}: {cnt:3d}  {'█' * min(cnt, 40)}")
        lines.append("=" * 72)

        report = "\n".join(lines)
        print(report)

        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report + "\n")
            print(f"\nModel parameters saved → {out_path}")

    def credible_interval(
        self, posterior: np.ndarray, date_grid: np.ndarray, frac: float = 0.68
    ) -> tuple[float, float]:
        """Return equal-tailed credible interval."""
        cdf   = np.cumsum(posterior)
        low   = date_grid[np.searchsorted(cdf, (1 - frac) / 2)]
        high  = date_grid[np.searchsorted(cdf, 1 - (1 - frac) / 2)]
        return float(low), float(high)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MVN Bayesian dating of Greek corpus.")
    parser.add_argument("--ridge",     type=float, default=0.05,
                        help="Ridge regularization for covariance inversion.")
    parser.add_argument("--grid-step", type=int,   default=5,
                        help="Step size (years) for date evaluation grid.")
    parser.add_argument("--max-feats", type=int,   default=80,
                        help="Maximum robust features to include in MVN (avoid over-fitting; "
                             "ridge regularisation on Σ makes n/2 ≈ 80 safe with 58 texts).")
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    post_dir = os.path.join(RESULTS, "posteriors")
    os.makedirs(post_dir, exist_ok=True)

    # Load data
    feat_path   = os.path.join(FEAT_DIR, "feature_matrix.csv")
    robust_path = os.path.join(RESULTS, "robust_feature_names.json")

    if not os.path.exists(feat_path):
        print("feature_matrix.csv not found — run 03_feature_extraction.py first.")
        sys.exit(1)
    if not os.path.exists(robust_path):
        print("robust_feature_names.json not found — run 04_feature_screening.py first.")
        sys.exit(1)

    df_all = pd.read_csv(feat_path, index_col="id")

    with open(robust_path, encoding="utf-8") as f:
        robust_names = json.load(f)
    with open(MANIFEST, encoding="utf-8") as f:
        corpus = json.load(f)

    holdout_ids = {c["id"] for c in corpus if c["holdout"]}
    meta_by_id  = {c["id"]: c for c in corpus}

    # Limit to max_feats most-robust features (avoid overfitting with small n)
    # Load scan to get ordering by |rho|
    scan_path = os.path.join(RESULTS, "feature_scan_robust.csv")
    if os.path.exists(scan_path):
        df_scan   = pd.read_csv(scan_path)
        ordered   = df_scan.sort_values("abs_rho", ascending=False)["feature"].tolist()
        # Keep only names that exist in the feature matrix
        ordered   = [f for f in ordered if f in df_all.columns]
        feat_cols = ordered[:args.max_feats]
    else:
        feat_cols = [f for f in robust_names if f in df_all.columns][:args.max_feats]

    print(f"Using {len(feat_cols)} robust features for MVN model.\n")

    # Split training / all
    df_train = df_all[~df_all.index.isin(holdout_ids)].copy()
    df_test  = df_all.copy()   # score everything (train + holdout)

    # Date evaluation grid (BCE expressed as negative, CE as positive)
    date_min  = int(df_train["date_ce"].min()) - 50
    date_max  = int(df_train["date_ce"].max()) + 50
    date_grid = np.arange(date_min, date_max + 1, args.grid_step, dtype=float)

    # Fit model
    print("Fitting MVN dating model …")
    model = MVNDatingModel(ridge=args.ridge)
    model.fit(df_train, feat_cols)
    param_path = os.path.join(RESULTS, "mvn_model_parameters.txt")
    model.print_parameters(top_n=25, out_path=param_path)

    # Score all entries
    print("\nComputing posteriors …")
    results = []
    for eid in df_test.index:
        entry = meta_by_id.get(eid, {})
        x = df_test.loc[eid, feat_cols].values.astype(float)
        # Replace NaN with column mean from training
        nan_mask = np.isnan(x)
        if nan_mask.any():
            col_means = df_train[feat_cols].mean().values
            x[nan_mask] = col_means[nan_mask]

        # Use scholarly Gaussian prior for ALL texts.
        # This is the correct Bayesian posterior: it combines the linguistic
        # evidence (likelihood) with our prior knowledge of when the text
        # was written.  Using a uniform prior for training texts caused MAP
        # to land at grid boundaries whenever the likelihood vertex fell
        # outside the grid — the posterior was correct but the MAP was
        # misleading (it reflected a truncated, not peaked, distribution).
        prior_mu    = entry.get("date_ce")
        prior_sigma = entry.get("date_sigma")

        post = model.posterior(x, date_grid,
                               date_prior_mu=prior_mu,
                               date_prior_sigma=prior_sigma)
        map_date = float(date_grid[np.argmax(post)])
        ci68_lo, ci68_hi = model.credible_interval(post, date_grid, 0.68)
        ci95_lo, ci95_hi = model.credible_interval(post, date_grid, 0.95)

        # Also compute likelihood-only posterior (no prior) so the holdout
        # validator can report the model's raw discriminating power.
        post_lik = model.likelihood_only_posterior(x, date_grid)

        # Save full posterior (combined) + likelihood-only
        post_record = {
            "id"                  : eid,
            "date_grid"           : date_grid.tolist(),
            "posterior"           : post.tolist(),          # prior × likelihood
            "posterior_lik_only"  : post_lik.tolist(),      # likelihood only (no prior)
        }
        with open(os.path.join(post_dir, f"{eid}.json"), "w") as f:
            json.dump(post_record, f)

        result = {
            "id"           : eid,
            "author"       : entry.get("author", ""),
            "work"         : entry.get("work", ""),
            "holdout"      : eid in holdout_ids,
            "scholarly_date_ce": entry.get("date_ce"),
            "scholarly_sigma"  : entry.get("date_sigma"),
            "map_date_ce"  : round(map_date),
            "ci68_lo"      : round(ci68_lo),
            "ci68_hi"      : round(ci68_hi),
            "ci95_lo"      : round(ci95_lo),
            "ci95_hi"      : round(ci95_hi),
        }
        results.append(result)

        label  = "BCE" if map_date < 0 else "CE "
        s_label = "BCE" if (entry.get("date_ce", 0) or 0) < 0 else "CE "
        print(f"  {eid:40s}  MAP={abs(round(map_date)):4d}{label}  "
              f"68%=[{abs(round(ci68_lo)):4d},{abs(round(ci68_hi)):4d}]  "
              f"scholarly={abs(entry.get('date_ce',0)):4d}{s_label}"
              f"{'  [HOLDOUT]' if eid in holdout_ids else ''}")

    # Save results
    df_results = pd.DataFrame(results)
    res_path   = os.path.join(RESULTS, "dating_results.csv")
    df_results.to_csv(res_path, index=False)
    print(f"\nDating results → {res_path}")

    # ── Inline plot: posteriors for training set (chronologically ordered) ──
    if HAS_MPL:
        df_plot = df_results[~df_results["holdout"]].copy()
        df_plot = df_plot.sort_values("scholarly_date_ce")
        n_plot  = len(df_plot)

        fig, axes = plt.subplots(n_plot, 1,
                                 figsize=(12, max(4, n_plot * 0.6)),
                                 sharex=True)
        if n_plot == 1:
            axes = [axes]

        colors = cm.plasma(np.linspace(0.1, 0.9, n_plot))

        for ax, (_, row), color in zip(axes, df_plot.iterrows(), colors):
            eid  = row["id"]
            post_file = os.path.join(post_dir, f"{eid}.json")
            if not os.path.exists(post_file):
                continue
            with open(post_file) as pf:
                pr = json.load(pf)
            dg = np.array(pr["date_grid"])
            po = np.array(pr["posterior"])

            ax.fill_between(dg, po, alpha=0.5, color=color)
            ax.axvline(row["map_date_ce"], color=color, lw=1.5, ls="-")
            ax.axvline(row["scholarly_date_ce"], color="gray", lw=1, ls="--", alpha=0.6)
            ax.set_yticks([])
            short_name = f"{row['author'][:22]}"
            ax.set_ylabel(short_name, rotation=0, ha="right", va="center", fontsize=7)

        axes[-1].set_xlabel("Date (CE; negative = BCE)", fontsize=10)
        fig.suptitle("Greek Prose Corpus: MVN Posterior Distributions\n"
                     "(solid = MAP, dashed = scholarly consensus)", fontsize=11)
        plt.tight_layout(rect=[0.15, 0, 1, 0.97])
        plot_path = os.path.join(PLOTS_DIR, "dating_posteriors.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Posteriors plot → {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
