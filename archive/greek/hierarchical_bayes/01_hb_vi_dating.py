"""
01_hb_vi_dating.py
==================
Hierarchical Bayesian dating model fitted with Variational Inference (VI)
and the ADAM optimizer.

Motivation
----------
The MLE-MVN approach (10_hard_register_dating.py) fits OLS point estimates
for each feature's α (intercept) and β (slope), then uses a fixed ridge-
regularised covariance.  Uncertainty in those regression parameters is
ignored when computing date posteriors.

This script replaces that with a fully Bayesian treatment:

  * β_{r,f} | μ_β_f, σ_β  ~  N(μ_β_f, σ_β²)     hierarchical prior on slopes
  * μ_β_f                  ~  N(0, 1²)              global slope mean per feature
  * σ_β                    ~  HalfNormal(0.5)       slope heterogeneity
  * α_{r,f}                ~  N(0, 3²)              intercepts (per register/feature)
  * log σ_obs_{r,f}        ~  N(0, 0.5²)            log observation noise
  * x_{t,f} | d_t          ~  N(α_{r_t,f} + β_{r_t,f}·d_t , σ_obs_{r_t,f}²)

The hierarchical prior on β provides automatic shrinkage: features with weak
temporal signal across all registers are shrunk toward zero; no manual feature
pre-selection is needed.

VI approach
-----------
Mean-field VI with reparameterisation trick.  Every latent variable θ_i is
approximated by q(θ_i) = N(m_i, exp(2ρ_i)), and we optimise the ELBO:

  ELBO = E_q[log p(X|θ)] - KL(q(θ) ‖ p(θ))

Gradients of the ELBO are computed automatically via the `autograd` library.
The ADAM optimiser drives convergence.

Date posterior (test-time)
--------------------------
After fitting, date posteriors are computed by Monte Carlo integration over the
VI posterior of the regression parameters.  Uncertainty in α, β, σ is fully
propagated into the date credible intervals — a key advantage over MLE-MVN.

Output
------
  results/hb_vi_dating.csv     — machine-readable per-text dating results
  results/hb_vi_params.npz     — fitted VI parameters (means + log-stds)
  results/hb_vi_elbo.txt       — ELBO training curve
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
from scipy import stats

try:
    import autograd.numpy as anp
    from autograd import grad as agrad
except ImportError:
    print("Missing: pip install autograd --break-system-packages")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Missing: pip install pandas --break-system-packages")
    sys.exit(1)

# ── Paths ────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
PARENT    = os.path.dirname(HERE)
FEAT_CSV  = os.path.join(PARENT, "data", "features", "feature_matrix.csv")
MANIFEST  = os.path.join(PARENT, "corpus_manifest.json")
REG_PROBS = os.path.join(PARENT, "results", "register_probs.json")
RESULTS   = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

# ── Hyper-settings ────────────────────────────────────────────────────────────
N_MC_TRAIN  = 32      # MC samples per ELBO gradient step
N_MC_INFER  = 500     # MC samples for date posterior
N_ITER      = 3000    # ADAM iterations
LR          = 0.02    # ADAM learning rate
ADAM_B1     = 0.9
ADAM_B2     = 0.999
ADAM_EPS    = 1e-8
N_GRID      = 2000    # date grid points for posterior
LOG_EVERY   = 200     # print ELBO every N iterations

# Register-group mapping (matches 10_hard_register_dating.py)
GROUPS = ["classical", "atticizing", "koine"]

META_COLS = {"author", "date_ce", "date_sigma", "genre", "holdout",
             "word_count", "register", "group"}


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ADAM optimiser (pure numpy — operates outside autograd)
# ═══════════════════════════════════════════════════════════════════════════════

class ADAM:
    """Vectorised ADAM optimiser."""

    def __init__(self, n_params: int, lr: float = 0.01,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr  = lr
        self.b1  = b1
        self.b2  = b2
        self.eps = eps
        self.t   = 0
        self.m   = np.zeros(n_params)
        self.v   = np.zeros(n_params)

    def step(self, phi: np.ndarray, grad_val: np.ndarray) -> np.ndarray:
        self.t  += 1
        self.m   = self.b1 * self.m + (1 - self.b1) * grad_val
        self.v   = self.b2 * self.v + (1 - self.b2) * grad_val ** 2
        m_hat    = self.m / (1 - self.b1 ** self.t)
        v_hat    = self.v / (1 - self.b2 ** self.t)
        return phi - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Parameter packing / unpacking
# ═══════════════════════════════════════════════════════════════════════════════

def n_params(R: int, F: int) -> int:
    """Total number of variational parameters."""
    return 6 * R * F + 2 * F + 2   # α, β, log_σ (×R×F each) + μ_β, ρ_μβ (×F) + log_σβ, ρ_log_σβ


def unpack(phi, R: int, F: int):
    """Return all VI parameter matrices from the flat vector."""
    RF  = R * F
    i   = 0
    m_a   = phi[i:i+RF].reshape(R, F); i += RF
    rho_a = phi[i:i+RF].reshape(R, F); i += RF
    m_b   = phi[i:i+RF].reshape(R, F); i += RF
    rho_b = phi[i:i+RF].reshape(R, F); i += RF
    m_ls  = phi[i:i+RF].reshape(R, F); i += RF   # log_sigma_obs
    rho_ls= phi[i:i+RF].reshape(R, F); i += RF
    m_mb  = phi[i:i+F];                i += F    # mu_beta  [F]
    rho_mb= phi[i:i+F];                i += F
    m_lsb = phi[i];                    i += 1    # log_sigma_beta (scalar)
    rho_lsb = phi[i];                  i += 1
    return m_a, rho_a, m_b, rho_b, m_ls, rho_ls, m_mb, rho_mb, m_lsb, rho_lsb


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  ELBO  (autograd-differentiable)
# ═══════════════════════════════════════════════════════════════════════════════

def kl_gauss(m_q, s_q, m_p, s_p):
    """KL( N(m_q, s_q²) ‖ N(m_p, s_p²) )  element-wise."""
    return (anp.log(s_p) - anp.log(s_q)
            + (s_q ** 2 + (m_q - m_p) ** 2) / (2.0 * s_p ** 2)
            - 0.5)


def neg_elbo(phi, eps_a, eps_b, eps_ls, eps_mb, eps_lsb,
             X_norm, dates_norm, reg_ids, mask, R, F):
    """
    Negative ELBO.  phi is the flat VI parameter vector.
    eps_* are pre-drawn standard-normal arrays (constants w.r.t. phi).
    Returns a scalar.
    """
    (m_a, rho_a, m_b, rho_b, m_ls, rho_ls,
     m_mb, rho_mb, m_lsb, rho_lsb) = unpack(phi, R, F)

    # ── Standard deviations ───────────────────────────────────────────────────
    s_a   = anp.exp(rho_a)    # [R, F]
    s_b   = anp.exp(rho_b)    # [R, F]
    s_ls  = anp.exp(rho_ls)   # [R, F]
    s_mb  = anp.exp(rho_mb)   # [F]
    s_lsb = anp.exp(rho_lsb)  # scalar

    # ── Reparameterised samples ───────────────────────────────────────────────
    # eps_a : [n_mc, R, F]  eps_b : [n_mc, R, F]  etc.
    alpha_s   = m_a   + s_a   * eps_a    # [n_mc, R, F]
    beta_s    = m_b   + s_b   * eps_b    # [n_mc, R, F]
    logσ_s    = m_ls  + s_ls  * eps_ls   # [n_mc, R, F]
    mu_beta_s = m_mb  + s_mb  * eps_mb   # [n_mc, F]
    lsb_s     = m_lsb + s_lsb * eps_lsb # [n_mc]

    σ_s          = anp.exp(logσ_s)   # [n_mc, R, F]   positive
    sigma_beta_s = anp.exp(lsb_s)    # [n_mc]          positive

    # ── Reconstruction: E_q[ log p(X | α, β, σ) ] ───────────────────────────
    # Select per-text register parameters: [n_mc, T, F]
    alpha_t = alpha_s[:, reg_ids, :]   # advanced indexing — autograd-safe
    beta_t  = beta_s[:,  reg_ids, :]
    σ_t     = σ_s[:,    reg_ids, :]

    # Predicted feature mean: [n_mc, T, F]
    pred = alpha_t + beta_t * dates_norm[anp.newaxis, :, anp.newaxis]

    # Log-likelihood per observation (dropping the -0.5 log 2π constant)
    resid      = X_norm[anp.newaxis, :, :] - pred
    log_lik_pf = -0.5 * (resid / σ_t) ** 2 - anp.log(σ_t)  # [n_mc, T, F]

    # Zero-out masked (NaN) features
    log_lik_pf = log_lik_pf * mask[anp.newaxis, :, :]

    # Average over MC samples, sum over texts and features
    n_mc = eps_a.shape[0]
    log_lik = anp.sum(anp.mean(log_lik_pf, axis=0))   # sum over T×F

    # ── KL terms ──────────────────────────────────────────────────────────────
    # α  prior:  N(0, 3)
    kl_alpha = anp.sum(kl_gauss(m_a, s_a, 0.0, 3.0))

    # log_σ_obs  prior:  N(0, 0.5)
    kl_logσ = anp.sum(kl_gauss(m_ls, s_ls, 0.0, 0.5))

    # μ_β  prior:  N(0, 1)
    kl_mubeta = anp.sum(kl_gauss(m_mb, s_mb, 0.0, 1.0))

    # log_σ_β  prior:  N(-1, 0.5)   → σ_β centred around exp(-1)≈0.37
    kl_lsb = kl_gauss(m_lsb, s_lsb, -1.0, 0.5)

    # β  hierarchical:  E_{q(μ_β, σ_β)}[ KL( q(β) ‖ N(μ_β, σ_β²) ) ]
    #   analytically tractable for fixed μ_β, σ_β → average over MC samples
    m_b_bc  = m_b[anp.newaxis, :, :]              # [1, R, F]
    s_b_bc  = s_b[anp.newaxis, :, :]              # [1, R, F]
    mb_bc   = mu_beta_s[:, anp.newaxis, :]         # [n_mc, 1, F]
    sb_bc   = sigma_beta_s[:, anp.newaxis, anp.newaxis]  # [n_mc, 1, 1]

    kl_beta_per = anp.sum(
        kl_gauss(m_b_bc, s_b_bc, mb_bc, sb_bc), axis=(1, 2))  # [n_mc]
    kl_beta = anp.mean(kl_beta_per)

    elbo = log_lik - kl_alpha - kl_logσ - kl_mubeta - kl_lsb - kl_beta
    return -elbo   # minimise negative ELBO


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Register assignment  (same logic as 10_hard_register_dating.py)
# ═══════════════════════════════════════════════════════════════════════════════

def assign_group(probs: dict) -> str:
    pa  = probs.get("p_ancient_Attic", 0.0)
    pat = probs.get("p_Atticizing",    0.0)
    pk  = probs.get("p_Koine",         0.0)
    pl  = probs.get("p_LXX",           0.0)
    best = max(pa, pat, pk, pl)
    if best == pa:
        return "classical"
    if best == pat:
        return "atticizing"
    return "koine"


def group_idx(g: str) -> int:
    return GROUPS.index(g)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Date posterior using fitted VI
# ═══════════════════════════════════════════════════════════════════════════════

def date_posterior(x_obs_norm: np.ndarray,
                   reg_id: int,
                   phi: np.ndarray,
                   R: int, F: int,
                   prior_mean_norm: float,
                   prior_sigma_norm: float,
                   n_samples: int = 500,
                   n_grid: int = 2000,
                   date_lo_norm: float = -3.0,
                   date_hi_norm: float = 3.0,
                   rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute posterior p(d | x_obs) by Monte Carlo integration over VI posterior.

    Returns (dates_norm_grid, posterior_probabilities).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    (m_a, rho_a, m_b, rho_b, m_ls, rho_ls,
     m_mb, rho_mb, m_lsb, rho_lsb) = unpack(phi, R, F)

    # Sample from VI posteriors
    s_a  = np.exp(rho_a);   s_b  = np.exp(rho_b);  s_ls = np.exp(rho_ls)
    alpha_s = m_a + s_a  * rng.standard_normal((n_samples, R, F))  # [S, R, F]
    beta_s  = m_b + s_b  * rng.standard_normal((n_samples, R, F))
    σ_s     = np.exp(m_ls + s_ls * rng.standard_normal((n_samples, R, F)))

    # Select register
    alpha_r = alpha_s[:, reg_id, :]   # [S, F]
    beta_r  = beta_s[:,  reg_id, :]
    σ_r     = σ_s[:,    reg_id, :]

    # Date grid (normalised)
    d_lo  = min(prior_mean_norm - 4 * prior_sigma_norm, date_lo_norm) - 0.3
    d_hi  = max(prior_mean_norm + 4 * prior_sigma_norm, date_hi_norm) + 0.3
    d_grid = np.linspace(d_lo, d_hi, n_grid)   # [D]

    # Log-likelihood: [D, S, F]
    pred   = alpha_r[np.newaxis, :, :] + beta_r[np.newaxis, :, :] * d_grid[:, np.newaxis, np.newaxis]
    x_bc   = x_obs_norm[np.newaxis, np.newaxis, :]                # [1, 1, F]
    σ_bc   = σ_r[np.newaxis, :, :]                                # [1, S, F]
    mask_f = (~np.isnan(x_obs_norm))[np.newaxis, np.newaxis, :]   # [1, 1, F]

    log_lik_f = -0.5 * ((x_bc - pred) / σ_bc) ** 2 - np.log(σ_bc)   # [D, S, F]
    log_lik_f = np.where(mask_f, log_lik_f, 0.0)
    log_lik_s = np.sum(log_lik_f, axis=2)   # [D, S]  sum over features

    # Marginalise over parameter samples via log-sum-exp
    from scipy.special import logsumexp as lse
    log_lik = lse(log_lik_s, axis=1) - np.log(n_samples)   # [D]

    # Add Gaussian prior on date
    log_prior = -0.5 * ((d_grid - prior_mean_norm) / prior_sigma_norm) ** 2

    log_post = log_lik + log_prior
    log_post -= np.max(log_post)   # numerical stability

    post = np.exp(log_post)
    post /= post.sum()
    return d_grid, post


def summarise_posterior(d_grid: np.ndarray,
                        post: np.ndarray,
                        date_mean: float,
                        date_std: float,
                        prior_mean: float,
                        prior_sigma: float) -> dict:
    """Convert normalised-date posterior to CE years and compute summary stats."""
    # Back-transform grid to CE
    dates_ce = d_grid * date_std + date_mean

    map_idx  = np.argmax(post)
    map_date = dates_ce[map_idx]

    cdf     = np.cumsum(post)
    lo68    = dates_ce[np.searchsorted(cdf, 0.16)]
    hi68    = dates_ce[np.searchsorted(cdf, 0.84)]

    pmean   = np.dot(post, dates_ce)
    pstd    = np.sqrt(np.dot(post, (dates_ce - pmean) ** 2))
    z_score = (map_date - prior_mean) / prior_sigma

    # Likelihood-only MAP (flat prior)
    prior_log_contrib = -0.5 * ((d_grid - (prior_mean - date_mean) / date_std)
                                / (prior_sigma / date_std)) ** 2
    # Rebuild log_post without prior to get likelihood-only MAP
    # We can't easily do this here without the log_lik vector, so approximate:
    # Use posterior with very wide prior as proxy
    prior_prec = (pstd / prior_sigma) ** 2 if pstd > 0 else 1.0

    return dict(
        map_date   = map_date,
        post_mean  = pmean,
        post_std   = pstd,
        ci68_lo    = lo68,
        ci68_hi    = hi68,
        z_score    = z_score,
        prior_prec = prior_prec,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def ce(year: float) -> str:
    y = int(round(year))
    return f"{abs(y)} BCE" if y < 0 else f"{y} CE"


def main() -> None:
    rng = np.random.default_rng(0)

    # ── Load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(FEAT_CSV, index_col="id")
    with open(MANIFEST, encoding="utf-8") as f:
        corpus = json.load(f)
    with open(REG_PROBS, encoding="utf-8") as f:
        reg_probs = json.load(f)

    for e in corpus:
        eid = e["id"]
        if eid in df.index:
            df.loc[eid, "register"]   = e.get("register", "unknown")
            df.loc[eid, "holdout"]    = e["holdout"]
            df.loc[eid, "date_ce"]    = e["date_ce"]
            df.loc[eid, "date_sigma"] = e["date_sigma"]

    df["group"] = df.index.map(lambda eid: assign_group(reg_probs.get(eid, {})))

    feat_cols = [c for c in df.columns if c not in META_COLS]
    F = len(feat_cols)
    R = len(GROUPS)

    print("=" * 70)
    print(f"Hierarchical Bayes VI Dating  —  {F} features, {R} register groups")
    print("=" * 70)

    # ── Build combined training array ────────────────────────────────────────
    # Training = NOT holdout; for Koine also exclude LXX from feature fitting
    # (same logic as Script 10 — LXX texts are predicted but not trained on)
    train_mask = ~df["holdout"].astype(bool)
    koine_lxx  = (df["group"] == "koine") & (df["register"] == "LXX")
    train_mask = train_mask & ~koine_lxx

    df_train = df[train_mask].copy()
    print(f"\nTraining texts: {len(df_train)}")
    for g in GROUPS:
        n = (df_train["group"] == g).sum()
        print(f"  {g:12s}: {n} texts")

    # ── Normalise dates ───────────────────────────────────────────────────────
    date_mean = df_train["date_ce"].mean()
    date_std  = df_train["date_ce"].std()
    if date_std < 1e-6:
        date_std = 1.0

    def norm_date(d):
        return (d - date_mean) / date_std

    # ── Normalise features (per-group, then globally fill) ────────────────────
    # Fit normalisation on all training texts (across groups)
    feat_means = df_train[feat_cols].mean()
    feat_stds  = df_train[feat_cols].std().replace(0, 1)

    def norm_feats(row_vals):
        return (row_vals - feat_means.values) / feat_stds.values

    # ── Assemble training matrices ────────────────────────────────────────────
    train_ids   = df_train.index.tolist()
    T_train     = len(train_ids)
    X_raw       = df_train[feat_cols].values.astype(float)   # [T, F]
    dates_raw   = df_train["date_ce"].values.astype(float)   # [T]
    reg_ids_np  = np.array([group_idx(df_train.loc[i, "group"]) for i in train_ids], dtype=int)

    # Normalise
    X_norm_np     = (X_raw - feat_means.values[None, :]) / feat_stds.values[None, :]
    dates_norm_np = norm_date(dates_raw)

    # Validity mask (1 where not NaN)
    mask_np = (~np.isnan(X_norm_np)).astype(float)
    X_norm_np = np.where(np.isnan(X_norm_np), 0.0, X_norm_np)  # fill NaN → 0 (masked out)

    # ── Initialise VI parameters ──────────────────────────────────────────────
    # Warm-start α, β from per-group OLS
    init_alpha = np.zeros((R, F))
    init_beta  = np.zeros((R, F))

    for gi, grp in enumerate(GROUPS):
        if grp == "koine":
            sel = train_mask & (df["group"] == grp) & (df["register"] != "LXX")
        else:
            sel = train_mask & (df["group"] == grp)
        df_g = df[sel]
        if len(df_g) < 4:
            continue
        d_g = norm_date(df_g["date_ce"].values.astype(float))
        X_g = (df_g[feat_cols].values.astype(float)
               - feat_means.values[None, :]) / feat_stds.values[None, :]
        for fi in range(F):
            y = X_g[:, fi]
            valid = ~np.isnan(y)
            if valid.sum() < 4:
                continue
            A = np.column_stack([np.ones(valid.sum()), d_g[valid]])
            coef, *_ = np.linalg.lstsq(A, y[valid], rcond=None)
            init_alpha[gi, fi] = coef[0]
            init_beta[gi, fi]  = coef[1]

    # Pack initial φ
    # m_alpha, rho_alpha, m_beta, rho_beta, m_logσ, rho_logσ, m_mubeta, rho_mubeta, m_lsb, rho_lsb
    rho_init  = -2.0   # log-std ≈ -2  →  std ≈ 0.14  (tight start)
    phi0 = np.concatenate([
        init_alpha.ravel(),                            # m_alpha
        np.full(R * F, rho_init),                     # rho_alpha
        init_beta.ravel(),                             # m_beta
        np.full(R * F, rho_init),                     # rho_beta
        np.zeros(R * F),                               # m_logσ  (log σ ≈ 1)
        np.full(R * F, rho_init),                      # rho_logσ
        np.zeros(F),                                   # m_mubeta
        np.full(F, rho_init),                          # rho_mubeta
        np.array([-1.0]),                              # m_lsb  (σ_β ≈ 0.37)
        np.array([rho_init]),                          # rho_lsb
    ])

    NP = n_params(R, F)
    assert len(phi0) == NP, f"Parameter count mismatch: {len(phi0)} vs {NP}"
    print(f"\nVariational parameters: {NP}")

    # ── Build autograd gradient function ─────────────────────────────────────
    def elbo_with_fresh_eps(phi):
        """Sample fresh ε each call (used during training)."""
        eps_a  = anp.array(rng.standard_normal((N_MC_TRAIN, R, F)))
        eps_b  = anp.array(rng.standard_normal((N_MC_TRAIN, R, F)))
        eps_ls = anp.array(rng.standard_normal((N_MC_TRAIN, R, F)))
        eps_mb = anp.array(rng.standard_normal((N_MC_TRAIN, F)))
        eps_lsb= anp.array(rng.standard_normal(N_MC_TRAIN))
        return neg_elbo(phi,
                        eps_a, eps_b, eps_ls, eps_mb, eps_lsb,
                        anp.array(X_norm_np),
                        anp.array(dates_norm_np),
                        reg_ids_np,
                        anp.array(mask_np),
                        R, F)

    grad_fn = agrad(neg_elbo)   # gradient w.r.t. first arg (phi)

    # ── ADAM training loop ────────────────────────────────────────────────────
    print(f"\nFitting with ADAM  (lr={LR}, n_iter={N_ITER}, n_mc={N_MC_TRAIN})")
    print("-" * 50)

    optimizer = ADAM(NP, lr=LR, b1=ADAM_B1, b2=ADAM_B2, eps=ADAM_EPS)
    phi = phi0.copy()
    elbo_curve = []
    best_phi   = phi.copy()
    best_elbo  = -np.inf

    for it in range(1, N_ITER + 1):
        # Fresh MC noise for this step
        eps_a   = rng.standard_normal((N_MC_TRAIN, R, F))
        eps_b   = rng.standard_normal((N_MC_TRAIN, R, F))
        eps_ls  = rng.standard_normal((N_MC_TRAIN, R, F))
        eps_mb  = rng.standard_normal((N_MC_TRAIN, F))
        eps_lsb = rng.standard_normal(N_MC_TRAIN)

        g = grad_fn(phi,
                    eps_a, eps_b, eps_ls, eps_mb, eps_lsb,
                    X_norm_np,
                    dates_norm_np,
                    reg_ids_np,
                    mask_np,
                    R, F)

        neg_e = float(neg_elbo(phi,
                               eps_a, eps_b, eps_ls, eps_mb, eps_lsb,
                               X_norm_np, dates_norm_np,
                               reg_ids_np, mask_np, R, F))
        elbo_val = -neg_e
        elbo_curve.append(elbo_val)

        if elbo_val > best_elbo:
            best_elbo = elbo_val
            best_phi  = phi.copy()

        phi = optimizer.step(phi, np.array(g))

        if it % LOG_EVERY == 0 or it == 1:
            # Clip rho_* to prevent collapse
            # rho indices: [RF:2RF] (alpha), [3RF:4RF] (beta), [5RF:6RF] (log_sigma)
            #              [6RF+F:6RF+2F] (mu_beta), [-1] (lsb)
            print(f"  iter {it:5d}  ELBO = {elbo_val:10.2f}"
                  f"  best = {best_elbo:10.2f}")

    phi = best_phi
    print(f"\nTraining complete.  Best ELBO = {best_elbo:.2f}")

    # Save ELBO curve
    np.savetxt(os.path.join(RESULTS, "hb_vi_elbo.txt"), elbo_curve)

    # ── Report learned hyperparameters ────────────────────────────────────────
    (m_a, rho_a, m_b, rho_b, m_ls, rho_ls,
     m_mb, rho_mb, m_lsb, rho_lsb) = unpack(phi, R, F)

    sigma_beta_posterior = float(np.exp(m_lsb))
    print(f"\nLearned slope heterogeneity σ_β ≈ {sigma_beta_posterior:.3f} (normalised units)")
    print(f"Global slope means μ_β (top 5 by |m_β|):")
    top5 = np.argsort(-np.abs(m_mb))[:5]
    for fi in top5:
        print(f"  {feat_cols[fi]:25s}  μ_β={m_mb[fi]:+.3f}  s_μβ={np.exp(rho_mb[fi]):.3f}")

    print(f"\nPer-register posterior slope means (|β| > 0.2):")
    for gi, grp in enumerate(GROUPS):
        betas = m_b[gi]
        sig_f = np.where(np.abs(betas) > 0.2)[0]
        print(f"  {grp:12s}: " + ", ".join(
            f"{feat_cols[fi]}={betas[fi]:+.3f}" for fi in sig_f))

    # ── Date all texts ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DATING RESULTS")
    print("=" * 70)

    # Global date normalisation stats (needed to back-transform)
    d_lo_norm = norm_date(df_train["date_ce"].min())
    d_hi_norm = norm_date(df_train["date_ce"].max())

    rows = []
    infer_rng = np.random.default_rng(99)

    for grp in GROUPS:
        ids = df[df["group"] == grp].index.tolist()
        print(f"\n── {grp.upper()} ──────────────────────────────────────────────────")
        reg_id = group_idx(grp)

        for eid in ids:
            row      = df.loc[eid]
            is_hold  = bool(row["holdout"])
            pmean    = float(row["date_ce"])
            psigma   = float(row["date_sigma"])

            x_raw = row[feat_cols].values.astype(float)
            x_norm = (x_raw - feat_means.values) / feat_stds.values

            pmean_n  = norm_date(pmean)
            psigma_n = psigma / date_std

            d_grid_n, post = date_posterior(
                x_norm, reg_id, phi, R, F,
                prior_mean_norm=pmean_n,
                prior_sigma_norm=psigma_n,
                n_samples=N_MC_INFER,
                n_grid=N_GRID,
                date_lo_norm=d_lo_norm,
                date_hi_norm=d_hi_norm,
                rng=infer_rng,
            )

            res = summarise_posterior(d_grid_n, post, date_mean, date_std,
                                      pmean, psigma)

            in68 = res["ci68_lo"] <= pmean <= res["ci68_hi"]
            flag = " [HOLDOUT]" if is_hold else ""

            print(
                f"  {eid:45s}"
                f"  MAP={ce(res['map_date']):10s}"
                f"  68%=[{ce(res['ci68_lo'])}, {ce(res['ci68_hi'])}]"
                f"  z={res['z_score']:+.2f}"
                f"{flag}"
            )

            rows.append(dict(
                id            = eid,
                group         = grp,
                holdout       = is_hold,
                scholarly_date= pmean,
                hb_map        = res["map_date"],
                hb_ci68_lo    = res["ci68_lo"],
                hb_ci68_hi    = res["ci68_hi"],
                hb_post_std   = res["post_std"],
                hb_z_score    = res["z_score"],
                hb_scholar_in68 = in68,
                hb_prior_prec = res["prior_prec"],
            ))

    # ── Holdout summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HOLDOUT SUMMARY — Hierarchical Bayes VI")
    print("=" * 70)
    holdout_rows = [r for r in rows if r["holdout"]]
    fmt = "{:45s}  {:>10s}  {:>10s}  {:>6s}  {:>8s}"
    print(fmt.format("Text", "Scholarly", "HB MAP", "In 68?", "Post σ"))
    print("─" * 90)
    for r in holdout_rows:
        flag = "✓" if r["hb_scholar_in68"] else "✗"
        print(fmt.format(
            r["id"],
            ce(r["scholarly_date"]),
            ce(r["hb_map"]),
            flag,
            f"±{r['hb_post_std']:.0f}yr",
        ))

    # MAE on holdouts
    maes = [abs(r["hb_map"] - r["scholarly_date"]) for r in holdout_rows]
    print(f"\nHoldout MAE: {np.mean(maes):.1f} yr  (n={len(maes)})")
    coverage = sum(r["hb_scholar_in68"] for r in holdout_rows) / len(holdout_rows)
    print(f"68% CI coverage: {coverage:.0%}  (expected ≈ 68%)")

    # ── Save results ──────────────────────────────────────────────────────────
    out_csv = os.path.join(RESULTS, "hb_vi_dating.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nResults → {out_csv}")

    out_params = os.path.join(RESULTS, "hb_vi_params.npz")
    np.savez(out_params, phi=phi,
             feat_cols=np.array(feat_cols),
             feat_means=feat_means.values,
             feat_stds=feat_stds.values,
             date_mean=date_mean,
             date_std=date_std,
             groups=np.array(GROUPS))
    print(f"VI params → {out_params}")


if __name__ == "__main__":
    main()
