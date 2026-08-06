"""
01_hb_vi_dating.py
==================
Hierarchical Bayesian dating model fitted with Variational Inference (VI)
for Biblical Hebrew texts, using the BHSA/ETCBC corpus features.

Key adaptations from the Greek version
----------------------------------------
  * date_bce convention  : Hebrew features use positive BCE numbers; internally
    we negate to date_ce = -date_bce so that "later = larger" throughout.
  * Register groups      : SBH  /  Transitional  /  LBH
    (Standard Biblical Hebrew, Transitional, Late Biblical Hebrew)
  * Jeremiah replacement : The whole-book Jeremiah row is dropped from
    training and replaced by Jer_oracle (chapters without Dtr additions),
    extracted fresh from BHSA.  Jer_DTR is dated as a test-only text.
  * Sub-source dating    : 8 literary sub-units (D_Code, D_Frame, D_Song,
    Lev_Holiness, Lev_Priestly, Jer_DTR, Song_Sea, Song_Deborah) are
    extracted from BHSA and dated with wide scholarly-consensus priors.
  * Unknown-register texts : D_source, P_source, JE_source from the main
    feature matrix are dated using a best-group approach (highest marginal
    posterior wins).

Model
-----
  β_{r,f} | μ_β_f, σ_β  ~  N(μ_β_f, σ_β²)
  μ_β_f                  ~  N(0, 1²)
  σ_β                    ~  HalfNormal(0.5)
  α_{r,f}                ~  N(0, 3²)
  log σ_obs_{r,f}        ~  N(0, 0.5²)
  x_{t,f} | d_t          ~  N(α_{r_t,f} + β_{r_t,f}·d_t, σ_obs_{r_t,f}²)

Output
------
  results/hb_vi_dating.csv    — per-text dating results
  results/hb_vi_params.npz    — fitted VI parameters
  results/hb_vi_elbo.txt      — ELBO training curve
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from scipy import stats
from scipy.special import logsumexp as lse

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

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
PARENT     = os.path.dirname(HERE)
FEAT_CSV   = os.path.join(PARENT, "data", "feature_matrix_hbvi_v2.csv")
EXTR_CSV   = os.path.join(HERE,   "results", "extracted_features_v2.csv")
RESULTS = os.path.join(HERE, "results_v2")
os.makedirs(RESULTS, exist_ok=True)

# ── Hyper-settings ─────────────────────────────────────────────────────────────
N_MC_TRAIN = 32
N_MC_INFER = 500
N_ITER     = 3000
LR         = 0.02
ADAM_B1    = 0.9
ADAM_B2    = 0.999
ADAM_EPS   = 1e-8
N_GRID     = 2000
LOG_EVERY  = 200

# ── Register groups ────────────────────────────────────────────────────────────
GROUPS = ["SBH", "Transitional", "LBH"]

META_COLS = {"date_bce", "date_ce", "date_sigma", "register", "genre",
             "holdout", "group", "n_words"}

# ── Sub-source scholarly metadata ─────────────────────────────────────────────
# (date_bce_est, sigma_bce, group, description)
# Wide sigmas = genuinely uncertain; date_ce = -date_bce
SUBSOURCE_META = {
    "Jer_DTR":      (570,  60, "Transitional", "Jeremiah Deuteronomic prose"),
    "D_Code":       (620,  40, "SBH",          "Deuteronomic Code (Deut 12-26)"),
    "D_Frame":      (620,  40, "SBH",          "Deuteronomic framing narrative"),
    "D_Song":       (900, 300, "SBH",          "Song of Moses (Deut 32)"),
    "Lev_Holiness": (580,  80, "SBH",          "Holiness Code (Lev 17-26)"),
    "Lev_Priestly": (520,  80, "SBH",          "Priestly source (Lev, non-H)"),
    "Song_Sea":     (1100, 300, "SBH",         "Song of the Sea (Ex 15)"),
    "Song_Deborah": (1150, 200, "SBH",         "Song of Deborah (Judg 5)"),
}

# Unknown-register texts from main matrix → group assignment for dating
UNKNOWN_GROUP = {
    "D_source":  "SBH",
    "P_source":  "SBH",
    "JE_source": "SBH",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. ADAM optimiser
# ══════════════════════════════════════════════════════════════════════════════

class ADAM:
    def __init__(self, n_params: int, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.lr  = lr;  self.b1  = b1;  self.b2  = b2;  self.eps = eps
        self.t   = 0
        self.m   = np.zeros(n_params)
        self.v   = np.zeros(n_params)

    def step(self, phi, grad_val):
        self.t  += 1
        self.m   = self.b1 * self.m + (1 - self.b1) * grad_val
        self.v   = self.b2 * self.v + (1 - self.b2) * grad_val ** 2
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return phi - self.lr * mh / (np.sqrt(vh) + self.eps)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Parameter packing / unpacking
# ══════════════════════════════════════════════════════════════════════════════

def n_params(R: int, F: int) -> int:
    return 6 * R * F + 2 * F + 2


def unpack(phi, R: int, F: int):
    RF = R * F; i = 0
    m_a    = phi[i:i+RF].reshape(R, F); i += RF
    rho_a  = phi[i:i+RF].reshape(R, F); i += RF
    m_b    = phi[i:i+RF].reshape(R, F); i += RF
    rho_b  = phi[i:i+RF].reshape(R, F); i += RF
    m_ls   = phi[i:i+RF].reshape(R, F); i += RF
    rho_ls = phi[i:i+RF].reshape(R, F); i += RF
    m_mb   = phi[i:i+F];                i += F
    rho_mb = phi[i:i+F];                i += F
    m_lsb  = phi[i];                    i += 1
    rho_lsb= phi[i];                    i += 1
    return m_a, rho_a, m_b, rho_b, m_ls, rho_ls, m_mb, rho_mb, m_lsb, rho_lsb


# ══════════════════════════════════════════════════════════════════════════════
# 3. ELBO  (autograd-differentiable)
# ══════════════════════════════════════════════════════════════════════════════

def kl_gauss(m_q, s_q, m_p, s_p):
    return (anp.log(s_p) - anp.log(s_q)
            + (s_q**2 + (m_q - m_p)**2) / (2.0 * s_p**2) - 0.5)


def neg_elbo(phi, eps_a, eps_b, eps_ls, eps_mb, eps_lsb,
             X_norm, dates_norm, reg_ids, mask, R, F):
    (m_a, rho_a, m_b, rho_b, m_ls, rho_ls,
     m_mb, rho_mb, m_lsb, rho_lsb) = unpack(phi, R, F)

    s_a   = anp.exp(rho_a);   s_b   = anp.exp(rho_b)
    s_ls  = anp.exp(rho_ls);  s_mb  = anp.exp(rho_mb)
    s_lsb = anp.exp(rho_lsb)

    alpha_s   = m_a  + s_a  * eps_a
    beta_s    = m_b  + s_b  * eps_b
    logσ_s    = m_ls + s_ls * eps_ls
    mu_beta_s = m_mb + s_mb * eps_mb
    lsb_s     = m_lsb + s_lsb * eps_lsb

    σ_s          = anp.exp(logσ_s)
    sigma_beta_s = anp.exp(lsb_s)

    alpha_t = alpha_s[:, reg_ids, :]
    beta_t  = beta_s[:,  reg_ids, :]
    σ_t     = σ_s[:,    reg_ids, :]

    pred       = alpha_t + beta_t * dates_norm[anp.newaxis, :, anp.newaxis]
    resid      = X_norm[anp.newaxis, :, :] - pred
    log_lik_pf = -0.5 * (resid / σ_t)**2 - anp.log(σ_t)
    log_lik_pf = log_lik_pf * mask[anp.newaxis, :, :]
    log_lik    = anp.sum(anp.mean(log_lik_pf, axis=0))

    kl_alpha  = anp.sum(kl_gauss(m_a,   s_a,   0.0, 3.0))
    kl_logσ   = anp.sum(kl_gauss(m_ls,  s_ls,  0.0, 0.5))
    kl_mubeta = anp.sum(kl_gauss(m_mb,  s_mb,  0.0, 1.0))
    kl_lsb    = kl_gauss(m_lsb, s_lsb, -1.0, 0.5)

    m_b_bc = m_b[anp.newaxis, :, :]
    s_b_bc = s_b[anp.newaxis, :, :]
    mb_bc  = mu_beta_s[:, anp.newaxis, :]
    sb_bc  = sigma_beta_s[:, anp.newaxis, anp.newaxis]
    kl_beta_per = anp.sum(kl_gauss(m_b_bc, s_b_bc, mb_bc, sb_bc), axis=(1, 2))
    kl_beta = anp.mean(kl_beta_per)

    elbo = log_lik - kl_alpha - kl_logσ - kl_mubeta - kl_lsb - kl_beta
    return -elbo


# ══════════════════════════════════════════════════════════════════════════════
# 4. Date posterior
# ══════════════════════════════════════════════════════════════════════════════

def date_posterior(x_obs_norm, reg_id, phi, R, F,
                   prior_mean_norm, prior_sigma_norm,
                   n_samples=500, n_grid=2000,
                   date_lo_norm=-3.0, date_hi_norm=3.0,
                   rng=None):
    if rng is None:
        rng = np.random.default_rng(42)

    (m_a, rho_a, m_b, rho_b, m_ls, rho_ls,
     m_mb, rho_mb, m_lsb, rho_lsb) = unpack(phi, R, F)

    s_a  = np.exp(rho_a);  s_b  = np.exp(rho_b);  s_ls = np.exp(rho_ls)
    alpha_s = m_a + s_a  * rng.standard_normal((n_samples, R, F))
    beta_s  = m_b + s_b  * rng.standard_normal((n_samples, R, F))
    σ_s     = np.exp(m_ls + s_ls * rng.standard_normal((n_samples, R, F)))

    alpha_r = alpha_s[:, reg_id, :]
    beta_r  = beta_s[:,  reg_id, :]
    σ_r     = σ_s[:,    reg_id, :]

    d_lo   = min(prior_mean_norm - 4*prior_sigma_norm, date_lo_norm) - 0.3
    d_hi   = max(prior_mean_norm + 4*prior_sigma_norm, date_hi_norm) + 0.3
    d_grid = np.linspace(d_lo, d_hi, n_grid)

    pred   = alpha_r[np.newaxis,:,:] + beta_r[np.newaxis,:,:] * d_grid[:,np.newaxis,np.newaxis]
    x_bc   = x_obs_norm[np.newaxis, np.newaxis, :]
    σ_bc   = σ_r[np.newaxis, :, :]
    mask_f = (~np.isnan(x_obs_norm))[np.newaxis, np.newaxis, :]

    log_lik_f = -0.5 * ((x_bc - pred) / σ_bc)**2 - np.log(σ_bc)
    log_lik_f = np.where(mask_f, log_lik_f, 0.0)
    log_lik_s = np.sum(log_lik_f, axis=2)
    log_lik   = lse(log_lik_s, axis=1) - np.log(n_samples)

    log_prior = -0.5 * ((d_grid - prior_mean_norm) / prior_sigma_norm)**2
    log_post  = log_lik + log_prior
    log_post -= np.max(log_post)

    post = np.exp(log_post)
    post /= post.sum()
    return d_grid, post, log_lik  # return log_lik for best-group selection


def best_group_date_posterior(x_obs_norm, phi, R, F,
                              prior_mean_norm, prior_sigma_norm,
                              n_samples=500, n_grid=2000,
                              date_lo_norm=-3.0, date_hi_norm=3.0, rng=None):
    """Try all 3 groups; return (best_group_idx, d_grid, post) for the one
    with the highest marginal log-likelihood."""
    if rng is None:
        rng = np.random.default_rng(42)
    best_gi, best_marg, best_grid, best_post = 0, -np.inf, None, None
    for gi in range(R):
        d_grid, post, log_lik = date_posterior(
            x_obs_norm, gi, phi, R, F,
            prior_mean_norm, prior_sigma_norm,
            n_samples=n_samples, n_grid=n_grid,
            date_lo_norm=date_lo_norm, date_hi_norm=date_hi_norm,
            rng=rng)
        marg = lse(log_lik)  # marginal log-likelihood for this group
        if marg > best_marg:
            best_marg  = marg
            best_gi    = gi
            best_grid  = d_grid
            best_post  = post
    return best_gi, best_grid, best_post


def summarise_posterior(d_grid, post, date_mean, date_std,
                        prior_mean, prior_sigma):
    """Back-transform normalised-date posterior to CE/BCE years."""
    dates_ce = d_grid * date_std + date_mean

    map_idx  = np.argmax(post)
    map_date = dates_ce[map_idx]

    cdf   = np.cumsum(post)
    lo68  = dates_ce[np.searchsorted(cdf, 0.16)]
    hi68  = dates_ce[np.searchsorted(cdf, 0.84)]

    pmean = np.dot(post, dates_ce)
    pstd  = np.sqrt(np.dot(post, (dates_ce - pmean)**2))
    z     = (map_date - prior_mean) / prior_sigma

    return dict(map_date=map_date, post_mean=pmean, post_std=pstd,
                ci68_lo=lo68, ci68_hi=hi68, z_score=z)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Helpers
# ══════════════════════════════════════════════════════════════════════════════

def bce(year_ce: float) -> str:
    y = int(round(-year_ce))
    if y > 0:
        return f"{y} BCE"
    return f"{-y} CE"


def group_idx(g: str) -> int:
    return GROUPS.index(g)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(0)

    # ── Load feature matrices ─────────────────────────────────────────────────
    df_main = pd.read_csv(FEAT_CSV, index_col="id")
    df_extr = pd.read_csv(EXTR_CSV, index_col="id")

    # Convert date_bce → date_ce (negate)
    df_main["date_ce"] = -df_main["date_bce"].astype(float)

    # Drop full-book Jeremiah; replace with Jer_oracle from extracted features
    if "Jeremiah" in df_main.index:
        df_main = df_main.drop("Jeremiah")
        print("Dropped Jeremiah; adding Jer_oracle from BHSA extraction.")

    # Build Jer_oracle row to insert into main training set
    jer_oracle_features = df_extr.loc["Jer_oracle"].drop("n_words", errors="ignore")
    jer_row = jer_oracle_features.copy()
    jer_row["date_bce"]    = 605
    jer_row["date_ce"]     = -605.0
    jer_row["date_sigma"]  = 30
    jer_row["register"]    = "SBH"
    jer_row["genre"]       = "prophecy"
    jer_row["holdout"]     = True   # AUDIT FIX: was False -> HB-VI trained on its own holdout
    df_main = pd.concat([df_main, jer_row.to_frame().T.rename(index={"Jer_oracle": "Jer_oracle"})])
    df_main.index.name = "id"

    # Assign groups from register column
    reg_to_group = {"SBH": "SBH", "Transitional": "Transitional", "LBH": "LBH",
                    "unknown": None}
    df_main["group"] = df_main["register"].map(reg_to_group)

    # Feature columns (same in both matrices, minus metadata)
    feat_cols = [c for c in df_extr.columns if c != "n_words" and c not in META_COLS]
    # Keep only feature cols present in main df too
    feat_cols = [c for c in feat_cols if c in df_main.columns]
    F = len(feat_cols)
    R = len(GROUPS)

    print("=" * 70)
    print(f"Hebrew HB-VI Dating  —  {F} features, {R} register groups")
    print("=" * 70)

    # ── Training set: known-register, non-holdout texts ───────────────────────
    df_known   = df_main[df_main["group"].notna()].copy()
    train_mask = (~df_known["holdout"].astype(bool)) & (df_known["group"].notna())
    df_train   = df_known[train_mask].copy()

    print(f"\nTraining texts: {len(df_train)}")
    for g in GROUPS:
        texts = df_train[df_train["group"] == g].index.tolist()
        print(f"  {g:14s} ({len(texts)}): {', '.join(texts)}")

    # ── Date normalisation ────────────────────────────────────────────────────
    date_mean = float(df_train["date_ce"].mean())
    date_std  = float(df_train["date_ce"].std())
    if date_std < 1e-6:
        date_std = 1.0

    def norm_date(d):
        return (d - date_mean) / date_std

    # ── Feature normalisation ─────────────────────────────────────────────────
    feat_means = df_train[feat_cols].apply(pd.to_numeric, errors="coerce").mean()
    feat_stds  = df_train[feat_cols].apply(pd.to_numeric, errors="coerce").std().replace(0, 1)

    # ── Training matrices ─────────────────────────────────────────────────────
    train_ids    = df_train.index.tolist()
    X_raw        = df_train[feat_cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
    dates_raw    = pd.to_numeric(df_train["date_ce"], errors="coerce").values.astype(np.float64)
    reg_ids_np   = np.array([group_idx(df_train.loc[i, "group"]) for i in train_ids], dtype=int)

    X_norm_np     = (X_raw - feat_means.values[None,:]) / feat_stds.values[None,:]
    X_norm_np     = X_norm_np.astype(np.float64)
    dates_norm_np = norm_date(dates_raw).astype(np.float64)
    mask_np       = (~np.isnan(X_norm_np)).astype(np.float64)
    X_norm_np     = np.where(np.isnan(X_norm_np), 0.0, X_norm_np)

    # ── Warm-start α, β from per-group OLS ───────────────────────────────────
    init_alpha = np.zeros((R, F))
    init_beta  = np.zeros((R, F))
    for gi, grp in enumerate(GROUPS):
        df_g = df_train[df_train["group"] == grp]
        if len(df_g) < 4:
            continue
        d_g = norm_date(df_g["date_ce"].values.astype(float))
        X_g = (df_g[feat_cols].values.astype(float) - feat_means.values[None,:]) / feat_stds.values[None,:]
        for fi in range(F):
            y = X_g[:, fi]; valid = ~np.isnan(y)
            if valid.sum() < 4:
                continue
            A = np.column_stack([np.ones(valid.sum()), d_g[valid]])
            coef, *_ = np.linalg.lstsq(A, y[valid], rcond=None)
            init_alpha[gi, fi] = coef[0]
            init_beta[gi, fi]  = coef[1]

    # ── Initial VI parameter vector ───────────────────────────────────────────
    rho_init = -2.0
    phi0 = np.concatenate([
        init_alpha.ravel(),
        np.full(R * F, rho_init),
        init_beta.ravel(),
        np.full(R * F, rho_init),
        np.zeros(R * F),
        np.full(R * F, rho_init),
        np.zeros(F),
        np.full(F, rho_init),
        np.array([-1.0]),
        np.array([rho_init]),
    ])
    NP = n_params(R, F)
    assert len(phi0) == NP, f"Param count mismatch: {len(phi0)} vs {NP}"
    print(f"\nVariational parameters: {NP}")

    # ── ADAM training ─────────────────────────────────────────────────────────
    grad_fn   = agrad(neg_elbo)
    optimizer = ADAM(NP, lr=LR, b1=ADAM_B1, b2=ADAM_B2, eps=ADAM_EPS)
    phi       = phi0.copy()
    elbo_curve= []
    best_phi  = phi.copy()
    best_elbo = -np.inf

    print(f"\nFitting with ADAM  (lr={LR}, n_iter={N_ITER}, n_mc={N_MC_TRAIN})")
    print("-" * 50)

    for it in range(1, N_ITER + 1):
        eps_a   = rng.standard_normal((N_MC_TRAIN, R, F))
        eps_b   = rng.standard_normal((N_MC_TRAIN, R, F))
        eps_ls  = rng.standard_normal((N_MC_TRAIN, R, F))
        eps_mb  = rng.standard_normal((N_MC_TRAIN, F))
        eps_lsb = rng.standard_normal(N_MC_TRAIN)

        args = (eps_a, eps_b, eps_ls, eps_mb, eps_lsb,
                X_norm_np, dates_norm_np, reg_ids_np, mask_np, R, F)

        g      = grad_fn(phi, *args)
        neg_e  = float(neg_elbo(phi, *args))
        elbo_v = -neg_e
        elbo_curve.append(elbo_v)

        if elbo_v > best_elbo:
            best_elbo = elbo_v
            best_phi  = phi.copy()

        phi = optimizer.step(phi, np.array(g))

        if it % LOG_EVERY == 0 or it == 1:
            print(f"  iter {it:5d}  ELBO = {elbo_v:10.2f}  best = {best_elbo:10.2f}")

    phi = best_phi
    print(f"\nTraining complete.  Best ELBO = {best_elbo:.2f}")
    np.savetxt(os.path.join(RESULTS, "hb_vi_elbo.txt"), elbo_curve)

    # ── Learned parameters ────────────────────────────────────────────────────
    (m_a, rho_a, m_b, rho_b, m_ls, rho_ls,
     m_mb, rho_mb, m_lsb, rho_lsb) = unpack(phi, R, F)

    sigma_beta = float(np.exp(m_lsb))
    print(f"\nLearned σ_β ≈ {sigma_beta:.3f}  (normalised units)")
    print("Top-5 global slope means |μ_β|:")
    for fi in np.argsort(-np.abs(m_mb))[:5]:
        print(f"  {feat_cols[fi]:28s}  μ_β={m_mb[fi]:+.3f}")

    print("\nPer-register significant slopes (|β| > 0.2):")
    for gi, grp in enumerate(GROUPS):
        sig = np.where(np.abs(m_b[gi]) > 0.2)[0]
        if len(sig):
            print(f"  {grp:14s}: " +
                  ", ".join(f"{feat_cols[fi]}={m_b[gi,fi]:+.3f}" for fi in sig))

    # ── Date all texts ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DATING RESULTS")
    print("=" * 70)

    d_lo_norm = norm_date(df_train["date_ce"].min())
    d_hi_norm = norm_date(df_train["date_ce"].max())
    infer_rng = np.random.default_rng(99)
    rows = []

    def do_date(eid, x_raw, group, prior_bce, prior_sigma_bce, is_holdout):
        """Date a single text; returns result dict."""
        prior_ce    = -float(prior_bce)
        prior_sig   = float(prior_sigma_bce)
        x_norm      = (x_raw - feat_means.values) / feat_stds.values
        pmean_n     = norm_date(prior_ce)
        psigma_n    = prior_sig / date_std
        reg_id      = group_idx(group)

        d_grid_n, post, _ = date_posterior(
            x_norm, reg_id, phi, R, F,
            pmean_n, psigma_n,
            n_samples=N_MC_INFER, n_grid=N_GRID,
            date_lo_norm=d_lo_norm, date_hi_norm=d_hi_norm,
            rng=infer_rng)

        res = summarise_posterior(d_grid_n, post, date_mean, date_std,
                                  prior_ce, prior_sig)
        in68 = res["ci68_lo"] <= prior_ce <= res["ci68_hi"]
        flag = " [HOLDOUT]" if is_holdout else ""
        print(f"  {eid:30s}  MAP={bce(res['map_date']):10s}"
              f"  68%=[{bce(res['ci68_lo'])},{bce(res['ci68_hi'])}]"
              f"  z={res['z_score']:+.2f}  grp={group}{flag}")
        return dict(
            id=eid, group=group, holdout=is_holdout,
            scholarly_date_bce=prior_bce,
            scholarly_date_ce=prior_ce,
            hb_map_ce=res["map_date"],
            hb_map_bce=-res["map_date"],
            hb_ci68_lo_ce=res["ci68_lo"],
            hb_ci68_hi_ce=res["ci68_hi"],
            hb_post_std=res["post_std"],
            hb_z_score=res["z_score"],
            hb_scholar_in68=in68,
        )

    # 1. Known-register texts from main matrix
    for grp in GROUPS:
        ids = df_known[df_known["group"] == grp].index.tolist()
        print(f"\n── {grp} ──────────────────────────────────────────────────────")
        for eid in ids:
            row_data = df_known.loc[eid]
            is_hold  = bool(row_data["holdout"])
            pbce     = float(row_data["date_bce"])
            psig     = float(row_data["date_sigma"])
            x_raw    = row_data[feat_cols].values.astype(float)
            rows.append(do_date(eid, x_raw, grp, pbce, psig, is_hold))

    # 2. Unknown-register texts from main matrix (best-group)
    print(f"\n── UNKNOWN REGISTER (best-group) ───────────────────────────────")
    df_unk = df_main[df_main["group"].isna()].copy()
    for eid, row_data in df_unk.iterrows():
        pbce  = float(row_data["date_bce"])
        psig  = float(row_data["date_sigma"])
        # AUDIT FIX: out-of-sample targets carry no scholarly date; give them
        # the agnostic Mode-B prior rather than a NaN prior (which silently
        # produced an all-NaN grid and an empty posterior).
        import math as _m
        if _m.isnan(pbce):
            pbce, psig = 575.0, 400.0
        if _m.isnan(psig) or psig <= 0:
            psig = 400.0
        x_raw = row_data[feat_cols].values.astype(float)
        x_norm = (x_raw - feat_means.values) / feat_stds.values
        pmean_n  = norm_date(-pbce)
        psigma_n = psig / date_std
        best_gi, d_grid_n, post = best_group_date_posterior(
            x_norm, phi, R, F, pmean_n, psigma_n,
            n_samples=N_MC_INFER, n_grid=N_GRID,
            date_lo_norm=d_lo_norm, date_hi_norm=d_hi_norm,
            rng=infer_rng)
        best_grp = GROUPS[best_gi]
        res = summarise_posterior(d_grid_n, post, date_mean, date_std, -pbce, psig)
        in68 = res["ci68_lo"] <= -pbce <= res["ci68_hi"]
        print(f"  {eid:30s}  MAP={bce(res['map_date']):10s}"
              f"  68%=[{bce(res['ci68_lo'])},{bce(res['ci68_hi'])}]"
              f"  best_grp={best_grp}")
        rows.append(dict(
            id=eid, group=best_grp, holdout=False,
            scholarly_date_bce=pbce,
            scholarly_date_ce=-pbce,
            hb_map_ce=res["map_date"],
            hb_map_bce=-res["map_date"],
            hb_ci68_lo_ce=res["ci68_lo"],
            hb_ci68_hi_ce=res["ci68_hi"],
            hb_post_std=res["post_std"],
            hb_z_score=res["z_score"],
            hb_scholar_in68=in68,
        ))

    # 3. Extracted sub-sources (Jer_DTR, D_Code, etc.)
    print(f"\n── SUB-SOURCES (extracted from BHSA) ──────────────────────────")
    for eid, (pbce, psig, grp, desc) in SUBSOURCE_META.items():
        if eid not in df_extr.index:
            print(f"  WARNING: {eid} not in extracted features; skipping.")
            continue
        x_raw = df_extr.loc[eid, feat_cols].values.astype(float)
        rows.append(do_date(eid, x_raw, grp, pbce, psig, is_holdout=False))

    # ── Holdout summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("HOLDOUT SUMMARY — Hierarchical Bayes VI")
    print("=" * 70)
    holdout_rows = [r for r in rows if r["holdout"]]
    fmt = "{:30s}  {:>10s}  {:>10s}  {:>6s}  {:>8s}"
    print(fmt.format("Text", "Scholarly", "HB MAP", "In 68?", "Post σ"))
    print("─" * 80)
    for r in holdout_rows:
        flag = "✓" if r["hb_scholar_in68"] else "✗"
        print(fmt.format(
            r["id"],
            f"{r['scholarly_date_bce']:.0f} BCE",
            bce(r["hb_map_ce"]),
            flag,
            f"±{r['hb_post_std']:.0f}yr",
        ))

    maes = [abs(r["hb_map_ce"] - r["scholarly_date_ce"]) for r in holdout_rows]
    print(f"\nHoldout MAE: {np.mean(maes):.1f} yr  (n={len(maes)})")
    cov = sum(r["hb_scholar_in68"] for r in holdout_rows) / len(holdout_rows)
    print(f"68% CI coverage: {cov:.0%}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_csv = os.path.join(RESULTS, "hb_vi_dating.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nResults → {out_csv}")

    np.savez(os.path.join(RESULTS, "hb_vi_params.npz"),
             phi=phi,
             feat_cols=np.array(feat_cols),
             feat_means=feat_means.values,
             feat_stds=feat_stds.values,
             date_mean=np.array(date_mean),
             date_std=np.array(date_std),
             groups=np.array(GROUPS))
    print(f"VI params → {os.path.join(RESULTS, 'hb_vi_params.npz')}")


if __name__ == "__main__":
    main()
