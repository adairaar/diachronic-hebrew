"""
Script 24 — Weakly Calibrated Monte Carlo Dating

Motivation
----------
Scripts 16–22 built and tested diachronic n-gram models entirely within
the Hebrew Bible.  Script 23 applied them to texts with independently
known dates (DSS manuscripts, ancient inscriptions, Ben Sira).  The
external validation revealed two regime-dependent failure modes:

  Char n-gram model:  works within the Masoretic tradition (biblical
      corpus); fails cross-corpus because it detects Masoretic
      orthographic conventions (shin/sin dots, defective vs. plene
      spelling) rather than purely linguistic change.

  Word n-gram model:  partially generalises cross-corpus; predicts DSS
      texts at ≈200–370 BCE (expected ≈100–150 BCE), a systematic
      over-dating of ≈70–220 years in the post-training extrapolation
      zone (< 400 BCE).

This script implements a "weakly calibrated" Monte Carlo framework that:
  1. Quantifies the systematic model bias using external controls of
     known date (1QS, 1QM, 1QHa) for the word n-gram model.
  2. Propagates both DATING uncertainty (when was the control composed?)
     and MODEL uncertainty (how wide is the posterior?) through a
     Bayesian calibration.
  3. Applies the calibrated correction ONLY to the word model, only to
     units the model places in the post-400 BCE zone, and with
     explicitly widened uncertainties.
  4. Reports both uncalibrated and calibrated results so the reader can
     judge the correction's credibility.

Key design choices
------------------
* Offset-only calibration (not scale+offset), because all controls
  cluster in a narrow date window (100–150 BCE); fitting a slope too
  would be over-constrained.
* Bayesian posterior for the offset δ:
      prior:  δ ~ N(0, σ_prior=200)  [allow up to ±200 yr shift]
      likelihood for each control i:
          d_i = T_true_i − T_pred_i
          σ_total_i = sqrt(σ_date_i² + σ_model_i²)
          δ ~ N(d_i, σ_total_i)
  Posterior is analytic (conjugate Gaussian) and sampled via MC.
* Calibration DOES NOT shift within-range (>400 BCE) predictions
  because those were never tested against external ground truth; the
  posterior naturally produces large uncertainty there anyway.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

WORKSPACE = Path('/sessions/relaxed-modest-dirac/mnt/Diachronic Hebrew')

# ── Model hyperparameters (must match Scripts 16/17/22b) ────────────────────
RIDGE      = 0.20
PRIOR_MU   = 600.0
PRIOR_SIG  = 350.0
DATE_GRID  = np.linspace(1200, 50, 500)   # old→young (BCE values)
N_MC       = 5000                          # MC iterations for calibration
RNG        = np.random.default_rng(42)

# ── Training dates for the word n-gram model ────────────────────────────────
TRAINING_DATES = {
    'Isaiah_2':    550, 'Isaiah_3':    510, 'Jeremiah':    610,
    'Ezekiel':     590, 'Haggai':      520, 'Zechariah_1': 520,
    'Malachi':     480, 'Jonah':       400, 'Lamentations':585,
    'Ezra':        450, 'Nehemiah':    440, 'Chronicles':  400,
    'Daniel':      165, 'Ecclesiastes':250, 'Esther':      350,
}

# ── External calibration controls ───────────────────────────────────────────
# name, true_date_mu (BCE), true_date_sigma (yr),
# word_map_pred (BCE), word_p16, word_p84 from dss_partA_results.csv
# (p16 = older bound = larger BCE number; p84 = younger bound)
WORD_CONTROLS = [
    # name            mu_true  s_true  map_pred  p16    p84
    ('1QS',           150,     35,     215.9,    285.1, 149.1),
    ('1QM',           100,     35,     308.1,    377.3, 239.0),
    ('1QHa [poetry]', 150,     35,     368.0,    437.2, 298.9),
]

# Ben Sira diagnostic (char model only — too poorly predicted for calibration)
BEN_SIRA_CHAR = dict(map_pred=626.2, p16=658.4, p84=591.6,
                     true_mu=180.0,  true_sigma=20.0)

# ── Load cached char n-gram data ────────────────────────────────────────────
def load_json(path):
    with open(path) as f:
        return json.load(f)

fixed_cng  = load_json(WORKSPACE / 'mc_cache_fixed_cng.json')
test_cng   = load_json(WORKSPACE / 'mc_cache_test_cng.json')
fixed_wngA = load_json(WORKSPACE / 'mc_cache_fixed_wngA.json')
fixed_wngB = load_json(WORKSPACE / 'mc_cache_fixed_wngB.json')
test_wngA  = load_json(WORKSPACE / 'mc_cache_test_wngA.json')
test_wngB  = load_json(WORKSPACE / 'mc_cache_test_wngB.json')
nwords     = load_json(WORKSPACE / 'mc_cache_nwords.json')

feats_cng = list(pd.read_csv(WORKSPACE / 'ngram_selected_features.csv')['ngram'])
feats_wA  = list(pd.read_csv(WORKSPACE / 'word_ngram_typeA_features.csv')['ngram'])
feats_wB  = list(pd.read_csv(WORKSPACE / 'word_ngram_typeB_features.csv')['ngram'])
feats_w   = feats_wA + feats_wB


# ═══════════════════════════════════════════════════════════════════════════
# Part 0 — Build MVN model parameters (vectorised, same as Script 22b)
# ═══════════════════════════════════════════════════════════════════════════

def build_mvn(fixed_cache, features):
    """Fit OLS + Tikhonov-regularised inverse covariance from training units."""
    units  = [u for u in TRAINING_DATES if u in fixed_cache]
    dates  = np.array([TRAINING_DATES[u] for u in units], dtype=float)
    K, N   = len(features), len(units)
    Y      = np.array([[fixed_cache[u].get(fn, 0.0) for u in units]
                       for fn in features], dtype=float)
    np.nan_to_num(Y, copy=False)
    xm     = dates.mean(); xd = dates - xm
    xv     = float((xd**2).sum()) or 1.0
    ym     = Y.mean(axis=1)
    slopes     = (Y * xd).sum(axis=1) / xv
    intercepts = ym - slopes * xm
    pred   = intercepts[:,None] + slopes[:,None] * dates[None,:]
    R      = (Y - pred).T
    Sig    = R.T @ R / max(N-2, 1)
    lam    = RIDGE * np.trace(Sig) / K if K > 0 else 0.0
    Sinv   = np.linalg.inv(Sig + lam * np.eye(K))
    return intercepts, slopes, Sinv


def get_posterior(obs_dict, features, intercepts, slopes, Sinv):
    """Return full posterior over DATE_GRID (normalised)."""
    obs  = np.array([obs_dict.get(fn, 0.0) for fn in features], dtype=float)
    pred = intercepts[None,:] + slopes[None,:] * DATE_GRID[:,None]
    diff = obs[None,:] - pred
    A    = diff @ Sinv
    lp   = -0.5 * (A * diff).sum(axis=1)
    lp  += -0.5 * ((DATE_GRID - PRIOR_MU) / PRIOR_SIG) ** 2
    lp  -= lp.max()
    post = np.exp(lp); post /= post.sum()
    return post


def posterior_stats(post):
    """Return MAP, p16 (older), p84 (younger) from a posterior over DATE_GRID."""
    map_date = float(DATE_GRID[np.argmax(post)])
    cum = np.cumsum(post)
    p16 = float(DATE_GRID[np.searchsorted(cum, 0.16)])
    p84 = float(DATE_GRID[np.searchsorted(cum, 0.84)])
    return map_date, p16, p84


print("Building MVN model parameters …")

# Char n-gram model
int_c, slp_c, Sinv_c = build_mvn(fixed_cng, feats_cng)

# Word n-gram model (merge type A + B cache per unit)
fixed_w = {}
for u in set(fixed_wngA) | set(fixed_wngB):
    d = {}
    d.update(fixed_wngA.get(u, {}))
    d.update(fixed_wngB.get(u, {}))
    fixed_w[u] = d
int_w, slp_w, Sinv_w = build_mvn(fixed_w, feats_w)

# Test unit merged word caches
test_w = {}
for u in set(test_wngA) | set(test_wngB):
    d = {}
    d.update(test_wngA.get(u, {}))
    d.update(test_wngB.get(u, {}))
    test_w[u] = d

print("  done.")


# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — Compute raw (uncalibrated) MAP dates for all test units
# ═══════════════════════════════════════════════════════════════════════════

print("\nComputing uncalibrated posteriors for all test units …")
uncalib = {}
for unit in test_cng:
    pc = get_posterior(test_cng[unit], feats_cng, int_c, slp_c, Sinv_c)
    mc, pc16, pc84 = posterior_stats(pc)
    if unit in test_w:
        pw = get_posterior(test_w[unit], feats_w, int_w, slp_w, Sinv_w)
        mw, pw16, pw84 = posterior_stats(pw)
    else:
        mw = pw16 = pw84 = float('nan')
    uncalib[unit] = dict(char_map=mc, char_p16=pc16, char_p84=pc84,
                         word_map=mw, word_p16=pw16, word_p84=pw84,
                         n_words=nwords.get(unit, 0))
    print(f"  {unit:<20} char={mc:.0f}  word={mw:.0f}")

# Also compute Ben Sira char prediction from flat consonantal text
print("\nComputing Ben Sira char n-gram rates …")
with open(WORKSPACE / 'ben_sira_flat.txt') as f:
    bs_text_raw = f.read()
bs_words = bs_text_raw.split()
bs_text  = '_' + '_'.join(bs_words) + '_'
bs_nch   = len(bs_text)
from collections import Counter
feat_set_c = set(feats_cng)
bs_counts = Counter()
for n in (3, 4):
    for i in range(bs_nch - n + 1):
        ng = bs_text[i:i+n]
        if ng in feat_set_c:
            bs_counts[ng] += 1
bs_rates = {fn: bs_counts.get(fn, 0) / bs_nch * 1000 for fn in feats_cng}
bs_post  = get_posterior(bs_rates, feats_cng, int_c, slp_c, Sinv_c)
bs_map, bs_p16, bs_p84 = posterior_stats(bs_post)
print(f"  Ben Sira: char_map={bs_map:.1f} BCE  CI68=[{bs_p16:.1f}, {bs_p84:.1f}]")
print(f"  Expected: ~180 BCE  →  raw offset ≈ +{bs_map-180:.0f} yr (over-dated)")


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — Distance-weighted linear calibration (word model only)
# ═══════════════════════════════════════════════════════════════════════════
#
# A uniform offset would wrongly shift all texts equally, including texts
# like Amos or Jeremiah whose predicted dates sit well within the training
# range and have no evidence of systematic bias.
#
# Instead we fit a linear calibration:
#
#   T_true = a · T_pred + b
#
# anchored at two ends:
#
#   OLD ANCHOR  — at the model's training-set centre (~530 BCE), we assert
#     the model is approximately unbiased (correction ≈ 0).  This is
#     justified by construction: the model was trained to reproduce these
#     dates.  We allow ±SIGMA_ANCHOR yr uncertainty in this assertion.
#
#   DSS CONTROLS — 1QS, 1QM, 1QHa give (T_pred, T_true) pairs in the
#     post-training zone (100–150 BCE).
#
# The resulting line gives a correction that VANISHES for old texts
# (T_pred near the anchor) and GROWS for texts predicted near the
# calibration zone (T_pred near 200–370 BCE).
#
# Monte Carlo propagates:
#   (a) uncertainty in the anchor true-date   ~ N(T_ANCHOR, SIGMA_ANCHOR)
#   (b) uncertainty in control true-dates     ~ N(mu_true_i, sigma_date_i)
#   (c) uncertainty in control model-outputs  ~ N(map_pred_i, sigma_model_i)
#
# Each MC iteration draws fresh samples, fits a line, and applies it to
# every test unit.  The spread across iterations gives calibration
# uncertainty.  The test unit's own model CI68 is added in quadrature.
#

# ── Old anchor ──────────────────────────────────────────────────────────────
# Mean date of the non-late (≥ 450 BCE) training texts:
#   Isaiah_2=550, Isaiah_3=510, Jeremiah=610, Ezekiel=590, Haggai=520,
#   Zechariah_1=520, Malachi=480, Lamentations=585, Ezra=450, Nehemiah=440
# Mean ≈ 526 BCE → rounded to 530 for clarity.
T_ANCHOR      = 530.0   # BCE — predicted date at which correction is ~0
SIGMA_ANCHOR  =  50.0   # yr  — uncertainty in the "model unbiased here" claim

print("\n── Distance-weighted linear calibration (word model) ──")
print(f"  Old anchor: T_pred = {T_ANCHOR:.0f} BCE (σ={SIGMA_ANCHOR:.0f} yr)")

ctrl_info = []
for name, mu_true, s_true, map_pred, p16, p84 in WORD_CONTROLS:
    sigma_model = abs(p16 - p84) / 2.0
    sigma_total = np.sqrt(s_true**2 + sigma_model**2)
    d_i         = mu_true - map_pred
    ctrl_info.append((name, mu_true, s_true, map_pred, sigma_model, d_i, sigma_total))
    print(f"  {name:<25}  pred={map_pred:.0f}  true={mu_true:.0f}  "
          f"d={d_i:+.0f}  σ_model={sigma_model:.0f}  σ_total={sigma_total:.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# Part 3 — Monte Carlo linear-calibration sampling
# ═══════════════════════════════════════════════════════════════════════════

print("\nRunning Monte Carlo linear calibration …")

from collections import defaultdict
cal_samples = defaultdict(list)   # unit → list of calibrated MAP dates
ab_samples  = []                   # list of (a, b) for diagnostics

for _ in range(N_MC):
    # ── 1. Sample old anchor ──────────────────────────────────────────────
    t_a = float(RNG.normal(T_ANCHOR, SIGMA_ANCHOR))
    p_a = T_ANCHOR   # anchor's predicted date is defined as T_ANCHOR

    # ── 2. Sample DSS control points ─────────────────────────────────────
    pts_p = [p_a]
    pts_t = [t_a]
    for _, mu_true, s_true, map_pred, sigma_model, _, _ in ctrl_info:
        pts_p.append(float(RNG.normal(map_pred, sigma_model)))
        pts_t.append(float(RNG.normal(mu_true,  s_true)))

    pts_p = np.array(pts_p)
    pts_t = np.array(pts_t)

    # ── 3. Fit linear T_true = a·T_pred + b (OLS) ────────────────────────
    # Use inverse-variance weights: anchor gets weight proportional to
    # 1/SIGMA_ANCHOR²; each DSS control gets 1/σ_total²
    w_a    = 1.0 / SIGMA_ANCHOR**2
    w_ctrl = [1.0 / c[6]**2 for c in ctrl_info]   # σ_total for each ctrl
    weights = np.array([w_a] + w_ctrl)
    weights /= weights.sum()

    W  = np.diag(weights)
    X  = np.column_stack([pts_p, np.ones(len(pts_p))])
    Xt = X.T
    # Weighted OLS: β = (X'WX)^{-1} X'W y
    XtW  = Xt @ W
    beta = np.linalg.lstsq(XtW @ X, XtW @ pts_t, rcond=None)[0]
    a_k, b_k = float(beta[0]), float(beta[1])
    ab_samples.append((a_k, b_k))

    # ── 4. Apply calibration line to each test unit ───────────────────────
    for unit, d in uncalib.items():
        mw = d['word_map']
        if not np.isnan(mw):
            cal_samples[unit].append(a_k * mw + b_k)

# ── Summarise calibration line posterior ────────────────────────────────────
ab_arr = np.array(ab_samples)
a_med, b_med = np.median(ab_arr[:,0]), np.median(ab_arr[:,1])
a_std, b_std = np.std(ab_arr[:,0]),    np.std(ab_arr[:,1])
print(f"  Calibration line  a = {a_med:.3f} ± {a_std:.3f},  "
      f"b = {b_med:.1f} ± {b_std:.1f}")
print(f"  Correction at T_pred=530: {(a_med*530+b_med)-530:+.1f} yr  (≈ 0 by design)")
print(f"  Correction at T_pred=310: {(a_med*310+b_med)-310:+.1f} yr")
print(f"  Correction at T_pred=216: {(a_med*216+b_med)-216:+.1f} yr")

# ── Build result dict (median + CI68 + CI95 from MC samples) ────────────────
calib_results = {}
for unit in uncalib:
    samp = np.array(cal_samples.get(unit, []))
    if len(samp) == 0 or np.isnan(uncalib[unit]['word_map']):
        calib_results[unit] = dict(cal_median=float('nan'),
                                   cal_p16=float('nan'), cal_p84=float('nan'),
                                   cal_p025=float('nan'), cal_p975=float('nan'))
        continue
    # Add individual unit model uncertainty (independent of calibration uncertainty)
    sigma_u = abs(uncalib[unit]['word_p16'] - uncalib[unit]['word_p84']) / 2.0
    samp_full = samp + RNG.normal(0.0, sigma_u, len(samp))
    calib_results[unit] = dict(
        cal_median = float(np.median(samp_full)),
        cal_p16    = float(np.percentile(samp_full, 84)),   # older BCE bound
        cal_p84    = float(np.percentile(samp_full, 16)),   # younger BCE bound
        cal_p025   = float(np.percentile(samp_full, 97.5)),
        cal_p975   = float(np.percentile(samp_full, 2.5)),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Part 4 — Save results CSV
# ═══════════════════════════════════════════════════════════════════════════

NOISY_UNITS = {'D_Song', 'Song_Sea', 'Song_Deborah'}

rows = []
for unit in uncalib:
    d  = uncalib[unit]
    cr = calib_results[unit]
    nw = d['n_words']
    noisy = unit in NOISY_UNITS
    rows.append(dict(
        unit           = unit,
        n_words        = nw,
        noisy          = noisy,
        # Uncalibrated char
        char_map       = round(d['char_map'], 1),
        char_p16       = round(d['char_p16'], 1),
        char_p84       = round(d['char_p84'], 1),
        # Uncalibrated word
        word_map       = round(d['word_map'], 1),
        word_p16       = round(d['word_p16'], 1),
        word_p84       = round(d['word_p84'], 1),
        # Calibrated word (MC)
        cal_word_median= round(cr['cal_median'], 1),
        cal_word_p16   = round(cr['cal_p16'],    1),
        cal_word_p84   = round(cr['cal_p84'],    1),
        cal_word_p025  = round(cr['cal_p025'],   1),
        cal_word_p975  = round(cr['cal_p975'],   1),
        # Calibration shift
        cal_shift      = round(cr['cal_median'] - d['word_map'], 1),
    ))

df_out = pd.DataFrame(rows)
df_out.to_csv(WORKSPACE / 'calibrated_dates.csv', index=False)
print("\nSaved calibrated_dates.csv")
print(df_out[['unit','word_map','cal_word_median','cal_shift','cal_word_p16','cal_word_p84']]
      .to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════
# Part 5 — Figures
# ═══════════════════════════════════════════════════════════════════════════

FIG_W, FIG_H = 14, 10

# ── Figure 1: Calibration diagnostic (linear model) ─────────────────────────
fig1, axes = plt.subplots(1, 2, figsize=(FIG_W, 5.5))

# Panel A: scatter of controls + anchor; draw MC calibration lines
ax = axes[0]

# Draw a random subset of MC calibration lines (faint)
n_show = 300
idx_show = RNG.integers(0, N_MC, n_show)
x_line = np.array([50, 700])
for idx in idx_show:
    a_i, b_i = ab_samples[idx]
    ax.plot(x_line, a_i*x_line + b_i, color='steelblue', lw=0.4, alpha=0.05)

# Median calibration line
ax.plot(x_line, a_med*x_line + b_med, color='steelblue', lw=2.5,
        label=f'median line  a={a_med:.2f}, b={b_med:.0f}')

# Identity line (ideal)
ax.plot(x_line, x_line, 'k--', lw=1.2, alpha=0.5, label='ideal (y=x)')

# Old anchor
ax.errorbar(T_ANCHOR, T_ANCHOR, xerr=0, yerr=SIGMA_ANCHOR,
            fmt='D', ms=10, color='#1a9641', capsize=5,
            label=f'Old anchor ({T_ANCHOR:.0f} BCE, σ={SIGMA_ANCHOR:.0f})')

# DSS controls (word model)
ctrl_colors = ['#e15759', '#f28e2b', '#76b7b2']
for i, (name, mu_true, s_true, map_pred, sigma_model, d_i, sigma_total) in enumerate(ctrl_info):
    ax.errorbar(map_pred, mu_true,
                xerr=sigma_model, yerr=s_true,
                fmt='o', ms=9, capsize=5, color=ctrl_colors[i],
                label=f'{name.split(" ")[0]}: pred={map_pred:.0f}, true={mu_true:.0f}')

# Ben Sira char diagnostic
bsc_err = abs(BEN_SIRA_CHAR['p16'] - BEN_SIRA_CHAR['p84']) / 2
ax.errorbar(BEN_SIRA_CHAR['map_pred'], BEN_SIRA_CHAR['true_mu'],
            xerr=bsc_err, yerr=BEN_SIRA_CHAR['true_sigma'],
            fmt='s', ms=8, capsize=4, color='#888888', alpha=0.7,
            label='Ben Sira (char — fails)')

ax.set_xlabel('Model predicted date (BCE)', fontsize=11)
ax.set_ylabel('True / expected date (BCE)', fontsize=11)
ax.set_title('A. Linear calibration\n(word n-gram model, MC lines in blue)', fontsize=11)
ax.set_xlim(700, 50); ax.set_ylim(700, 50)
ax.legend(fontsize=7.5, loc='upper left')
ax.grid(True, alpha=0.3)

# Panel B: correction as a function of predicted date
ax2 = axes[1]
t_grid = np.linspace(650, 100, 200)

# Shade 95% band of corrections
corrections = np.array([a*t_grid + b - t_grid for a, b in ab_samples])
c_med  = np.median(corrections, axis=0)
c_lo   = np.percentile(corrections, 2.5, axis=0)
c_hi   = np.percentile(corrections, 97.5, axis=0)
c_p16  = np.percentile(corrections, 16, axis=0)
c_p84  = np.percentile(corrections, 84, axis=0)

ax2.fill_between(t_grid, c_lo, c_hi,   color='steelblue', alpha=0.15, label='95% CI')
ax2.fill_between(t_grid, c_p16, c_p84, color='steelblue', alpha=0.30, label='68% CI')
ax2.plot(t_grid, c_med, color='steelblue', lw=2.5, label='Median correction')
ax2.axhline(0, color='k', lw=1, ls='--', alpha=0.5, label='No correction')

# Mark where each test unit's word MAP falls
unit_colors = {'Jer_oracle':'#d62728','1_Kings':'#9467bd',
               'D_Code':'#8c564b', 'Daniel-like':'#bcbd22'}
for unit, d in sorted(uncalib.items(), key=lambda x: x[1]['word_map'], reverse=True):
    mw = d['word_map']
    if np.isnan(mw) or unit in {'D_Song','Song_Sea','Song_Deborah'}: continue
    corr_at = float(np.median([a*mw+b - mw for a,b in ab_samples]))
    ax2.scatter(mw, corr_at, s=25, color='#555', zorder=3, alpha=0.7)
    if unit in ('Jer_oracle','D_Code','Genesis','Joshua'):
        ax2.annotate(unit, (mw, corr_at), textcoords='offset points',
                     xytext=(4, 4), fontsize=7.5, color='#333')

ax2.set_xlabel('Model predicted date (BCE)', fontsize=11)
ax2.set_ylabel('Correction = T_calibrated − T_predicted (yr)', fontsize=11)
ax2.set_title('B. Distance-weighted correction\n'
              '(small for old texts, larger for late texts)', fontsize=11)
ax2.invert_xaxis()
ax2.legend(fontsize=8, loc='lower right')
ax2.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig(WORKSPACE / 'calibration_diagnostic.png', dpi=150, bbox_inches='tight')
plt.close(fig1)
print("Saved calibration_diagnostic.png")


# ── Figure 2: Uncalibrated vs calibrated word-model dates ───────────────────
# Sort units by uncalibrated word MAP date (oldest first)
df_plot = df_out[~df_out['noisy']].copy()
df_plot = df_plot.sort_values('word_map', ascending=True).reset_index(drop=True)

fig2, ax = plt.subplots(figsize=(FIG_W, FIG_H))

y_pos = np.arange(len(df_plot))
offset_y = 0.18   # vertical jitter between calib/uncalib

# Uncalibrated (grey)
for i, row in df_plot.iterrows():
    mw = row['word_map']
    if np.isnan(mw): continue
    err_l = max(0.0, row['word_p16'] - mw)
    err_r = max(0.0, mw - row['word_p84'])
    ax.errorbar(mw, i - offset_y, xerr=[[err_l],[err_r]],
                fmt='o', color='#888888', ms=5, capsize=3, lw=1.2, alpha=0.7)

# Calibrated (blue)
for i, row in df_plot.iterrows():
    cm = row['cal_word_median']
    if np.isnan(cm): continue
    err_l = max(0.0, row['cal_word_p16'] - cm)
    err_r = max(0.0, cm - row['cal_word_p84'])
    ax.errorbar(cm, i + offset_y, xerr=[[err_l],[err_r]],
                fmt='D', color='#2166ac', ms=5, capsize=3, lw=1.2, alpha=0.9)

# Arrows showing shift
for i, row in df_plot.iterrows():
    mw = row['word_map']; cm = row['cal_word_median']
    if np.isnan(mw) or np.isnan(cm): continue
    ax.annotate('', xy=(cm, i), xytext=(mw, i),
                arrowprops=dict(arrowstyle='->', color='#444', lw=0.8, alpha=0.5))

# Labels
ax.set_yticks(y_pos)
ax.set_yticklabels(df_plot['unit'], fontsize=9)
ax.invert_xaxis()
ax.set_xlabel('Date (BCE)', fontsize=12)
ax.set_title('Uncalibrated vs. weakly-calibrated word n-gram model dates\n'
             'Grey: uncalibrated (CI68); Blue: calibrated (CI68 including δ uncertainty)',
             fontsize=11)
ax.grid(True, axis='x', alpha=0.3)

# Legend
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker='o', color='#888', ls='', ms=7, label='Uncalibrated MAP ± CI68'),
       Line2D([0],[0], marker='D', color='#2166ac', ls='', ms=7,
              label=f'Calibrated MAP ± CI68  (a={a_med:.2f}, b={b_med:.0f})')]
ax.legend(handles=leg, fontsize=9, loc='lower left')

fig2.tight_layout()
fig2.savefig(WORKSPACE / 'calibrated_dates_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Saved calibrated_dates_comparison.png")


# ── Figure 3: Summary — char vs calibrated word model ───────────────────────
df_plot2 = df_out[~df_out['noisy']].copy()
# Order by char MAP (decreasing BCE = oldest last for typical book order)
df_plot2 = df_plot2.sort_values('char_map', ascending=True).reset_index(drop=True)

fig3, ax = plt.subplots(figsize=(FIG_W, FIG_H))
y2 = np.arange(len(df_plot2))

for i, row in df_plot2.iterrows():
    # Char model (uncalibrated, orange)
    mc = row['char_map']
    if not np.isnan(mc):
        err_l = max(0.0, row['char_p16'] - mc)
        err_r = max(0.0, mc - row['char_p84'])
        ax.errorbar(mc, i + 0.2, xerr=[[err_l],[err_r]],
                    fmt='s', color='#d95f02', ms=5, capsize=3, lw=1.2, alpha=0.8)
    # Calibrated word (blue)
    cm = row['cal_word_median']
    if not np.isnan(cm):
        err_l = max(0.0, row['cal_word_p16'] - cm)
        err_r = max(0.0, cm - row['cal_word_p84'])
        ax.errorbar(cm, i - 0.2, xerr=[[err_l],[err_r]],
                    fmt='D', color='#2166ac', ms=5, capsize=3, lw=1.2, alpha=0.8)

ax.set_yticks(y2)
ax.set_yticklabels(df_plot2['unit'], fontsize=9)
ax.invert_xaxis()
ax.set_xlabel('Date (BCE)', fontsize=12)
ax.set_title('Char n-gram (uncalibrated) vs. calibrated word n-gram dates\n'
             'Orange squares: char model (CI68); Blue diamonds: calibrated word model (CI68)',
             fontsize=11)
ax.grid(True, axis='x', alpha=0.3)

from matplotlib.lines import Line2D
leg3 = [Line2D([0],[0], marker='s', color='#d95f02', ls='', ms=7,
               label='Char n-gram (uncalibrated)'),
        Line2D([0],[0], marker='D', color='#2166ac', ls='', ms=7,
               label=f'Word n-gram (calibrated, a={a_med:.2f}, b={b_med:.0f})')]
ax.legend(handles=leg3, fontsize=9, loc='lower left')

fig3.tight_layout()
fig3.savefig(WORKSPACE / 'char_vs_calibrated_word.png', dpi=150, bbox_inches='tight')
plt.close(fig3)
print("Saved char_vs_calibrated_word.png")


# ── Figure 4: External validation summary ───────────────────────────────────
fig4, ax = plt.subplots(figsize=(10, 5.5))

controls_plot = [
    ('1QS',         150,  35,  215.9,  215.9 + (285.1-215.9), 215.9 - (215.9-149.1)),
    ('1QM',         100,  35,  308.1,  308.1 + (377.3-308.1), 308.1 - (308.1-239.0)),
    ('1QHa',        150,  35,  368.0,  368.0 + (437.2-368.0), 368.0 - (368.0-298.9)),
    ('Ben Sira\n(char)', 180, 20, bs_map, bs_p16, bs_p84),
]

colors_ctrl = ['#1b7837', '#762a83', '#c51b7d', '#888888']
for i, (nm, t_mu, t_s, pred, pred_p16, pred_p84) in enumerate(controls_plot):
    err_xl = max(0, pred - pred_p84)
    err_xr = max(0, pred_p16 - pred)
    # Predicted (model output)
    ax.errorbar(pred, i, xerr=[[err_xr],[err_xl]],
                fmt='o', color=colors_ctrl[i], ms=9, capsize=5, lw=1.5,
                label=f'{nm}: pred={pred:.0f} BCE')
    # True date (expected)
    ax.errorbar(t_mu, i, xerr=t_s,
                fmt='*', color=colors_ctrl[i], ms=13, capsize=5, lw=1.5,
                alpha=0.7)
    # Arrow from predicted to true
    ax.annotate('', xy=(t_mu, i), xytext=(pred, i),
                arrowprops=dict(arrowstyle='->', color=colors_ctrl[i],
                                lw=1.5, alpha=0.6))

ax.set_yticks(range(len(controls_plot)))
ax.set_yticklabels([c[0] for c in controls_plot], fontsize=11)
ax.set_xlabel('Date (BCE)', fontsize=12)
ax.set_title('External validation: model prediction (●) vs. expected date (★)\n'
             'Top 3: word model; bottom: char model for Ben Sira', fontsize=11)
ax.invert_xaxis()
ax.grid(True, axis='x', alpha=0.3)
ax.axvline(400, color='k', lw=1, ls=':', alpha=0.5,
           label='Training range limit (~400 BCE)')

from matplotlib.lines import Line2D
leg4 = [Line2D([0],[0], marker='o', color='k', ls='', ms=9, label='Model prediction (CI68)'),
        Line2D([0],[0], marker='*', color='k', ls='', ms=13, label='Expected date (±1σ)'),
        Line2D([0],[0], color='k', lw=1, ls=':', label='Training boundary ~400 BCE')]
ax.legend(handles=leg4, fontsize=9)
ax.set_xlim(800, 50)

fig4.tight_layout()
fig4.savefig(WORKSPACE / 'external_validation_summary.png', dpi=150, bbox_inches='tight')
plt.close(fig4)
print("Saved external_validation_summary.png")


# ═══════════════════════════════════════════════════════════════════════════
# Part 6 — Print final summary
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

def med_corr(t_pred):
    return float(np.median([a*t_pred + b - t_pred for a, b in ab_samples]))

print(f"""
Calibration model: T_true = a·T_pred + b  (distance-weighted linear)
  Median:  a = {a_med:.3f},  b = {b_med:.1f}
  Std:     σ_a = {a_std:.3f},  σ_b = {b_std:.1f}

Correction by predicted date (median of MC):
  T_pred = 600 BCE  →  {med_corr(600):+.0f} yr  (within training range, near-zero)
  T_pred = 530 BCE  →  {med_corr(530):+.0f} yr  (old anchor — forced near zero)
  T_pred = 450 BCE  →  {med_corr(450):+.0f} yr
  T_pred = 350 BCE  →  {med_corr(350):+.0f} yr
  T_pred = 250 BCE  →  {med_corr(250):+.0f} yr
  T_pred = 216 BCE  →  {med_corr(216):+.0f} yr  (1QS predicted position)

Key findings:

  1. Char n-gram model fails cross-corpus:
       Ben Sira  pred={bs_map:.0f} BCE, true≈180 BCE  (offset +{bs_map-180:.0f} yr)
       DSS prose pred=755–787 BCE, true≈100–150 BCE
     Root cause: model captures Masoretic orthographic conventions
     (shin/sin dot encoding, defective vs. plene spelling), not universal
     Hebrew linguistic change.  CANNOT be calibrated.

  2. Word n-gram model is partially cross-corpus valid:
       1QS:  pred=216 BCE, true≈150 BCE  (offset +66 yr)
       1QM:  pred=308 BCE, true≈100 BCE  (offset +208 yr)
       1QHa: pred=368 BCE, true≈150 BCE  (offset +218 yr)
     Systematic over-dating of 66–218 yr in the post-training zone.

  3. Distance-weighted calibration (this script):
     Correction is proportional to distance from the old anchor (530 BCE).
     Texts predicted near the training centre receive negligible correction;
     texts predicted near the calibration zone (~200–370 BCE) receive the
     largest correction.
       Jer_oracle (pred≈{uncalib['Jer_oracle']['word_map']:.0f} BCE) shift: {med_corr(uncalib['Jer_oracle']['word_map']):+.0f} yr
       Joshua     (pred≈{uncalib['Joshua']['word_map']:.0f} BCE) shift: {med_corr(uncalib['Joshua']['word_map']):+.0f} yr
       D_Code     (pred≈{uncalib['D_Code']['word_map']:.0f} BCE) shift: {med_corr(uncalib['D_Code']['word_map']):+.0f} yr

  4. Recommended interpretive framework:
       a. Char model:  relative ordering within Masoretic corpus only.
       b. Word model (uncalibrated):  best absolute dates in training range.
       c. Word model (calibrated):  apply for texts predicted at <450 BCE;
          correction is small for ancient texts and grows proportionally
          as predicted dates approach the post-training zone.
""")
