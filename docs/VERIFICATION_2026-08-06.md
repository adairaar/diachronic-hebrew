# Independent verification — 2026-08-06

Re-derived from `feature_matrix_v2.csv`, `corpus_manifest_v2.json`,
`hb_vi_dating.csv` (results_v2) and `greek/results/hard_register_dating.csv`.
This supersedes the diagnostic sections of `RERUN_STATUS.md`.

Scope note: Kings/Chronicles work excluded at user's direction (2026-08-06).

---

## Finding 1 — Holdout validation is circular in BOTH pipelines

Every holdout is dated using **its own scholarly date as the prior mean**, with
the scholarly sigma as the prior width.

- Hebrew: `hierarchical_bayes/01_hb_vi_dating_v2.py` line 542 —
  `pbce = row_data["date_bce"]`, `psig = row_data["date_sigma"]`, passed
  straight into `do_date(...)` as `prior_bce`/`prior_sigma_bce`.
- Greek: `greek/10_hard_register_dating.py` lines 404–408 —
  `prior_mean = float(row["date_ce"])`, `prior_sigma = float(row["date_sigma"])`,
  passed into `mdl.map_and_ci(x, prior_mean, prior_sigma)`.

The audit fix at Hebrew line 339 (`jer_row["holdout"] = True`) correctly removed
Jer_oracle from *training*. It did not address the prior. Training leakage was
the smaller of the two problems.

### Consequence

| | reported holdout MAE | likelihood-only MAE |
|---|---|---|
| Hebrew (n=4) | 8.1 yr | **103.7 yr** |
| Greek (n=5) | 8.7 yr | **120.7 yr** |

README currently claims HB-VI holdout MAE = 16.8 yr and Greek recovery "within
~30 yr". Per-text:

| text | established | reported MAP | err | likelihood-only | err |
|---|---|---|---|---|---|
| Habakkuk | 605 BCE | 606 | 1 | 792 | 187 |
| Haggai | 520 BCE | 520 | 0 | 640 | 120 |
| Daniel | 167 BCE | 172 | 5 | 205 | 38 |
| Jer_oracle | 605 BCE | 631 | 26 | 675 | 70 |
| Polybius | 160 BCE | 148 | 12 | 29 CE | 189 |
| Mark | 70 CE | 69 | 1 | 16 | 54 |
| Matthew | 85 CE | 80 | 5 | −104 | 189 |
| Luke | 120 CE | 120 | 0 | 118 | 2 |
| Diogenes Laërtius | 230 CE | 256 | 26 | 400 | 170 |

The README's own note — that the unconditioned likelihood is "80–345 yr off for
these same texts" — is currently attributed to the absence of register
conditioning. That attribution is wrong. The gap is the prior.

## Finding 2 — How little the data contributes

Posterior sd is an almost deterministic function of prior sd. Fraction of
posterior precision coming from the linguistic features:

| text | prior sd | data weight |
|---|---|---|
| Haggai | 10 yr | 2.8% |
| Zechariah_1 | 10 yr | 3.4% |
| Daniel | 10 yr | 5.5% |
| Amos | 20 yr | 12.9% |
| Habakkuk | 20 yr | 15.7% |
| Jer_oracle | 30 yr | 27.6% |
| Joel | 150 yr | 90.4% |
| Zechariah_2 | 150 yr | 93.1% |

Median across 25 dated texts: posterior sd / prior sd = **0.851**; the data
removes a median of **27.6%** of prior variance. Median |MAP − prior mean| =
**10.4 yr**.

The three near-perfect "recoveries" (Haggai, Zechariah_1, Daniel) are exactly
the three texts with the tightest priors. That is the whole effect.

## Finding 3 — The likelihood is overconfident ~2×

Decomposing `1/post_var − 1/prior_var` gives an implied likelihood sigma of
**48.6 yr** (IQR 43.3–53.1), remarkably stable across texts. But the
likelihood-only holdout error is ~104 yr (Hebrew) / ~121 yr (Greek).

Claimed ±49, delivers ±104–121. This is the mechanism behind the documented
coverage failure (`calibration_v3.csv`: 68% intervals achieve 52–61%;
manuscript reports 32%).

`multiverse.csv` already identified the driver: chance-correlated features get
large fitted beta and small residual sigma, and the MVN likelihood weights by
1/sigma, so adding noise features *shrinks* the HDI (P_source 329 → 285 → 246).

## Finding 4 — LOO filter has zero rejection power (confirmed)

`loo_diagnostic.csv`: `loo_survival_rate = 1.00` at every alpha; FDP identical
with and without the filter. `screen_sweep_v2.csv`: retained feature count is
identical at loo50, loo68, loo75, loo90 and loo100 — the gate never fires at
any setting. Confirmed as documented.

---

## Implication for the manuscript

The two load-bearing validation claims — Hebrew holdout recovery and Greek
cross-language generalisation — do not survive. Both reduce to reading the
prior back out. Everything downstream that cites holdout accuracy as evidence
the method works needs rewriting, not renumbering.

What is *not* affected: the specification-curve / multiverse result. That
compares targets across model specifications and does not depend on holdout
accuracy. `multiverse.csv` remains the strongest surviving evidence
(P_source stable 285–351 across seven specs; D_source 100–491, not identified).

## Required fix

Date every holdout under an agnostic prior (the Mode-B `N(575, 400²)` already
used for out-of-sample targets at line 555), never its own scholarly date.
Report `lik_only` as the headline validation number. Expect Hebrew holdout MAE
~104 yr and Greek ~121 yr, and rewrite the validation claims to match.
