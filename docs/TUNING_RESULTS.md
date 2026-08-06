# Hyperparameter tuning and the n-gram test under variance matching

Two questions, both answered negatively — which is useful.

## 1. Tuning α, bandwidth and λ by inner LOBO makes things worse

Selection criterion was Spearman ρ on inner out-of-fold predictions (ρ is
scale-invariant, so it selects for the property variance matching preserves;
selecting on MAE would favour the shrinkage that variance matching then undoes).
Grid: λ ∈ {10², 10³, 10⁴, 10⁵}, α ∈ {0, 0.5, 1.0, 1.5}, bandwidth ∈ {60, 120, 200}.

| config | MAE fixed | MAE tuned | ρ fixed | ρ tuned |
|---|---|---|---|---|
| 500w base | **120.8** | 125.9 | **+0.700** | +0.613 |
| 1000w both | **109.9** | 130.8 | **+0.650** | +0.590 |

Tuning lost on every configuration and every metric. The mechanism is visible
in the selected values: the modal α varies across configurations as 0.5, 0.0,
1.5 — it is being driven by noise, not signal. Selecting among 48 combinations
using 24 books overfits the selection criterion itself.

**This is worth stating in the paper.** At n = 25 books, hyperparameter tuning
is a source of overfitting rather than a remedy for it, and the honest move is
to fix defensible defaults a priori and say so.

## 2. More features hurt under variance matching

| feature set | p | MAE | ρ | coverage |
|---|---|---|---|---|
| base | 579 | **120.8** | **+0.700** | 1.08 |
| base + n-gram | 1268 | 135.4 | +0.595 | 1.18 |

The n-gram features helped when the objective was MAE with shrinkage intact
(they gave the project's best MAE, 91.2 yr). Under variance matching they hurt
on every axis. The reason is mechanical: variance matching amplifies whatever
scatter the model produces, so extra weakly-informative features are amplified
along with the signal, and coverage overshoots to 1.18.

Feature-set choice is therefore not independent of the calibration choice. That
interaction was not obvious in advance and is worth reporting.

## Recommended final model

**~500-word chunks, 579 base features, inverse-density weighted (α = 1.0,
bandwidth 90), ridge with λ by inner LOBO, variance-matched.**

| metric | value |
|---|---|
| Spearman ρ | **+0.700** (p = 0.0001) |
| pairwise ordering | **75.3%** |
| range coverage | **1.08** |
| predicted range | 719–126 BCE (true 760–167) |
| side-of-586 accuracy | **80%** (base rate 68%) |
| pre-exilic books placed pre-exilic | **6 / 8** |
| post-exilic books placed post-exilic | **14 / 17** |
| MAE | 120.8 yr |

Books are the unit of analysis and are independent, so the ordinary Spearman
p-value is valid here — no clustering correction is needed at book level.

Everything except λ is fixed a priori rather than tuned, which given the result
above is a feature of the design, not a shortcut.
