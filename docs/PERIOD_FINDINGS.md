# Can diachronic change be detected in dated Biblical Hebrew?

Honest leave-one-out evaluation, 2026-08-06. Supersedes all holdout numbers in
the current manuscript.

## Method

25 texts with scholarly dates from `corpus_manifest_v2.json`; 53 usable
features from `feature_matrix_v2.csv`.

For each text in turn: refit everything on the other 24 — feature screening
(Spearman, alpha = 0.05), standardisation, and regression — then predict the
held-out text under an **agnostic prior**. The held-out text never influences
its own prediction, and its scholarly date is never used as a prior.

Two pre-specified model families, **both reported**:

- **generative** — invert per-feature OLS, flat prior over a 900–100 BCE grid.
  This is the manuscript's MLE-MVN family.
- **ridge** — ridge regression of date on screened features, lambda by inner LOO.

Significance by permutation: shuffle the date vector and re-run the *entire*
pipeline (2000 draws), so the null absorbs selection effects exactly as the
real run does.

Periods: Pre-exilic > 586 > Exilic > 539 > Persian > 332 > Hellenistic.

---

## Result 1 — The ordering signal is real

| family | Spearman rho (LOO) | null | p |
|---|---|---|---|
| generative | **+0.575** | −0.069 ± 0.421 | 0.016 |
| ridge | **+0.494** | −0.203 ± 0.418 | 0.024 |

Replicated across two independent model families. Hebrew morphosyntactic
features do carry information about *relative* chronological ordering.

## Result 2 — Absolute dating does not work

| predictor | LOO MAE |
|---|---|
| predict corpus mean (503 BCE) for every text | **137.3 yr** |
| predict corpus median (520 BCE) for every text | 135.4 yr |
| generative model | 156.2 yr |
| ridge model | 120.9 yr |

The manuscript's own model family performs **worse than a constant**. Ridge
beats the constant by 16 years. Any MAE figure must be read against 137, not
against zero.

## Result 3 — Period assignment is marginal at best

| metric | generative | ridge | baseline | best p |
|---|---|---|---|---|
| 4-period exact | 0.400 | 0.480 | 0.360 | 0.037 |
| 4-period within one | 0.840 | 0.840 | — | 0.028 |
| pre-exilic vs post-exilic | 0.720 | 0.760 | 0.680 | 0.070 |

The classic CBH/LBH binary is the **weakest** result, not the strongest —
neither family separates pre- from post-exilic significantly.

Confusion is structured, not random (ridge):

```
                 Pre-ex  Exilic  Persia  Hellen
  Pre-exilic         5       2       1       0
  Exilic             1       1       1       1
  Persian            2       2       5       0
  Hellenistic        0       0       3       1
```

- Pre-exilic is the only well-recovered class.
- **Exilic is unrecoverable.** It is a 47-year window against ~130 yr of
  irreducible error; 4 texts, 1 correct. It should be merged or dropped.
- Hellenistic collapses into Persian: Daniel (167 BCE) predicts 438, Ecclesiastes
  (250) predicts 450.

## Result 4 — Multiple comparisons

Ten tests (5 metrics x 2 families), highly correlated. Under Benjamini-Hochberg:

- at q = 0.05: **nothing survives**
- at q = 0.10: generative/within-one, ridge/4-period, ridge/MAE, ridge/within-one

The rho result has the smallest p-values (0.016, 0.024) and is the only finding
replicated across both families, but it does not clear a strict BH correction on
all ten. The defensible statement is that the ordinal signal replicates; no
single accuracy claim is individually robust.

## Result 5 — Direct confirmation of the leakage finding

Haggai and Zechariah_1 carry the tightest priors (sigma = 10) and were the
paper's most impressive recoveries. Under honest LOO with an agnostic prior:

| text | scholarly | paper's reported MAP | honest LOO (generative) | error |
|---|---|---|---|---|
| Haggai | 520 BCE | 520 (err 0) | 186 | **334 yr** |
| Zechariah_1 | 518 BCE | 517 (err 1) | 282 | **236 yr** |

---

## What this supports

**Yes:** Hebrew texts can be placed in relative chronological order at better
than chance (rho ~ 0.5, replicated).

**No:** they cannot be assigned absolute dates more accurately than guessing the
middle of the corpus, and they cannot be sorted into historical periods with
useful reliability at n = 25.

This is a coherent and defensible paper, and it engages the Rezetko / Young /
Ehrensvard / Naaijer critique directly on its own terms — with a
pre-registered, permutation-tested, leakage-free design rather than an
assertion. It is not the paper currently drafted.

## Recommended next steps

1. Merge Exilic into an adjacent class, or adopt a 3-period scheme. The 47-year
   window cannot be resolved and its presence depresses every 4-way metric.
2. Report ordinal metrics (Spearman, Kendall, pairwise-ordering accuracy) as
   primary; demote absolute dates to a secondary, explicitly-caveated table.
3. Add pairwise-ordering accuracy — "given two texts, does the model order them
   correctly?" It matches what the data supports and has a clean 50% baseline.
4. Re-run the Greek pipeline under the same LOO-with-agnostic-prior design. Its
   n = 63 may support conclusions Hebrew's n = 25 cannot, which would make the
   cross-language section a genuine strength rather than a repeat of the same
   artifact.
