# Compression fix, n-grams, and chunk-size search

35 configurations: 6 chunk sizes × 3 feature sets × with/without de-attenuation.
Leave-one-book-out throughout, ridge in dual form, hyperparameter and
calibration slope both fitted by an inner leave-one-book-out on training books
only.

## 1. The compression fix works, and it is close to free

De-attenuation = fit `truth ~ a + s·pred` on out-of-fold training predictions
(s > 1 expands the range) and apply to the held-out book.

| effect, averaged over all 17 paired configs | |
|---|---|
| change in MAE | **+0.2 yr** (no cost) |
| change in range coverage | **+0.12** (0.49 → 0.64 typical) |

On the base feature set it improves *both*: MAE down 1.2–7.5 yr and coverage up
0.04–0.18. On the n-gram sets it trades ~2–4 yr of MAE for ~0.13 of coverage.

Best coverage achieved: **0.685**, predicting 647–211 BCE against a true range
of 760–167. Before any of this work the model spanned 358–627.

## 2. N-grams help, but only in combination

689 n-gram features added: character 3- and 4-grams over the consonantal
skeleton, POS trigrams, closed-class lexeme bigrams, clause-relation trigrams,
phrase-function bigrams. Vocabularies chosen by raw frequency only, so
selection cannot leak the label.

| feature set | best MAE | best rho |
|---|---|---|
| base (579) | 96.6 | **+0.666** |
| n-gram alone (689) | 95.8 | +0.604 |
| both (1268) | **91.2** | +0.621 |

N-grams alone are worse than the base set on rank correlation. Combined they
give the single best MAE in the project (91.2 yr at ~400 w). The gain is real
but modest — this is not the order-of-magnitude change the feature count might
suggest.

## 3. Chunk size has no sharp optimum

| chunk | best MAE | best rho | best coverage |
|---|---|---|---|
| ~200 w | 98.5 | +0.591 | 0.64 |
| ~300 w | 99.1 | +0.576 | 0.64 |
| ~400 w | **91.2** | +0.638 | 0.66 |
| ~500 w | 97.4 | **+0.666** | 0.66 |
| ~700 w | 97.5 | +0.609 | 0.67 |
| ~1000 w | 94.0 | +0.621 | **0.685** |

400–500 w is best for accuracy and rank correlation; 1000 w for range
coverage. The curve is flat — chunk size is not where the leverage is.

## Where this leaves the project

Best numbers achieved, against the starting point:

| metric | book-level (start) | now | constant baseline |
|---|---|---|---|
| MAE | 120.9 / 156.2 yr | **91.2 yr** | 137.3 yr |
| Spearman rho | +0.494 / +0.575 | **+0.666** | — |
| pairwise ordering | 67.8 / 69.8% | **72.9%** | 50% |
| range coverage | ~0.50 | **0.685** | — |

Every metric improved substantially. MAE is now 34% better than a constant,
where the original book-level generative model was *worse* than a constant.

**But the asymmetry is not fixed.** Across all 35 configurations the model
makes at most 4 confident pre-exilic calls and gets at most 2 right. The
persistent failures are the same texts:

| unit | true | best-config pred | err |
|---|---|---|---|
| Daniel | 167 | 414–477 | 247–310 |
| Zechariah 9–14 | 350 | 556–614 | 206–264 |
| Hosea | 745 | 530–547 | 198–215 |
| Isaiah 1–39 | 720 | 516–524 | 196–204 |

The model still cannot reach 167 BCE or 760 BCE. Coverage of 0.685 is a large
improvement on 0.50 but it is not 1.0, and the residual compression lands
exactly on the endpoints that any pre-exilic or Hellenistic claim depends on.
