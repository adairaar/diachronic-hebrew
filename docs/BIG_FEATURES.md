# Expanded feature space — results

## What was built

634 candidate features per chunk, extracted directly from BHSA node features,
none of them inherited from the hand-picked 64:

| family | count | content |
|---|---|---|
| LEX | 250 | top-250 lexeme rates per 1,000 words |
| POS | ~210 | part-of-speech unigrams, bigrams, phrase-dependent POS |
| VERB | ~60 | tense, stem, and tense×stem cross rates and fractions |
| AGR | ~25 | person / number / gender / state / pronominal suffix |
| PHR | ~28 | phrase type and phrase function distributions |
| CLS | ~48 | clause relation rates, clause-type bigrams |
| STRUCT | 5 | phrase length, clause length, clauses per sentence, etc. |

579 survive the sparsity filter (std > 0, <20% missing).

## Power analysis — the number that governs this

Chunks inside a book share its date and are correlated, so chunk count is not
sample size. Effective n = n_chunks / (1 + (m̄−1)·ICC):

| chunk size | chunks | median ICC | design effect | **n_eff** | gain vs n=25 |
|---|---|---|---|---|---|
| ~300 w | 470 | 0.09 | 2.68 | **175** | 7.0× |
| ~500 w | 285 | 0.13 | 2.38 | **120** | 4.8× |
| ~1000 w | 145 | 0.20 | 1.96 | **74** | 3.0× |

Chunking buys a genuine 3–7× in effective sample size. It does not buy the
naive 6–19× the raw chunk counts suggest.

### Screening is not viable at this dimensionality

Against a **book-level** permutation null (the only permutation that respects
the design), at ~500 w:

| alpha | retained | expected under null | est. FDP |
|---|---|---|---|
| 0.001 | 214 | 100.0 | **0.47** |
| 0.01 | 280 | 154.8 | 0.55 |
| 0.05 | 341 | 222.2 | 0.65 |

Even at α = 0.001 nearly half the "significant" features are noise. With
p ≈ 580 against n_eff ≈ 120, hard selection cannot be made honest. Minimum
detectable |ρ| at 80% power, n_eff = 74: **0.32–0.45**.

**Conclusion: do not screen. Regularise.** Ridge and PLS take p > n without
incurring a selection penalty; hyperparameters chosen by inner
leave-one-book-out.

## Leave-one-book-out results

| model | MAE | Spearman ρ | pairwise | range coverage |
|---|---|---|---|---|
| **ridge, ~500 w** | **99.2 yr** | +0.596 (p=0.0017) | 70.5% | 0.50 |
| PLS, ~300 w | 99.8 | +0.555 | 69.2% | 0.55 |
| PLS, ~1000 w | 103.1 | +0.593 | 71.2% | **0.75** |
| ridge, ~1000 w | 103.8 | **+0.613 (p=0.0011)** | **71.5%** | 0.62 |
| — | | | | |
| book-level ridge (n=25, 64 feats) | 120.9 | +0.494 | 67.8% | — |
| book-level generative (n=25) | 156.2 | +0.575 | 69.8% | — |
| constant predictor | 137.3 | — | 50% | — |

Every metric improved. MAE down 18% against the best previous model and 28%
against a constant; ρ up from +0.575 to +0.613; pairwise from 69.8% to 71.5%;
p-values an order of magnitude smaller (0.001 vs 0.016).

## What is still broken

**Range compression.** Best-MAE model predicts only 358–627 BCE against a
true range of 167–760 — coverage 0.50. It cannot reach the endpoints, which
is why extreme texts stay wrong:

| unit | true | pred | err |
|---|---|---|---|
| Zechariah 9–14 | 350 | 613 | **263** |
| Isaiah 1–39 | 720 | 514 | 206 |
| Hosea | 745 | 547 | 198 |
| Amos | 760 | 586 | 174 |

PLS at ~1000 w is notably better here — coverage 0.75, predicting 224–743 —
at a cost of 4 yr MAE. That trade is probably worth taking, since a model
that cannot output early dates can never support a pre-exilic finding.

## Where the remaining headroom is

1. **More features.** 634 was a first pass. Character n-grams on the
   consonantal text, syntactic n-grams over clause structures, and a larger
   lexeme vocabulary are all untried and all cheap now that extraction works.
2. **Attack compression directly** rather than accepting it — variance
   calibration or inverse regression, evaluated on range coverage as a
   reported metric alongside MAE.
3. **Smaller chunks** raise n_eff (175 at ~300 w) but lower ICC. The optimum
   is somewhere in this range and has not been searched properly.
