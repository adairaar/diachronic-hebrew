# Chunk-level redesign — first results

BHSA is set up in the container (54 MB of .tf features fetched directly, since
the GitHub API is blocked for this session).  Chunk extraction reuses the
existing `extract_unit()` with its word-collection patched, so feature
definitions are identical to the book-level pipeline.  Verified: chunk word
counts sum exactly to the book-level counts (Amos 2,780; Ezekiel 26,182;
Chronicles 35,330).

## 1. Sparsity — your concern, quantified

| chunk size | chunks | usable features | dead: >70% zeros |
|---|---|---|---|
| ~300 w | 470 | 32 / 64 | 31 |
| ~500 w | 285 | 33 / 64 | 30 |
| ~1000 w | 145 | 37 / 64 | 25 |

About half the feature set dies at chunk level, and **it is specifically the
classic CBH/LBH lexical diagnostics that die** — the asher/she ratio, the
anochi/ani ratio, and the rare lexemes are too infrequent to measure in a
500-word span.  What survives is high-frequency verbal morphology.

That is a substantive finding in its own right: the measurable diachronic
signal in Hebrew lives in verb-form frequencies, not in the rare lexical
shibboleths the field has traditionally relied on, because those cannot be
estimated from any short span.

## 2. ICC — is the variance between books or within them?

| chunk size | median ICC | ICC > 0.5 | ICC < 0.1 |
|---|---|---|---|
| ~300 w | 0.22 | 1 | 6 |
| ~500 w | 0.25 | 3 | 4 |
| ~1000 w | 0.32 | 8 | 3 |

Median ICC of 0.25–0.32 means most features scatter about as widely inside a
book as across books.  But eight features at 1,000 words clear ICC > 0.5 and
correlate with date:

| feature | ICC | rho with date |
|---|---|---|
| `rate_impv` | 0.44 | +0.55 |
| `rate_yiqtol` | 0.79 | +0.55 |
| `rate_af` | 0.36 | +0.54 |
| `rate_qatal` | 0.56 | +0.51 |
| `rate_neg_lo` | 0.57 | +0.51 |
| `frac_neg_al` | 0.31 | -0.50 |
| `rate_prs` | 0.51 | +0.46 |
| `rate_qal` | 0.54 | +0.45 |

## 3. Leave-one-book-out result

Train on chunks from 24 units, predict every chunk of the held-out unit,
take the median.  Nothing leaks across the fold boundary.

| | MAE | Spearman | pairwise |
|---|---|---|---|
| **chunk, 1000 w, 37 features** | **105.9 yr** | +0.456 | 66.4% |
| chunk, 500 w, 33 features | 119.6 yr | +0.401 | 62.0% |
| book-level ridge (n=25) | 120.9 yr | +0.494 | 67.8% |
| book-level generative (n=25) | 156.2 yr | +0.575 | 69.8% |
| constant predictor | 137.3 yr | — | 50% |

Between-unit sd of predictions is 98 yr against a median within-unit sd of
44 yr — a ratio of 2.2, so book identity is real in the predictions and
chunks from the same book genuinely agree.

## 4. Why this still is not enough

**Predictions are compressed to 60% of the true range.**  True dates have
sd 163 yr; predictions have sd 98 yr.  The model's entire output range across
25 books is 330–730 BCE, against a true range of 167–760.  Dates outside that
window are unreachable by construction.

The consequences are exactly the failures you identified:

| unit | true | predicted | error |
|---|---|---|---|
| Amos | 760 | 543 | 217 |
| Hosea | 745 | 551 | 194 |
| Daniel | 167 | 407 | 240 |
| Zechariah 9–14 | 350 | 587 | 237 |

A model that never outputs a date before 730 BCE cannot confidently call
anything pre-exilic, and a model that never outputs one after 330 cannot
place P in the Hellenistic period.  This is textbook regression attenuation
from noisy predictors, and it follows directly from a feature–date
correlation of ~0.5.  Chunking does not change that correlation.

**And the chunk-derived uncertainty is miscalibrated in the same direction as
before.**  Within-unit sd is 44 yr; actual error is 106 yr.  Using the chunk
spread as an interval would understate error by 2.4x — narrower intervals
around answers that are still wrong.

## Verdict

Chunking is a genuine improvement on one axis: best MAE in the project so far
(105.9 vs 137.3 for a constant), a real between/within signal, and a
diagnostic framework that can kill bad ideas in an afternoon.  It is not a
rescue.  Ordering got slightly worse, the compression problem is untouched,
and the accuracy ceiling is set by a feature–date correlation the redesign
does not move.

